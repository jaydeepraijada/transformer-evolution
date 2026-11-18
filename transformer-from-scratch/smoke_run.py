"""
Smoke run (roadmap step 4): overfit 100 OPUS Books en-it sentence pairs to prove the wiring.

Baseline is the paper's setup (warmup + inverse-sqrt LR schedule, post-norm, base model size,
dropout 0.1, label smoothing 0.1, Adam betas (0.9, 0.98)). Then one variable at a time is switched
to our alternative, plus a run with both switched (our current default config).

Because label smoothing (eps = 0.1) puts probability on every vocab word, the smoothed training
loss can never reach 0. "Near-zero" is therefore checked with:
  - plain (unsmoothed) cross-entropy, evaluated in eval mode -> should approach 0
  - teacher-forced token accuracy                            -> should approach 100%
  - greedy-decoding exact match on all 100 pairs             -> proves the causal mask doesn't leak

Each invocation saves a self-contained folder experiments/smoke_run/<timestamp>/:
  README.md                  summary table, what was run, how to reproduce
  meta.json                  command, seed, device, library versions, git commit, data selection
  log.txt                    full console output
  pairs.json                 the exact 100 sentence pairs used
  tokenizers/{en,it}.json    the tokenizers used
  code/                      snapshot of every .py file + git diff of uncommitted changes
  plots/*.png                loss / CE / accuracy / LR curves, all variants overlaid
  variants/<name>/
      config.json            fully resolved config (incl. model + optimizer hyperparameters)
      metrics.json           per-step train loss + lr, periodic eval loss / CE / accuracy
      predictions.json       greedy decode of all 100 pairs vs. reference
      tensorboard/           TensorBoard logs

Model weights are not saved (~240 MB per variant); the run is seeded, so re-running reproduces it
(up to small floating-point nondeterminism on GPU / MPS).

Usage (from transformer-from-scratch/):  ../.venv/bin/python smoke_run.py [--steps 2000]
"""
import argparse
import inspect
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from config import get_config
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from train import get_or_build_tokenizer, get_model, get_paper_lr_lambda

HERE = Path(__file__).resolve().parent

N_PAIRS = 100
MIN_TOKENS = 3
MAX_TOKENS = 30      # only keep pairs where both sides are 3..30 tokens, keeps the run fast
BATCH_SIZE = 10      # 100 pairs / 10 = 10 steps per epoch
EVAL_EVERY = 50
LOG_EVERY = 5
SEED = 0
LABEL_SMOOTHING = 0.1
PAPER_BETAS = (0.9, 0.98)
ADAM_EPS = 1e-9

# (slug, description, overrides on top of get_config()).
# First = the paper; the next two flip exactly one variable; the last flips both (our default).
VARIANTS = [
    ('paper',              'paper (paper_lr + post-norm)', {'use_paper_lr_schedule': True,  'pre_norm': False}),
    ('fixed_lr_postnorm',  'fixed_lr + post-norm',         {'use_paper_lr_schedule': False, 'pre_norm': False}),
    ('paper_lr_prenorm',   'paper_lr + pre-norm',          {'use_paper_lr_schedule': True,  'pre_norm': True}),
    ('ours',               'fixed_lr + pre-norm (ours)',   {'use_paper_lr_schedule': False, 'pre_norm': True}),
]


class Tee:
    """Write console output to both the terminal and log.txt."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def git(*args):
    try:
        return subprocess.run(['git', *args], cwd=HERE, capture_output=True, text=True, check=True).stdout
    except Exception as e:  # not a git repo / git missing
        return f'<unavailable: {e}>'


def package_versions():
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for pkg in ('torch', 'datasets', 'tokenizers', 'tensorboard', 'matplotlib', 'numpy', 'tqdm'):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def label_smoothing_floor(vocab_size, eps):
    # PyTorch's smoothed target: (1 - eps) on the true token + eps / V spread over all V tokens.
    # The lowest possible loss is the entropy of that target distribution.
    p_true = 1 - eps + eps / vocab_size
    p_other = eps / vocab_size
    return -(p_true * math.log(p_true) + (vocab_size - 1) * p_other * math.log(p_other))


def select_pairs(ds_raw, config, tokenizer_src, tokenizer_tgt):
    # first N_PAIRS pairs (in dataset order) whose both sides are MIN_TOKENS..MAX_TOKENS tokens
    pairs, indices, max_len = [], [], 0
    for idx, item in enumerate(ds_raw):
        src_len = len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids)
        tgt_len = len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids)
        if MIN_TOKENS <= src_len <= MAX_TOKENS and MIN_TOKENS <= tgt_len <= MAX_TOKENS:
            pairs.append(item)
            indices.append(idx)
            max_len = max(max_len, src_len, tgt_len)
            if len(pairs) == N_PAIRS:
                break
    return pairs, indices, max_len + 2  # + [SOS]/[EOS]


def resolved_config(config):
    # everything that determines this variant: our config + build_transformer's defaults
    # (get_model doesn't pass N, h, d_ff, dropout, so they come from the function signature)
    model_defaults = {k: v.default for k, v in inspect.signature(build_transformer).parameters.items()
                      if v.default is not inspect.Parameter.empty}
    model_hparams = {**model_defaults, 'd_model': config['d_model'], 'pre_norm': config['pre_norm'],
                     'src_seq_len': config['seq_len'], 'tgt_seq_len': config['seq_len']}
    if config['use_paper_lr_schedule']:
        optim = {'type': 'Adam', 'base_lr': 1.0, 'betas': list(PAPER_BETAS), 'eps': ADAM_EPS,
                 'schedule': 'd_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)',
                 'warmup_steps': config['warmup_steps']}
    else:
        optim = {'type': 'Adam', 'lr': config['lr'], 'betas': [0.9, 0.999], 'eps': ADAM_EPS,
                 'schedule': 'constant'}
    return {
        'config': config,
        'model': model_hparams,
        'optimizer': optim,
        'loss': {'type': 'CrossEntropyLoss', 'label_smoothing': LABEL_SMOOTHING, 'ignore_index': '[PAD]'},
        'training': {'batch_size': BATCH_SIZE, 'shuffle': True, 'seed': SEED,
                     'eval_every': EVAL_EVERY, 'log_every': LOG_EVERY},
    }


@torch.no_grad()
def evaluate(model, full_batch, loss_fn, ce_fn, vocab_size, pad_id, device):
    model.eval()
    enc_in = full_batch['encoder_input'].to(device)
    dec_in = full_batch['decoder_input'].to(device)
    enc_mask = full_batch['encoder_mask'].to(device)
    dec_mask = full_batch['decoder_mask'].to(device)
    label = full_batch['label'].to(device)

    logits = model.project(model.decode(model.encode(enc_in, enc_mask), enc_mask, dec_in, dec_mask))
    flat_logits, flat_label = logits.view(-1, vocab_size), label.view(-1)
    smoothed = loss_fn(flat_logits, flat_label).item()
    ce = ce_fn(flat_logits, flat_label).item()

    real = flat_label != pad_id
    acc = (flat_logits.argmax(-1)[real] == flat_label[real]).float().mean().item()
    model.train()
    return smoothed, ce, acc


@torch.no_grad()
def greedy_decode_all(model, full_batch, tokenizer_tgt, seq_len, device):
    # Batched greedy decoding: no teacher forcing, the model only sees its own previous outputs.
    # If the causal mask leaked future tokens, teacher-forced accuracy would look perfect but
    # this would fail.
    model.eval()
    sos_id = tokenizer_tgt.token_to_id('[SOS]')
    eos_id = tokenizer_tgt.token_to_id('[EOS]')
    enc_in = full_batch['encoder_input'].to(device)
    enc_mask = full_batch['encoder_mask'].to(device)
    encoder_output = model.encode(enc_in, enc_mask)

    dec = torch.full((enc_in.size(0), 1), sos_id, dtype=torch.int64, device=device)  # (B, 1)
    for _ in range(seq_len - 1):
        dec_mask = causal_mask(dec.size(1)).to(device)  # (1, t, t), broadcasts over batch and heads
        out = model.decode(encoder_output, enc_mask, dec, dec_mask)  # (B, t, d_model)
        next_token = model.project(out[:, -1]).argmax(-1, keepdim=True)  # (B, 1)
        dec = torch.cat([dec, next_token], dim=1)

    predictions = []
    for row in dec[:, 1:].tolist():  # drop [SOS], cut at the first [EOS]
        predictions.append(row[:row.index(eos_id)] if eos_id in row else row)
    model.train()
    return predictions


def run_variant(slug, name, overrides, base_config, pairs, seq_len, tokenizer_src, tokenizer_tgt,
                steps, device, out_dir):
    # values the smoke run sets itself override config.py, so config.json shows what was actually used
    config = {**base_config, **overrides, 'seq_len': seq_len, 'batch_size': BATCH_SIZE,
              'num_epochs': None, 'preload': None}
    vdir = out_dir / 'variants' / slug
    vdir.mkdir(parents=True, exist_ok=True)
    resolved = resolved_config(config)
    resolved['training']['steps'] = steps
    (vdir / 'config.json').write_text(json.dumps({'name': name, 'overrides': overrides, **resolved}, indent=2))

    torch.manual_seed(SEED)  # same init and same batch order for every variant

    ds = BilingualDataset(pairs, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    full_batch = next(iter(DataLoader(ds, batch_size=len(ds))))  # all 100 pairs, for eval

    vocab_size = tokenizer_tgt.get_vocab_size()
    pad_id = tokenizer_src.token_to_id('[PAD]')
    model = get_model(config, tokenizer_src.get_vocab_size(), vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if config['use_paper_lr_schedule']:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=PAPER_BETAS, eps=ADAM_EPS)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_paper_lr_lambda(config['d_model'], config['warmup_steps']))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=ADAM_EPS)
        scheduler = None

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=LABEL_SMOOTHING).to(device)
    ce_fn = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)  # unsmoothed, for the "near zero" check
    writer = SummaryWriter(str(vdir / 'tensorboard'))

    history = {'step': [], 'train_loss': [], 'lr': []}
    evals = {'step': [], 'loss_smoothed': [], 'ce': [], 'acc': []}
    diverged_at = None
    start = time.time()

    def do_eval(step):
        smoothed, ce, acc = evaluate(model, full_batch, loss_fn, ce_fn, vocab_size, pad_id, device)
        for key, val in (('step', step), ('loss_smoothed', smoothed), ('ce', ce), ('acc', acc)):
            evals[key].append(val)
        writer.add_scalar('eval/loss_smoothed', smoothed, step)
        writer.add_scalar('eval/ce_unsmoothed', ce, step)
        writer.add_scalar('eval/token_acc', acc, step)
        return smoothed, ce, acc

    print(f'  {n_params / 1e6:.1f}M parameters')
    do_eval(0)
    step = 0
    model.train()
    while step < steps and diverged_at is None:
        for batch in loader:
            encoder_input = batch['encoder_input'].to(device)  # (B, Seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, Seq_len)
            encoder_mask = batch['encoder_mask'].to(device)    # (B, 1, 1, Seq_len)
            decoder_mask = batch['decoder_mask'].to(device)    # (B, 1, Seq_len, Seq_len)
            label = batch['label'].to(device)                  # (B, Seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)  # (B, Seq_len, vocab)
            loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))

            lr_now = optimizer.param_groups[0]['lr']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            step += 1

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                diverged_at = step
                print(f'  diverged at step {step}')
                break
            if step % LOG_EVERY == 0 or step == 1:
                history['step'].append(step)
                history['train_loss'].append(loss_val)
                history['lr'].append(lr_now)
                writer.add_scalar('train/loss', loss_val, step)
                writer.add_scalar('train/lr', lr_now, step)
            if step % EVAL_EVERY == 0:
                smoothed, ce, acc = do_eval(step)
                if step % (EVAL_EVERY * 4) == 0:
                    print(f'  step {step:5d}  train {loss_val:.3f}  eval-ce {ce:.4f}  acc {acc:6.1%}  '
                          f'lr {lr_now:.2e}  ({time.time() - start:.0f}s)')
            if step >= steps:
                break
    train_seconds = time.time() - start

    # final greedy decode on all 100 pairs
    predictions = greedy_decode_all(model, full_batch, tokenizer_tgt, seq_len, device)
    targets = [tokenizer_tgt.encode(p['translation'][config['lang_tgt']]).ids for p in pairs]
    pred_records = [{
        'i': i,
        'src': p['translation'][config['lang_src']],
        'tgt': p['translation'][config['lang_tgt']],
        'pred': tokenizer_tgt.decode(pred),
        'exact_match': pred == tgt,
    } for i, (p, pred, tgt) in enumerate(zip(pairs, predictions, targets))]
    exact = sum(r['exact_match'] for r in pred_records)
    writer.close()

    summary = {
        'name': name, 'slug': slug, 'overrides': overrides, 'n_params': n_params,
        'steps_completed': step, 'diverged_at': diverged_at, 'train_seconds': round(train_seconds, 1),
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_eval_loss_smoothed': evals['loss_smoothed'][-1],
        'final_eval_ce': evals['ce'][-1],
        'final_eval_acc': evals['acc'][-1],
        'greedy_exact_match': exact, 'n_pairs': len(pairs),
        # first eval step where unsmoothed CE < 0.05 (i.e. "memorised"); None if never
        'step_ce_below_0.05': next((s for s, c in zip(evals['step'], evals['ce']) if c < 0.05), None),
    }
    (vdir / 'metrics.json').write_text(json.dumps({'summary': summary, 'train': history, 'eval': evals}, indent=1))
    (vdir / 'predictions.json').write_text(json.dumps(pred_records, indent=1, ensure_ascii=False))

    print(f'  done in {train_seconds:.0f}s  final eval-ce {summary["final_eval_ce"]:.4f}  '
          f'acc {summary["final_eval_acc"]:.1%}  greedy exact match {exact}/{len(pairs)}')
    return {'summary': summary, 'train': history, 'eval': evals}


def plot(results, floor, steps, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ('train_loss.png', 'Training loss (label-smoothed, eps=0.1)', 'train', 'train_loss', True),
        ('eval_ce_unsmoothed.png', 'Eval cross-entropy, no smoothing (should go to ~0)', 'eval', 'ce', True),
        ('eval_token_accuracy.png', 'Eval token accuracy (teacher-forced)', 'eval', 'acc', False),
        ('learning_rate.png', 'Learning rate', 'train', 'lr', True),
    ]
    fig_all, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax_all, (fname, title, source, key, log_y) in zip(axes.flat, panels):
        fig, ax = plt.subplots(figsize=(8, 5))
        for target_ax in (ax, ax_all):
            for r in results:
                target_ax.plot(r[source]['step'], r[source][key], label=r['summary']['name'], linewidth=1.4)
            if key == 'train_loss':
                target_ax.axhline(floor, color='gray', linestyle='--', linewidth=1,
                                  label=f'label-smoothing floor ({floor:.2f})')
            if log_y:
                target_ax.set_yscale('log')
            if key == 'acc':
                target_ax.set_ylim(0, 1.02)
            target_ax.set_title(title)
            target_ax.set_xlabel('step')
            target_ax.set_xlim(0, steps)
            target_ax.grid(alpha=0.3)
            target_ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=130)
        plt.close(fig)
    fig_all.suptitle(f'Smoke run: overfit {N_PAIRS} OPUS Books en-it pairs, base Transformer')
    fig_all.tight_layout()
    fig_all.savefig(plot_dir / 'all_curves.png', dpi=130)
    plt.close(fig_all)


def write_readme(out_dir, meta, results):
    rows = []
    for r in results:
        s = r['summary']
        rows.append(f"| {s['name']} | `{json.dumps(s['overrides'])}` | {s['final_train_loss']:.3f} | "
                    f"{s['final_eval_ce']:.4f} | {s['final_eval_acc']:.1%} | "
                    f"{s['greedy_exact_match']}/{s['n_pairs']} | {s['step_ce_below_0.05'] or 'never'} | "
                    f"{'step ' + str(s['diverged_at']) if s['diverged_at'] else 'no'} | {s['train_seconds']:.0f}s |")
    readme = f"""# Smoke run {meta['timestamp']}

Overfit {meta['data']['n_pairs']} OPUS Books {meta['data']['lang_pair']} sentence pairs with the base
Transformer to prove the wiring. Baseline = the paper's setup; other variants flip one variable at a time
(the last flips both = our default config).

- Steps per variant: {meta['steps']} (batch size {meta['batch_size']}, {meta['data']['n_pairs'] // meta['batch_size']} steps/epoch)
- seq_len: {meta['data']['seq_len']}, vocab: src {meta['data']['vocab_src']} / tgt {meta['data']['vocab_tgt']}
- Label-smoothing loss floor (lowest possible training loss with eps=0.1): **{meta['label_smoothing_floor']:.3f}**
- Device: {meta['device']}, torch {meta['packages']['torch']}, seed {meta['seed']}
- Git commit: `{meta['git']['commit'].strip()}` (uncommitted changes: {'yes, see code/uncommitted.diff' if meta['git']['dirty'] else 'none'})

## Results

| Variant | Overrides | Final train loss | Eval CE (unsmoothed) | Token acc | Greedy exact match | Step CE < 0.05 | Diverged | Time |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

![all curves](plots/all_curves.png)

## Files

- `meta.json`: command, environment, git, data selection
- `pairs.json`: the exact sentence pairs (with their index in the dataset)
- `tokenizers/`: tokenizers used (`<lang>.json`)
- `code/`: snapshot of all `.py` files at run time, plus `uncommitted.diff`
- `variants/<name>/config.json`: fully resolved config for that variant
- `variants/<name>/metrics.json`: summary + every logged train step + every eval
- `variants/<name>/predictions.json`: greedy decode of every pair vs. reference
- `variants/<name>/tensorboard/`: view with `../.venv/bin/tensorboard --logdir {out_dir.relative_to(HERE)}/variants`
- `plots/`: loss / CE / accuracy / learning-rate curves
- `log.txt`: console output

## Reproduce

```
cd transformer-from-scratch
git checkout {meta['git']['commit'].strip()}   # then apply code/uncommitted.diff, or copy code/*.py over
{meta['command']}
```
"""
    (out_dir / 'README.md').write_text(readme)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out_dir = HERE / 'experiments' / 'smoke_run' / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / 'log.txt', 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)

    # snapshot code + git state before anything runs
    (out_dir / 'code').mkdir()
    for py in sorted(HERE.glob('*.py')):
        shutil.copy2(py, out_dir / 'code' / py.name)
    status = git('status', '--porcelain', '.')
    (out_dir / 'code' / 'uncommitted.diff').write_text(git('diff', 'HEAD', '--', '.'))

    base_config = get_config()
    device = pick_device()
    print(f'Smoke run {timestamp}  ->  {out_dir}')
    print(f'Using device {device}')

    ds_raw = load_dataset('Helsinki-NLP/opus_books', f'{base_config["lang_src"]}-{base_config["lang_tgt"]}', split='train')
    tokenizer_src = get_or_build_tokenizer(base_config, ds_raw, base_config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(base_config, ds_raw, base_config['lang_tgt'])
    (out_dir / 'tokenizers').mkdir()
    tokenizer_src.save(str(out_dir / 'tokenizers' / f'{base_config["lang_src"]}.json'))
    tokenizer_tgt.save(str(out_dir / 'tokenizers' / f'{base_config["lang_tgt"]}.json'))

    pairs, indices, seq_len = select_pairs(ds_raw, base_config, tokenizer_src, tokenizer_tgt)
    (out_dir / 'pairs.json').write_text(json.dumps(
        [{'dataset_index': i, **p['translation']} for i, p in zip(indices, pairs)], indent=1, ensure_ascii=False))
    floor = label_smoothing_floor(tokenizer_tgt.get_vocab_size(), LABEL_SMOOTHING)
    print(f'{len(pairs)} pairs, seq_len {seq_len}, vocab src {tokenizer_src.get_vocab_size()} '
          f'tgt {tokenizer_tgt.get_vocab_size()}, label-smoothing loss floor {floor:.3f}')

    meta = {
        'timestamp': timestamp,
        'command': f'../.venv/bin/python smoke_run.py --steps {args.steps}',
        'argv': sys.argv,
        'steps': args.steps, 'batch_size': BATCH_SIZE, 'seed': SEED,
        'eval_every': EVAL_EVERY, 'log_every': LOG_EVERY,
        'label_smoothing': LABEL_SMOOTHING, 'label_smoothing_floor': floor,
        'device': str(device),
        'python': sys.version, 'platform': platform.platform(), 'packages': package_versions(),
        'git': {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(status.strip()), 'status': status},
        'data': {
            'dataset': 'Helsinki-NLP/opus_books', 'split': 'train',
            'lang_pair': f'{base_config["lang_src"]}-{base_config["lang_tgt"]}',
            'selection': f'first {N_PAIRS} pairs in dataset order with {MIN_TOKENS}..{MAX_TOKENS} tokens on both sides',
            'n_pairs': len(pairs), 'seq_len': seq_len,
            'vocab_src': tokenizer_src.get_vocab_size(), 'vocab_tgt': tokenizer_tgt.get_vocab_size(),
        },
        'base_config': base_config,
        'variants': [{'slug': s, 'name': n, 'overrides': o} for s, n, o in VARIANTS],
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    results = []
    for slug, name, overrides in VARIANTS:
        print(f'\n=== {name} ===')
        results.append(run_variant(slug, name, overrides, base_config, pairs, seq_len,
                                   tokenizer_src, tokenizer_tgt, args.steps, device, out_dir))
        # refresh plots + README after every variant so partial results survive an interruption
        plot(results, floor, args.steps, out_dir / 'plots')
        write_readme(out_dir, meta, results)

    print(f'\nSaved everything to {out_dir}')
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == '__main__':
    main()
