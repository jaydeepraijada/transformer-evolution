# From Scratch

Implementations of transformer and LLM architectures written from scratch in PyTorch, one paper at a time,
starting with *Attention Is All You Need* (2017) and working forward through the developments that followed.

Each implementation lives in its own folder and is written by hand to understand the mechanism, not to be a
production library.

## Implementations

| Folder | Paper | Status |
|---|---|---|
| [`transformer-from-scratch/`](transformer-from-scratch/) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — encoder–decoder, sinusoidal positions, multi-head attention | In progress |

## Layout

```
<name>-from-scratch/
    model.py        # the architecture
    train.py        # training loop
    dataset.py      # data loading and tokenisation
    config.py       # hyperparameters
    README.md       # what this one adds over the previous implementation
```

New implementations go in a new top-level folder. Anything shared across several of them (a training loop,
evaluation helpers) moves into a `common/` folder once it is actually needed by more than one.

## Conventions

- Written by hand. Editor autocomplete is disabled for Python in `.vscode/settings.json` on purpose.
- Shape comments on every tensor operation, in `(batch, seq_len, d_model)` form.
- Each folder's README notes what changed relative to the implementation before it.
