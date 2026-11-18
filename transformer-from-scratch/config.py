from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4, # only used when use_paper_lr_schedule is False
        # True  -> paper's schedule: lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5),
        #          with Adam betas (0.9, 0.98) as in the paper
        # False -> fixed learning rate "lr" for the whole run
        "use_paper_lr_schedule": False,
        "warmup_steps": 4000,
        # True  -> pre-norm:  x + Dropout(Sublayer(LayerNorm(x)))  (modern default, more stable)
        # False -> post-norm: LayerNorm(x + Dropout(Sublayer(x)))  (the original paper)
        "pre_norm": True,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
