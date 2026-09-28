import os
import wandb

def start_wandb(wandb_options, config=None, run_name=None):
    """
    Initialize Weights & Biases using user-specified options.

    Args:
        wandb_options: dict with keys ["wandb_mode", "wandb_entity", "wandb_project"]
        config: full experiment configuration (logged to wandb)
        run_name: optional name for wandb run
    """
    required = ["wandb_mode", "wandb_entity", "wandb_project"]
    for key in required:
        if key not in wandb_options:
            raise KeyError(f"Missing key '{key}' in wandb_options")

    # Set wandb runtime mode (online/offline/disabled)
    os.environ["WANDB_MODE"] = wandb_options["wandb_mode"]

    return wandb.init(
        project=wandb_options["wandb_project"],
        entity=wandb_options["wandb_entity"],
        config=config,
        name=run_name,
    )