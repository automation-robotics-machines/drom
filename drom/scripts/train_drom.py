import os
import json
import torch
import argparse

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# Ensures CUDA operations are executed synchronously (easier debugging)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Prevents HDF5 file-locking issues (common on network filesystems)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# DROM imports
import drom.demos
from drom import trainer
from drom.models import (
    UNET_DIM_MULTS,
    TemporalUnet,
    TaskModelIdentity,
)
from drom.trainer import (
    get_dataset,
    get_model,
    get_loss,
    get_summary,
)
from drom.trainer.trainer import get_num_epochs
from drom.torch_utils.seed import fix_random_seed
from drom.torch_utils.torch_utils import get_torch_device
from drom.utils.train_utils import get_exp_dir
from drom.utils.wandb_utils import start_wandb
from drom.utils.utils import select_tasks

# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def main(args):
    """
    Main training entry point.

    Loads configuration, creates dataset, builds models, loss functions,
    and launches the DROM training loop.
    """
    # -------------------------------------------------------------------------
    # 1. Load configuration
    # -------------------------------------------------------------------------
    if args.config is None:
        raise ValueError("You must provide a --config JSON file.")

    with open(args.config, "r") as f:
        config = json.load(f)

    train_cfg = config["train"]
    exp_cfg   = config["experiment"]
    wandb_cfg = config.get("wandb_options", {})

    # Create experiment directories (log, ckpt, video, time)
    log_dir, ckpt_dir, video_dir, time_dir = get_exp_dir(config, resume=False)

    # -------------------------------------------------------------------------
    # 2. Start WandB
    # -------------------------------------------------------------------------
    start_wandb(
        wandb_options=wandb_cfg,
        config=config,
        run_name=exp_cfg.get("name", "DROM-Experiment"),
    )

    # -------------------------------------------------------------------------
    # 3. Fix random seed
    # -------------------------------------------------------------------------
    fix_random_seed(train_cfg["seed"])

    # -------------------------------------------------------------------------
    # 4. Resolve compute device
    # -------------------------------------------------------------------------
    device = get_torch_device(
        device="cuda:0" if train_cfg["cuda"] else "cpu"
    )
    tensor_args = dict(device=device, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # 5. Determine conditioning type (attention vs. none)
    # -------------------------------------------------------------------------
    context_enabled = (train_cfg["conditioning_type"] == "attention")

    # -------------------------------------------------------------------------
    # 6. Resolve dataset paths and tasks
    # -------------------------------------------------------------------------
    # dataset_dir = os.path.join(drom.demos.demos_root, "hdf5")
    dataset_dir = drom.demos.demo_root
    config["train"]["mimicgen"] = False
    if args.mimicgen:
        config["train"]["mimicgen"] = True
        dataset_dir = os.path.join(dataset_dir, "mimicgen")

    # tasks = "all" → gather all folder names
    tasks = select_tasks(train_cfg.get("tasks"), dataset_dir)
    print(f"Selected tasks: {tasks}")

    # -------------------------------------------------------------------------
    # 7. Build dataset + dataloaders
    # -------------------------------------------------------------------------
    train_subset, train_loader, val_subset, val_loader = get_dataset(
        dataset_class="TrajectoryDataset",
        batch_size=train_cfg["batch_size"],
        results_dir=time_dir,
        save_indices=True,
        tensor_args=tensor_args,
        tasks=tasks,
        number_of_trajs=train_cfg["number_of_trajs_per_task"],
        horizon=train_cfg["horizon"],
        context=context_enabled,
        dataset_dir=dataset_dir,
        normalizer=train_cfg["normalizer"]
    )

    dataset = train_subset.dataset

    # -------------------------------------------------------------------------
    # 8. Build context model
    # -------------------------------------------------------------------------
    if context_enabled:
        context_dim = dataset.context_dim

        # Identity mapping for conditioning embedding
        context_model = TaskModelIdentity(
            input_dim=context_dim,
            out_dim=context_dim,
        )
        conditioning_dim = context_model.out_dim
        config["train"]["context_dim"] = context_dim
        config["train"]["conditioning_dim"] = conditioning_dim
    else:
        context_model = None
        conditioning_dim = 4  # default placeholder

    # -------------------------------------------------------------------------
    # 9. Derive number of epochs
    # -------------------------------------------------------------------------
    num_epochs = get_num_epochs(
        train_cfg["num_train_steps"],
        train_cfg["batch_size"],
        len(dataset),
    )

    # -------------------------------------------------------------------------
    # 10. Build model configuration dictionaries
    # -------------------------------------------------------------------------
    diffusion_cfg = dict(
        variance_schedule=train_cfg["variance_schedule"],
        n_diffusion_steps=train_cfg["n_diffusion_steps"],
        predict_epsilon=train_cfg["predict_epsilon"],
        context_model=context_model,
        loss_type=train_cfg["diffusion_loss"],
    )

    unet_cfg = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.horizon,
        unet_input_dim=train_cfg["unet_input_dim"],
        dim_mults=UNET_DIM_MULTS[train_cfg["unet_dim_mults_option"]],
        conditioning_type=train_cfg["conditioning_type"],
        conditioning_embed_dim=conditioning_dim,
        self_attention=train_cfg["self_attention"],
    )

    # -------------------------------------------------------------------------
    # 11. Instantiate Gaussian Diffusion Model
    # -------------------------------------------------------------------------
    model = get_model(
        model_class="GaussianDiffusionModel",
        model=TemporalUnet(**unet_cfg),
        tensor_args=tensor_args,
        **diffusion_cfg,
        **unet_cfg,
    )

    # -------------------------------------------------------------------------
    # 12. Loss + Summary
    # -------------------------------------------------------------------------
    loss_fn = get_loss("GaussianDiffusionLoss")
    # summary_fn = get_summary(summary_class="SummaryTrajectoryGeneration")
    summary_fn = None  # summary temporarily disabled

    # -------------------------------------------------------------------------
    # 13. Save the config as a json file
    # -------------------------------------------------------------------------
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # -------------------------------------------------------------------------
    # 14. Launch Training Loop
    # -------------------------------------------------------------------------
    trainer.train(
        model=model,
        train_dataloader=train_loader,
        train_subset=train_subset,
        val_dataloader=val_loader,
        val_subset=val_subset,
        epochs=num_epochs,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        video_dir=video_dir,
        time_dir=time_dir,
        summary_fn=summary_fn,
        lr=train_cfg["lr"],
        loss_fn=loss_fn,
        val_loss_fn=loss_fn,
        steps_til_summary=train_cfg["steps_til_summary"],
        steps_til_checkpoint=train_cfg["steps_til_ckpt"],
        clip_grad=True,
        use_ema=train_cfg["use_ema"],
        use_amp=train_cfg["use_amp"],
        debug=True,
        tensor_args=tensor_args,
    )


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DROM model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON config file containing training settings.",
    )
    parser.add_argument(
        "--mimicgen",
        action="store_true",
        help="Whether to use MimicGen datasets or not.",
    )

    args = parser.parse_args()
    main(args)
