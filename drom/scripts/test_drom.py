import os
import json
import argparse
from math import ceil
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from drom.models import TemporalUnet, UNET_DIM_MULTS, TaskModelIdentity
from drom.models.diffusion_models.sample_functions import ddpm_sample_fn
from drom.trainer import get_dataset, get_model
from drom.utils.one_hot import from_one_hot_to_index, from_one_hot
from drom.plotting.base import plot_drom_trajectories
from drom.torch_utils.seed import fix_random_seed
from drom.torch_utils.torch_timer import TimerCUDA
from drom.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from drom.utils.utils import select_tasks, get_env_metadata_from_dataset, select_best_traj, ask_subtask
import drom.demos

import robosuite
import robosuite.utils.transform_utils as T
import mimicgen


# ========================================================================== #
# Helper functions
# ========================================================================== #
def load_config(model_dir: str) -> dict:
    """Load model configuration JSON."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

def build_context_model(train_cfg):
    """Construct identity context model if attention conditioning is used."""
    if train_cfg["conditioning_type"] != "attention":
        return None, 4

    context_dim = train_cfg["context_dim"]
    model = TaskModelIdentity(input_dim=context_dim, out_dim=context_dim)
    return model, train_cfg["conditioning_dim"]


def resolve_checkpoint(train_cfg, ckpt_arg, model_dir):
    """Resolve which checkpoint filename to load."""
    if ckpt_arg == "last":
        ckpt = (
            "ema_model_current_state_dict.pth"
            if train_cfg["use_ema"]
            else "model_current_state_dict.pth"
        )
    else:
        ckpt = ckpt_arg

    ckpt_path = Path(model_dir) / "models" / ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return ckpt_path


# ========================================================================== #
# Main
# ========================================================================== #
def main(args):

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    if args.directory == "MugNewTrained":
        folder = "20251209175732"
    elif args.directory == "MugMimicGen":
        folder = "20251220175732"
    elif args.directory == "SquarePickPlace":
        folder = "20251221015146"
    elif args.directory == "SquarePickPlaceLow":
        folder = "20251221101249"
    elif args.directory == "MugVoice":
        folder = "20251216173456"
    elif args.directory == "MugHuman":
        folder = "20251212164555"
    elif args.directory == "MugMPD":
        folder = "20260129115516"
    elif args.directory == "StackReal":
        # folder = "20260119115708"
        folder = "20260223174333"
    elif args.directory == "StackNewReal":
        folder = "20260223191608"
    elif args.directory == "StackCleanupNewReal":
        folder = "20260223204550"
    elif args.directory == "StackCleanupNewRealMPD":
        folder = "20260225095605"
        
    model_dir = Path("drom/trained_models") / args.directory / folder

    config = load_config(model_dir)
    train_cfg = config["train"]

    fix_random_seed(args.seed)

    device = get_torch_device("cuda:0" if train_cfg["cuda"] else "cpu")
    tensor_args = dict(device=device, dtype=torch.float32)
    context_enabled = train_cfg["conditioning_type"] == "attention"

    # Directories
    results_dir = model_dir / "validation" / str(args.seed)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    dataset_dir = drom.demos.demo_root
    if train_cfg["mimicgen"]:
        dataset_dir = os.path.join(dataset_dir, "mimicgen")

    # tasks = "all" → gather all folder names
    tasks = select_tasks(train_cfg.get("tasks"), dataset_dir)
    print(f"\nSelected tasks: {tasks}")

    train_subset, train_loader, val_subset, val_loader = get_dataset(
        dataset_class="TrajectoryDataset",
        batch_size=train_cfg["batch_size"],
        results_dir=model_dir,
        save_indices=False,
        tensor_args=tensor_args,
        tasks=train_cfg["tasks"],
        number_of_trajs=train_cfg["number_of_trajs_per_task"],
        horizon=train_cfg["horizon"],
        context=context_enabled,
        dataset_dir=dataset_dir,
        normalizer=train_cfg["normalizer"]
    )

    dataset = train_subset.dataset
    context_model, conditioning_dim = build_context_model(train_cfg)
    
    if "Real" in args.directory:
        env_name = args.directory
    else:
        env_args = get_env_metadata_from_dataset(os.path.join(dataset_dir, tasks[0], "dmp", "image_200.hdf5"))
        env_name = env_args["env_name"]
        assert len(tasks) == 1, "The selected task must be equal to 1. It is not possible to load multiple environments to be tested in MuJoCo"

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    ckpt_path = resolve_checkpoint(train_cfg, args.ckpt, model_dir)

    diffusion_cfg = dict(
        variance_schedule=train_cfg["variance_schedule"],
        n_diffusion_steps=train_cfg["n_diffusion_steps"],
        predict_epsilon=train_cfg["predict_epsilon"],
        context_model=context_model,
    )

    unet_cfg = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.horizon,
        unet_input_dim=train_cfg["unet_input_dim"],
        dim_mults=UNET_DIM_MULTS[train_cfg["unet_dim_mults_option"]],
        conditioning_type=train_cfg["conditioning_type"],
        conditioning_embed_dim=conditioning_dim,
        self_attention=train_cfg["self_attention"],
        device=device,
    )

    model = get_model(
        model_class="GaussianDiffusionModel",
        model=TemporalUnet(**unet_cfg),
        tensor_args=tensor_args,
        **diffusion_cfg,
        **unet_cfg,
    )

    # Load weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    freeze_torch_model_params(model)

    # Compile + warmup
    model = torch.compile(model)
    model.warmup(horizon=train_cfg["horizon"], device=device)

    total_params = sum(param.numel() for param in model.parameters())
    # input(f"total_params: {total_params}")


    # ------------------------------------------------------------------ #
    # Validation Setup
    # ------------------------------------------------------------------ #
    val_indices = torch.load(model_dir / "val_subset_indices.pt")
    chosen_indices = torch.randperm(len(val_indices))[:args.validations]
    val_subset.indices = val_indices

    subtasks = dataset.subtask_list
    if args.directory == "StackReal":
        subtasks = ["Pick", "Lift", "Place"]

    # ------------------------------------------------------------------ #
    # Loop over validation samples
    # ------------------------------------------------------------------ #
    trajs_dict = {}
    init_state_dict = {}
    subtasks_dict = {}
    
    # for index in chosen_indices:
    print("")
    # success_count = 0
    success_dict = {}
    for s in subtasks:
        success_dict[s] = 0
    
    fail_dict = {}
    for s in subtasks:
        fail_dict[s] = 0
    
    elapsed_total = 0.0

    val_episodes = len(chosen_indices)
    for index in tqdm(chosen_indices, desc="Validating model...", unit="test"):
        data = val_loader.dataset[index]

        # Determine task name if context conditioning is active
        if train_cfg["conditioning_type"] == "attention":
            hard_conds_normalized = data["hard_conds_normalized"]
            context_normalized = {
                "tasks": hard_conds_normalized[-dataset.context_dim:]
                }
            context = dataset.unnormalize_hard_conds(hard_conds_normalized)[-dataset.context_dim:]
            subtask_index = from_one_hot(context).item()
            subtask = subtasks[subtask_index]
        else:
            context_normalized = None
            subtask = "Unknown"

        hard_conds_dict = data["hard_conds"]

        # Unnormalize states
        start_cond = dataset.unnormalize(hard_conds_dict[0], key=dataset.field_key_traj)
        goal_cond = dataset.unnormalize(hard_conds_dict[train_cfg["horizon"]-1], key=dataset.field_key_traj)

        t_start_guide = ceil(0.25 * model.n_diffusion_steps)

        sample_kwargs = dict(
            guide=None,
            n_guide_steps=5,
            t_start_guide=t_start_guide,
            noise_std_extra_schedule_fn=lambda _: 0.5,
            n_diffusion_steps_without_noise=5,
        )

        # ------------------------------------------------------------------ #
        # Sampling
        # ------------------------------------------------------------------ #
        with TimerCUDA() as timer:
            trajs_norm = model.run_inference(
                context=context_normalized,
                hard_conds=hard_conds_dict,
                n_samples=args.samples,
                horizon=train_cfg["horizon"],
                return_chain=False,
                sample_fn=ddpm_sample_fn,
                **sample_kwargs,
            )

        # print(f"\nTask: {subtask}")
        # print(f"{args.samples} trajectories sampled in {timer.elapsed:.3f} sec")
        elapsed_total += timer.elapsed

        # Unnormalize + plot
        trajs = dataset.unnormalize_trajectories(trajs_norm)
        states = dataset.unnormalize_states(data["states_normalized"])
        if "Square" in env_name:
            states[:, 17] = 10.0
            states[:, 18] = 10.0
            states[:, 19] = 10.0
        demo = dataset.unnormalize_trajectories(data["actions_normalized"])

        data_path = results_dir / subtask
        data_path.mkdir(parents=True, exist_ok=True)
        torch.save(trajs, os.path.join(data_path, f"trajs_{index.item()}.pt"))
        torch.save(states, os.path.join(data_path, f"states_{index.item()}.pt"))

        plot_drom_trajectories(
            trajectories=trajs,
            hard_conds=[start_cond, goal_cond],
            task=subtask,
            demo=demo,
            results_dir=results_dir,
            val_index=index.item(),
        )

        trajs_dict[index.item()] = trajs
        states = np.array(states.cpu().detach())
        init_state_dict[index.item()] = states[0].copy()

        subtasks_dict[index.item()] = subtask
    
    print(f"\nAverage generation time for {args.samples} samples across {val_episodes} test episodes: {elapsed_total / val_episodes:.4f} seconds")
    
    if args.render:
        # ------------------------------------------------------------------ #
        # Robosuite environment setup
        # ------------------------------------------------------------------ #
        # Environment arguments
        env_kwargs = env_args["env_kwargs"]
        env_args.pop("env_version")
        env_args.pop("type")
        env_args.pop("env_kwargs")
        for k, v in env_kwargs.items():
            env_args[k] = v
        env_args["has_renderer"] = True
        env_args["has_offscreen_renderer"] = True
        env = robosuite.make(**env_args, render_camera=args.camera)
        env.reset()

        for ep, index in enumerate(list(trajs_dict.keys())):
            flag = False
            while not flag:
                env.sim.set_state_from_flattened(init_state_dict[index])
                env.sim.forward()

                trajectories = trajs_dict[index]

                traj = select_best_traj(trajectories)
                s = subtasks_dict[index]

                for action in traj:
                    action = np.array(action.cpu().detach())
                    env.step(action)
                    env.render()
                
                print(f"\nEpisode: {ep}, Subtask: {s}")
                ans = input("Success? [y:yes, n:no, r:restart] ")
                while not ans.lower() in ["y", "yes", "no", "n", "r", "restart", "res"]:
                    print("Answer not valid!")
                    ans = input("Success? [y:yes, n:no, r:restart] ")

                if ans.lower() not in ["r", "restart", "res"]:
                    s = ask_subtask(subtasks)
                    if ans.lower() in ["yes", "y"]:
                        success_dict[s] += 1
                    else:
                        fail_dict[s] += 1
                    flag = True


        env.close()
    
    if args.render:
        success_count = 0
        for v in success_dict.values():
            success_count += v
        success_rate = success_count/val_episodes
        print(f"Success rate: {success_rate}")
        sr_path = os.path.join(model_dir, "validation", str(args.seed), "SuccessRate.yaml")
        with open(sr_path, "w") as f:
            json.dump({
                "ValEpisodes": val_episodes,
                "Successes": success_count,
                "Fails": val_episodes - success_count,
                "SuccessRate": success_rate,
                "SubtaskSuccesses": success_dict,
                "SubtaskFails":fail_dict,
                }, f, indent=2)


# ========================================================================== #
# Entry Point
# ========================================================================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate DROM model.")
    parser.add_argument("--directory", required=True, help="Model directory inside trained_models.")
    parser.add_argument("--ckpt", default="last", help="Checkpoint name or 'last'.")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--samples", type=int, default=10, help="Trajectories to generate per validation.")
    parser.add_argument("--validations", type=int, default=5, help="Number of validation cases to run.")
    parser.add_argument("--camera", type=str, default="frontview", help="Which camera to use for rendering.")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()
    main(args)
