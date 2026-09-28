import abc
import os
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import drom.demos
from tqdm import tqdm
from drom.datasets.normalization import DatasetNormalizer

class TrajectoryDatasetBase(Dataset, abc.ABC):

    def __init__(
        self,
        include_velocity=False,
        normalizer='LimitsNormalizer',
        tensor_args=None,
        tasks=None,
        number_of_trajs=2000,
        horizon=64,
        context=False,
        dataset_dir=None,
    ):
        self.tensor_args = tensor_args or {"device": "cpu"}
        self.base_dir = dataset_dir
        self.include_velocity = include_velocity
        self.tasks = tasks

        self.fields = {}
        self.field_key_traj = 'actions'
        self.field_key_state = 'states'
        self.field_key_hard_conds = 'hard_conds'
        self.field_key_task_names = 'task_names'
        self.map_subtasks_to_id = {}

        # self.voice_embeddings = np.load("drom/voice_data/queries_embedding_matrix.npy")
        # self.voice_embeddings = torch.tensor(self.voice_embeddings, dtype=self.tensor_args["dtype"], device=self.tensor_args["device"])

        self.load_subtask_trajectories(N=number_of_trajs, H=horizon, context=context)

        # Dimensions and shapes
        b, h, d = self.dataset_shape = self.fields[self.field_key_traj].shape
        self.n_trajs = b
        self.horizon = h
        self.state_dim = d
        self.trajectory_dim = (h, d)
        self.context_dim = self.fields[self.field_key_hard_conds].shape[1] - 2 * d

        # Normalization
        self.normalizer_keys = [self.field_key_traj, self.field_key_state, self.field_key_hard_conds]
        self.normalizer = DatasetNormalizer({k: self.fields[k] for k in self.normalizer_keys}, normalizer=normalizer)
        self.normalize_all_data(*self.normalizer_keys)

    def load_task_trajectories(self, N, H, context):
        print("\nLoading trajectories from HDF5 files...")
        trajectories = []
        one_hot_vectors = []

        for task_idx, task in enumerate(self.tasks):
            task_path = os.path.join(self.base_dir, task, "dmp", "demo.hdf5")
            with h5py.File(task_path, "r") as f:
                demos = list(f["data"].keys())
                trajs = []

                for i in range(min(N, len(demos))):
                    actions = torch.tensor(
                        f[f"data/{demos[i]}/actions"][()],
                        device=self.tensor_args["device"]
                    )
                    indexes = torch.linspace(0, actions.size(0) - 1, H).long()
                    traj = actions[indexes].float()
                    trajs.append(traj)

                task_trajs = torch.stack(trajs)
                trajectories.append(task_trajs)

                one_hot = torch.zeros((1, len(self.tasks)), device=self.tensor_args["device"])
                one_hot[0, task_idx] = 1.0
                one_hot = one_hot.expand(task_trajs.size(0), -1)
                one_hot_vectors.append(one_hot)

        trajectories = torch.cat(trajectories, dim=0)
        one_hot_vectors = torch.cat(one_hot_vectors, dim=0)

        print(f"Trajectories loaded: {trajectories.shape}")
        print(f"One-hot task vectors: {one_hot_vectors.shape}")

        self.fields[self.field_key_traj] = trajectories

        if context:
            print(f"one_hot_vectors: {one_hot_vectors.shape}")
            input(f"trajectories[:, 0]: {trajectories[:, 0].shape}")
            hard_conds = torch.cat(
                (trajectories[:, 0], trajectories[:, -1], one_hot_vectors),
                dim=-1
            )
        else:
            hard_conds = torch.cat(
                (trajectories[:, 0], trajectories[:, -1]),
                dim=-1
            )

        self.fields[self.field_key_hard_conds] = hard_conds

    def load_subtask_trajectories(self, N: int, H: int, context: bool):
        """
        Load up to N trajectories per subtask.
        Each trajectory is resampled to fixed horizon H.

        Args:
            N: max number of samples per subtask
            H: trajectory horizon
            context: whether to append one-hot context to hard conditions
        """

        device = self.tensor_args["device"]

        print("\n🔹 Collecting subtask names...")

        # ============================================================
        # 1️⃣ Collect unique subtask names
        # ============================================================

        subtask_set = set()

        task = self.tasks[0]
        traj_path = os.path.join(self.base_dir, task, "dmp", "image_200.hdf5")
        with h5py.File(traj_path, "r") as f:
            for demo_key in f["data"].keys():
                demo = f["data"][demo_key]
                if "subtask_data" not in demo.attrs:
                    continue

                subtask_data = json.loads(demo.attrs["subtask_data"])
                for name in subtask_data["subtask_names"]:
                    name = "Lift" if name.lower() == "lift" else name
                    subtask_set.add(name)

        subtask_list = sorted(list(subtask_set))
        self.subtask_list = subtask_list
        subtask_to_id = {name: i for i, name in enumerate(subtask_list)}

        print(f"Found {len(subtask_list)} unique subtasks: {subtask_list}")

        # ============================================================
        # 2️⃣ Load trajectories
        # ============================================================

        subtask_counter = {name: 0 for name in subtask_list}

        traj_buffer = []
        state_buffer = []
        name_indices = []

        print("\n🔹 Loading trajectories...")

        traj_path = os.path.join(self.base_dir, task, "dmp", "image_200.hdf5")

        with h5py.File(traj_path, "r") as f:
            for demo_key in tqdm(f["data"].keys(), desc="Demos", unit="demo"):
                demo = f["data"][demo_key]
                if "subtask_data" not in demo.attrs:
                    continue

                subtask_data = json.loads(demo.attrs["subtask_data"])
                names = subtask_data["subtask_names"]
                ts_split = subtask_data["ts_split"]

                actions_np = f[f"data/{demo_key}/actions_abs"][()]
                states_np = f[f"data/{demo_key}/states"][()]

                for idx, raw_name in enumerate(names):

                    name = "Lift" if raw_name.lower() == "lift" else raw_name

                    # Skip if enough samples already collected
                    if subtask_counter[name] >= N:
                        continue

                    start = ts_split[idx]
                    end = ts_split[idx + 1] if idx < len(names) - 1 else actions_np.shape[0]

                    if end <= start:
                        continue  # invalid slice

                    segment_actions = actions_np[start:end]
                    segment_states = states_np[start:end]

                    length = segment_actions.shape[0]

                    # Reject empty or too-short segments
                    # if length < H:
                    #     continue

                    # ------------------------------------------------
                    # Resample to fixed horizon H
                    # ------------------------------------------------
                    if length == H:
                        resampled_actions = segment_actions
                        resampled_states = segment_states
                    else:
                        idxs = np.linspace(0, length - 1, H)
                        idxs = np.round(idxs).astype(np.int64)

                        resampled_actions = segment_actions[idxs]
                        resampled_states = segment_states[idxs]
                    

                    # Convert to torch
                    traj_buffer.append(torch.from_numpy(resampled_actions))
                    state_buffer.append(torch.from_numpy(resampled_states))
                    name_indices.append(subtask_to_id[name])

                    subtask_counter[name] += 1

                # Early stop if all subtasks full
                if all(v >= N for v in subtask_counter.values()):
                    break
        
        # ============================================================
        # 3️⃣ Stack tensors (single GPU transfer)
        # ============================================================

        trajectories = torch.stack(traj_buffer).float().to(device)
        total_states = torch.stack(state_buffer).float().to(device)
        name_indices = torch.tensor(name_indices, device=device)

        # ============================================================
        # 4️⃣ Build one-hot vectors
        # ============================================================

        one_hot_vectors = torch.zeros(
            (len(name_indices), len(subtask_list)),
            device=device,
        )
        one_hot_vectors[torch.arange(len(name_indices)), name_indices] = 1.0

        # ============================================================
        # 5️⃣ Build hard conditions
        # ============================================================

        if context:
            hard_conds = torch.cat(
                (
                    trajectories[:, 0],      # start
                    trajectories[:, -1],     # goal
                    one_hot_vectors,
                ),
                dim=-1,
            )
        else:
            hard_conds = torch.cat(
                (
                    trajectories[:, 0],
                    trajectories[:, -1],
                ),
                dim=-1,
            )

        # ============================================================
        # 6️⃣ Store fields
        # ============================================================

        self.fields[self.field_key_traj] = trajectories
        self.fields[self.field_key_state] = total_states
        self.fields[self.field_key_task_names] = name_indices
        self.fields[self.field_key_hard_conds] = hard_conds

        # ============================================================
        # 7️⃣ Diagnostics
        # ============================================================

        print("\n✅ Dataset loaded")
        print(f"Trajectories: {trajectories.shape}")
        print(f"States:       {total_states.shape}")
        print(f"Hard conds:   {hard_conds.shape}")
        print(f"Subtask distribution:\n{subtask_counter}")

    def normalize_all_data(self, *keys):
        for key in keys:
            self.fields[f'{key}_normalized'] = self.normalizer(self.fields[key], key)

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, index):
        field_traj_norm = f'{self.field_key_traj}_normalized'
        field_state_norm = f'{self.field_key_state}_normalized'
        field_hard_conds_norm = f'{self.field_key_hard_conds}_normalized'
        field_task_names = self.field_key_task_names

        traj = self.fields[field_traj_norm][index]
        state = self.fields[field_state_norm][index]
        hard_conds = self.fields[field_hard_conds_norm][index]
        task_names = self.fields[field_task_names][index]

        return {
            field_traj_norm: traj,
            field_state_norm: state,
            field_hard_conds_norm: hard_conds,
            field_task_names: task_names,
            "hard_conds": self.get_hard_conditions(traj, horizon=len(traj))
        }

    def __repr__(self):
        return (
            f'{self.__class__.__name__}\n'
            f'  Number of Trajectories: {self.n_trajs}\n'
            f'  Trajectory Shape: {self.trajectory_dim}\n'
            f'  Context Dimension: {self.context_dim}\n'
        )

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        raise NotImplementedError

    def get_context_conditions(self, one_hot_encoded, horizon=None, normalize=False):
        raise NotImplementedError

    def get_unnormalized(self, index):
        raise NotImplementedError

    def normalize(self, x, key):
        return self.normalizer.normalize(x, key)

    def unnormalize(self, x, key):
        return self.normalizer.unnormalize(x, key)

    def normalize_trajectories(self, x):
        return self.normalize(x, self.field_key_traj)

    def unnormalize_trajectories(self, x):
        return self.unnormalize(x, self.field_key_traj)
    
    def normalize_states(self, x):
        return self.normalize(x, self.field_key_state)

    def unnormalize_states(self, x):
        return self.unnormalize(x, self.field_key_state)

    def normalize_hard_conds(self, x):
        return self.normalize(x, self.field_key_hard_conds)

    def unnormalize_hard_conds(self, x):
        return self.unnormalize(x, self.field_key_hard_conds)



class TrajectoryDataset(TrajectoryDatasetBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        # start and goal positions
        start_state_pos = traj[0]
        goal_state_pos = traj[-1]

        if self.include_velocity:
            # If velocities are part of the state, then set them to zero at the beggining and end of a trajectory
            start_state = torch.cat((start_state_pos, torch.zeros_like(start_state_pos)), dim=-1)
            goal_state = torch.cat((goal_state_pos, torch.zeros_like(goal_state_pos)), dim=-1)
        else:
            start_state = start_state_pos
            goal_state = goal_state_pos

        if normalize:
            start_state = self.normalizer.normalize(start_state, key=self.field_key_traj)
            goal_state = self.normalizer.normalize(goal_state, key=self.field_key_traj)

        if horizon is None:
            horizon = self.horizon
        hard_conds = {
            0: start_state,
            horizon - 1: goal_state
        }
        return hard_conds
    
    def get_context_conditions(self, context_conds, horizon=None, normalize=False):
        if normalize:
            # start_state = context_conds[:7]
            # goal_state = context_conds[7:14]
            # start_state = self.normalizer.normalize(start_state, key=self.field_key_traj)
            # goal_state = self.normalizer.normalize(goal_state, key=self.field_key_traj)
            # context_conds = torch.cat((start_state, goal_state, context_conds[14:]), dim=-1)
            context_conds = context_conds[14:]
        
        context_conds = {
            "tasks": context_conds,
        }
        return context_conds
