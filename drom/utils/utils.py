import os
import h5py
import json

def select_tasks(cfg_tasks, dataset_dir):
    """Select tasks based on the 'tasks' config field."""
    if isinstance(cfg_tasks, str) and cfg_tasks == "all":
        return [
            t for t in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, t))
        ]

    if not cfg_tasks:
        raise ValueError("Config field 'tasks' must be non-empty or equal to 'all'.")

    return cfg_tasks

def ask_subtask(subtasks):
    """Select subtask based on the task is being performed"""
    print("\nPlease, select the subtask:")
    for i, s in enumerate(subtasks, start=1):
        print(f"{i}: {s}")
    idx = int(input("Select: "))
    while not idx in range(1, len(subtasks)+1):
        print("Answer not valid.")
        idx = int(input("Select: "))
    s = subtasks[idx-1]
    print(f"Selected subtask: {s}")
    return s

def get_env_metadata_from_dataset(dataset_path):
    """
    Credits:
        This function is adapted from the RoboMimic implementation:

            https://github.com/ARISE-Initiative/robomimic/blob/v0.5/robomimic/utils/file_utils.py

        The original version was authored by the RoboMimic team as part of their
        training utilities. It has been copied here (with minor modifications)
        to ensure compatibility with the DROM project without requiring direct
        modification of the RoboMimic source code.

        By duplicating and adapting this function in our own repository, we
        maintain full isolation from the upstream RoboMimic codebase while still
        preserving the intended behavior of the original implementation.

        All credit for the initial design and logic goes to the RoboMimic authors.

        Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

        set_env_specific_obs_processors (bool): environment might have custom rules for how to process
            observations - if this flag is true, make sure ObsUtils will use these custom settings. This
            is a good place to do this operation to make sure it happens before loading data, running a 
            trained model, etc.

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    # env_info = json.loads(f["data"].attrs["env_info"])
    if "env_lang" in env_args["env_kwargs"]: del env_args["env_kwargs"]["env_lang"]

    f.close()
    return env_args

def select_best_traj(trajecotries):
    return trajecotries[0, ...]