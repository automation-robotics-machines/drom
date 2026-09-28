"""
This function is adapted from the RoboMimic implementation:

    https://github.com/ARISE-Initiative/robomimic/blob/v0.5/robomimic/utils/train_utils.py

The original version was authored by the RoboMimic team as part of their
training utilities. It has been copied here (with minor modifications)
to ensure compatibility with the DROM project without requiring direct
modification of the RoboMimic source code.

By duplicating and adapting this function in our own repository, we
maintain full isolation from the upstream RoboMimic codebase while still
preserving the intended behavior of the original implementation.

All credit for the initial design and logic goes to the RoboMimic authors.
"""

import os
import time
import datetime
import shutil

def get_exp_dir(config, auto_remove_exp_dir=False, resume=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
        resume (bool): if True, resume an existing training run instead of creating a 
            new experiment directory
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config["train"]["output_dir"])
    base_output_dir = os.path.join(base_output_dir, config["experiment"]["name"])
    if resume:
        assert os.path.exists(base_output_dir), "Resuming training run, but output dir {} does not exist".format(base_output_dir)
        subdir_lst = os.listdir(base_output_dir)
        assert len(subdir_lst) == 1, "Found more than one subdir {} in output dir {}".format(subdir_lst, base_output_dir)
        time_str = subdir_lst[0]
        assert os.path.isdir(os.path.join(base_output_dir, time_str)), "Found item {} that is not a subdirectory in {}".format(time_str, base_output_dir)
    elif os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config["experiment"]["save"]["enabled"]:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir, exist_ok=resume)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir, exist_ok=resume)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir, exist_ok=resume)

    time_dir = os.path.join(base_output_dir, time_str)
    
    return log_dir, output_dir, video_dir, time_dir