import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch
import os
import scipy.stats
from matplotlib.patches import Ellipse

def plot_drom_trajectories(
    trajectories, hard_conds, task,
    demo=None, cartesian=True,
    results_dir=None, val_index=None
):
    n_channels = trajectories.shape[-1]
    if cartesian:
        if n_channels == 7:
            titles = ["PosX", "PosY", "PosZ", "RotX", "RotY", "RotZ", "Gripper"]
        elif n_channels == 8:
            titles = ["PosX", "PosY", "PosZ", "QuatX", "QuatY", "QuatZ", "QuatW", "Gripper"]
        else:
            raise ValueError(f"Unexpected number of channels: {n_channels}")
    else:
        titles = [f"J{i+1}" for i in range(n_channels)]
    
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.detach().cpu().numpy()

    if isinstance(demo, torch.Tensor):
        demo = demo.detach().cpu().numpy()
    
    start_cond = hard_conds[0]
    goal_cond = hard_conds[1]
    if isinstance(start_cond,torch.Tensor):
        start_cond = start_cond.detach().cpu().numpy()
    if isinstance(goal_cond,torch.Tensor):
        goal_cond = goal_cond.detach().cpu().numpy()

    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    fig.suptitle(f"Task: {task}")

    for traj_index in range(len(trajectories)):
        axs[0,0].plot(trajectories[traj_index, :, 0])
        axs[0,1].plot(trajectories[traj_index, :, 1])
        axs[1,0].plot(trajectories[traj_index, :, 2])

        axs[1,1].plot(trajectories[traj_index, :, 3])
        axs[2,0].plot(trajectories[traj_index, :, 4])
        axs[2,1].plot(trajectories[traj_index, :, 5])
        axs[3,0].plot(trajectories[traj_index, :, 6])

        axs[3,1].plot(trajectories[traj_index, :, 7])

    # --- Hard conditions ---
    # Pos X axis subplot
    axs[0,0].scatter(0, start_cond[0])
    axs[0,0].scatter(trajectories.shape[1], goal_cond[0])
    axs[0,0].set_title(titles[0])
    # Pos Y axis subplot
    axs[0,1].scatter(0, start_cond[1])
    axs[0,1].scatter(trajectories.shape[1], goal_cond[1])
    axs[0,1].set_title(titles[1])
    # Pos Z axis subplot
    axs[1,0].scatter(0, start_cond[2])
    axs[1,0].scatter(trajectories.shape[1], goal_cond[2])
    axs[1,0].set_title(titles[2])
    # Quat X axis subplot
    axs[1,1].scatter(0, start_cond[3])
    axs[1,1].scatter(trajectories.shape[1], goal_cond[3])
    axs[1,1].set_title(titles[3])
    # Quat Y axis subplot
    axs[2,0].scatter(0, start_cond[4])
    axs[2,0].scatter(trajectories.shape[1], goal_cond[4])
    axs[2,0].set_title(titles[4])
    # Quat Z axis subplot
    axs[2,1].scatter(0, start_cond[5])
    axs[2,1].scatter(trajectories.shape[1], goal_cond[5])
    axs[2,1].set_title(titles[5])
    # Quat W axis subplot
    axs[3,0].scatter(0, start_cond[6])
    axs[3,0].scatter(trajectories.shape[1], goal_cond[6])
    axs[3,0].set_title(titles[6])
    # Gripper axis subplot
    axs[3,1].scatter(0, start_cond[7])
    axs[3,1].scatter(trajectories.shape[1], goal_cond[7])
    axs[3,1].set_title(titles[7])

    # --- Demo overlay ---
    if demo is not None:
        axs[0,0].plot(demo[:, 0], "r--")
        axs[0,1].plot(demo[:, 1], "r--")
        axs[1,0].plot(demo[:, 2], "r--")
        axs[1,1].plot(demo[:, 3], "r--")
        axs[2,0].plot(demo[:, 4], "r--")
        axs[2,1].plot(demo[:, 5], "r--")
        axs[3,0].plot(demo[:, 6], "r--")
        axs[3,1].plot(demo[:, 7], "r--")

    # ==========================
    # AXIS LIMITS (NEW PART)
    # ==========================
    pos_lim = (-1.1, 1.1)
    rot_lim = (-1.1, 1.1)
    grip_lim = (-1.1, 1.1)

    # Positions
    for ax in [axs[0,0], axs[0,1], axs[1,0]]:
        ax.set_ylim(pos_lim)

    # Rotations
    for ax in [axs[1,1], axs[2,0], axs[2,1], axs[3,0]]:
        ax.set_ylim(rot_lim)

    # Gripper
    axs[3,1].set_ylim(grip_lim)

    # ==========================
    # SAVE FIGURE
    # ==========================
    if results_dir is not None:
        index_str = str(val_index)
        path = os.path.join(results_dir, task)
        os.makedirs(path, exist_ok=True)

        save_path = os.path.join(path, f"val_index_{index_str}.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

def save_fig(fig, name, dir=f"./{os.path.basename(__file__).split('.py')[0]}"):
    fig.tight_layout()
    os.makedirs(dir, exist_ok=True)
    fig.savefig(f"{dir}/{name}.png")
    fig.savefig(f"{dir}/{name}.pdf")


def export_legend(ax, filename="legend.pdf", plot_dir='', ncol=10, linewidth=7):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=ncol)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(linewidth)
    fig1 = legend.figure
    fig1.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    fig1.savefig(os.path.join(plot_dir, filename), dpi="figure", bbox_inches=bbox)
def export_legendv2(plot_options_d, filename="legend.pdf", plot_dir='', ncol=10, linewidth=7, ):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    for k, v in plot_options_d.items():
        v['linewidth'] = linewidth
        ax2.plot([], [], label=k, **v)
    ax2.axis('off')
    legend = ax2.legend(frameon=False, loc='lower center', ncol=ncol)
    #for legobj in legend.legendHandles:
    #    legobj.set_linewidth(linewidth)
    fig1 = legend.figure
    fig1.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    fig1.savefig(os.path.join(plot_dir, filename), dpi="figure", bbox_inches=bbox)
    plt.close(fig1)
    plt.close(fig2)


def set_small_ticks(ax, fontsize=6, set_minor_ticks=False):
    if set_minor_ticks:
        ax.tick_params(which='minor', grid_linestyle='--')
    else:
        ax.get_yaxis().set_tick_params(which='minor', size=0)
        ax.get_yaxis().set_tick_params(which='minor', width=0)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.xaxis.get_minor_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_minor_ticks():
        tick.label.set_fontsize(fontsize)


def remove_borders(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)


def remove_axes_labels_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def mean_confidence_interval(data, confidence=0.95, axis=0):
    n = data.shape[axis]
    m, se = np.mean(data, axis=axis), scipy.stats.sem(data, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


