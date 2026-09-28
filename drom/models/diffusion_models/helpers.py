import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from drom.torch_utils.torch_utils import to_torch

#-----------------------------------------------------------------------------#
#---------------------------- variance schedules -----------------------------#
#-----------------------------------------------------------------------------#

def linear_beta_schedule(n_diffusion_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_diffusion_steps)


def quadratic_beta_schedule(n_diffusion_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, n_diffusion_steps) ** 2


def sigmoid_beta_schedule(n_diffusion_steps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, n_diffusion_steps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = n_diffusion_steps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=a_min, a_max=a_max)
    return to_torch(betas_clipped, dtype=dtype)


def exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0):
    # exponential increasing noise from t=0 to t=T
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps)
    beta_start = to_torch(beta_start)
    beta_end = to_torch(beta_end)
    a = 1 / n_diffusion_steps * torch.log(beta_end / beta_start)
    return beta_start * torch.exp(a * x)


def constant_fraction_beta_schedule(n_diffusion_steps):
    # exponential increasing noise from t=0 to t=T
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps)
    return 1 / (n_diffusion_steps-x+1)


def variance_preserving_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0):
    # Works only with a small number of diffusion steps
    # https://arxiv.org/abs/2112.07804
    # https://openreview.net/pdf?id=AHvFDPi-FA
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps)
    alphas = torch.exp(-beta_start*(1/n_diffusion_steps) - 0.5*(beta_end-beta_start)*(2*x-1)/(n_diffusion_steps**2))
    betas = 1 - alphas
    return betas




#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights=None):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        if self.weights is not None:
            weighted_loss = (loss * self.weights).mean()
        else:
            weighted_loss = loss.mean()
        return weighted_loss, {}


class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        total_loss = F.mse_loss(pred, targ, reduction='none')
        return total_loss

class WeightedLossAgumented(WeightedLoss):
    """
    Loss function class adapted from 3D Diffuser Actor (https://github.com/nickgkan/3d_diffuser_actor)
    """
    def __init__(self):
        super().__init__()
        self.position_loss = "mse"
        self.rotation_parametrization = "euler"
        self.compute_loss_at_all_layers = False
        self.ground_truth_gaussian_spread = 1000.0
        self.label_smoothing = 0.0
        self.position_loss_coeff = 1.0
        self.position_offset_loss_coeff = 1000.0
        self.rotation_loss_coeff = 1.0
        self.gripper_loss_coeff = 1.0
        self.symmetric_rotation_loss = False

    def _loss(self, pred, targ):
        position_loss = self._compute_position_loss(pred[:, :, :3], targ[:, :, :3])
        rotation_loss = self._compute_rotation_loss(pred[:, :, 3:6], targ[:, :, 3:6])
        total_loss = position_loss * self.position_loss_coeff + rotation_loss * self.rotation_loss_coeff
        if pred.shape[2] == 7 and targ.shape[2] == 7:
            gripper_loss = F.binary_cross_entropy(pred[:, :, 6], targ[:, :, 6], reduction='none')
            total_loss += gripper_loss * self.gripper_loss_coeff
        return total_loss

    def _compute_rotation_loss(self, pred, gt_quat):
        if "quat" in self.rotation_parametrization:
            if self.symmetric_rotation_loss:
                gt_quat_ = -gt_quat.clone()
                quat_loss = F.mse_loss(pred["rotation"], gt_quat, reduction='none').mean(1)
                quat_loss_ = F.mse_loss(pred["rotation"], gt_quat_, reduction='none').mean(1)
                select_mask = (quat_loss < quat_loss_).float()
                loss = (select_mask * quat_loss + (1 - select_mask) * quat_loss_).mean()
        else:
            loss = F.mse_loss(pred, gt_quat, reduction='none')

        loss *= self.rotation_loss_coeff
        return loss

    def _compute_position_loss(self, pred, gt_position):
        if self.position_loss == "mse":
            # Only used for original HiveFormer
            return F.mse_loss(pred, gt_position, reduction='none')

        elif self.position_loss in ["ce", "ce+mse"]:
            # Select a normalized Gaussian ball around the ground-truth
            # as a proxy label for a soft cross-entropy loss
            l2_pyramid = []
            label_pyramid = []
            for ghost_pcd_i in pred['ghost_pcd_pyramid']:
                l2_i = ((ghost_pcd_i - gt_position.unsqueeze(-1)) ** 2).sum(1).sqrt()
                label_i = torch.softmax(-l2_i / self.ground_truth_gaussian_spread, dim=-1).detach()
                l2_pyramid.append(l2_i)
                label_pyramid.append(label_i)

            loss_layers = range(len(pred['ghost_pcd_masks_pyramid'][0])) if self.compute_loss_at_all_layers else [-1]

            for j in loss_layers:
                for i, ghost_pcd_masks_i in enumerate(pred["ghost_pcd_masks_pyramid"]):
                    losses[f"position_ce_level{i}"] = F.cross_entropy(
                        ghost_pcd_masks_i[j], label_pyramid[i],
                        label_smoothing=self.label_smoothing
                    ).mean() * self.position_loss_coeff / len(pred["ghost_pcd_masks_pyramid"])

            # Supervise offset from the ghost point's position to the predicted position
            num_sampling_level = len(pred['ghost_pcd_masks_pyramid'])
            if pred.get("fine_ghost_pcd_offsets") is not None:
                if pred["ghost_pcd_pyramid"][-1].shape[-1] != pred["ghost_pcd_pyramid"][0].shape[-1]:
                    npts = pred["ghost_pcd_pyramid"][-1].shape[-1] // num_sampling_level
                    pred_with_offset = (pred["ghost_pcd_pyramid"][-1] + pred["fine_ghost_pcd_offsets"])[:, :, -npts:]
                else:
                    pred_with_offset = (pred["ghost_pcd_pyramid"][-1] + pred["fine_ghost_pcd_offsets"])
                losses["position_offset"] = F.mse_loss(
                    pred_with_offset,
                    gt_position.unsqueeze(-1).repeat(1, 1, pred_with_offset.shape[-1])
                )
                losses["position_offset"] *= (self.position_offset_loss_coeff * self.position_loss_coeff)

            if self.position_loss == "ce":
                # Clear gradient on pred["position"] to avoid a memory leak since we don't
                # use it in the loss
                pred["position"] = pred["position"].detach()
            else:
                losses["position_mse"] = (
                    F.mse_loss(pred["position"], gt_position)
                    * self.position_loss_coeff
                )

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'augmented': WeightedLossAgumented,
}
