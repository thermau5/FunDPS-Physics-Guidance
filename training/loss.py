# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM) with sampler.


@persistence.persistent_class
class EDMLossWithSampler:
    def __init__(self, sampler, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.sampler = sampler
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):  # TODO provide sampler when calling
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        if labels is not None:
            # We want to augment pipe both x and y at once.
            x_dim = images.size(1)
            all_images = torch.cat((images, labels), dim=1)
            all_images_augmented, augment_labels = augment_pipe(all_images) if augment_pipe is not None else (images, None)
            # Extract out the x and y components
            y = all_images_augmented[:, 0:x_dim]  # JIACHEN: previously it was all_images[:, 0:x_dim]
            labels = all_images_augmented[:, x_dim::]
        else:
            y, augment_labels = images, None

        # n = torch.randn_like(y) * sigma
        n = self.sampler.sample(y.size(0)) * sigma

        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

class PI_EDMLossWithSampler:
    def __init__(self, sampler, fno_surrogate, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.sampler = sampler
        self.fno_surrogate = fno_surrogate  # FNO surrogate model
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None, gt_images=None):
        # Data loss calculation (same as EDMLossWithSampler)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        if labels is not None:
            x_dim = images.size(1)
            all_images = torch.cat((images, labels), dim=1)
            all_images_augmented, augment_labels = augment_pipe(all_images) if augment_pipe is not None else (images, None)
            y = all_images_augmented[:, 0:x_dim]
            labels = all_images_augmented[:, x_dim::]
        else:
            y, augment_labels = images, None

        n = self.sampler.sample(y.size(0)) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        data_loss = weight * ((D_yn - y) ** 2)  # Data loss on DM input channels

        # Physics loss calculation (requires gt_images)
        if gt_images is not None:
            assert gt_images.shape[1] == 2, "Ground truth images must have 2 channels (param, solution)"
            param_gt = gt_images[:, 0:1, :, :]  # Ground truth parameter
            solution_gt = gt_images[:, 1:2, :, :]  # Ground truth solution
            
            # FNO surrogate prediction (1-channel input, 1-channel output)
            self.fno_surrogate.eval()
            surrogate_of_param_pred = self.fno_surrogate(
                D_yn.to(dtype=torch.float32, device=next(self.fno_surrogate.parameters()).device)
            )
            surrogate_of_param_pred = surrogate_of_param_pred.to(solution_gt.device, dtype=solution_gt.dtype)

            # Physics loss: compare surrogate's prediction to ground truth solution
            physics_loss = (surrogate_of_param_pred - solution_gt) ** 2  # MSE per-pixel
        else:
            # If no gt_images provided, physics loss is zero
            physics_loss = torch.zeros_like(data_loss)

        # print(f'Data Loss:{data_loss.sum()}. Phys Loss: {physics_loss.sum()}.')
        # Return sum of data loss and physics loss
        return data_loss + physics_loss
