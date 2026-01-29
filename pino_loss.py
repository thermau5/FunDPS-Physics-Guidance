"""Physics-Informed Neural Operator (PINO) loss functions for different PDEs.

This module provides loss classes for various partial differential equations (PDEs)
including Poisson equation and Navier-Stokes equation. Each loss class computes
the PDE residual loss and boundary condition loss as needed.
"""

import torch
import torch.nn.functional as F
from neuralop.losses import LpLoss


class PoissonLoss(object):
    """Class to handle PDE loss calculations for the Poisson equation.

    The Poisson equation is: ∇²u = a
    where a is the source term and u is the solution field.

    This class computes:
    1. PDE residual loss: ||∇²u - a||
    2. Boundary condition loss: ||u - 0|| on boundaries
    """

    def __init__(self, dataset_ref):
        """Initialize Poisson loss with dataset normalizer.

        Args:
            dataset_ref: Reference to dataset for creating normalizer
        """
        self.normalizer = dataset_ref.create_normalizer()
        self.loss_func = LpLoss(d=2, p=2)

    def __call__(self, x_pred):
        """Calculate the PDE loss for Poisson equation.

        Args:
            x_pred (torch.Tensor): Predicted data containing both fields
                First channel ([:,0:1]) is source term a
                Second channel ([:,1:2]) is solution field u

        Returns:
            tuple: (pde_loss, bc_loss)
                - pde_loss: PDE residual loss in interior domain
                - bc_loss: Boundary condition loss
        """
        x_pred = self.normalizer.denormalize(x_pred)
        a_pred = x_pred[:, 0:1]
        u_pred = x_pred[:, 1:2]

        # Calculate grid spacing
        length = a_pred.shape[-1]
        h = 1 / (length - 1)

        # Calculate Laplacian using second order finite difference (interior points only)
        laplacian = (u_pred[..., :-2, 1:-1] + u_pred[..., 2:, 1:-1] + u_pred[..., 1:-1, :-2] + u_pred[..., 1:-1, 2:] - 4 * u_pred[..., 1:-1, 1:-1]) / h**2

        # Match a_pred to laplacian size (interior points only)
        a_pred_interior = a_pred[..., 1:-1, 1:-1]

        pde_loss = self.loss_func(laplacian, a_pred_interior)

        # Boundary condition loss: u should be 0 on all boundaries
        # Extract boundaries: top, bottom, left, right
        u_top = u_pred[..., 0, :]  # First row
        u_bottom = u_pred[..., -1, :]  # Last row
        u_left = u_pred[..., :, 0]  # First column
        u_right = u_pred[..., :, -1]  # Last column

        # Target is zero for all boundaries
        target_zero = torch.zeros_like(u_top)

        bc_loss = (self.loss_func.abs(u_top, target_zero) + self.loss_func.abs(u_bottom, target_zero) + self.loss_func.abs(u_left, target_zero) + self.loss_func.abs(u_right, target_zero)) / 4.0

        return pde_loss, bc_loss


class NavierStokesLoss(object):
    def __init__(self, dataset_ref):
        """Initialize Navier-Stokes loss with dataset normalizer.

        Args:
            dataset_ref: Reference to dataset for creating normalizer
        """
        self.normalizer = dataset_ref.create_normalizer()
        self.loss_func = LpLoss(d=2, p=2)

    def __call__(self, x_pred):
        """Calculate the PDE loss for non-bounded Navier-Stokes equation."""
        x_pred = self.normalizer.denormalize(x_pred)
        # u0_pred_sum = x_pred[:, 0:1].sum(dim=(-2, -1), keepdim=True)
        u1_pred_sum = x_pred[:, 1:2].sum(dim=(-2, -1), keepdim=True)
        zeros = torch.zeros_like(u1_pred_sum)
        pde_loss = self.loss_func.abs(u1_pred_sum, zeros)

        # No boundary conditions for non-bounded case
        bc_loss = torch.tensor(0.0)

        return pde_loss, bc_loss


class HelmholtzLoss(object):
    """Class to handle PDE loss calculations for the Helmholtz equation.

    The Helmholtz equation is: ∇²u + u = a
    where a is the source term and u is the solution field.

    This class computes:
    1. PDE residual loss: ||∇²u + u - a||
    2. Boundary condition loss: ||u - 0|| on boundaries
    """

    def __init__(self, dataset_ref):
        """Initialize Helmholtz loss with dataset normalizer.

        Args:
            dataset_ref: Reference to dataset for creating normalizer
        """
        self.normalizer = dataset_ref.create_normalizer()
        self.loss_func = LpLoss(d=2, p=2)

    def __call__(self, x_pred):
        """Calculate the PDE loss for Helmholtz equation.

        Args:
            x_pred (torch.Tensor): Predicted data containing both fields
                First channel ([:,0:1]) is source term a
                Second channel ([:,1:2]) is solution field u

        Returns:
            tuple: (pde_loss, bc_loss)
                - pde_loss: PDE residual loss in interior domain
                - bc_loss: Boundary condition loss
        """
        x_pred = self.normalizer.denormalize(x_pred)
        a_pred = x_pred[:, 0:1]
        u_pred = x_pred[:, 1:2]

        # Calculate grid spacing
        length = a_pred.shape[-1]
        h = 1 / (length - 1)

        # Calculate Laplacian using second order finite difference (interior points only)
        laplacian = (u_pred[..., :-2, 1:-1] + u_pred[..., 2:, 1:-1] + u_pred[..., 1:-1, :-2] + u_pred[..., 1:-1, 2:] - 4 * u_pred[..., 1:-1, 1:-1]) / h**2

        # Match a_pred and u_pred to laplacian size (interior points only)
        a_pred_interior = a_pred[..., 1:-1, 1:-1]
        u_pred_interior = u_pred[..., 1:-1, 1:-1]

        # PDE residual: ∇²u + u - a = 0
        pde_loss = self.loss_func(laplacian + u_pred_interior, a_pred_interior)
        pde_residual = laplacian + u_pred_interior - a_pred_interior

        # Plot PDE residual
        # import matplotlib.pyplot as plt

        # plt.imshow(pde_residual[0, 0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.title("Helmholtz PDE Residual")
        # plt.savefig("helmholtz_pde_residual.png")
        # plt.close()
        # exit()

        # Boundary condition loss: u should be 0 on all boundaries
        # Extract boundaries: top, bottom, left, right
        u_top = u_pred[..., 0, :]  # First row
        u_bottom = u_pred[..., -1, :]  # Last row
        u_left = u_pred[..., :, 0]  # First column
        u_right = u_pred[..., :, -1]  # Last column

        # Target is zero for all boundaries
        target_zero = torch.zeros_like(u_top)

        bc_loss = (self.loss_func.abs(u_top, target_zero) + self.loss_func.abs(u_bottom, target_zero) + self.loss_func.abs(u_left, target_zero) + self.loss_func.abs(u_right, target_zero)) / 4.0

        return pde_loss, bc_loss


def get_pde_loss(dataset_name, dataset_ref):
    """Factory function to get the appropriate PDE loss for a given dataset.

    Args:
        dataset_name (str): Name of the dataset/PDE type
            Supported values: 'poisson', 'ns-nonbounded', 'helmholtz'
        dataset_ref: Reference to dataset for creating normalizer

    Returns:
        Loss object for the specified PDE type

    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name == "poisson":
        return PoissonLoss(dataset_ref)
    elif dataset_name == "ns-nonbounded":
        return NavierStokesLoss(dataset_ref)
    elif dataset_name == "helmholtz":
        return HelmholtzLoss(dataset_ref)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}. " f"Supported types: 'poisson', 'ns-nonbounded', 'helmholtz'")
