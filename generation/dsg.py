import torch
from tqdm import tqdm

from .dps import PDESolverDPS


class PDESolverDSG(PDESolverDPS):
    """PDE solver using Diffusion State-Guided Projected Gradient."""

    def __init__(self, config):
        super().__init__(config)
        self.variance_threshold = config["guidance"]["threshold"]

    def project_gradient(self, x, g):
        """Project gradient update using SVD-based dimensionality reduction.

        Args:
            x (torch.Tensor): Current diffusion state tensor of shape [B, C, H, W]
            g (torch.Tensor): Gradient tensor of same shape as x

        Returns:
            torch.Tensor: Projected gradient tensor of same shape as input
        """
        B, C, H, W = x.shape
        g_projected = torch.zeros_like(g)

        # Process each batch and channel separately
        for b in range(B):
            for c in range(C):
                # Get current state and gradient for this channel
                # Keep as 2D matrices (no reshaping needed)
                Z = x[b, c]  # State matrix [H, W]
                G = g[b, c]  # Gradient matrix [H, W]

                # Perform SVD on current diffusion state
                U, S, V = torch.linalg.svd(Z, full_matrices=False)

                # Calculate eigenvalues (square of singular values)
                eigenvals = S * S

                # Calculate cumulative sum of eigenvalues (normalized)
                cumsum = torch.cumsum(eigenvals, dim=0)
                cumsum = cumsum / cumsum[-1]
                # TODO compare with the official implementation
                # TODO determine rank by the average across channels
                # TODO decompose only every n steps

                # Find rank r based on variance retention threshold
                tau = self.variance_threshold
                r = torch.where(cumsum >= tau)[0][0].item() + 1

                # Get truncated singular vectors
                Ur = U[:, :r]  # First r left singular vectors
                Vr = V[:r, :]  # First r right singular vectors

                # Project gradient
                # R = Ur^T * G * Vr^T
                R = torch.matmul(torch.matmul(Ur.T, G), Vr.T)

                # Reconstruct approximated gradient
                # G' = Ur * R * Vr
                G_proj = torch.matmul(torch.matmul(Ur, R), Vr)

                # Store projected gradient
                g_projected[b, c] = G_proj

        return g_projected
