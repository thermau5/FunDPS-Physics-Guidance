"""
Optimize a single data sample using Poisson PDE loss.

This script loads one sample from the Poisson dataset and optimizes it
to satisfy the Poisson equation constraint for 100 steps.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from neuralop.losses import LpLoss
from training.dataset_hf import PDEDataset


class PoissonLoss(object):
    """Class to handle PDE loss calculations for the Poisson equation.

    The Poisson equation is: ∇²u = a
    where a is the source term and u is the solution field.
    Boundary condition: u = 0 on all boundaries
    """

    def __init__(self, dataset_ref):
        self.normalizer = dataset_ref.create_normalizer()
        self.loss_func = LpLoss(d=2, p=2)

    def __call__(self, x_pred):
        """Calculate the PDE loss and boundary condition loss for Poisson equation.

        Args:
            x_pred (torch.Tensor): Predicted data containing both fields
                First channel ([:,0:1]) is source term a
                Second channel ([:,1:2]) is solution field u

        Returns:
            tuple: (pde_loss, bc_loss)
                pde_loss: PDE residual in the interior
                bc_loss: Boundary condition loss (u should be 0 on boundaries)
        """
        x_pred_denorm = self.normalizer.denormalize(x_pred)
        a_pred = x_pred_denorm[:, 0:1]
        u_pred = x_pred_denorm[:, 1:2]

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


def visualize_results(data_initial, data_optimized, losses, pde_losses=None, bc_losses=None, save_path="optimization_results.png"):
    """Visualize the optimization results."""
    data_initial_np = data_initial[0].detach().cpu().numpy()
    data_optimized_np = data_optimized[0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Initial state
    im1 = axes[0, 0].imshow(data_initial_np[0], cmap="viridis")
    axes[0, 0].set_title("Initial: Source term (a)")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(data_initial_np[1], cmap="viridis")
    axes[0, 1].set_title("Initial: Solution field (u)")
    plt.colorbar(im2, ax=axes[0, 1])

    # Optimized state
    im3 = axes[1, 0].imshow(data_optimized_np[0], cmap="viridis")
    axes[1, 0].set_title("Optimized: Source term (a)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(data_optimized_np[1], cmap="viridis")
    axes[1, 1].set_title("Optimized: Solution field (u)")
    plt.colorbar(im4, ax=axes[1, 1])

    # Difference
    diff_a = data_optimized_np[0] - data_initial_np[0]
    diff_u = data_optimized_np[1] - data_initial_np[1]

    im5 = axes[0, 2].imshow(diff_a, cmap="bwr", vmin=-np.abs(diff_a).max(), vmax=np.abs(diff_a).max())
    axes[0, 2].set_title("Change in source term (a)")
    plt.colorbar(im5, ax=axes[0, 2])

    im6 = axes[1, 2].imshow(diff_u, cmap="bwr", vmin=-np.abs(diff_u).max(), vmax=np.abs(diff_u).max())
    axes[1, 2].set_title("Change in solution field (u)")
    plt.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {save_path}")
    plt.close()

    # Plot loss curve
    if pde_losses is not None and bc_losses is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Total loss
        axes[0].plot(losses, linewidth=2, color="blue")
        axes[0].set_xlabel("Optimization Step")
        axes[0].set_ylabel("Total Loss")
        axes[0].set_title("Total Loss (PDE + BC)")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale("log")

        # PDE loss
        axes[1].plot(pde_losses, linewidth=2, color="red")
        axes[1].set_xlabel("Optimization Step")
        axes[1].set_ylabel("PDE Loss")
        axes[1].set_title("PDE Loss (∇²u = a)")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("log")

        # BC loss
        axes[2].plot(bc_losses, linewidth=2, color="green")
        axes[2].set_xlabel("Optimization Step")
        axes[2].set_ylabel("BC Loss")
        axes[2].set_title("Boundary Condition Loss (u = 0)")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale("log")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(losses, linewidth=2)
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("PDE Loss")
        ax.set_title("Poisson PDE Loss During Optimization")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    plt.tight_layout()
    loss_path = save_path.replace(".png", "_loss_curve.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    print(f"Loss curve saved to {loss_path}")
    plt.close()


def plot_boundary_lines(data_initial, data_optimized, save_path="boundary_lines.png"):
    """Plot the four boundary lines (top, bottom, left, right) for initial and optimized states."""
    # Extract solution field (channel 1) in denormalized space
    u_initial = data_initial[0, 1].detach().cpu().numpy()
    u_optimized = data_optimized[0, 1].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top boundary
    axes[0, 0].plot(u_initial[0, :], label="Initial", linewidth=2, alpha=0.7)
    axes[0, 0].plot(u_optimized[0, :], label="Optimized", linewidth=2, alpha=0.7)
    axes[0, 0].axhline(y=1.0, color="red", linestyle="--", label="Target=1", alpha=0.5)
    axes[0, 0].set_title("Top Boundary (row 0)")
    axes[0, 0].set_xlabel("x position")
    axes[0, 0].set_ylabel("u value")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Bottom boundary
    axes[0, 1].plot(u_initial[-1, :], label="Initial", linewidth=2, alpha=0.7)
    axes[0, 1].plot(u_optimized[-1, :], label="Optimized", linewidth=2, alpha=0.7)
    axes[0, 1].axhline(y=1.0, color="red", linestyle="--", label="Target=1", alpha=0.5)
    axes[0, 1].set_title("Bottom Boundary (row -1)")
    axes[0, 1].set_xlabel("x position")
    axes[0, 1].set_ylabel("u value")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Left boundary
    axes[1, 0].plot(u_initial[:, 0], label="Initial", linewidth=2, alpha=0.7)
    axes[1, 0].plot(u_optimized[:, 0], label="Optimized", linewidth=2, alpha=0.7)
    axes[1, 0].axhline(y=1.0, color="red", linestyle="--", label="Target=1", alpha=0.5)
    axes[1, 0].set_title("Left Boundary (col 0)")
    axes[1, 0].set_xlabel("y position")
    axes[1, 0].set_ylabel("u value")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Right boundary
    axes[1, 1].plot(u_initial[:, -1], label="Initial", linewidth=2, alpha=0.7)
    axes[1, 1].plot(u_optimized[:, -1], label="Optimized", linewidth=2, alpha=0.7)
    axes[1, 1].axhline(y=1.0, color="red", linestyle="--", label="Target=1", alpha=0.5)
    axes[1, 1].set_title("Right Boundary (col -1)")
    axes[1, 1].set_xlabel("y position")
    axes[1, 1].set_ylabel("u value")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Boundary lines plot saved to {save_path}")
    plt.close()


def main():
    # Configuration
    DATASET_NAME = "poisson"
    DATA_RESOLUTION = 128
    NUM_STEPS = 10000
    LEARNING_RATE = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset: data/DiffPDE/{DATASET_NAME}_hf")
    dataset = PDEDataset(
        path=f"data/DiffPDE/{DATASET_NAME}_hf",
        resolution=DATA_RESOLUTION,
        max_size=10,
    )

    # Load one sample
    data_sample, _ = dataset[0]
    data_sample = torch.from_numpy(data_sample).unsqueeze(0).to(device).float()  # Add batch dimension
    print(f"Data sample shape: {data_sample.shape}")

    # Store initial state
    data_initial = data_sample.clone()

    # Fix channel 0 (source term 'a') and initialize channel 1 (solution 'u') with random values
    source_term = data_sample[:, 0:1].clone()  # Keep the source term fixed
    solution_field = torch.randn_like(data_sample[:, 1:2]) * 1.0  # Random initialization for solution

    # Combine into optimizable data - only solution field requires grad
    data_optimized = torch.cat([source_term.detach(), solution_field], dim=1)
    data_optimized[:, 1:2].requires_grad_(True)

    print(f"Fixed source term (channel 0), initialized solution (channel 1) with random values")
    print(f"Solution field initial range: [{solution_field.min().item():.4f}, {solution_field.max().item():.4f}]")

    # Initialize Poisson loss
    criterion_pde = PoissonLoss(dataset)

    # Optimizer - only optimize the solution field (channel 1)
    solution_param = data_optimized[:, 1:2].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([solution_param], lr=LEARNING_RATE)

    # Optimization loop
    print(f"\nStarting optimization for {NUM_STEPS} steps...")
    losses = []
    pde_losses = []
    bc_losses = []

    for step in range(NUM_STEPS):
        optimizer.zero_grad()

        # Reconstruct data with fixed source term and optimized solution
        data_optimized = torch.cat([source_term, solution_param], dim=1)

        # Calculate PDE loss and boundary condition loss
        pde_loss, bc_loss = criterion_pde(data_optimized)

        # Total loss is combination of PDE and boundary losses
        loss = 1 * pde_loss + 1000 * bc_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pde_losses.append(pde_loss.item())
        bc_losses.append(bc_loss.item())

        # if (step + 1) % 10 == 0:
        #     print(f"Step {step+1}/{NUM_STEPS}, Loss: {loss.item():.6f}, PDE: {pde_loss.item():.6f}, BC: {bc_loss.item():.6f}")

    # Final data with optimized solution
    data_optimized = torch.cat([source_term, solution_param], dim=1)

    print(f"\nOptimization complete!")
    print(f"Initial total loss: {losses[0]:.6f} (PDE: {pde_losses[0]:.6f}, BC: {bc_losses[0]:.6f})")
    print(f"Final total loss: {losses[-1]:.6f} (PDE: {pde_losses[-1]:.6f}, BC: {bc_losses[-1]:.6f})")
    print(f"Total loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")
    print(f"PDE loss reduction: {(pde_losses[0] - pde_losses[-1]) / pde_losses[0] * 100:.2f}%")
    print(f"BC loss reduction: {(bc_losses[0] - bc_losses[-1]) / bc_losses[0] * 100:.2f}%")

    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(data_initial, data_optimized.detach(), losses, pde_losses, bc_losses)

    # Plot boundary lines - need to denormalize first
    criterion_pde = PoissonLoss(dataset)
    data_initial_denorm = criterion_pde.normalizer.denormalize(data_initial)
    data_optimized_denorm = criterion_pde.normalizer.denormalize(data_optimized.detach())
    plot_boundary_lines(data_initial_denorm, data_optimized_denorm)

    # Print statistics
    print("\nStatistics:")
    data_initial_np = data_initial[0].detach().cpu().numpy()
    data_optimized_np = data_optimized[0].detach().cpu().numpy()

    print(f"Initial - Source term (a) range: [{data_initial_np[0].min():.4f}, {data_initial_np[0].max():.4f}]")
    print(f"Initial - Solution field (u) range: [{data_initial_np[1].min():.4f}, {data_initial_np[1].max():.4f}]")
    print(f"Optimized - Source term (a) range: [{data_optimized_np[0].min():.4f}, {data_optimized_np[0].max():.4f}]")
    print(f"Optimized - Solution field (u) range: [{data_optimized_np[1].min():.4f}, {data_optimized_np[1].max():.4f}]")

    print("\nMean absolute change:")
    print(f"Source term (a): {np.abs(data_optimized_np[0] - data_initial_np[0]).mean():.6f}")
    print(f"Solution field (u): {np.abs(data_optimized_np[1] - data_initial_np[1]).mean():.6f}")


if __name__ == "__main__":
    main()
