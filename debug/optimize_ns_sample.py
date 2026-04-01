"""
Optimize a single data sample using Navier-Stokes PDE loss.

This script loads one sample from the Navier-Stokes non-bounded dataset and optimizes it
to satisfy the incompressibility constraint (divergence-free vorticity) for multiple steps.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from neuralop.losses import LpLoss
from training.dataset_hf import PDEDataset
from pino_loss import get_pde_loss


def visualize_results(data_initial, data_optimized, losses, pde_losses=None, bc_losses=None, save_path="optimization_results_ns.png"):
    """Visualize the optimization results for Navier-Stokes."""
    data_initial_np = data_initial[0].detach().cpu().numpy()
    data_optimized_np = data_optimized[0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Initial state
    im1 = axes[0, 0].imshow(data_initial_np[0], cmap="RdBu_r")
    axes[0, 0].set_title("Initial: Input field")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(data_initial_np[1], cmap="RdBu_r")
    axes[0, 1].set_title("Initial: Vorticity field (ω)")
    plt.colorbar(im2, ax=axes[0, 1])

    # Optimized state
    im3 = axes[1, 0].imshow(data_optimized_np[0], cmap="RdBu_r")
    axes[1, 0].set_title("Optimized: Input field")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(data_optimized_np[1], cmap="RdBu_r")
    axes[1, 1].set_title("Optimized: Vorticity field (ω)")
    plt.colorbar(im4, ax=axes[1, 1])

    # Difference
    diff_input = data_optimized_np[0] - data_initial_np[0]
    diff_vorticity = data_optimized_np[1] - data_initial_np[1]

    im5 = axes[0, 2].imshow(diff_input, cmap="bwr", vmin=-np.abs(diff_input).max(), vmax=np.abs(diff_input).max())
    axes[0, 2].set_title("Change in input field")
    plt.colorbar(im5, ax=axes[0, 2])

    im6 = axes[1, 2].imshow(diff_vorticity, cmap="bwr", vmin=-np.abs(diff_vorticity).max(), vmax=np.abs(diff_vorticity).max())
    axes[1, 2].set_title("Change in vorticity field (ω)")
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
        axes[0].set_title("Total Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale("log")

        # PDE loss
        axes[1].plot(pde_losses, linewidth=2, color="red")
        axes[1].set_xlabel("Optimization Step")
        axes[1].set_ylabel("PDE Loss")
        axes[1].set_title("PDE Loss (∇ · ω = 0)")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("log")

        # BC loss (should be zero for non-bounded NS)
        axes[2].plot(bc_losses, linewidth=2, color="green")
        axes[2].set_xlabel("Optimization Step")
        axes[2].set_ylabel("BC Loss")
        axes[2].set_title("Boundary Condition Loss (N/A for non-bounded)")
        axes[2].grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(losses, linewidth=2)
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("PDE Loss")
        ax.set_title("Navier-Stokes PDE Loss During Optimization")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    loss_path = save_path.replace(".png", "_loss_curve.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    print(f"Loss curve saved to {loss_path}")
    plt.close()


def visualize_divergence(data_optimized, criterion_pde, save_path="divergence_field_ns.png"):
    """Visualize the divergence of vorticity field to verify incompressibility."""
    # Denormalize data
    data_denorm = criterion_pde.normalizer.denormalize(data_optimized)
    vorticity = data_denorm[:, 1:2]

    device = vorticity.device
    dx = 1.0
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * dx)
    deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * dx)

    deriv_x = deriv_x.to(dtype=vorticity.dtype, device=vorticity.device)
    deriv_y = deriv_y.to(dtype=vorticity.dtype, device=vorticity.device)

    # Calculate divergence
    div_vort_x = F.conv2d(vorticity, deriv_x, padding=(0, 1))
    div_vort_y = F.conv2d(vorticity, deriv_y, padding=(1, 0))
    divergence = div_vort_x + div_vort_y
    divergence_interior = divergence[:, :, 1:-1, 1:-1]

    divergence_np = divergence_interior[0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Vorticity field
    vorticity_np = vorticity[0, 0].detach().cpu().numpy()
    im1 = axes[0].imshow(vorticity_np, cmap="RdBu_r")
    axes[0].set_title("Vorticity Field (ω)")
    plt.colorbar(im1, ax=axes[0])

    # Divergence field
    max_abs = np.abs(divergence_np).max()
    im2 = axes[1].imshow(divergence_np, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    axes[1].set_title(f"Divergence of Vorticity (∇ · ω)\nMax: {max_abs:.2e}")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Divergence field visualization saved to {save_path}")
    plt.close()

    return divergence_np


def main():
    # Configuration
    DATASET_NAME = "ns-nonbounded"
    DATA_RESOLUTION = 128  # NS dataset is typically 64x64
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

    # Fix channel 0 (input field) and initialize channel 1 (vorticity) with random values
    input_field = data_sample[:, 0:1].clone()  # Keep the input field fixed
    # vorticity_field = torch.randn_like(data_sample[:, 1:2]) * 0.5  # Random initialization for vorticity
    vorticity_field = data_sample[:, 1:2].clone()  # Random initialization for vorticity

    # Combine into optimizable data - only vorticity field requires grad
    data_optimized = torch.cat([input_field.detach(), vorticity_field.detach()], dim=1)
    data_optimized[:, 1:2].requires_grad_(True)

    print(f"Fixed input field (channel 0), initialized vorticity (channel 1) with random values")
    print(f"Vorticity field initial range: [{vorticity_field.min().item():.4f}, {vorticity_field.max().item():.4f}]")

    # Initialize Navier-Stokes loss
    criterion_pde = get_pde_loss(DATASET_NAME, dataset)
    print(f"Using PDE loss: {type(criterion_pde).__name__}")

    # Optimizer - only optimize the vorticity field (channel 1)
    vorticity_param = data_optimized[:, 1:2].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([vorticity_param], lr=LEARNING_RATE)

    # Optimization loop
    print(f"\nStarting optimization for {NUM_STEPS} steps...")
    losses = []
    pde_losses = []
    bc_losses = []

    for step in range(NUM_STEPS):
        optimizer.zero_grad()

        # Reconstruct data with fixed input field and optimized vorticity
        data_optimized = torch.cat([input_field, vorticity_param], dim=1)

        # Calculate PDE loss and boundary condition loss
        pde_loss, bc_loss = criterion_pde(data_optimized)

        # Total loss (bc_loss should be 0 for non-bounded NS)
        loss = pde_loss + bc_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pde_losses.append(pde_loss.item())
        bc_losses.append(bc_loss.item())

        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}/{NUM_STEPS}, Loss: {loss.item():.6f}, PDE: {pde_loss.item():.6f}, BC: {bc_loss.item():.6f}")

    # Final data with optimized vorticity
    data_optimized = torch.cat([input_field, vorticity_param], dim=1)

    print(f"\nOptimization complete!")
    print(f"Initial total loss: {losses[0]:.6f} (PDE: {pde_losses[0]:.6f}, BC: {bc_losses[0]:.6f})")
    print(f"Final total loss: {losses[-1]:.6f} (PDE: {pde_losses[-1]:.6f}, BC: {bc_losses[-1]:.6f})")
    print(f"Total loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")
    print(f"PDE loss reduction: {(pde_losses[0] - pde_losses[-1]) / pde_losses[0] * 100:.2f}%")

    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(data_initial, data_optimized.detach(), losses, pde_losses, bc_losses)

    # Visualize divergence field
    divergence_np = visualize_divergence(data_optimized.detach(), criterion_pde)

    # Print statistics
    print("\nStatistics:")
    data_initial_np = data_initial[0].detach().cpu().numpy()
    data_optimized_np = data_optimized[0].detach().cpu().numpy()

    print(f"Initial - Input field range: [{data_initial_np[0].min():.4f}, {data_initial_np[0].max():.4f}]")
    print(f"Initial - Vorticity field range: [{data_initial_np[1].min():.4f}, {data_initial_np[1].max():.4f}]")
    print(f"Optimized - Input field range: [{data_optimized_np[0].min():.4f}, {data_optimized_np[0].max():.4f}]")
    print(f"Optimized - Vorticity field range: [{data_optimized_np[1].min():.4f}, {data_optimized_np[1].max():.4f}]")

    print("\nMean absolute change:")
    print(f"Input field: {np.abs(data_optimized_np[0] - data_initial_np[0]).mean():.6f}")
    print(f"Vorticity field: {np.abs(data_optimized_np[1] - data_initial_np[1]).mean():.6f}")

    print("\nDivergence statistics (interior points):")
    print(f"Mean absolute divergence: {np.abs(divergence_np).mean():.2e}")
    print(f"Max absolute divergence: {np.abs(divergence_np).max():.2e}")
    print(f"RMS divergence: {np.sqrt(np.mean(divergence_np**2)):.2e}")


if __name__ == "__main__":
    main()
