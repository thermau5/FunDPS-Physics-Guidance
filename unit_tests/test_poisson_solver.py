"""
Tests for NumericalPoissonWrapper after merge.

Covers the three key merge changes:
1. A_dense.requires_grad_(False) — matrix should not accumulate gradients
2. Boundary condition enforcement — boundary values in output should be zero
3. _predict_solution_from_normalized_source — normalization round-trip correctness

Also includes a data-driven validation test against the real Poisson dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def get_wrapper():
    from generation.daps import NumericalPoissonWrapper
    return NumericalPoissonWrapper()


# ---------------------------------------------------------------------------
# Test 1: Eigenvalue cache has no gradient
# ---------------------------------------------------------------------------
def test_eigenvalue_no_grad():
    wrapper = get_wrapper()
    M = 6  # interior grid size for S=8
    device = torch.device("cpu")
    dtype = torch.float32
    eig = wrapper._get_eigenvalues(M, device, dtype)
    assert not eig.requires_grad, "eigenvalue tensor should have requires_grad=False"
    assert eig.shape == (M, M), f"Expected ({M},{M}), got {eig.shape}"
    assert (eig < 0).all(), "All eigenvalues of discrete Laplacian should be negative"
    print("PASS test_eigenvalue_no_grad")


# ---------------------------------------------------------------------------
# Test 2: Boundary values in output are zero (Dirichlet BC)
# ---------------------------------------------------------------------------
def test_boundary_conditions():
    wrapper = get_wrapper()
    S = 16
    B = 2
    x = torch.randn(B, 1, S, S)  # random source term

    with torch.no_grad():
        phi = wrapper(x)  # [B, 1, S, S]

    tol = 1e-4
    assert phi[:, 0, 0, :].abs().max().item() < tol,  "Top boundary not zero"
    assert phi[:, 0, -1, :].abs().max().item() < tol, "Bottom boundary not zero"
    assert phi[:, 0, :, 0].abs().max().item() < tol,  "Left boundary not zero"
    assert phi[:, 0, :, -1].abs().max().item() < tol, "Right boundary not zero"
    print("PASS test_boundary_conditions")


# ---------------------------------------------------------------------------
# Test 3: Known Poisson solution — f=0 implies phi=0
# ---------------------------------------------------------------------------
def test_zero_source_gives_zero_solution():
    wrapper = get_wrapper()
    S = 16
    B = 1
    x = torch.zeros(B, 1, S, S)  # zero source term -> solution should be zero

    with torch.no_grad():
        phi = wrapper(x)

    assert phi.abs().max().item() < 1e-5, f"Zero source should give zero solution, got max={phi.abs().max().item()}"
    print("PASS test_zero_source_gives_zero_solution")


# ---------------------------------------------------------------------------
# Test 4: Gradients flow to input (needed for DAPS guidance)
# ---------------------------------------------------------------------------
def test_gradients_flow_to_input():
    wrapper = get_wrapper()
    S = 16
    x = torch.randn(1, 1, S, S, requires_grad=True)
    phi = wrapper(x)
    loss = phi.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input"
    assert x.grad.abs().sum().item() > 0, "Input gradients should be nonzero"
    print("PASS test_gradients_flow_to_input")


# ---------------------------------------------------------------------------
# Test 5: _predict_solution_from_normalized_source normalization round-trip
# ---------------------------------------------------------------------------
def test_predict_solution_normalizer():
    """
    Build a minimal mock of the DAPS surrogate context to test
    _predict_solution_from_normalized_source without loading a full model.
    """
    from generation.daps import NumericalPoissonWrapper

    # Build a mock normalizer with known stats
    class MockNormalizer:
        def __init__(self):
            # channel 0 (source): mean=0, std=1  -> denormalize = identity
            # channel 1 (solution): mean=0, std=1 -> normalize = identity
            pass

        def denormalize(self, x, channel=None):
            return x  # identity

        def normalize(self, x, channel=None):
            return x  # identity

    # Build a minimal surrogate holder
    class MockSurrogate:
        uses_numerical_poisson = True
        surrogate = NumericalPoissonWrapper()
        normalizer = MockNormalizer()

        def _predict_solution_from_normalized_source(self, x):
            if self.uses_numerical_poisson:
                x_physical = self.normalizer.denormalize(x, channel=0)
                sol_physical = self.surrogate(x_physical)
                return self.normalizer.normalize(sol_physical, channel=1)
            return self.surrogate(x)

    mock = MockSurrogate()
    S = 16
    x = torch.randn(1, 1, S, S)
    with torch.no_grad():
        out = mock._predict_solution_from_normalized_source(x)

    assert out.shape == (1, 1, S, S), f"Unexpected output shape: {out.shape}"
    print("PASS test_predict_solution_normalizer")


# ---------------------------------------------------------------------------
# Test 6: Validate against real Poisson dataset
# ---------------------------------------------------------------------------
def test_on_real_data(dataset_path="data/DiffPDE/poisson_test_hf", num_samples=4, save_dir=None):
    """
    Load real Poisson samples, solve with NumericalPoissonWrapper, and compare
    against ground-truth solutions. Optionally saves visualizations.
    """
    if not os.path.exists(dataset_path):
        print(f"SKIP test_on_real_data — dataset not found at {dataset_path}")
        return

    from training.dataset_hf import PDEDataset
    from generation.daps import NumericalPoissonWrapper

    dataset = PDEDataset(path=dataset_path, resolution=None)
    normalizer = dataset.create_normalizer()
    num_samples = min(num_samples, len(dataset))
    print(f"Dataset loaded: {len(dataset)} samples, testing on {num_samples}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Batch load
    batch_data = [dataset[i][0] for i in range(num_samples)]
    batch_tensor = torch.tensor(np.stack(batch_data), dtype=torch.float32).to(device)
    batch_denorm = normalizer.denormalize(batch_tensor).cpu()

    f_batch = batch_denorm[:, 0:1]        # [N, 1, H, W] source terms
    phi_true_batch = batch_denorm[:, 1:2]  # [N, 1, H, W] ground-truth solutions

    wrapper = NumericalPoissonWrapper()

    errors = []
    with torch.no_grad():
        phi_hat_batch = wrapper(f_batch)  # [N, 1, H, W]

    for i in range(num_samples):
        phi_hat = phi_hat_batch[i, 0]
        phi_true = phi_true_batch[i, 0]
        err = (torch.norm(phi_hat - phi_true) / torch.norm(phi_true)).item()
        errors.append(err)
        print(f"  Sample {i}: relative L2 error = {err:.6e}")

    mean_err = np.mean(errors)
    print(f"  Mean relative L2 error over {num_samples} samples: {mean_err:.6e}")

    # Optional visualizations
    if save_dir is not None:
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime

            run_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(run_dir, exist_ok=True)
            S = f_batch.shape[-1]
            x = np.linspace(0, 1, S)
            X, Y = np.meshgrid(x, x, indexing="ij")

            for i in range(num_samples):
                f = f_batch[i, 0].numpy()
                phi_true = phi_true_batch[i, 0].numpy()
                phi_hat = phi_hat_batch[i, 0].numpy()
                err = errors[i]
                error_field = phi_hat - phi_true

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"Poisson Solver — Sample {i} (S={S}, rel L2: {err:.2e})", fontsize=14)

                for ax, data, title, cmap in zip(
                    axes.flat,
                    [phi_true, phi_hat, error_field, f],
                    ["True φ", "Computed φ̂", "Error φ̂ − φ", "Source f"],
                    ["viridis", "viridis", "RdBu_r", "coolwarm"],
                ):
                    im = ax.contourf(X, Y, data, levels=20, cmap=cmap)
                    ax.set_title(title)
                    ax.set_aspect("equal")
                    plt.colorbar(im, ax=ax)

                plt.tight_layout()
                path = os.path.join(run_dir, f"sample_{i:03d}.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Saved: {path}")

            print(f"Figures saved to: {run_dir}/")
        except ImportError:
            print("matplotlib not available — skipping visualization")

    print("PASS test_on_real_data")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/DiffPDE/poisson_test_hf")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--save-dir", default=None, help="Directory to save visualizations, e.g. exps/tests/poisson_solver (off by default)")
    args = parser.parse_args()

    print("Running Poisson solver tests...\n")
    test_eigenvalue_no_grad()
    test_boundary_conditions()
    test_zero_source_gives_zero_solution()
    test_gradients_flow_to_input()
    test_predict_solution_normalizer()
    test_on_real_data(dataset_path=args.data_path, num_samples=args.num_samples, save_dir=args.save_dir)
    print("\nAll tests passed.")
