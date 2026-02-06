import numpy as np
import os
from datetime import datetime

def forward_poisson(f: np.ndarray, S: int):
    """
    Solve the 2D Poisson problem on [0,1]x[0,1]:
        Δφ = f
    with homogeneous Dirichlet boundary conditions:
        φ = 0 on the boundary.

    Discretization: uniform SxS grid, 5-point finite-difference stencil.
    Ordering (matches your MATLAB code): index = (i)*S + j with i,j = 0..S-1 (row-major).
    """
    f = np.asarray(f, dtype=float)
    assert f.shape == (S, S), "f must be an SxS array"

    h = 1.0 / (S - 1)
    N = S * S

    # --- assemble sparse matrix A and vector B ---
    try:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve
    except Exception as e:
        raise ImportError(
            "This solver uses SciPy (scipy.sparse, scipy.sparse.linalg). "
            "Install via: pip install scipy"
        ) from e

    A = lil_matrix((N, N), dtype=float)
    B = f.reshape(N).copy()  # row-major flattening consistent with index = i*S + j

    for i in range(S):
        for j in range(S):
            k = i * S + j  # linear index

            # Dirichlet boundary: φ = 0
            if i == 0 or i == S - 1 or j == 0 or j == S - 1:
                A[k, k] = 1.0
                B[k] = 0.0
            else:
                A[k, k] = -4.0 / h**2
                A[k, k - 1] =  1.0 / h**2     # left  (i, j-1)
                A[k, k + 1] =  1.0 / h**2     # right (i, j+1)
                A[k, k - S] =  1.0 / h**2     # down  (i-1, j)
                A[k, k + S] =  1.0 / h**2     # up    (i+1, j)

    phi_vec = spsolve(A.tocsr(), B)
    phi = phi_vec.reshape(S, S)
    return phi


# ---- test on real dataset ----
if __name__ == "__main__":
    # Configuration
    batch_size = 4  # Number of samples to process and visualize
    
    # Load dataset
    try:
        from training.dataset_hf import PDEDataset
    except ImportError as e:
        raise ImportError("Cannot import PDEDataset. Make sure you're running from the project root.") from e

    dataset_path = "data/DiffPDE/poisson_test_hf"
    dataset = PDEDataset(path=dataset_path, resolution=None)  # Use original resolution
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Image shape: {dataset.image_shape}")
    print(f"Resolution: {dataset.resolution}")
    print(f"Processing {batch_size} samples (0-{batch_size-1})...")
    
    # Batch load and denormalize data on GPU
    import torch
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    num_samples = min(batch_size, len(dataset))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    normalizer = dataset.create_normalizer()
    
    # Batch load all samples and stack them
    print("Loading and denormalizing samples in batch...")
    batch_data = []
    for sample_idx in range(num_samples):
        data, _ = dataset[sample_idx]
        batch_data.append(data)
    
    # Stack and convert to torch tensor: (batch_size, 2, H, W)
    batch_tensor = torch.tensor(np.stack(batch_data), dtype=torch.float32)
    if device.type == 'cuda':
        batch_tensor = batch_tensor.to(device)
    
    # Denormalize the entire batch at once (normalizer handles device internally)
    batch_denorm = normalizer.denormalize(batch_tensor)
    
    # Extract to CPU numpy arrays
    if device.type == 'cuda':
        batch_denorm = batch_denorm.cpu()
    batch_denorm = batch_denorm.numpy()
    
    # Extract channels: f and phi_true for all samples
    f_batch = batch_denorm[:, 0, :, :]  # (batch_size, H, W) - source terms
    phi_true_batch = batch_denorm[:, 1, :, :]  # (batch_size, H, W) - true solutions
    
    S = phi_true_batch.shape[1]
    print(f"Using resolution S={S}")
    
    # Create coordinate grids (same for all samples)
    x = np.linspace(0.0, 1.0, S)
    y = np.linspace(0.0, 1.0, S)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    # Solve Poisson equations in parallel
    print("Solving Poisson equations in parallel...")
    
    def solve_poisson_wrapper(args):
        """Wrapper function for parallel solving"""
        f, sample_idx = args
        phi_hat = forward_poisson(f, S)
        return sample_idx, phi_hat
    
    # Prepare arguments for parallel processing
    solve_args = [(f_batch[i], i) for i in range(num_samples)]
    
    # Solve in parallel using ProcessPoolExecutor
    phi_hat_batch = {}
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(solve_poisson_wrapper, args): args[1] for args in solve_args}
        for future in as_completed(futures):
            sample_idx, phi_hat = future.result()
            phi_hat_batch[sample_idx] = phi_hat
            err = np.linalg.norm(phi_hat - phi_true_batch[sample_idx]) / np.linalg.norm(phi_true_batch[sample_idx])
            print(f"Sample {sample_idx} solved - relative L2 error: {err:.6e}")
    
    # Sort results by sample index
    phi_hat_batch = [phi_hat_batch[i] for i in range(num_samples)]
    
    # Save separate figures for each sample
    try:
        import matplotlib.pyplot as plt
        
        base_exp_dir = "exps/poisson_solver"
        os.makedirs(base_exp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a new folder with timestamp for this run
        exp_dir = os.path.join(base_exp_dir, timestamp)
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"\nSaving {num_samples} separate figures to: {exp_dir}/")
        for sample_idx in range(num_samples):
            f = f_batch[sample_idx]
            phi_true = phi_true_batch[sample_idx]
            phi_hat = phi_hat_batch[sample_idx]
            err = np.linalg.norm(phi_hat - phi_true) / np.linalg.norm(phi_true)
            error_field = phi_hat - phi_true
            
            # Create figure for this sample
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Poisson Solver Test - Sample {sample_idx} (S={S}, relative L2 error: {err:.2e})', 
                         fontsize=14, fontweight='bold')
            
            # True solution
            im1 = axes[0, 0].contourf(X, Y, phi_true, levels=20, cmap='viridis')
            axes[0, 0].set_title('True Solution φ(x,y)')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('y')
            axes[0, 0].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Computed solution
            im2 = axes[0, 1].contourf(X, Y, phi_hat, levels=20, cmap='viridis')
            axes[0, 1].set_title('Computed Solution φ̂(x,y)')
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('y')
            axes[0, 1].set_aspect('equal')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Error
            im3 = axes[1, 0].contourf(X, Y, error_field, levels=20, cmap='RdBu_r')
            axes[1, 0].set_title('Error: φ̂ - φ')
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('y')
            axes[1, 0].set_aspect('equal')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Source term
            im4 = axes[1, 1].contourf(X, Y, f, levels=20, cmap='coolwarm')
            axes[1, 1].set_title('Source Term f(x,y) = Δφ')
            axes[1, 1].set_xlabel('x')
            axes[1, 1].set_ylabel('y')
            axes[1, 1].set_aspect('equal')
            plt.colorbar(im4, ax=axes[1, 1])
            
            plt.tight_layout()
            
            # Save figure
            filename = os.path.join(exp_dir, f"poisson_solver_sample{sample_idx}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {filename}")
        
        print(f"\nAll figures saved to: {exp_dir}/")

    except ImportError:
        print("matplotlib not available - skipping visualization")
        print("Install via: pip install matplotlib")
