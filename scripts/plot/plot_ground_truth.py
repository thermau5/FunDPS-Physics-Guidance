import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from matplotlib.colors import Normalize, PowerNorm
import numpy as np
import matplotlib.pyplot as plt


def get_norm(vmin, vmax, gamma=1.0):
    """Get normalization object. If gamma != 1.0, use PowerNorm for enhanced contrast.
    
    Args:
        vmin: Minimum value
        vmax: Maximum value
        gamma: Gamma correction factor (default: 1.0)
               - gamma < 1.0: Enhances contrast in middle ranges (makes differences more visible)
               - gamma = 1.0: Linear normalization (default)
               - gamma > 1.0: Enhances contrast at extremes
    
    Returns:
        Normalize or PowerNorm object
    """
    if gamma == 1.0:
        return Normalize(vmin=vmin, vmax=vmax)
    else:
        return PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)


def plot_ground_truth_individual(gt_data_list, save_dir=None, problem_type="both", gamma=1.0):
    """
    Plot individual ground truth images, one per figure, without colorbars.
    
    Args:
        gt_data_list: List of dictionaries, each containing:
            - inverse: Inverse problem ground truth array (2D)
            - forward: Forward problem ground truth array (2D)
            - sample_idx: Sample index (optional, for labeling)
        save_dir: Optional directory to save the figures (if None, figures are only displayed)
        problem_type: Which problems to plot. Options:
            - "both": Plot both inverse and forward (default)
            - "inverse": Plot only inverse problems
            - "forward": Plot only forward problems
        gamma: Gamma correction factor for color contrast
    """
    for gt_data in gt_data_list:
        inverse = gt_data["inverse"]
        forward = gt_data["forward"]
        sample_idx = gt_data.get("sample_idx", 0)
        
        # Get min/max for each problem type
        inv_min, inv_max = np.min(inverse), np.max(inverse)
        fwd_min, fwd_max = np.min(forward), np.max(forward)
        
        # Plot inverse problem if requested
        if problem_type in ["both", "inverse"]:
            fig_inv = plt.figure(figsize=(6, 6))
            ax_inv = fig_inv.add_subplot(111)
            ax_inv.imshow(inverse, cmap="viridis", norm=get_norm(inv_min, inv_max, gamma=gamma), interpolation="none")
            ax_inv.set_xticks([])
            ax_inv.set_yticks([])
            ax_inv.axis('off')  # Remove all axes and borders
            
            if save_dir:
                save_path_inv = os.path.join(save_dir, f"gt_sample_{sample_idx}_inverse.png")
                plt.savefig(save_path_inv, bbox_inches="tight", pad_inches=0, dpi=300)
                print(f"Saved: {save_path_inv}")
            
            plt.show()
        
        # Plot forward problem if requested
        if problem_type in ["both", "forward"]:
            fig_fwd = plt.figure(figsize=(6, 6))
            ax_fwd = fig_fwd.add_subplot(111)
            ax_fwd.imshow(forward, cmap="viridis", norm=get_norm(fwd_min, fwd_max, gamma=gamma), interpolation="none")
            ax_fwd.set_xticks([])
            ax_fwd.set_yticks([])
            ax_fwd.axis('off')  # Remove all axes and borders
            
            if save_dir:
                save_path_fwd = os.path.join(save_dir, f"gt_sample_{sample_idx}_forward.png")
                plt.savefig(save_path_fwd, bbox_inches="tight", pad_inches=0, dpi=300)
                print(f"Saved: {save_path_fwd}")
            
            plt.show()


def plot_ground_truth_with_sparse_observations(gt_data_list, save_dir=None, n_obs=None, obs_ratio=None, mask=None, gamma=1.0, sparse_bg_color='black', random_seed=None):
    """
    Plot ground truth images along with sparse observations used for inverse problems.
    Shows: GT Forward, Sparse Observation (masked forward), GT Inverse.
    
    Args:
        gt_data_list: List of dictionaries, each containing:
            - inverse: Inverse problem ground truth array (2D)
            - forward: Forward problem ground truth array (2D)
            - sample_idx: Sample index (optional, for labeling)
        save_dir: Optional directory to save the figures (if None, figures are only displayed)
        n_obs: Number of observation points for sparse mask (if mask is not provided)
              Can be an integer or a float between 0 and 1 (ratio of total points)
        obs_ratio: Alternative to n_obs - ratio of observed points (0-1)
                   If both n_obs and obs_ratio are provided, n_obs takes precedence
        mask: Optional pre-computed mask array (2D binary array). If provided, n_obs/obs_ratio are ignored.
              If None, a random mask will be generated using n_obs or obs_ratio
        gamma: Gamma correction factor for color contrast
        sparse_bg_color: Background color for sparse observation plot. Options: 'black', 'white', or 'gray' (default: 'black')
        random_seed: Random seed for mask generation. If None (default), uses truly random generation.
                     If provided, ensures reproducible masks.
    """
    for gt_data in gt_data_list:
        inverse = gt_data["inverse"]
        forward = gt_data["forward"]
        sample_idx = gt_data.get("sample_idx", 0)
        
        # Get min/max for each problem type
        inv_min, inv_max = np.min(inverse), np.max(inverse)
        fwd_min, fwd_max = np.min(forward), np.max(forward)
        
        # Generate or use provided mask
        if mask is not None:
            # Use provided mask
            obs_mask = mask
        else:
            # Generate random mask
            resolution = forward.shape[0]
            total_points = resolution ** 2
            
            # Determine number of observation points
            if n_obs is not None:
                if 0 < n_obs < 1:
                    n_obs_points = int(n_obs * total_points)
                else:
                    n_obs_points = int(n_obs)
            elif obs_ratio is not None:
                n_obs_points = int(obs_ratio * total_points)
            else:
                # Default: use 1% of points
                n_obs_points = int(0.01 * total_points)
            
            # Set random seed if provided (for reproducibility)
            # Use sample_idx as offset so each sample gets a unique but reproducible mask
            if random_seed is not None:
                np.random.seed(random_seed + sample_idx)
            
            # Generate random mask
            indices = np.random.choice(total_points, n_obs_points, replace=False)
            indices_2d = np.unravel_index(indices, (resolution, resolution))
            obs_mask = np.zeros((resolution, resolution))
            obs_mask[indices_2d] = 1
        
        # Create sparse observation: masked forward channel
        # Set unobserved points to NaN so they appear as white background
        sparse_observation = forward.copy()
        sparse_observation[obs_mask == 0] = np.nan
        
        # Get min/max for sparse observation (use forward scale for consistency)
        sparse_min, sparse_max = fwd_min, fwd_max
        
        # Plot 1: Ground Truth Forward
        fig_fwd = plt.figure(figsize=(6, 6))
        ax_fwd = fig_fwd.add_subplot(111)
        ax_fwd.imshow(forward, cmap="viridis", norm=get_norm(fwd_min, fwd_max, gamma=gamma), interpolation="none")
        ax_fwd.set_xticks([])
        ax_fwd.set_yticks([])
        ax_fwd.axis('off')
        
        if save_dir:
            save_path_fwd = os.path.join(save_dir, f"gt_sample_{sample_idx}_forward.png")
            plt.savefig(save_path_fwd, bbox_inches="tight", pad_inches=0, dpi=300)
            print(f"Saved: {save_path_fwd}")
        
        plt.show()
        
        # Plot 2: Sparse Observation (masked forward)
        # Validate background color and convert to matplotlib color
        if sparse_bg_color not in ['black', 'white', 'gray']:
            raise ValueError(f"sparse_bg_color must be 'black', 'white', or 'gray', got '{sparse_bg_color}'")
        
        # Convert 'gray' to a proper gray color value
        bg_color_map = {
            'black': 'black',
            'white': 'white',
            'gray': '0.5'  # 50% gray
        }
        bg_color = bg_color_map[sparse_bg_color]
        
        fig_sparse = plt.figure(figsize=(6, 6), facecolor=bg_color)
        ax_sparse = fig_sparse.add_subplot(111, facecolor=bg_color)
        im_sparse = ax_sparse.imshow(sparse_observation, cmap="viridis", norm=get_norm(sparse_min, sparse_max, gamma=gamma), interpolation="none")
        ax_sparse.set_xticks([])
        ax_sparse.set_yticks([])
        ax_sparse.axis('off')
        
        if save_dir:
            save_path_sparse = os.path.join(save_dir, f"gt_sample_{sample_idx}_sparse_observation.png")
            plt.savefig(save_path_sparse, bbox_inches="tight", pad_inches=0, dpi=300, facecolor=bg_color)
            print(f"Saved: {save_path_sparse}")
        
        plt.show()
        
        # Plot 3: Ground Truth Inverse
        fig_inv = plt.figure(figsize=(6, 6))
        ax_inv = fig_inv.add_subplot(111)
        ax_inv.imshow(inverse, cmap="viridis", norm=get_norm(inv_min, inv_max, gamma=gamma), interpolation="none")
        ax_inv.set_xticks([])
        ax_inv.set_yticks([])
        ax_inv.axis('off')
        
        if save_dir:
            save_path_inv = os.path.join(save_dir, f"gt_sample_{sample_idx}_inverse.png")
            plt.savefig(save_path_inv, bbox_inches="tight", pad_inches=0, dpi=300)
            print(f"Saved: {save_path_inv}")
        
        plt.show()


if __name__ == "__main__":
    import torch
    from datasets import load_from_disk
    from training.dataset_hf import PDEDataset

    # ========== USER INPUTS ==========
    dataset_name = "poisson"  # Options: "poisson", "helmholtz", etc.
    sample_indices = [0, 1, 2, 3]  # Which samples from the dataset to plot
    problem_type = "both"  # Options: "both", "inverse", "forward"
    
    # Plot mode: "simple" (just GT) or "with_sparse" (GT + sparse observations)
    plot_mode = "with_sparse"  # Options: "simple", "with_sparse"
    
    # For sparse observations (only used if plot_mode == "with_sparse"):
    n_obs = None  # Number of observation points (int) or ratio (float 0-1). If None, uses obs_ratio
    obs_ratio = 0.03  # Ratio of observed points (0-1). Default: 0.01 (1%)
    sparse_bg_color = "gray"  # Background color for sparse observation plot. Options: "black", "white", or "gray"
    random_seed = None  # Random seed for mask generation. None = truly random, int = reproducible
    # You can also provide a pre-computed mask instead (see function call below)
    
    # Color contrast enhancement (gamma correction)
    # gamma < 1.0: Enhances contrast in middle ranges (makes differences more visible)
    # gamma = 1.0: Linear normalization (default, no enhancement)
    # gamma > 1.0: Enhances contrast at extremes
    gamma = 1.0  # Default: 1.0 (no enhancement), try 0.5-0.7 for stronger contrast
    
    # Save directory (set to None to only display, not save)
    save_dir = "exps/ground_truth_individual"  # Directory to save individual plots
    # ==================================
    
    # Derive paths from inputs
    gt_dataset_path = f"data/DiffPDE/{dataset_name}_test_hf"
    
    # Load ground truth data and create normalizer for denormalization
    print(f"Loading ground truth dataset from {gt_dataset_path}...")
    gt_dataset_raw = load_from_disk(gt_dataset_path)
    dataset_obj = PDEDataset(path=gt_dataset_path, max_size=len(gt_dataset_raw))
    normalizer = dataset_obj.create_normalizer()
    
    dataset_size = len(gt_dataset_raw)
    print(f"Dataset size: {dataset_size} samples")
    
    # Validate sample indices
    sample_indices = [idx for idx in sample_indices if idx < dataset_size]
    if not sample_indices:
        raise ValueError(f"All sample_indices are out of range. Dataset size is {dataset_size}.")
    
    print(f"Plotting samples: {sample_indices}")
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving plots to: {save_dir}")
    
    # Prepare ground truth data for plotting
    gt_data_list = []
    for i in sample_indices:
        # Load ground truth for this sample (shape: (2, 128, 128)) and denormalize
        gt_data_normalized = np.array(gt_dataset_raw[i]['data'])
        gt_data_torch = torch.from_numpy(gt_data_normalized[np.newaxis, ...])  # Add batch dim
        gt_data_denorm = normalizer.denormalize(gt_data_torch)[0].cpu().numpy()  # Denormalize and remove batch dim
        
        # Channel 0 = Inverse problem (inferring parameters), Channel 1 = Forward problem
        gt_inverse = gt_data_denorm[0]  # Inverse problem ground truth (channel 0)
        gt_forward = gt_data_denorm[1]  # Forward problem ground truth (channel 1)
        
        gt_data_list.append({
            "inverse": gt_inverse,
            "forward": gt_forward,
            "sample_idx": i
        })
    
    # Generate plots
    if plot_mode == "simple":
        # Simple plots: just ground truth images
        plot_ground_truth_individual(gt_data_list, save_dir=save_dir, problem_type=problem_type, gamma=gamma)
    elif plot_mode == "with_sparse":
        # Plots with sparse observations: GT Forward, Sparse Observation, GT Inverse
        plot_ground_truth_with_sparse_observations(
            gt_data_list, 
            save_dir=save_dir, 
            n_obs=n_obs, 
            obs_ratio=obs_ratio, 
            mask=None,  # You can provide a pre-computed mask here if needed
            gamma=gamma,
            sparse_bg_color=sparse_bg_color,
            random_seed=random_seed
        )
    else:
        raise ValueError(f"Unknown plot_mode: {plot_mode}. Must be 'simple' or 'with_sparse'")

