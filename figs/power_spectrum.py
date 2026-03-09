"""
Script to compute and plot spectral energy comparison for Poisson Groups 1, 2, 4.
Compares ground truth, DDIS (DAPS), and FunDPS (DPS) methods.
Creates figures with 3 subplots (one per group) for each sample.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset_hf import PDEDataset


def compute_spectral_energy(field):
    """Compute the power spectral density of a 2D field using FFT.

    Args:
        field (np.ndarray): 2D field array [height, width]

    Returns:
        tuple: (frequencies, power_spectral_density)
            - frequencies: 1D array of wave numbers
            - power_spectral_density: 1D array of spectral energy values
    """
    # Compute 2D FFT
    fft_field = np.fft.fft2(field)

    # Compute power spectral density
    power_spectrum = np.abs(fft_field) ** 2

    # Get field dimensions
    ny, nx = field.shape

    # Create frequency grids
    kx = np.fft.fftfreq(nx, d=1.0)
    ky = np.fft.fftfreq(ny, d=1.0)
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # Compute radial wave number
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)

    # Define radial bins for averaging based on actual frequency range
    # Note: fftfreq returns normalized frequencies in [-0.5, 0.5), so k_max ≈ 0.707 for square grids
    k_max = 0.5  # Maximum frequency magnitude
    n_bins = min(nx, ny) // 2  # Number of bins
    k_bins = np.linspace(0, k_max, n_bins + 1)[1:]
    k_centers = (k_bins[1:] + k_bins[:-1]) / 2

    # Radially average the power spectrum
    power_radial = np.zeros(len(k_centers))
    for i, k_center in enumerate(k_centers):
        # Find pixels within this radial bin
        mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i + 1])
        if np.any(mask):
            power_radial[i] = np.mean(power_spectrum[mask])

    return k_centers, power_radial


def load_ground_truth(config_path):
    """Load ground truth data using PDEDataset.

    Args:
        config_path (str): Path to config.json file

    Returns:
        np.ndarray: Ground truth data [batch_size, channels, height, width]
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset = PDEDataset(
        path=config["data_path"],
        offset=config.get("data_offset"),
        resolution=config["resolution"],
        max_size=config.get("max_size", config["batch_size"]),
        shuffle=False,
        use_labels=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # Get first batch
    data, label = next(iter(dataloader))
    gt = dataset.denormalize(data)

    return gt.cpu().numpy()


def load_predictions(result_dir):
    """Load prediction results from npy files.

    Args:
        result_dir (str): Directory containing batch_*.npy files

    Returns:
        list: List of numpy arrays, one for each batch
    """
    results = []
    batch_idx = 0
    while True:
        batch_file = os.path.join(result_dir, f"batch_{batch_idx}.npy")
        if not os.path.exists(batch_file):
            break
        results.append(np.load(batch_file))
        batch_idx += 1

    if len(results) == 0:
        raise FileNotFoundError(f"No batch files found in {result_dir}")

    # Concatenate all batches
    return np.concatenate(results, axis=0)


def plot_spectral_energy_comparison(gt_groups, daps_groups, dps_groups, save_path):
    """Plot spectral energy comparison for all three methods across multiple groups.

    Args:
        gt_groups (dict): Dictionary mapping group names to ground truth arrays [batch_size, channels, height, width]
        daps_groups (dict): Dictionary mapping group names to DAPS predictions [batch_size, channels, height, width]
        dps_groups (dict): Dictionary mapping group names to DPS predictions [batch_size, channels, height, width]
        save_path (str): Path to save the plot
    """
    groups = ["Group 1", "Group 2", "Group 4"]
    group_labels = {
        "Group 1": "Time Budget 16s",
        "Group 2": "Time Budget 32s",
        "Group 4": "Time Budget 128s",
    }

    # Get the minimum number of samples across all groups
    min_samples = min(gt_groups[g].shape[0] for g in groups)

    # Plot for first 4 samples
    for sample_idx in range(min(4, min_samples)):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # fig.suptitle(f"Poisson Spectral Energy Comparison - Sample {sample_idx}", fontsize=16, y=1.02)

        for group_idx, group_name in enumerate(groups):
            ax = axes[group_idx]

            # Get data for this group and sample (assuming single channel)
            gt_field = gt_groups[group_name][sample_idx, 0]
            daps_field = daps_groups[group_name][sample_idx, 0]
            dps_field = dps_groups[group_name][sample_idx, 0]

            # Compute spectral energy
            k_gt, power_gt = compute_spectral_energy(gt_field)
            k_daps, power_daps = compute_spectral_energy(daps_field)
            k_dps, power_dps = compute_spectral_energy(dps_field)

            # Compute relative errors in spectral energy using geometric mean
            def compute_geometric_error(power_pred, power_gt, k_pred, k_gt):
                if len(k_pred) == len(k_gt):
                    relative_error = np.abs(power_pred - power_gt) / (power_gt + 1e-10)

                    valid_mask = (relative_error > 0) & np.isfinite(relative_error)
                    if np.any(valid_mask):
                        valid_errors = relative_error[valid_mask]
                        geometric_mean = np.exp(np.mean(np.log(valid_errors + 1e-10)))
                        median_error = np.median(valid_errors)
                        return geometric_mean, median_error
                return 0.0, 0.0

            spectral_error_daps, median_error_daps = compute_geometric_error(power_daps, power_gt, k_daps, k_gt)
            spectral_error_dps, median_error_dps = compute_geometric_error(power_dps, power_gt, k_dps, k_gt)

            # Plot on log-log scale
            ax.loglog(k_gt, power_gt, "b-", label="Ground Truth", linewidth=3, alpha=0.8)
            ax.loglog(k_daps, power_daps, "r--", label=f"DDIS (Error: {spectral_error_daps:.3f})", linewidth=3, alpha=0.8)
            ax.loglog(k_dps, power_dps, "g-.", label=f"FunDPS (Error: {spectral_error_dps:.3f})", linewidth=3, alpha=0.8)

            ax.set_xlabel("Wave number k", fontsize=18)
            ax.set_ylabel("Power Spectral Density", fontsize=18)
            ax.set_title(f"{group_labels[group_name]}", fontsize=20)
            ax.legend(fontsize=14, loc="best")
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.grid(True, alpha=0.3)

            # Print to console
            print(f"Sample {sample_idx}, {group_name}:")
            print(f"  DDIS Spectral Geom. Mean Error: {spectral_error_daps:.4f}, Median: {median_error_daps:.4f}")
            print(f"  FunDPS Spectral Geom. Mean Error: {spectral_error_dps:.4f}, Median: {median_error_dps:.4f}")

        plt.tight_layout()

        # Save as PNG
        save_file = save_path.replace(".png", f"_sample{sample_idx}.png")
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_file}")

        # Save as PDF in pdf folder
        pdf_dir = os.path.join(os.path.dirname(save_path), "pdf")
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_file = os.path.join(pdf_dir, os.path.basename(save_file).replace(".png", ".pdf"))
        plt.savefig(pdf_file, dpi=300, bbox_inches="tight", format="pdf")
        print(f"Saved PDF to: {pdf_file}")

        plt.close()


def main():
    """Main function to generate spectral energy comparison plots."""

    # Define paths for all three datasets
    base_dir = "/home/chen/lab/FunDPS-Physics-Guidance/figs/main-table-12batches-groups124"

    dataset_configs = {
        "poisson": {
            "Group 1": {"daps": "G0101_001937-poisson-128-daps - Poisson Group 1", "dps": "FunDPS/G0101_003350-poisson-128-dps - Poisson Group 1"},
            "Group 2": {"daps": "G0101_094225-poisson-128-daps - Poisson Group 2", "dps": "FunDPS/G0102_052629-poisson-128-dps - Poisson Group 2"},
            "Group 4": {"daps": "G0101_082347-poisson-128-daps - Poisson Group 4", "dps": "FunDPS/G0101_125410-poisson-128-dps - Poisson Group 4"},
        },
        "helmholtz": {
            "Group 1": {"daps": "G0102_053914-helmholtz-128-daps - Helmholtz Group 1", "dps": "FunDPS/G0102_054053-helmholtz-128-dps - Helmholtz Group 1"},
            "Group 2": {"daps": "G0103_063122-helmholtz-128-daps - Helmholtz Group 2", "dps": "FunDPS/G0103_121345-helmholtz-128-dps - Helmholtz Group 2"},
            "Group 4": {"daps": "G0102_055044-helmholtz-128-daps - Helmholtz Group 4", "dps": "FunDPS/G0102_224359-helmholtz-128-dps - Helmholtz Group 4"},
        },
        "ns": {
            "Group 1": {"daps": "G0104_045307-ns-nonbounded-128-daps - NS Group 1", "dps": "FunDPS/G0104_041632-ns-nonbounded-128-dps - NS Group 1"},
            "Group 2": {"daps": "G0104_050359-ns-nonbounded-128-daps - NS Group 2", "dps": "FunDPS/G0104_045346-ns-nonbounded-128-dps - NS Group 2"},
            "Group 4": {"daps": "G0104_210717-ns-nonbounded-128-daps - NS Group 4", "dps": "FunDPS/G0104_210820-ns-nonbounded-128-dps - NS Group 4"},
        },
    }

    # Process each dataset
    for dataset_name, group_configs in dataset_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'='*60}")

        gt_groups = {}
        daps_groups = {}
        dps_groups = {}

        # Load data for all groups
        for group_name, paths in group_configs.items():
            print(f"\n=== Processing {dataset_name} {group_name} ===")

            daps_dir = os.path.join(base_dir, paths["daps"])
            dps_dir = os.path.join(base_dir, paths["dps"])

            # Load configurations
            daps_config = os.path.join(daps_dir, "config.json")

            print(f"Loading ground truth data for {group_name}...")
            gt = load_ground_truth(daps_config)
            print(f"Ground truth shape: {gt.shape}")
            gt_groups[group_name] = gt

            print(f"Loading DDIS predictions for {group_name}...")
            daps_results_dir = os.path.join(daps_dir, "results")
            daps_pred = load_predictions(daps_results_dir)
            print(f"DDIS predictions shape: {daps_pred.shape}")
            daps_groups[group_name] = daps_pred

            print(f"Loading FunDPS predictions for {group_name}...")
            dps_results_dir = os.path.join(dps_dir, "results")
            dps_pred = load_predictions(dps_results_dir)
            print(f"FunDPS predictions shape: {dps_pred.shape}")
            dps_groups[group_name] = dps_pred

            # Ensure all have the same shape for this group
            min_samples = min(gt.shape[0], daps_pred.shape[0], dps_pred.shape[0])
            gt_groups[group_name] = gt[:min_samples]
            daps_groups[group_name] = daps_pred[:min_samples]
            dps_groups[group_name] = dps_pred[:min_samples]

            print(f"Using {min_samples} samples for {group_name}")

        # Generate plots for this dataset
        print(f"\n=== Generating spectral energy comparison plots for {dataset_name} ===")
        save_path = f"/home/chen/lab/FunDPS-Physics-Guidance/figs/spectral_energy_comparison_{dataset_name}_all_groups.png"
        plot_spectral_energy_comparison(gt_groups, daps_groups, dps_groups, save_path)

    print("\n" + "=" * 60)
    print("All datasets completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
