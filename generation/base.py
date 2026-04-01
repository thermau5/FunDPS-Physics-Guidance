import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from training.dataset_hf import PDEDataset
from training.noise_samplers import RBFKernel

from .observation import get_observation_class


class PDESolver:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")
        self.dataloader = None
        self.net = None
        self.noise_sampler = None
        self.batch_size = self.config["batch_size"]
        self.resolution = self.config["resolution"]
        self.num_steps = self.config["iterations"]
        self.save_dir = self.config["outdir"]

        self.n_plots = self.config["n_plots"]
        self.cnt_result_plots = 0
        self.cnt_process_plots = 0
        self.cnt_losses_plots = 0
        self.cnt_spectral_plots = 0
        self.n_process_steps = self.config["n_process_steps"]
        # Initialize storage for spectral errors across all batches
        self.all_spectral_geometric_means = []
        self.all_spectral_medians = []

        # Configuration for spectral energy plotting (default to True for backward compatibility)
        self.plot_spectral_energy_enabled = self.config.get("plot_spectral_energy", False)
        self.save_indices = np.linspace(0, self.num_steps - 1, self.n_process_steps, dtype=int)
        self.observations = [get_observation_class(c, self.config["dataset"]) for c in self.config["observation"]]

    def load_data(self):
        max_size = self.config["max_size"]
        if max_size is None:
            max_size = self.batch_size
        elif max_size == -1:
            max_size = None

        self.dataset = PDEDataset(
            path=self.config["data_path"],
            offset=self.config["data_offset"],
            resolution=self.resolution,
            max_size=max_size,
            shuffle=False,
            use_labels=False,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

    def load_network(self):
        f = open(self.config["pkl_path"], "rb")
        self.net = pickle.load(f)["ema"].to(self.device)
        self.n_channels = self.net.img_channels
        if self.net.img_resolution != self.resolution:
            print(f"Warning: Network resolution {self.net.img_resolution} does not match data resolution {self.resolution}.")

        # Print network parameters
        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"Loaded network with {n_params} parameters.")

    def generate_latents(self):
        if self.config["init_latents"] == "white_noise":
            return torch.randn([self.batch_size, self.n_channels, self.resolution, self.resolution], device=self.device)
        elif self.config["init_latents"] == "rbf":
            return self.noise_sampler.sample(self.batch_size)
        else:
            raise ValueError("Invalid init_latents value")

    def generate(self):
        if self.dataloader is None:
            self.load_data()
        if self.net is None:
            self.load_network()

        self.init_stats()

        if self.config["init_latents"] == "rbf":
            self.noise_sampler = RBFKernel(self.n_channels, self.resolution, self.resolution, scale=self.config["rbf_scale"], device=self.device)

        for i, data in enumerate(tqdm(self.dataloader)):
            data, label = data
            data = data.to(self.device)
            gt = self.dataset.denormalize(data)
            for obs in self.observations:
                obs.init(gt, self.dataset._normalizer)

            # if self.flag_plot_results:
            #     self.init_plot_results()
            # if self.flag_plot_process:
            #     self.init_plot_process()

            self.batch_size = data.shape[0]  # Update batch size in case it's different for the last batch

            pred, aux = self.generate_single_batch(self.observations)
            metrics = self.calculate_metrics(pred, gt)

            self.update_stats(metrics)
            # if i == 0:
            #     print("Metrics for first batch:")
            #     self.finalize_stats(save_dir=None)
            print(f"Completed batch {i+1}/{len(self.dataloader)}")
            self.finalize_stats(save_dir=None)

            self.save_results(pred, f"{self.save_dir}/results/batch_{i}.npy")
            self.plot_results(pred, gt, metrics, self.save_dir)
            if self.plot_spectral_energy_enabled:
                self.plot_spectral_energy(pred, gt, self.save_dir, is_first_batch_iteration=(i == 0))
            if "loss_history" in aux:
                self.plot_losses(aux["loss_history"], self.save_dir)
            if "intermediates" in aux and "intermediates_channel_index" in aux:
                self.plot_process(aux["intermediates"], aux["intermediates_channel_index"], gt, self.save_dir)

        self.finalize_stats(self.save_dir)
        
        # Print spectral errors summary after processing all batches
        if self.plot_spectral_energy_enabled:
            self.print_spectral_errors_summary()

    def generate_single_batch(self, observation):
        raise NotImplementedError

    def calculate_metrics(self, pred, gt):
        """Calculate metrics for each channel.

        Args:
            pred (torch.Tensor): Predicted tensor of shape [batch_size, channels, height, width]
            gt (torch.Tensor): Ground truth tensor of same shape as pred

        Returns:
            dict: Dictionary containing metrics for each channel
        """
        from .loss import sobolev_h1_loss

        metrics = {}
        batch_size, n_channels = pred.shape[:2]

        for c in range(n_channels):
            pred_c = pred[:, c]
            gt_c = gt[:, c]

            # For binary/discrete fields, use error rate
            if torch.allclose(gt_c, torch.round(gt_c)):
                gt_c = torch.round(gt_c)
                error = 1 - torch.sum(pred_c == gt_c, dim=(1, 2)) / (pred_c.shape[-1] * pred_c.shape[-2])
                metrics[f"error_rate_channel{c}"] = error
            # For continuous fields, use relative error and Sobolev H1 loss
            else:
                # Calculate relative error (L2 norm)
                relative_error = torch.norm(pred_c - gt_c, p=2, dim=(1, 2)) / torch.norm(gt_c, p=2, dim=(1, 2))
                metrics[f"rel_error_channel{c}"] = relative_error

                # # Calculate Sobolev H1 loss
                # error_field = pred_c - gt_c
                # n_obs = error_field.shape[-1] * error_field.shape[-2]  # resolution^2
                # sobolev_h1_error = sobolev_h1_loss(error_field, n_obs)
                # # Normalize by the Sobolev H1 norm of the ground truth
                # gt_sobolev_norm = sobolev_h1_loss(gt_c, n_obs)
                # relative_sobolev_error = sobolev_h1_error / (gt_sobolev_norm + 1e-8)  # Add small epsilon to avoid division by zero
                # metrics[f"sobolev_h1_error_channel{c}"] = relative_sobolev_error

        return metrics

    def plot_results(self, pred, gt, metrics, save_dir):
        """Plot final results for all channels.

        Args:
            pred (torch.Tensor): Predicted tensor [batch_size, channels, height, width]
            gt (torch.Tensor): Ground truth tensor [batch_size, channels, height, width]
            metrics (dict): Dictionary containing metrics for each channel
        """
        if self.cnt_result_plots >= self.n_plots:
            return

        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        batch_size, n_channels = pred.shape[:2]

        for batch_idx in range(batch_size):
            # Create a figure with 2 rows (pred/GT) and n_channels columns
            fig, axs = plt.subplots(2, n_channels, figsize=(6 * n_channels, 10), squeeze=False)
            fig.suptitle("PDE Solution Results", fontsize=16)

            # Plot each channel
            for c in range(n_channels):
                # Get data for this channel
                pred_c = pred[batch_idx, c]
                gt_c = gt[batch_idx, c]

                # Get common color range for this channel
                vmin = gt_c.min()
                vmax = gt_c.max()

                # Plot prediction
                im = axs[0, c].imshow(pred_c, cmap="viridis", vmin=vmin, vmax=vmax)
                axs[0, c].set_title(f"Predicted (Channel {c})")
                plt.colorbar(im, ax=axs[0, c])

                # Plot ground truth
                im = axs[1, c].imshow(gt_c, cmap="viridis", vmin=vmin, vmax=vmax)
                axs[1, c].set_title(f"Ground Truth (Channel {c})")
                plt.colorbar(im, ax=axs[1, c])

            # Add metrics as text
            metrics_text = []
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    metric_value = value[batch_idx].item()
                metrics_text.append(f"{key}: {metric_value:.4f}")

            fig.text(0.5, 0.02, "\n".join(metrics_text), ha="center", fontsize=12, bbox=dict(facecolor="white", alpha=0.8))

            # Save plot
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(f"{save_dir}/results_{batch_idx}.png", dpi=300, bbox_inches="tight")
            plt.close()

            self.cnt_result_plots += 1
            if self.cnt_result_plots == self.n_plots:
                break

    def plot_process(self, intermediates, intermediate_channels_index, gt, save_dir):
        """Plot intermediate results during inference.

        Args:
            intermediates (list): List of tensors showing intermediate results
            gt (torch.Tensor): Ground truth tensor [batch_size, channels, height, width]
        """
        if self.cnt_process_plots >= self.n_plots:
            return

        intermediates = [x.detach().cpu().numpy() for x in intermediates]
        gt = gt.detach().cpu().numpy()
        # gt = np.tile(gt, (1, n_channels // gt.shape[1], 1, 1))

        batch_size, n_channels = intermediates[0].shape[:2]
        n_steps = len(intermediates)

        for batch_idx in range(batch_size):
            fig = plt.figure(figsize=(4 * n_steps, 4 * n_channels))
            gs = fig.add_gridspec(n_channels, n_steps + 1, width_ratios=[1] * n_steps + [0.05], height_ratios=[1] * n_channels, hspace=0.3, wspace=0.1, top=0.95, bottom=0.05, left=0.05, right=0.95)

            fig.suptitle("PDE Solution Process", fontsize=16, y=0.98)

            # For each channel
            for c in range(n_channels):
                # Get channel data
                channel_data = [step[batch_idx, c] for step in intermediates]
                gt_c = gt[batch_idx, intermediate_channels_index[c]]

                # Get global min/max for consistent colormaps
                vmin = gt_c.min()
                vmax = gt_c.max()

                # Create axes for this channel
                axs = [fig.add_subplot(gs[c, i]) for i in range(n_steps)]

                # Plot each step
                for step in range(n_steps):
                    im = axs[step].imshow(channel_data[step], cmap="viridis", vmin=vmin, vmax=vmax)
                    axs[step].set_title(f"Step {step / (n_steps - 1) * 100:.0f}%")
                    axs[step].axis("off")

                # Add colorbar for this channel
                cax = fig.add_subplot(gs[c, -1])
                plt.colorbar(im, cax=cax, label=f"Channel {c}")

            # Save figure
            plt.savefig(f"{save_dir}/process_{batch_idx}.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
            plt.close()

            self.cnt_process_plots += 1
            if self.cnt_process_plots == self.n_plots:
                break

    def plot_losses(self, loss_history, save_dir):
        """Plot loss curves for each observation channel.

        Args:
            loss_history (list): List of tensors containing losses
                Each tensor has shape [total_channels] containing losses for each observation channel
            save_dir (str): Directory to save the plot
        """
        if self.cnt_losses_plots >= self.n_plots:
            return

        # Convert list of tensors to numpy array
        losses = torch.stack(loss_history).cpu().numpy()  # [steps, batches, total_channels]

        # Get channel counts for each observation
        channel_counts = [obs.n_channels for obs in self.observations]
        total_channels = sum(channel_counts)
        assert losses.shape[-1] == total_channels, f"Expected {total_channels} channels, got {losses.shape[-1]}"

        for batch_idx in range(losses.shape[1]):
            # Create plot
            plt.figure(figsize=(12, 6))
            for c in range(total_channels):
                label = f"Loss {c}"
                plt.plot(range(len(loss_history)), losses[:, batch_idx, c], label=label)

            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Observation Losses During Sampling")
            plt.legend()
            plt.grid(True)

            # Use log scale if loss values span multiple orders of magnitude
            if np.any(losses > 10 * np.min(losses[losses > 0])):
                plt.yscale("log")

            plt.tight_layout()
            plt.savefig(f"{save_dir}/losses_{batch_idx}.png", dpi=200, bbox_inches="tight")
            plt.close()

            self.cnt_losses_plots += 1
            if self.cnt_losses_plots == self.n_plots:
                break

    def init_stats(self):
        """Initialize statistics tracking for metrics."""
        self.stats = {}

    def update_stats(self, metrics):
        """Update statistics by appending new metrics to lists.

        Args:
            metrics (dict): Dictionary containing metrics for current batch
                Each value can be a tensor of batch_size length
        """
        for metric_name, values in metrics.items():
            # Initialize list if this is the first batch
            if metric_name not in self.stats:
                self.stats[metric_name] = []

            # Extend list with individual values from this batch
            self.stats[metric_name].extend(values.tolist())

    def finalize_stats(self, save_dir):
        """Calculate final statistics, log results, and save all metrics to JSON.

        Returns:
            dict: Dictionary containing final statistics for each metric
        """
        final_stats = self.stats.copy()

        # Calculate statistics for each metric
        for metric_name, values in self.stats.items():
            values_array = np.array(values)
            mean = np.mean(values_array)
            std = np.std(values_array)

            # Log results
            print(f"{metric_name}:")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std:  {std:.4f}")

            final_stats[f"{metric_name}_mean"] = float(mean)
            final_stats[f"{metric_name}_std"] = float(std)

        # Add spectral relative errors (geometric mean) if computed
        if getattr(self, "all_spectral_geometric_means", None):
            for c, values in enumerate(self.all_spectral_geometric_means):
                if len(values) > 0:
                    avg_geo_mean = float(np.mean(values))
                    final_stats[f"spectral_rel_error_channel{c}_geometric_mean"] = avg_geo_mean

        # Save complete results to JSON
        if save_dir is not None:
            output_path = f"{self.save_dir}/metrics.json"
            with open(output_path, "w") as f:
                json.dump(final_stats, f, indent=2)
            print(f"Saved complete metrics to: {output_path}")

        return final_stats

    def compute_spectral_energy(self, field):
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

    def plot_spectral_energy(self, pred, gt, save_dir, is_first_batch_iteration=False):
        """Plot spectral energy comparison between predicted and ground truth fields.

        Args:
            pred (torch.Tensor): Predicted tensor [batch_size, channels, height, width]
            gt (torch.Tensor): Ground truth tensor [batch_size, channels, height, width]
            save_dir (str): Directory to save the plot
            is_first_batch_iteration (bool): Whether this is the first batch iteration (i==0)
        """
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        batch_size, n_channels = pred.shape[:2]

        # Initialize storage if this is the first call or if channel count changed
        if not hasattr(self, 'all_spectral_geometric_means'):
            self.all_spectral_geometric_means = [[] for _ in range(n_channels)]
            self.all_spectral_medians = [[] for _ in range(n_channels)]
        else:
            # Ensure lists have correct number of channels (resize if needed)
            while len(self.all_spectral_geometric_means) < n_channels:
                self.all_spectral_geometric_means.append([])
                self.all_spectral_medians.append([])

        for batch_idx in range(batch_size):
            # Check if we should create plots (only up to n_plots limit)
            should_plot = self.cnt_spectral_plots < self.n_plots
            
            # Create figure if we're plotting this batch
            if should_plot:
                fig, axes = plt.subplots(2, n_channels, figsize=(6 * n_channels, 10), squeeze=False)
                fig.suptitle("Spectral Energy Comparison", fontsize=16)
            
            for c in range(n_channels):
                # Get data for this channel and batch
                pred_field = pred[batch_idx, c]
                gt_field = gt[batch_idx, c]

                # Compute spectral energy for both fields
                k_pred, power_pred = self.compute_spectral_energy(pred_field)
                k_gt, power_gt = self.compute_spectral_energy(gt_field)

                # Compute relative error
                if len(k_pred) == len(k_gt):
                    k_common = k_gt
                    relative_error = np.abs(power_pred - power_gt) / (power_gt + 1e-10)
                    
                    valid_mask = (relative_error > 0) & np.isfinite(relative_error)
                    if np.any(valid_mask):
                        valid_errors = relative_error[valid_mask]
                        
                        geometric_mean = np.exp(np.mean(np.log(valid_errors + 1e-10)))
                        median_error = np.median(valid_errors)
                        
                        # Always store errors for averaging across all batches
                        self.all_spectral_geometric_means[c].append(geometric_mean)
                        self.all_spectral_medians[c].append(median_error)
                        
                        # Print spectral errors for first batch (similar to L2 errors)
                        if is_first_batch_iteration and batch_idx == 0:
                            if c == 0:
                                print("Spectral Rel. Error for first batch:")
                            print(f"  Channel {c}: Geometric Mean: {geometric_mean:.3f}, Median: {median_error:.3f}")
                        
                        # Only create plots up to n_plots limit
                        if should_plot:
                            # Top row: Spectral energy comparison
                            ax_energy = axes[0, c]
                            # Bottom row: Relative error as function of k
                            ax_error = axes[1, c]

                            # Plot spectral energy on log-log scale (top row)
                            ax_energy.loglog(k_pred, power_pred, "r-", label="Predicted", linewidth=2, alpha=0.8)
                            ax_energy.loglog(k_gt, power_gt, "b-", label="Ground Truth", linewidth=2, alpha=0.8)

                            ax_energy.set_xlabel("Wave number k")
                            ax_energy.set_ylabel("Power Spectral Density")
                            ax_energy.set_title(f"Channel {c} - Spectral Energy")
                            ax_energy.legend()
                            ax_energy.grid(True, alpha=0.3)

                            # Plot relative error vs k
                            ax_error.loglog(k_common, relative_error, "g-", label="Relative Error", linewidth=2, alpha=0.8)
                            ax_error.set_xlabel("Wave number k")
                            ax_error.set_ylabel("Relative Error |P_pred - P_gt| / P_gt")
                            ax_error.set_title(f"Channel {c} - Spectral Error vs k")
                            ax_error.legend()
                            ax_error.grid(True, alpha=0.3)
                            
                            # Add geometric mean to plot
                            ax_error.text(0.98, 0.02, f"Geom. Mean: {geometric_mean:.3f}", 
                                         transform=ax_error.transAxes, fontsize=10,
                                         verticalalignment='bottom', horizontalalignment='right',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    if should_plot:
                        print(f"Warning: k_pred and k_gt have different lengths for channel {c}, batch {batch_idx}")

            # Save plot only if we haven't reached the limit
            if should_plot:
                plt.tight_layout()
                plt.savefig(f"{save_dir}/spectral_energy_{self.cnt_spectral_plots}.png", dpi=300, bbox_inches="tight")
                plt.close()
                self.cnt_spectral_plots += 1

    def print_spectral_errors_summary(self):
        """Print average spectral errors across all batches."""
        if not hasattr(self, 'all_spectral_geometric_means') or len(self.all_spectral_geometric_means) == 0:
            return
        
        n_channels = len(self.all_spectral_geometric_means)
        for c in range(n_channels):
            if len(self.all_spectral_geometric_means[c]) > 0:
                avg_geometric_mean = np.mean(self.all_spectral_geometric_means[c])
                avg_median = np.mean(self.all_spectral_medians[c])
                print(f"Spectral Rel. Error (Channel {c}): Geometric Mean: {avg_geometric_mean:.3f}, Median: {avg_median:.3f}")

    def save_results(self, pred, save_path):
        """Save results to disk.

        Args:
            pred (torch.Tensor): Predicted tensor [batch_size, channels, height, width]
            save_path (str): Path to save the results
        """
        pred = pred.detach().cpu().numpy()
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save(save_path, pred)
