from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def create_multiple_comparison_plots(data_pairs, save_path=None, l2_errors=None):
    """
    Create comparison plots for multiple pairs of forward and inverse problems in a 2x2 grid.
    Args:
        data_pairs: List of tuples, each containing (forward_data, inverse_data) where each contains:
            - ground_truth: True ground truth array
            - model1: First baseline model predictions (e.g., FunDPS)
            - model2: Second baseline model predictions (e.g., FunDAPS)
            - model3: Third model predictions (e.g., DDIS - ours)
            - mask: Binary mask array (1 for observed points, 0 for non-observed) [optional]
        save_path: Optional path to save the figure
        l2_errors: Optional dict with keys 'model1', 'model2', 'model3' containing L2 error percentages
    """
    # Create figure
    fig = plt.figure(figsize=(58, 15))  # Increased width to accommodate 7 columns + 3 colorbars + 2x2 layout

    # Create outer GridSpec for the 2x2 layout
    outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1)

    # Plot titles
    titles = ["Ground Truth", "FunDAPS", "FunDPS", "DDIS (Ours)", "FunDAPS Error", "FunDPS Error", "DDIS Error (Ours)"]
    row_titles = ["Inverse\nProblem", "Forward\nProblem"]

    def add_colorbar_matched(im, cax, ax):
        """Add a colorbar with height matched to the corresponding image axes"""
        ax_pos = ax.get_position()
        cbar = plt.colorbar(im, cax=cax)
        cax_pos = cax.get_position()
        new_x0 = ax_pos.x1 + 0.005
        cax.set_position([new_x0, ax_pos.y0, cax_pos.width, ax_pos.height])
        return cbar

    def plot_pair(forward_data, inverse_data, outer_spec, pair_idx):
        # Create inner GridSpec for this pair (2 rows for forward/inverse)
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_spec, height_ratios=[1, 1], hspace=0.1)

        # Create GridSpec for each row
        gs_rows = []
        for row in range(2):
            # 10 columns: GT, Model1, Model2, Model3, cbar, Error1, cbar, Error2, Error3, cbar
            # Increased colorbar widths from 0.05 to 0.08 for better label visibility
            gs_row = gridspec.GridSpecFromSubplotSpec(1, 10, subplot_spec=inner_gs[row], width_ratios=[1, 1, 1, 1, 0.08, 1, 0.08, 1, 1, 0.08], wspace=0.3)
            gs_rows.append(gs_row)

        def plot_row(data, row_idx, gs_row):
            # Get data with descriptive names (swapped: model1=FunDAPS, model2=FunDPS)
            ground_truth = data["ground_truth"]
            model1_pred = data["model1"]  # FunDAPS
            model2_pred = data["model2"]  # FunDPS
            model3_pred = data["model3"]  # DDIS (ours)
            mask = data.get("mask", np.ones_like(ground_truth))

            # Calculate errors: |model_prediction - ground_truth|
            # Note: order doesn't matter for absolute value, so |pred - gt| = |gt - pred|
            model1_error = np.abs(model1_pred - ground_truth)  # FunDAPS error vs true GT
            model2_error = np.abs(model2_pred - ground_truth)  # FunDPS error vs true GT
            model3_error = np.abs(model3_pred - ground_truth)  # DDIS error vs true GT

            # Get global min/max for consistent coloring
            vmin = np.min(model2_pred)  # Use FunDPS for vmin/vmax
            vmax = np.max(model2_pred)
            gt_min = np.min(ground_truth)
            gt_max = np.max(ground_truth)
            # Separate error scales: FunDAPS has larger errors, FunDPS/DDIS share scale
            err_max_fundaps = np.max(model1_error)  # FunDAPS error scale (larger)
            err_max_others = max(np.max(model2_error), np.max(model3_error))  # FunDPS/DDIS scale (smaller)

            plot_positions = [0, 1, 2, 3, 5, 7, 8]  # GT, Model1, Model2, Model3, Error1, Error2, Error3
            cbar_positions = [4, 6, 9]  # Colorbars for Model3, Error1 (FunDAPS), Error3 (DDIS)
            axes_dict = {}

            # Column 0: Ground Truth (no masking)
            ax = fig.add_subplot(gs_row[plot_positions[0]])
            ax.imshow(ground_truth, cmap="viridis", norm=Normalize(vmin=gt_min, vmax=gt_max), interpolation="none")
            axes_dict[0] = ax

            # Column 1: Model 1 (FunDPS)
            ax = fig.add_subplot(gs_row[plot_positions[1]])
            ax.imshow(model1_pred, cmap="viridis", norm=Normalize(vmin=vmin, vmax=vmax), interpolation="none")
            axes_dict[1] = ax

            # Column 2: Model 2 (FunDAPS)
            ax = fig.add_subplot(gs_row[plot_positions[2]])
            ax.imshow(model2_pred, cmap="viridis", norm=Normalize(vmin=vmin, vmax=vmax), interpolation="none")
            axes_dict[2] = ax

            # Column 3: Model 3 (DDIS - ours) with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[3]])
            im3 = ax.imshow(model3_pred, cmap="viridis", norm=Normalize(vmin=vmin, vmax=vmax), interpolation="none")
            cax3 = fig.add_subplot(gs_row[cbar_positions[0]])
            add_colorbar_matched(im3, cax3, ax)
            axes_dict[3] = ax

            # Column 4: Model 1 Error (FunDAPS) - uses separate larger scale with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[4]])
            im4 = ax.imshow(model1_error, cmap="viridis", norm=Normalize(vmin=0, vmax=err_max_fundaps), interpolation="none")
            cax4 = fig.add_subplot(gs_row[cbar_positions[1]])
            add_colorbar_matched(im4, cax4, ax)
            axes_dict[4] = ax

            # Column 5: Model 2 Error (FunDPS) - uses smaller scale shared with DDIS
            ax = fig.add_subplot(gs_row[plot_positions[5]])
            ax.imshow(model2_error, cmap="viridis", norm=Normalize(vmin=0, vmax=err_max_others), interpolation="none")
            axes_dict[5] = ax

            # Column 6: Model 3 Error (DDIS - ours) - uses smaller scale shared with FunDPS with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[6]])
            im6 = ax.imshow(model3_error, cmap="viridis", norm=Normalize(vmin=0, vmax=err_max_others), interpolation="none")
            cax6 = fig.add_subplot(gs_row[cbar_positions[2]])
            add_colorbar_matched(im6, cax6, ax)
            axes_dict[6] = ax

            # Remove ticks from all plots
            for ax in axes_dict.values():
                ax.set_xticks([])
                ax.set_yticks([])

            # Add row title to first plot with pair number
            row_label = f"{row_titles[row_idx]}"
            axes_dict[0].set_ylabel(row_label, fontsize=20)

            # Add titles only to first pair and first row (inverse problem)
            if pair_idx == 0 and row_idx == 0:
                for i, title in enumerate(titles):
                    axes_dict[i].set_title(title, pad=10, fontsize=20)
            
            # Add L2 error percentage below first row (inverse problem) of error columns
            if pair_idx == 0 and row_idx == 0:
                # Use provided L2 errors if available, otherwise calculate them
                if l2_errors is not None:
                    l2_error_model1 = l2_errors['model1']
                    l2_error_model2 = l2_errors['model2']
                    l2_error_model3 = l2_errors['model3']
                else:
                    # Calculate L2 error as percentage: ||pred - gt||_2 / ||gt||_2 * 100
                    l2_error_model1 = np.linalg.norm(model1_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
                    l2_error_model2 = np.linalg.norm(model2_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
                    l2_error_model3 = np.linalg.norm(model3_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
                
                # Add text below error columns (closer to the plot) with upward arrow
                axes_dict[4].text(0.5, -0.02, f'↑ L2 Error: {l2_error_model1:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[4].transAxes, fontsize=16)
                axes_dict[5].text(0.5, -0.02, f'↑ L2 Error: {l2_error_model2:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[5].transAxes, fontsize=16)
                axes_dict[6].text(0.5, -0.02, f'↑ L2 Error: {l2_error_model3:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[6].transAxes, fontsize=16)

        # Plot inverse and forward problems (matching row_titles order)
        plot_row(inverse_data, 0, gs_rows[0])  # Row 0: Inverse Problem
        plot_row(forward_data, 1, gs_rows[1])  # Row 1: Forward Problem
        
        # Add vertical separator lines by directly accessing the colorbar axes from grid positions
        # Get the outer grid position for this pair
        outer_pos = outer_spec.get_position(fig)
        
        # Colorbar positions in grid: [4, 6, 9]  # DDIS pred cbar, FunDAPS error cbar, DDIS error cbar
        cbar_grid_positions = [4, 6, 9]
        
        # Separator 1: Right after DDIS predictions colorbar (grid position 4)
        temp_ax1 = fig.add_subplot(gs_rows[0][cbar_grid_positions[0]])
        pos1 = temp_ax1.get_position()
        temp_ax1.remove()
        line1_x = pos1.x1 + 0.0075  # Right after colorbar
        
        # Separator 2: Right after FunDAPS error colorbar (grid position 6)  
        temp_ax2 = fig.add_subplot(gs_rows[0][cbar_grid_positions[1]])
        pos2 = temp_ax2.get_position()
        temp_ax2.remove()
        line2_x = pos2.x1 + 0.0075  # Right after colorbar
        
        # Get y-extent from outer grid (spans both rows)
        y_bottom = outer_pos.y0 - 0.01
        y_top = outer_pos.y1 + 0.01
        
        # Draw the separator lines with academic styling (light gray, dashed)
        fig.add_artist(plt.Line2D([line1_x, line1_x], [y_bottom, y_top], 
                                   transform=fig.transFigure, color='gray', linewidth=2, 
                                   linestyle='--', alpha=0.6, zorder=10))
        fig.add_artist(plt.Line2D([line2_x, line2_x], [y_bottom, y_top], 
                                   transform=fig.transFigure, color='gray', linewidth=2, 
                                   linestyle='--', alpha=0.6, zorder=10))

    # Plot all pairs in 2x2 grid
    for i, (forward_data, inverse_data) in enumerate(data_pairs):
        row = i // 2
        col = i % 2
        plot_pair(forward_data, inverse_data, outer_gs[row, col], i)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()

if __name__ == "__main__":
    import torch
    from datasets import load_from_disk
    from training.dataset_hf import PDEDataset

    dataset_name = "helmholtz"
    
    # Load model predictions
    # ddis_pred = np.load("exps/Z_test_plot_poisson/ddis_200_100_multi.npy")  # Shape: (8, 2, 128, 128)
    # fundaps_pred = np.load("exps/Z_test_plot_poisson/fundaps_100_20.npy")  # Shape: (8, 2, 128, 128)
    # fundps_pred = np.load("exps/Z_test_plot_poisson/fundps_3000.npy")  # Shape: (8, 2, 128, 128)

    ddis_pred = np.load(f"exps/Z_test_plot_{dataset_name}/ddis_100_100.npy")  # Shape: (8, 2, 128, 128)
    fundaps_pred = np.load(f"exps/Z_test_plot_{dataset_name}/fundaps_500.npy")  # Shape: (8, 2, 128, 128)
    fundps_pred = np.load(f"exps/Z_test_plot_{dataset_name}/fundps_5000.npy")  # Shape: (8, 2, 128, 128)
    
    # Load ground truth data and create normalizer for denormalization
    gt_dataset_raw = load_from_disk(f"data/DiffPDE/{dataset_name}_test_hf")
    dataset_obj = PDEDataset(path=f"data/DiffPDE/{dataset_name}_test_hf", max_size=len(gt_dataset_raw))
    normalizer = dataset_obj.create_normalizer()
    
    save_path = f"exps/Z_test_plot_{dataset_name}/comparison_plots.png"
    
    # L2 Error values (hardcoded - to display below error columns)
    # Swapped order: model1=FunDAPS, model2=FunDPS, model3=DDIS
    error_dict = {
        "poisson": {
            "l2_error_fundaps": 98.69,
            "l2_error_fundps": 19.34,
            "l2_error_ddis": 14.60
        },
        "helmholtz": {
            "l2_error_fundaps": 0.00,    # TODO: add actual values
            "l2_error_fundps": 0.00,    # TODO: add actual values
            "l2_error_ddis": 0.00,       # TODO: add actual values
        }
    }
    l2_error_fundaps = error_dict[dataset_name]["l2_error_fundaps"]  # FunDAPS L2 error percentage (model1)
    l2_error_fundps = error_dict[dataset_name]["l2_error_fundps"]   # FunDPS L2 error percentage (model2)
    l2_error_ddis = error_dict[dataset_name]["l2_error_ddis"]     # DDIS L2 error percentage (model3)
    
    # Create data pairs for visualization
    data_pairs = []
    batch_size = ddis_pred.shape[0]
    
    for i in range(min(1, batch_size)):  # Show up to 4 pairs in 2x2 grid
        # Load ground truth for this sample (shape: (2, 128, 128)) and denormalize
        gt_data_normalized = np.array(gt_dataset_raw[i]['data'])
        gt_data_torch = torch.from_numpy(gt_data_normalized[np.newaxis, ...])  # Add batch dim
        gt_data_denorm = normalizer.denormalize(gt_data_torch)[0].cpu().numpy()  # Denormalize and remove batch dim
        
        # Channel 0 = Inverse problem (inferring parameters), Channel 1 = Forward problem
        gt_inverse = gt_data_denorm[0]  # Inverse problem ground truth (channel 0)
        gt_forward = gt_data_denorm[1]  # Forward problem ground truth (channel 1)
        
        # Extract predictions for inverse and forward problems
        ddis_inverse = ddis_pred[i, 0]   # Inverse problem (channel 0)
        ddis_forward = ddis_pred[i, 1]   # Forward problem (channel 1)
        fundaps_inverse = fundaps_pred[i, 0]
        fundaps_forward = fundaps_pred[i, 1]
        fundps_inverse = fundps_pred[i, 0]
        fundps_forward = fundps_pred[i, 1]
        
        # Create forward problem data with clear naming (swapped: model1=FunDAPS, model2=FunDPS)
        forward_data = {
            "ground_truth": gt_forward,     # True ground truth from dataset
            "model1": fundaps_forward,      # FunDAPS predictions (model1)
            "model2": fundps_forward,       # FunDPS predictions (model2)
            "model3": ddis_forward,         # DDIS predictions (ours, model3)
            "mask": np.ones_like(gt_forward)  # Full mask (all observed)
        }
        
        # Create inverse problem data with clear naming (swapped: model1=FunDAPS, model2=FunDPS)
        inverse_data = {
            "ground_truth": gt_inverse,     # True ground truth from dataset
            "model1": fundaps_inverse,      # FunDAPS predictions (model1)
            "model2": fundps_inverse,       # FunDPS predictions (model2)
            "model3": ddis_inverse,         # DDIS predictions (ours, model3)
            "mask": np.ones_like(gt_inverse)  # Full mask (all observed)
        }
        
        data_pairs.append((forward_data, inverse_data))
    
    # Create L2 error dictionary (swapped: model1=FunDAPS, model2=FunDPS)
    l2_errors = {
        'model1': l2_error_fundaps,  # FunDAPS
        'model2': l2_error_fundps,   # FunDPS
        'model3': l2_error_ddis      # DDIS
    }
    
    create_multiple_comparison_plots(data_pairs, save_path=save_path, l2_errors=l2_errors)