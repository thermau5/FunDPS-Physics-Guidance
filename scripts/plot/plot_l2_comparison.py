import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from matplotlib.colors import Normalize, PowerNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def create_multiple_comparison_plots(data_pairs, save_path=None, l2_errors=None, shift_x=0.0, shift_y=0.0, gamma=1.0):
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

    def add_colorbar_matched(im, cax, ax, vmin=None, vmax=None, shift_x=0.0, shift_y=0.0):
        """Add a colorbar with height matched to the corresponding image axes.
        Automatically uses scientific notation for very small or large values.
        Args:
            shift_x: Horizontal offset for scientific notation (default: 0.0)
            shift_y: Vertical offset for scientific notation (default: 0.0)
        """
        ax_pos = ax.get_position()
        cbar = plt.colorbar(im, cax=cax)
        # Get figure for adding text
        fig = ax.figure
        
        # Match height exactly to the image axes (fix height bug)
        cax.set_position([ax_pos.x1 + 0.005, ax_pos.y0, cax.get_position().width, ax_pos.height])
        cax_pos = cax.get_position()
        
        # Format colorbar with scientific notation if needed
        if vmin is not None and vmax is not None:
            # Use scientific notation for better readability when values are very small or large
            # matplotlib's ScalarFormatter automatically switches to scientific notation
            # when values are outside the range [10^-2, 10^3] or when needed for clarity
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))  # Use scientific notation for exponents outside [-2, 3]
            # Use offset notation (scientific notation above colorbar) if values are very small or large
            max_abs = max(abs(vmin), abs(vmax))
            if max_abs > 0 and (max_abs < 0.01 or max_abs > 1000):
                # Calculate the exponent based on the order of magnitude
                # Find the order of magnitude that will normalize values to a reasonable range
                if max_abs < 1:
                    exponent = int(np.floor(np.log10(max_abs)))
                else:
                    exponent = int(np.floor(np.log10(max_abs)))
                
                # Get current ticks and normalize them
                ticks = cbar.get_ticks()
                if len(ticks) > 0:
                    scale_factor = 10 ** exponent
                    
                    # Set colorbar limits to exactly vmin and vmax FIRST to prevent blank areas
                    # Set limits on the mappable (image) first
                    cbar.mappable.set_clim(vmin, vmax)
                    
                    # Then set the colorbar axis limits to match exactly
                    cbar.ax.set_ylim(vmin, vmax)
                    
                    # Filter ticks to only include those within (vmin, vmax) range (exclude vmin and vmax)
                    # This ensures colorbar fills completely but doesn't show vmin/vmax as tick labels
                    ticks_in_range = ticks[(ticks > vmin) & (ticks < vmax)]
                    
                    # Use ticks within range (excluding vmin and vmax from display)
                    all_ticks = np.unique(ticks_in_range) if len(ticks_in_range) > 0 else np.array([])
                    normalized_ticks = all_ticks / scale_factor
                    
                    # Format tick labels (show normalized values with 1 decimal place)
                    tick_labels = []
                    for t in normalized_ticks:
                        tick_labels.append(f"{t:.1f}")  # Always show 1 decimal place (e.g., 4.2, 6.0, -2.0)
                    
                    # Fix UserWarning: set ticks before ticklabels
                    # Set ticks AFTER setting limits to ensure they respect the range
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Force update to ensure limits are applied, then restore tick labels
                    cbar.update_normal(cbar.mappable)
                    # Re-apply tick labels after update (update_normal may reset them)
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Position exponent above the top tick label
                    # For vertical colorbar, top tick is at y1, labels are to the right
                    # Align with the right edge where tick labels are, at the top
                    # Note: cax_pos is calculated per colorbar, so this works correctly for both rows
                    # (inverse problem row 0 and forward problem row 1)
                    exp_x = cax_pos.x1 + shift_x  # To the right of colorbar, aligned with tick labels
                    exp_y = cax_pos.y1 + shift_y  # Above the top of colorbar (relative to each row's colorbar)
                else:
                    # Fallback if no ticks
                    exp_x = cax_pos.x1 + shift_x
                    exp_y = cax_pos.y1 + shift_y
                
                exp_str = f"×10$^{{{exponent}}}$"
                # Add scientific notation text - this is applied to each colorbar individually
                # so it will appear on both rows (inverse and forward) when values trigger it
                fig.text(exp_x, exp_y, exp_str, ha='left', va='bottom', fontsize=9)
            else:
                # For normal values, use standard formatting
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 3))
                cbar.formatter = formatter
                cbar.update_ticks()
        
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
            ax.imshow(ground_truth, cmap="viridis", norm=get_norm(gt_min, gt_max, gamma=gamma), interpolation="none")
            axes_dict[0] = ax

            # Column 1: Model 1 (FunDPS)
            ax = fig.add_subplot(gs_row[plot_positions[1]])
            ax.imshow(model1_pred, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
            axes_dict[1] = ax

            # Column 2: Model 2 (FunDAPS)
            ax = fig.add_subplot(gs_row[plot_positions[2]])
            ax.imshow(model2_pred, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
            axes_dict[2] = ax

            # Column 3: Model 3 (DDIS - ours) with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[3]])
            im3 = ax.imshow(model3_pred, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
            cax3 = fig.add_subplot(gs_row[cbar_positions[0]])
            add_colorbar_matched(im3, cax3, ax, vmin=vmin, vmax=vmax, shift_x=shift_x, shift_y=shift_y)
            axes_dict[3] = ax

            # Column 4: Model 1 Error (FunDAPS) - uses separate larger scale with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[4]])
            im4 = ax.imshow(model1_error, cmap="viridis", norm=get_norm(0, err_max_fundaps, gamma=gamma), interpolation="none")
            cax4 = fig.add_subplot(gs_row[cbar_positions[1]])
            add_colorbar_matched(im4, cax4, ax, vmin=0, vmax=err_max_fundaps, shift_x=shift_x, shift_y=shift_y)
            axes_dict[4] = ax

            # Column 5: Model 2 Error (FunDPS) - uses smaller scale shared with DDIS
            ax = fig.add_subplot(gs_row[plot_positions[5]])
            ax.imshow(model2_error, cmap="viridis", norm=get_norm(0, err_max_others, gamma=gamma), interpolation="none")
            axes_dict[5] = ax

            # Column 6: Model 3 Error (DDIS - ours) - uses smaller scale shared with FunDPS with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[6]])
            im6 = ax.imshow(model3_error, cmap="viridis", norm=get_norm(0, err_max_others, gamma=gamma), interpolation="none")
            cax6 = fig.add_subplot(gs_row[cbar_positions[2]])
            add_colorbar_matched(im6, cax6, ax, vmin=0, vmax=err_max_others, shift_x=shift_x, shift_y=shift_y)
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
                axes_dict[4].text(0.5, -0.02, f'↑L2 Error: {l2_error_model1:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[4].transAxes, fontsize=16)
                axes_dict[5].text(0.5, -0.02, f'↑L2 Error: {l2_error_model2:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[5].transAxes, fontsize=16)
                axes_dict[6].text(0.5, -0.02, f'↑L2 Error: {l2_error_model3:.2f}%', 
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


def create_error_only_plots(data_pairs, save_path=None, l2_errors=None, shift_x=0.0, shift_y=0.0, gamma=1.0):
    """
    Create error-only plots for multiple pairs of forward and inverse problems in a 2x2 grid.
    Shows only error columns (3 columns: FunDAPS Error, FunDPS Error, DDIS Error) without separator lines.
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
    # Create figure - reduced width since we only have 3 error columns + 2 colorbars
    fig = plt.figure(figsize=(26, 11))

    # Create outer GridSpec for the 2x2 layout
    outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1)

    # Plot titles - only error columns
    titles = ["FunDAPS Error", "FunDPS Error", "DDIS Error (Ours)"]
    row_titles = ["Inverse\nProblem", "Forward\nProblem"]

    def add_colorbar_matched(im, cax, ax, vmin=None, vmax=None, shift_x=0.0, shift_y=0.0):
        """Add a colorbar with height matched to the corresponding image axes.
        Automatically uses scientific notation for very small or large values.
        Args:
            shift_x: Horizontal offset for scientific notation (default: 0.0)
            shift_y: Vertical offset for scientific notation (default: 0.0)
        """
        ax_pos = ax.get_position()
        cbar = plt.colorbar(im, cax=cax)
        cbar = plt.colorbar(im, cax=cax)
        # Get figure for adding text
        fig = ax.figure

        # Match height exactly to the image axes (fix height bug)
        cax.set_position([ax_pos.x1 + 0.005, ax_pos.y0, cax.get_position().width, ax_pos.height])
        cax_pos = cax.get_position()
        
        # Format colorbar with scientific notation if needed
        if vmin is not None and vmax is not None:
            # Use scientific notation for better readability when values are very small or large
            # matplotlib's ScalarFormatter automatically switches to scientific notation
            # when values are outside the range [10^-2, 10^3] or when needed for clarity
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))  # Use scientific notation for exponents outside [-2, 3]
            # Use offset notation (scientific notation above colorbar) if values are very small or large
            max_abs = max(abs(vmin), abs(vmax))
            if max_abs > 0 and (max_abs < 0.01 or max_abs > 1000):
                # Calculate the exponent based on the order of magnitude
                # Find the order of magnitude that will normalize values to a reasonable range
                if max_abs < 1:
                    exponent = int(np.floor(np.log10(max_abs)))
                else:
                    exponent = int(np.floor(np.log10(max_abs)))
                
                # Get current ticks and normalize them
                ticks = cbar.get_ticks()
                if len(ticks) > 0:
                    scale_factor = 10 ** exponent
                    
                    # Set colorbar limits to exactly vmin and vmax FIRST to prevent blank areas
                    # Set limits on the mappable (image) first
                    cbar.mappable.set_clim(vmin, vmax)
                    
                    # Then set the colorbar axis limits to match exactly
                    cbar.ax.set_ylim(vmin, vmax)
                    
                    # Filter ticks to only include those within (vmin, vmax) range (exclude vmin and vmax)
                    # This ensures colorbar fills completely but doesn't show vmin/vmax as tick labels
                    ticks_in_range = ticks[(ticks > vmin) & (ticks < vmax)]
                    
                    # Use ticks within range (excluding vmin and vmax from display)
                    all_ticks = np.unique(ticks_in_range) if len(ticks_in_range) > 0 else np.array([])
                    normalized_ticks = all_ticks / scale_factor
                    
                    # Format tick labels (show normalized values with 1 decimal place)
                    tick_labels = []
                    for t in normalized_ticks:
                        tick_labels.append(f"{t:.1f}")  # Always show 1 decimal place (e.g., 4.2, 6.0, -2.0)
                    
                    # Fix UserWarning: set ticks before ticklabels
                    # Set ticks AFTER setting limits to ensure they respect the range
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Force update to ensure limits are applied, then restore tick labels
                    cbar.update_normal(cbar.mappable)
                    # Re-apply tick labels after update (update_normal may reset them)
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Position exponent above the top tick label
                    # For vertical colorbar, top tick is at y1, labels are to the right
                    # Align with the right edge where tick labels are, at the top
                    exp_x = cax_pos.x1 + shift_x  # To the right of colorbar, aligned with tick labels
                    exp_y = cax_pos.y1 + shift_y  # Above the top of colorbar
                else:
                    # Fallback if no ticks
                    exp_x = cax_pos.x1 + shift_x
                    exp_y = cax_pos.y1 + shift_y
                
                exp_str = f"×10$^{{{exponent}}}$"
                fig.text(exp_x, exp_y, exp_str, ha='left', va='bottom', fontsize=9)
            else:
                # For normal values, use standard formatting
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 3))
                cbar.formatter = formatter
                cbar.update_ticks()
        
        return cbar

    def plot_pair(forward_data, inverse_data, outer_spec, pair_idx):
        # Create inner GridSpec for this pair (2 rows for forward/inverse)
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_spec, height_ratios=[1, 1], hspace=0.1)

        # Create GridSpec for each row
        gs_rows = []
        for row in range(2):
            # 5 columns: Error1, cbar, Error2, Error3, cbar
            gs_row = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=inner_gs[row], width_ratios=[1, 0.08, 1, 1, 0.08], wspace=0.3)
            gs_rows.append(gs_row)

        def plot_row(data, row_idx, gs_row):
            # Get data with descriptive names (swapped: model1=FunDAPS, model2=FunDPS)
            ground_truth = data["ground_truth"]
            model1_pred = data["model1"]  # FunDAPS
            model2_pred = data["model2"]  # FunDPS
            model3_pred = data["model3"]  # DDIS (ours)

            # Calculate errors: |model_prediction - ground_truth|
            model1_error = np.abs(model1_pred - ground_truth)  # FunDAPS error vs true GT
            model2_error = np.abs(model2_pred - ground_truth)  # FunDPS error vs true GT
            model3_error = np.abs(model3_pred - ground_truth)  # DDIS error vs true GT

            # Separate error scales: FunDAPS has larger errors, FunDPS/DDIS share scale
            err_max_fundaps = np.max(model1_error)  # FunDAPS error scale (larger)
            err_max_others = max(np.max(model2_error), np.max(model3_error))  # FunDPS/DDIS scale (smaller)

            # Only error columns: Error1 (FunDAPS), Error2 (FunDPS), Error3 (DDIS)
            plot_positions = [0, 2, 3]  # Error1, Error2, Error3
            cbar_positions = [1, 4]  # Colorbars for Error1 (FunDAPS), Error3 (DDIS)
            axes_dict = {}

            # Column 0: Model 1 Error (FunDAPS) - uses separate larger scale with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[0]])
            im1 = ax.imshow(model1_error, cmap="viridis", norm=get_norm(0, err_max_fundaps, gamma=gamma), interpolation="none")
            cax1 = fig.add_subplot(gs_row[cbar_positions[0]])
            add_colorbar_matched(im1, cax1, ax, vmin=0, vmax=err_max_fundaps, shift_x=shift_x, shift_y=shift_y)
            axes_dict[0] = ax

            # Column 1: Model 2 Error (FunDPS) - uses smaller scale shared with DDIS
            ax = fig.add_subplot(gs_row[plot_positions[1]])
            ax.imshow(model2_error, cmap="viridis", norm=get_norm(0, err_max_others, gamma=gamma), interpolation="none")
            axes_dict[1] = ax

            # Column 2: Model 3 Error (DDIS - ours) - uses smaller scale shared with FunDPS with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[2]])
            im3 = ax.imshow(model3_error, cmap="viridis", norm=get_norm(0, err_max_others, gamma=gamma), interpolation="none")
            cax3 = fig.add_subplot(gs_row[cbar_positions[1]])
            add_colorbar_matched(im3, cax3, ax, vmin=0, vmax=err_max_others, shift_x=shift_x, shift_y=shift_y)
            axes_dict[2] = ax

            # Remove ticks from all plots
            for ax in axes_dict.values():
                ax.set_xticks([])
                ax.set_yticks([])

            # Add row title to first plot
            row_label = f"{row_titles[row_idx]}"
            axes_dict[0].set_ylabel(row_label, fontsize=20)

            # Add titles only to first pair and first row (inverse problem)
            if pair_idx == 0 and row_idx == 0:
                for i, title in enumerate(titles):
                    axes_dict[i].set_title(title, pad=10, fontsize=20)
            
            # Add L2 error percentage below error columns for both rows (inverse and forward problems)
            if pair_idx == 0:
                # Calculate L2 error as percentage: ||pred - gt||_2 / ||gt||_2 * 100
                l2_error_model1 = np.linalg.norm(model1_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
                l2_error_model2 = np.linalg.norm(model2_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
                l2_error_model3 = np.linalg.norm(model3_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
                
                # Add text below error columns (closer to the plot) with upward arrow
                axes_dict[0].text(0.5, -0.02, f'↑L2 Error: {l2_error_model1:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[0].transAxes, fontsize=16)
                axes_dict[1].text(0.5, -0.02, f'↑L2 Error: {l2_error_model2:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[1].transAxes, fontsize=16)
                axes_dict[2].text(0.5, -0.02, f'↑L2 Error: {l2_error_model3:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[2].transAxes, fontsize=16)

        # Plot inverse and forward problems (matching row_titles order)
        plot_row(inverse_data, 0, gs_rows[0])  # Row 0: Inverse Problem
        plot_row(forward_data, 1, gs_rows[1])  # Row 1: Forward Problem

    # Plot all pairs in 2x2 grid
    for i, (forward_data, inverse_data) in enumerate(data_pairs):
        row = i // 2
        col = i % 2
        plot_pair(forward_data, inverse_data, outer_gs[row, col], i)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def create_comparison_plots_no_fundaps(data_pairs, save_path=None, l2_errors=None, color_scale="gt", custom_vmin=None, custom_vmax=None, shift_x=0.0, shift_y=0.0, gamma=1.0):
    """
    Create comparison plots without FunDAPS columns for multiple pairs of forward and inverse problems in a 2x2 grid.
    Shows: Ground Truth, FunDPS, DDIS (Ours), FunDPS Error, DDIS Error (Ours).
    Error numbers are displayed in both rows (inverse and forward problems).
    No grey dashed separator lines.
    Args:
        data_pairs: List of tuples, each containing (forward_data, inverse_data) where each contains:
            - ground_truth: True ground truth array
            - model1: First baseline model predictions (e.g., FunDAPS) - will be ignored
            - model2: Second baseline model predictions (e.g., FunDPS)
            - model3: Third model predictions (e.g., DDIS - ours)
            - mask: Binary mask array (1 for observed points, 0 for non-observed) [optional]
        save_path: Optional path to save the figure
        l2_errors: Optional dict with keys 'model2', 'model3' containing L2 error percentages (model1 ignored)
        color_scale: Color scale method. Options:
            - "gt": Use GT scale for all columns (good for absolute comparison)
            - "ddis": Use DDIS scale for all columns (DDIS as standard)
            - "global": Use global min/max across GT, FunDPS, DDIS (better for showing over-smoothing)
            - "individual": Each column uses its own scale (shows full detail but harder to compare)
            - "self-def": Use custom min/max values (requires custom_vmin and custom_vmax)
        custom_vmin: Custom minimum value for color scale (required when color_scale="self-def")
        custom_vmax: Custom maximum value for color scale (required when color_scale="self-def")
    """
    # Create figure - reduced width since we only have 5 columns + 2 colorbars
    fig = plt.figure(figsize=(35, 12))

    # Create outer GridSpec for the 2x2 layout
    outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1)

    # Plot titles - without FunDAPS
    titles = ["Ground Truth", "FunDPS", "DDIS (Ours)", "FunDPS Error", "DDIS Error (Ours)"]
    row_titles = ["Inverse\nProblem", "Forward\nProblem"]

    def add_colorbar_matched(im, cax, ax, vmin=None, vmax=None, shift_x=0.0, shift_y=0.0):
        """Add a colorbar with height matched to the corresponding image axes.
        Automatically uses scientific notation for very small or large values.
        Args:
            shift_x: Horizontal offset for scientific notation (default: 0.0)
            shift_y: Vertical offset for scientific notation (default: 0.0)
        """
        ax_pos = ax.get_position()
        cbar = plt.colorbar(im, cax=cax)
        cbar = plt.colorbar(im, cax=cax)
        # Get figure for adding text
        fig = ax.figure

        # Match height exactly to the image axes (fix height bug)
        cax.set_position([ax_pos.x1 + 0.005, ax_pos.y0, cax.get_position().width, ax_pos.height])
        cax_pos = cax.get_position()
        
        # Format colorbar with scientific notation if needed
        if vmin is not None and vmax is not None:
            # Use scientific notation for better readability when values are very small or large
            # matplotlib's ScalarFormatter automatically switches to scientific notation
            # when values are outside the range [10^-2, 10^3] or when needed for clarity
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))  # Use scientific notation for exponents outside [-2, 3]
            # Use offset notation (scientific notation above colorbar) if values are very small or large
            max_abs = max(abs(vmin), abs(vmax))
            if max_abs > 0 and (max_abs < 0.01 or max_abs > 1000):
                # Calculate the exponent based on the order of magnitude
                # Find the order of magnitude that will normalize values to a reasonable range
                if max_abs < 1:
                    exponent = int(np.floor(np.log10(max_abs)))
                else:
                    exponent = int(np.floor(np.log10(max_abs)))
                
                # Get current ticks and normalize them
                ticks = cbar.get_ticks()
                if len(ticks) > 0:
                    scale_factor = 10 ** exponent
                    
                    # Set colorbar limits to exactly vmin and vmax FIRST to prevent blank areas
                    # Set limits on the mappable (image) first
                    cbar.mappable.set_clim(vmin, vmax)
                    
                    # Then set the colorbar axis limits to match exactly
                    cbar.ax.set_ylim(vmin, vmax)
                    
                    # Filter ticks to only include those within (vmin, vmax) range (exclude vmin and vmax)
                    # This ensures colorbar fills completely but doesn't show vmin/vmax as tick labels
                    ticks_in_range = ticks[(ticks > vmin) & (ticks < vmax)]
                    
                    # Use ticks within range (excluding vmin and vmax from display)
                    all_ticks = np.unique(ticks_in_range) if len(ticks_in_range) > 0 else np.array([])
                    normalized_ticks = all_ticks / scale_factor
                    
                    # Format tick labels (show normalized values with 1 decimal place)
                    tick_labels = []
                    for t in normalized_ticks:
                        tick_labels.append(f"{t:.1f}")  # Always show 1 decimal place (e.g., 4.2, 6.0, -2.0)
                    
                    # Fix UserWarning: set ticks before ticklabels
                    # Set ticks AFTER setting limits to ensure they respect the range
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Force update to ensure limits are applied, then restore tick labels
                    cbar.update_normal(cbar.mappable)
                    # Re-apply tick labels after update (update_normal may reset them)
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Position exponent above the top tick label
                    # For vertical colorbar, top tick is at y1, labels are to the right
                    # Align with the right edge where tick labels are, at the top
                    # Note: cax_pos is calculated per colorbar, so this works correctly for both rows
                    # (inverse problem row 0 and forward problem row 1)
                    exp_x = cax_pos.x1 + shift_x  # To the right of colorbar, aligned with tick labels
                    exp_y = cax_pos.y1 + shift_y  # Above the top of colorbar (relative to each row's colorbar)
                else:
                    # Fallback if no ticks
                    exp_x = cax_pos.x1 + shift_x
                    exp_y = cax_pos.y1 + shift_y
                
                exp_str = f"×10$^{{{exponent}}}$"
                # Add scientific notation text - this is applied to each colorbar individually
                # so it will appear on both rows (inverse and forward) when values trigger it
                fig.text(exp_x, exp_y, exp_str, ha='left', va='bottom', fontsize=9)
            else:
                # For normal values, use standard formatting
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 3))
                cbar.formatter = formatter
                cbar.update_ticks()
        
        return cbar

    def plot_pair(forward_data, inverse_data, outer_spec, pair_idx):
        # Create inner GridSpec for this pair (2 rows for forward/inverse)
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_spec, height_ratios=[1, 1], hspace=0.1)

        # Create GridSpec for each row
        gs_rows = []
        for row in range(2):
            if color_scale == "individual":
                # 9 columns: GT, cbar, FunDPS, cbar, DDIS, cbar, FunDPS Error, DDIS Error, cbar
                gs_row = gridspec.GridSpecFromSubplotSpec(1, 9, subplot_spec=inner_gs[row], 
                    width_ratios=[1, 0.08, 1, 0.08, 1, 0.08, 1, 1, 0.08], wspace=0.3)
            else:
                # 7 columns: GT, FunDPS, DDIS, cbar, FunDPS Error, DDIS Error, cbar
                gs_row = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=inner_gs[row], 
                    width_ratios=[1, 1, 1, 0.08, 1, 1, 0.08], wspace=0.3)
            gs_rows.append(gs_row)

        def plot_row(data, row_idx, gs_row):
            # Get data - model1 (FunDAPS) is ignored
            ground_truth = data["ground_truth"]
            model2_pred = data["model2"]  # FunDPS
            model3_pred = data["model3"]  # DDIS (ours)

            # Calculate errors: |model_prediction - ground_truth|
            model2_error = np.abs(model2_pred - ground_truth)  # FunDPS error vs true GT
            model3_error = np.abs(model3_pred - ground_truth)  # DDIS error vs true GT

            # Get min/max for consistent coloring based on color_scale method
            gt_min = np.min(ground_truth)
            gt_max = np.max(ground_truth)
            fundps_min = np.min(model2_pred)
            fundps_max = np.max(model2_pred)
            ddis_min = np.min(model3_pred)
            ddis_max = np.max(model3_pred)
            
            if color_scale == "gt":
                # Use GT scale for all columns
                vmin, vmax = gt_min, gt_max
            elif color_scale == "ddis":
                # Use DDIS scale for all columns (DDIS as standard)
                vmin, vmax = ddis_min, ddis_max
            elif color_scale == "global":
                # Use global min/max across all three (better for showing over-smoothing)
                vmin = min(gt_min, fundps_min, ddis_min)
                vmax = max(gt_max, fundps_max, ddis_max)
            elif color_scale == "self-def":
                # Use custom min/max values
                if custom_vmin is None or custom_vmax is None:
                    raise ValueError("custom_vmin and custom_vmax must be provided when color_scale='self-def'")
                vmin, vmax = custom_vmin, custom_vmax
            elif color_scale == "individual":
                # Each column will use its own scale (set per column below)
                vmin, vmax = None, None  # Will be set individually
            else:
                raise ValueError(f"Unknown color_scale: {color_scale}. Must be 'gt', 'ddis', 'global', 'self-def', or 'individual'")
            
            # Print scale information for all three columns
            if pair_idx == 0 and row_idx == 0:  # Only print once for the first pair, first row
                print(f"\nColor scale information ({color_scale} mode):")
                print(f"  Ground Truth:  min={gt_min:.6f}, max={gt_max:.6f}, range={gt_max-gt_min:.6f}")
                print(f"  FunDPS:        min={fundps_min:.6f}, max={fundps_max:.6f}, range={fundps_max-fundps_min:.6f}")
                print(f"  DDIS:          min={ddis_min:.6f}, max={ddis_max:.6f}, range={ddis_max-ddis_min:.6f}")
                if color_scale == "self-def":
                    print(f"  Using custom scale: min={custom_vmin:.6f}, max={custom_vmax:.6f}, range={custom_vmax-custom_vmin:.6f}")
                elif color_scale in ["gt", "ddis", "global"]:
                    print(f"  Applied scale:     min={vmin:.6f}, max={vmax:.6f}, range={vmax-vmin:.6f}")
            
            # Error scales: FunDPS and DDIS share scale
            err_max = max(np.max(model2_error), np.max(model3_error))

            # Column positions depend on color_scale mode
            if color_scale == "individual":
                # 9 columns: GT, cbar, FunDPS, cbar, DDIS, cbar, FunDPS Error, DDIS Error, cbar
                plot_positions = [0, 2, 4, 6, 7]  # GT, FunDPS, DDIS, FunDPS Error, DDIS Error
                cbar_positions = [1, 3, 5, 8]  # Colorbars for GT, FunDPS, DDIS pred, DDIS Error
            else:
                # 7 columns: GT, FunDPS, DDIS, cbar, FunDPS Error, DDIS Error, cbar
                # For "gt", "ddis", "global", "self-def": all use shared scale, only DDIS has colorbar
                plot_positions = [0, 1, 2, 4, 5]  # GT, FunDPS, DDIS, FunDPS Error, DDIS Error
                cbar_positions = [3, 6]  # Colorbars for DDIS pred, DDIS Error
            axes_dict = {}

            # Column 0: Ground Truth
            ax = fig.add_subplot(gs_row[plot_positions[0]])
            if color_scale == "individual":
                im0 = ax.imshow(ground_truth, cmap="viridis", norm=get_norm(gt_min, gt_max, gamma=gamma), interpolation="none")
                # Add colorbar for GT when using individual scale
                cax0 = fig.add_subplot(gs_row[cbar_positions[0]])
                add_colorbar_matched(im0, cax0, ax, vmin=gt_min, vmax=gt_max, shift_x=shift_x, shift_y=shift_y)
            else:
                # Use shared scale (gt, ddis, global, or self-def)
                ax.imshow(ground_truth, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
            axes_dict[0] = ax

            # Column 1: FunDPS
            ax = fig.add_subplot(gs_row[plot_positions[1]])
            if color_scale == "individual":
                im1 = ax.imshow(model2_pred, cmap="viridis", norm=get_norm(fundps_min, fundps_max, gamma=gamma), interpolation="none")
                # Add colorbar for FunDPS when using individual scale
                cax1 = fig.add_subplot(gs_row[cbar_positions[1]])
                add_colorbar_matched(im1, cax1, ax, vmin=fundps_min, vmax=fundps_max, shift_x=shift_x, shift_y=shift_y)
            else:
                # Use shared scale (gt, ddis, global, or self-def)
                ax.imshow(model2_pred, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
            axes_dict[1] = ax

            # Column 2: DDIS (ours) with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[2]])
            if color_scale == "individual":
                im3 = ax.imshow(model3_pred, cmap="viridis", norm=get_norm(ddis_min, ddis_max, gamma=gamma), interpolation="none")
                # Add colorbar for DDIS when using individual scale
                cax3 = fig.add_subplot(gs_row[cbar_positions[2]])
                add_colorbar_matched(im3, cax3, ax, vmin=ddis_min, vmax=ddis_max, shift_x=shift_x, shift_y=shift_y)
            else:
                # Use shared scale (gt, ddis, global, or self-def) - only DDIS shows colorbar
                im3 = ax.imshow(model3_pred, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
                # Show colorbar for DDIS when using shared scales (gt, ddis, global, or self-def)
                cax3 = fig.add_subplot(gs_row[cbar_positions[0]])
                add_colorbar_matched(im3, cax3, ax, vmin=vmin, vmax=vmax, shift_x=shift_x, shift_y=shift_y)
            axes_dict[2] = ax

            # Column 3: FunDPS Error
            ax = fig.add_subplot(gs_row[plot_positions[3]])
            ax.imshow(model2_error, cmap="viridis", norm=get_norm(0, err_max, gamma=gamma), interpolation="none")
            axes_dict[3] = ax

            # Column 4: DDIS Error (ours) with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[4]])
            im5 = ax.imshow(model3_error, cmap="viridis", norm=get_norm(0, err_max, gamma=gamma), interpolation="none")
            if color_scale == "individual":
                cax5 = fig.add_subplot(gs_row[cbar_positions[3]])  # Last colorbar position for individual mode
            else:
                cax5 = fig.add_subplot(gs_row[cbar_positions[1]])  # Second colorbar position for shared scale mode
            add_colorbar_matched(im5, cax5, ax, vmin=0, vmax=err_max, shift_x=shift_x, shift_y=shift_y)
            axes_dict[4] = ax

            # Remove ticks from all plots
            for ax in axes_dict.values():
                ax.set_xticks([])
                ax.set_yticks([])

            # Add row title to first plot
            row_label = f"{row_titles[row_idx]}"
            axes_dict[0].set_ylabel(row_label, fontsize=20)

            # Add titles only to first pair and first row (inverse problem)
            if pair_idx == 0 and row_idx == 0:
                for i, title in enumerate(titles):
                    axes_dict[i].set_title(title, pad=10, fontsize=20)
            
            # Add L2 error percentage below error columns for both rows (inverse and forward problems)
            if pair_idx == 0:
                # Calculate L2 error as percentage: ||pred - gt||_2 / ||gt||_2 * 100
                l2_error_model2 = np.linalg.norm(model2_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
                l2_error_model3 = np.linalg.norm(model3_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
                
                # Add text below error columns (closer to the plot) with upward arrow
                axes_dict[3].text(0.5, -0.02, f'↑L2 Error: {l2_error_model2:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[3].transAxes, fontsize=16)
                axes_dict[4].text(0.5, -0.02, f'↑L2 Error: {l2_error_model3:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[4].transAxes, fontsize=16)

        # Plot inverse and forward problems (matching row_titles order)
        plot_row(inverse_data, 0, gs_rows[0])  # Row 0: Inverse Problem
        plot_row(forward_data, 1, gs_rows[1])  # Row 1: Forward Problem

    # Plot all pairs in 2x2 grid
    for i, (forward_data, inverse_data) in enumerate(data_pairs):
        row = i // 2
        col = i % 2
        plot_pair(forward_data, inverse_data, outer_gs[row, col], i)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def create_comparison_plots_fundps_only(data_pairs, save_path=None, l2_errors=None, color_scale="gt", custom_vmin=None, custom_vmax=None, shift_x=0.0, shift_y=0.0, gamma=1.0):
    """
    Create comparison plots with FunDPS only for multiple pairs of forward and inverse problems in a 2x2 grid.
    Shows: Ground Truth, FunDPS, FunDPS Error.
    Error numbers are displayed in both rows (inverse and forward problems).
    Args:
        data_pairs: List of tuples, each containing (forward_data, inverse_data) where each contains:
            - ground_truth: True ground truth array
            - model1: First baseline model predictions (e.g., FunDAPS) - will be ignored
            - model2: Second baseline model predictions (e.g., FunDPS)
            - model3: Third model predictions (e.g., DDIS - ours) - will be ignored
            - mask: Binary mask array (1 for observed points, 0 for non-observed) [optional]
        save_path: Optional path to save the figure
        l2_errors: Optional dict with key 'model2' containing L2 error percentage (model1 and model3 ignored)
        color_scale: Color scale method. Options:
            - "gt": Use GT scale for all columns (good for absolute comparison)
            - "fundps": Use FunDPS scale for all columns (FunDPS as standard)
            - "global": Use global min/max across GT and FunDPS (better for showing over-smoothing)
            - "individual": Each column uses its own scale (shows full detail but harder to compare)
            - "self-def": Use custom min/max values (requires custom_vmin and custom_vmax)
        custom_vmin: Custom minimum value for color scale (required when color_scale="self-def")
        custom_vmax: Custom maximum value for color scale (required when color_scale="self-def")
    """
    # Create figure - reduced width since we only have 3 columns + 1 colorbar
    fig = plt.figure(figsize=(20, 12))

    # Create outer GridSpec for the 2x2 layout
    outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1)

    # Plot titles - FunDPS only
    titles = ["Ground Truth", "FunDPS", "FunDPS Error"]
    row_titles = ["Inverse\nProblem", "Forward\nProblem"]

    def add_colorbar_matched(im, cax, ax, vmin=None, vmax=None, shift_x=0.0, shift_y=0.0):
        """Add a colorbar with height matched to the corresponding image axes.
        Automatically uses scientific notation for very small or large values.
        Args:
            shift_x: Horizontal offset for scientific notation (default: 0.0)
            shift_y: Vertical offset for scientific notation (default: 0.0)
        """
        ax_pos = ax.get_position()
        cbar = plt.colorbar(im, cax=cax)
        cbar = plt.colorbar(im, cax=cax)
        # Get figure for adding text
        fig = ax.figure

        # Match height exactly to the image axes (fix height bug)
        cax.set_position([ax_pos.x1 + 0.005, ax_pos.y0, cax.get_position().width, ax_pos.height])
        cax_pos = cax.get_position()
        
        # Format colorbar with scientific notation if needed
        if vmin is not None and vmax is not None:
            # Use scientific notation for better readability when values are very small or large
            # matplotlib's ScalarFormatter automatically switches to scientific notation
            # when values are outside the range [10^-2, 10^3] or when needed for clarity
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))  # Use scientific notation for exponents outside [-2, 3]
            # Use offset notation (scientific notation above colorbar) if values are very small or large
            max_abs = max(abs(vmin), abs(vmax))
            if max_abs > 0 and (max_abs < 0.01 or max_abs > 1000):
                # Calculate the exponent based on the order of magnitude
                # Find the order of magnitude that will normalize values to a reasonable range
                if max_abs < 1:
                    exponent = int(np.floor(np.log10(max_abs)))
                else:
                    exponent = int(np.floor(np.log10(max_abs)))
                
                # Get current ticks and normalize them
                ticks = cbar.get_ticks()
                if len(ticks) > 0:
                    scale_factor = 10 ** exponent
                    
                    # Set colorbar limits to exactly vmin and vmax FIRST to prevent blank areas
                    # Set limits on the mappable (image) first
                    cbar.mappable.set_clim(vmin, vmax)
                    
                    # Then set the colorbar axis limits to match exactly
                    cbar.ax.set_ylim(vmin, vmax)
                    
                    # Filter ticks to only include those within (vmin, vmax) range (exclude vmin and vmax)
                    # This ensures colorbar fills completely but doesn't show vmin/vmax as tick labels
                    ticks_in_range = ticks[(ticks > vmin) & (ticks < vmax)]
                    
                    # Use ticks within range (excluding vmin and vmax from display)
                    all_ticks = np.unique(ticks_in_range) if len(ticks_in_range) > 0 else np.array([])
                    normalized_ticks = all_ticks / scale_factor
                    
                    # Format tick labels (show normalized values with 1 decimal place)
                    tick_labels = []
                    for t in normalized_ticks:
                        tick_labels.append(f"{t:.1f}")  # Always show 1 decimal place (e.g., 4.2, 6.0, -2.0)
                    
                    # Fix UserWarning: set ticks before ticklabels
                    # Set ticks AFTER setting limits to ensure they respect the range
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Force update to ensure limits are applied, then restore tick labels
                    cbar.update_normal(cbar.mappable)
                    # Re-apply tick labels after update (update_normal may reset them)
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Position exponent above the top tick label
                    # For vertical colorbar, top tick is at y1, labels are to the right
                    # Align with the right edge where tick labels are, at the top
                    # Note: cax_pos is calculated per colorbar, so this works correctly for both rows
                    # (inverse problem row 0 and forward problem row 1)
                    exp_x = cax_pos.x1 + shift_x  # To the right of colorbar, aligned with tick labels
                    exp_y = cax_pos.y1 + shift_y  # Above the top of colorbar (relative to each row's colorbar)
                else:
                    # Fallback if no ticks
                    exp_x = cax_pos.x1 + shift_x
                    exp_y = cax_pos.y1 + shift_y
                
                exp_str = f"×10$^{{{exponent}}}$"
                # Add scientific notation text - this is applied to each colorbar individually
                # so it will appear on both rows (inverse and forward) when values trigger it
                fig.text(exp_x, exp_y, exp_str, ha='left', va='bottom', fontsize=9)
            else:
                # For normal values, use standard formatting
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 3))
                cbar.formatter = formatter
                cbar.update_ticks()
        
        return cbar

    def plot_pair(forward_data, inverse_data, outer_spec, pair_idx):
        # Create inner GridSpec for this pair (2 rows for forward/inverse)
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_spec, height_ratios=[1, 1], hspace=0.1)

        # Create GridSpec for each row
        gs_rows = []
        for row in range(2):
            if color_scale == "individual":
                # 6 columns: GT, cbar, FunDPS, cbar, FunDPS Error, cbar
                gs_row = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=inner_gs[row], 
                    width_ratios=[1, 0.08, 1, 0.08, 1, 0.08], wspace=0.3)
            else:
                # 5 columns: GT, FunDPS, cbar, FunDPS Error, cbar
                gs_row = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=inner_gs[row], 
                    width_ratios=[1, 1, 0.08, 1, 0.08], wspace=0.3)
            gs_rows.append(gs_row)

        def plot_row(data, row_idx, gs_row):
            # Get data - only use model2 (FunDPS), ignore model1 (FunDAPS) and model3 (DDIS)
            ground_truth = data["ground_truth"]
            model2_pred = data["model2"]  # FunDPS

            # Calculate errors: |model_prediction - ground_truth|
            model2_error = np.abs(model2_pred - ground_truth)  # FunDPS error vs true GT

            # Get min/max for consistent coloring based on color_scale method
            gt_min = np.min(ground_truth)
            gt_max = np.max(ground_truth)
            fundps_min = np.min(model2_pred)
            fundps_max = np.max(model2_pred)
            
            if color_scale == "gt":
                # Use GT scale for all columns
                vmin, vmax = gt_min, gt_max
            elif color_scale == "fundps":
                # Use FunDPS scale for all columns (FunDPS as standard)
                vmin, vmax = fundps_min, fundps_max
            elif color_scale == "global":
                # Use global min/max across GT and FunDPS (better for showing over-smoothing)
                vmin = min(gt_min, fundps_min)
                vmax = max(gt_max, fundps_max)
            elif color_scale == "self-def":
                # Use custom min/max values
                if custom_vmin is None or custom_vmax is None:
                    raise ValueError("custom_vmin and custom_vmax must be provided when color_scale='self-def'")
                vmin, vmax = custom_vmin, custom_vmax
            elif color_scale == "individual":
                # Each column will use its own scale (set per column below)
                vmin, vmax = None, None  # Will be set individually
            else:
                raise ValueError(f"Unknown color_scale: {color_scale}. Must be 'gt', 'fundps', 'global', 'self-def', or 'individual'")
            
            # Print scale information
            if pair_idx == 0 and row_idx == 0:  # Only print once for the first pair, first row
                print(f"\nColor scale information ({color_scale} mode):")
                print(f"  Ground Truth:  min={gt_min:.6f}, max={gt_max:.6f}, range={gt_max-gt_min:.6f}")
                print(f"  FunDPS:        min={fundps_min:.6f}, max={fundps_max:.6f}, range={fundps_max-fundps_min:.6f}")
                if color_scale == "self-def":
                    print(f"  Using custom scale: min={custom_vmin:.6f}, max={custom_vmax:.6f}, range={custom_vmax-custom_vmin:.6f}")
                elif color_scale in ["gt", "fundps", "global"]:
                    print(f"  Applied scale:     min={vmin:.6f}, max={vmax:.6f}, range={vmax-vmin:.6f}")
            
            # Error scale
            err_max = np.max(model2_error)

            # Column positions depend on color_scale mode
            if color_scale == "individual":
                # 6 columns: GT, cbar, FunDPS, cbar, FunDPS Error, cbar
                plot_positions = [0, 2, 4]  # GT, FunDPS, FunDPS Error
                cbar_positions = [1, 3, 5]  # Colorbars for GT, FunDPS, FunDPS Error
            else:
                # 5 columns: GT, FunDPS, cbar, FunDPS Error, cbar
                plot_positions = [0, 1, 3]  # GT, FunDPS, FunDPS Error
                cbar_positions = [2, 4]  # Colorbars for FunDPS pred, FunDPS Error
            axes_dict = {}

            # Column 0: Ground Truth
            ax = fig.add_subplot(gs_row[plot_positions[0]])
            if color_scale == "individual":
                im0 = ax.imshow(ground_truth, cmap="viridis", norm=get_norm(gt_min, gt_max, gamma=gamma), interpolation="none")
                # Add colorbar for GT when using individual scale
                cax0 = fig.add_subplot(gs_row[cbar_positions[0]])
                add_colorbar_matched(im0, cax0, ax, vmin=gt_min, vmax=gt_max, shift_x=shift_x, shift_y=shift_y)
            else:
                # Use shared scale (gt, fundps, global, or self-def)
                ax.imshow(ground_truth, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
            axes_dict[0] = ax

            # Column 1: FunDPS with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[1]])
            if color_scale == "individual":
                im1 = ax.imshow(model2_pred, cmap="viridis", norm=get_norm(fundps_min, fundps_max, gamma=gamma), interpolation="none")
                # Add colorbar for FunDPS when using individual scale
                cax1 = fig.add_subplot(gs_row[cbar_positions[1]])
                add_colorbar_matched(im1, cax1, ax, vmin=fundps_min, vmax=fundps_max, shift_x=shift_x, shift_y=shift_y)
            else:
                # Use shared scale (gt, fundps, global, or self-def) - FunDPS shows colorbar
                im1 = ax.imshow(model2_pred, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
                # Show colorbar for FunDPS when using shared scales
                cax1 = fig.add_subplot(gs_row[cbar_positions[0]])
                add_colorbar_matched(im1, cax1, ax, vmin=vmin, vmax=vmax, shift_x=shift_x, shift_y=shift_y)
            axes_dict[1] = ax

            # Column 2: FunDPS Error with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[2]])
            im2 = ax.imshow(model2_error, cmap="viridis", norm=get_norm(0, err_max, gamma=gamma), interpolation="none")
            if color_scale == "individual":
                cax2 = fig.add_subplot(gs_row[cbar_positions[2]])  # Last colorbar position for individual mode
            else:
                cax2 = fig.add_subplot(gs_row[cbar_positions[1]])  # Last colorbar position for shared scale mode
            add_colorbar_matched(im2, cax2, ax, vmin=0, vmax=err_max, shift_x=shift_x, shift_y=shift_y)
            axes_dict[2] = ax

            # Remove ticks from all plots
            for ax in axes_dict.values():
                ax.set_xticks([])
                ax.set_yticks([])

            # Add row title to first plot
            row_label = f"{row_titles[row_idx]}"
            axes_dict[0].set_ylabel(row_label, fontsize=20)

            # Add titles only to first pair and first row (inverse problem)
            if pair_idx == 0 and row_idx == 0:
                for i, title in enumerate(titles):
                    axes_dict[i].set_title(title, pad=10, fontsize=20)
            
            # Add L2 error percentage below error column for both rows (inverse and forward problems)
            if pair_idx == 0:
                # Use provided L2 errors if available, otherwise calculate them
                if l2_errors is not None and 'model2' in l2_errors:
                    l2_error_model2 = l2_errors['model2']
                else:
                    # Calculate L2 error as percentage: ||pred - gt||_2 / ||gt||_2 * 100
                    l2_error_model2 = np.linalg.norm(model2_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
                
                # Add text below error column (closer to the plot) with upward arrow
                axes_dict[2].text(0.5, -0.02, f'↑L2 Error: {l2_error_model2:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[2].transAxes, fontsize=16)

        # Plot inverse and forward problems (matching row_titles order)
        plot_row(inverse_data, 0, gs_rows[0])  # Row 0: Inverse Problem
        plot_row(forward_data, 1, gs_rows[1])  # Row 1: Forward Problem

    # Plot all pairs in 2x2 grid
    for i, (forward_data, inverse_data) in enumerate(data_pairs):
        row = i // 2
        col = i % 2
        plot_pair(forward_data, inverse_data, outer_gs[row, col], i)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def create_comparison_plots_fundaps_only(data_pairs, save_path=None, l2_errors=None, color_scale="gt", custom_vmin=None, custom_vmax=None, shift_x=0.0, shift_y=0.0, gamma=1.0):
    """
    Create comparison plots with FunDAPS only for multiple pairs of forward and inverse problems in a 2x2 grid.
    Shows: Ground Truth, FunDAPS, FunDAPS Error.
    Error numbers are displayed in both rows (inverse and forward problems).
    Args:
        data_pairs: List of tuples, each containing (forward_data, inverse_data) where each contains:
            - ground_truth: True ground truth array
            - model1: First baseline model predictions (e.g., FunDAPS)
            - model2: Second baseline model predictions (e.g., FunDPS) - will be ignored
            - model3: Third model predictions (e.g., DDIS - ours) - will be ignored
            - mask: Binary mask array (1 for observed points, 0 for non-observed) [optional]
        save_path: Optional path to save the figure
        l2_errors: Optional dict with key 'model1' containing L2 error percentage (model2 and model3 ignored)
        color_scale: Color scale method. Options:
            - "gt": Use GT scale for all columns (good for absolute comparison)
            - "fundaps": Use FunDAPS scale for all columns (FunDAPS as standard)
            - "global": Use global min/max across GT and FunDAPS (better for showing over-smoothing)
            - "individual": Each column uses its own scale (shows full detail but harder to compare)
            - "self-def": Use custom min/max values (requires custom_vmin and custom_vmax)
        custom_vmin: Custom minimum value for color scale (required when color_scale="self-def")
        custom_vmax: Custom maximum value for color scale (required when color_scale="self-def")
    """
    # Create figure - reduced width since we only have 3 columns + 1 colorbar
    fig = plt.figure(figsize=(22, 12.5))

    # Create outer GridSpec for the 2x2 layout
    outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1)

    # Plot titles - FunDAPS only
    titles = ["Ground Truth", "FunDAPS", "FunDAPS Error"]
    row_titles = ["Inverse\nProblem", "Forward\nProblem"]

    def add_colorbar_matched(im, cax, ax, vmin=None, vmax=None, shift_x=0.0, shift_y=0.0):
        """Add a colorbar with height matched to the corresponding image axes.
        Automatically uses scientific notation for very small or large values.
        Args:
            shift_x: Horizontal offset for scientific notation (default: 0.0)
            shift_y: Vertical offset for scientific notation (default: 0.0)
        """
        ax_pos = ax.get_position()
        cbar = plt.colorbar(im, cax=cax)
        cbar = plt.colorbar(im, cax=cax)
        # Get figure for adding text
        fig = ax.figure

        # Match height exactly to the image axes (fix height bug)
        cax.set_position([ax_pos.x1 + 0.005, ax_pos.y0, cax.get_position().width, ax_pos.height])
        cax_pos = cax.get_position()
        
        # Format colorbar with scientific notation if needed
        if vmin is not None and vmax is not None:
            # Use scientific notation for better readability when values are very small or large
            # matplotlib's ScalarFormatter automatically switches to scientific notation
            # when values are outside the range [10^-2, 10^3] or when needed for clarity
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))  # Use scientific notation for exponents outside [-2, 3]
            # Use offset notation (scientific notation above colorbar) if values are very small or large
            max_abs = max(abs(vmin), abs(vmax))
            if max_abs > 0 and (max_abs < 0.01 or max_abs > 1000):
                # Calculate the exponent based on the order of magnitude
                # Find the order of magnitude that will normalize values to a reasonable range
                if max_abs < 1:
                    exponent = int(np.floor(np.log10(max_abs)))
                else:
                    exponent = int(np.floor(np.log10(max_abs)))
                
                # Get current ticks and normalize them
                ticks = cbar.get_ticks()
                if len(ticks) > 0:
                    scale_factor = 10 ** exponent
                    
                    # Set colorbar limits to exactly vmin and vmax FIRST to prevent blank areas
                    # Set limits on the mappable (image) first
                    cbar.mappable.set_clim(vmin, vmax)
                    
                    # Then set the colorbar axis limits to match exactly
                    cbar.ax.set_ylim(vmin, vmax)
                    
                    # Filter ticks to only include those within (vmin, vmax) range (exclude vmin and vmax)
                    # This ensures colorbar fills completely but doesn't show vmin/vmax as tick labels
                    ticks_in_range = ticks[(ticks > vmin) & (ticks < vmax)]
                    
                    # Use ticks within range (excluding vmin and vmax from display)
                    all_ticks = np.unique(ticks_in_range) if len(ticks_in_range) > 0 else np.array([])
                    normalized_ticks = all_ticks / scale_factor
                    
                    # Format tick labels (show normalized values with 1 decimal place)
                    tick_labels = []
                    for t in normalized_ticks:
                        tick_labels.append(f"{t:.1f}")  # Always show 1 decimal place (e.g., 4.2, 6.0, -2.0)
                    
                    # Fix UserWarning: set ticks before ticklabels
                    # Set ticks AFTER setting limits to ensure they respect the range
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Force update to ensure limits are applied, then restore tick labels
                    cbar.update_normal(cbar.mappable)
                    # Re-apply tick labels after update (update_normal may reset them)
                    if len(all_ticks) > 0:
                        cbar.set_ticks(all_ticks)
                        cbar.set_ticklabels(tick_labels)
                    
                    # Position exponent above the top tick label
                    # For vertical colorbar, top tick is at y1, labels are to the right
                    # Align with the right edge where tick labels are, at the top
                    # Note: cax_pos is calculated per colorbar, so this works correctly for both rows
                    # (inverse problem row 0 and forward problem row 1)
                    exp_x = cax_pos.x1 + shift_x  # To the right of colorbar, aligned with tick labels
                    exp_y = cax_pos.y1 + shift_y  # Above the top of colorbar (relative to each row's colorbar)
                else:
                    # Fallback if no ticks
                    exp_x = cax_pos.x1 + shift_x
                    exp_y = cax_pos.y1 + shift_y
                
                exp_str = f"×10$^{{{exponent}}}$"
                # Add scientific notation text - this is applied to each colorbar individually
                # so it will appear on both rows (inverse and forward) when values trigger it
                fig.text(exp_x, exp_y, exp_str, ha='left', va='bottom', fontsize=9)
            else:
                # For normal values, use standard formatting
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 3))
                cbar.formatter = formatter
                cbar.update_ticks()
        
        return cbar

    def plot_pair(forward_data, inverse_data, outer_spec, pair_idx):
        # Create inner GridSpec for this pair (2 rows for forward/inverse)
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_spec, height_ratios=[1, 1], hspace=0.1)

        # Create GridSpec for each row
        gs_rows = []
        for row in range(2):
            if color_scale == "individual":
                # 6 columns: GT, cbar, FunDAPS, cbar, FunDAPS Error, cbar
                gs_row = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=inner_gs[row], 
                    width_ratios=[1, 0.08, 1, 0.08, 1, 0.08], wspace=0.3)
            else:
                # 5 columns: GT, FunDAPS, cbar, FunDAPS Error, cbar
                gs_row = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=inner_gs[row], 
                    width_ratios=[1, 1, 0.08, 1, 0.08], wspace=0.3)
            gs_rows.append(gs_row)

        def plot_row(data, row_idx, gs_row):
            # Get data - only use model1 (FunDAPS), ignore model2 (FunDPS) and model3 (DDIS)
            ground_truth = data["ground_truth"]
            model1_pred = data["model1"]  # FunDAPS

            # Calculate errors: |model_prediction - ground_truth|
            model1_error = np.abs(model1_pred - ground_truth)  # FunDAPS error vs true GT

            # Get min/max for consistent coloring based on color_scale method
            gt_min = np.min(ground_truth)
            gt_max = np.max(ground_truth)
            fundaps_min = np.min(model1_pred)
            fundaps_max = np.max(model1_pred)
            
            if color_scale == "gt":
                # Use GT scale for all columns
                vmin, vmax = gt_min, gt_max
            elif color_scale == "fundaps":
                # Use FunDAPS scale for all columns (FunDAPS as standard)
                vmin, vmax = fundaps_min, fundaps_max
            elif color_scale == "global":
                # Use global min/max across GT and FunDAPS (better for showing over-smoothing)
                vmin = min(gt_min, fundaps_min)
                vmax = max(gt_max, fundaps_max)
            elif color_scale == "self-def":
                # Use custom min/max values
                if custom_vmin is None or custom_vmax is None:
                    raise ValueError("custom_vmin and custom_vmax must be provided when color_scale='self-def'")
                vmin, vmax = custom_vmin, custom_vmax
            elif color_scale == "individual":
                # Each column will use its own scale (set per column below)
                vmin, vmax = None, None  # Will be set individually
            else:
                raise ValueError(f"Unknown color_scale: {color_scale}. Must be 'gt', 'fundaps', 'global', 'self-def', or 'individual'")
            
            # Print scale information
            if pair_idx == 0 and row_idx == 0:  # Only print once for the first pair, first row
                print(f"\nColor scale information ({color_scale} mode):")
                print(f"  Ground Truth:  min={gt_min:.6f}, max={gt_max:.6f}, range={gt_max-gt_min:.6f}")
                print(f"  FunDAPS:       min={fundaps_min:.6f}, max={fundaps_max:.6f}, range={fundaps_max-fundaps_min:.6f}")
                if color_scale == "self-def":
                    print(f"  Using custom scale: min={custom_vmin:.6f}, max={custom_vmax:.6f}, range={custom_vmax-custom_vmin:.6f}")
                elif color_scale in ["gt", "fundaps", "global"]:
                    print(f"  Applied scale:     min={vmin:.6f}, max={vmax:.6f}, range={vmax-vmin:.6f}")
            
            # Error scale
            err_max = np.max(model1_error)

            # Column positions depend on color_scale mode
            if color_scale == "individual":
                # 6 columns: GT, cbar, FunDAPS, cbar, FunDAPS Error, cbar
                plot_positions = [0, 2, 4]  # GT, FunDAPS, FunDAPS Error
                cbar_positions = [1, 3, 5]  # Colorbars for GT, FunDAPS, FunDAPS Error
            else:
                # 5 columns: GT, FunDAPS, cbar, FunDAPS Error, cbar
                plot_positions = [0, 1, 3]  # GT, FunDAPS, FunDAPS Error
                cbar_positions = [2, 4]  # Colorbars for FunDAPS pred, FunDAPS Error
            axes_dict = {}

            # Column 0: Ground Truth
            ax = fig.add_subplot(gs_row[plot_positions[0]])
            if color_scale == "individual":
                im0 = ax.imshow(ground_truth, cmap="viridis", norm=get_norm(gt_min, gt_max, gamma=gamma), interpolation="none")
                # Add colorbar for GT when using individual scale
                cax0 = fig.add_subplot(gs_row[cbar_positions[0]])
                add_colorbar_matched(im0, cax0, ax, vmin=gt_min, vmax=gt_max, shift_x=shift_x, shift_y=shift_y)
            else:
                # Use shared scale (gt, fundaps, global, or self-def)
                ax.imshow(ground_truth, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
            axes_dict[0] = ax

            # Column 1: FunDAPS with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[1]])
            if color_scale == "individual":
                im1 = ax.imshow(model1_pred, cmap="viridis", norm=get_norm(fundaps_min, fundaps_max, gamma=gamma), interpolation="none")
                # Add colorbar for FunDAPS when using individual scale
                cax1 = fig.add_subplot(gs_row[cbar_positions[1]])
                add_colorbar_matched(im1, cax1, ax, vmin=fundaps_min, vmax=fundaps_max, shift_x=shift_x, shift_y=shift_y)
            else:
                # Use shared scale (gt, fundaps, global, or self-def) - FunDAPS shows colorbar
                im1 = ax.imshow(model1_pred, cmap="viridis", norm=get_norm(vmin, vmax, gamma=gamma), interpolation="none")
                # Show colorbar for FunDAPS when using shared scales
                cax1 = fig.add_subplot(gs_row[cbar_positions[0]])
                add_colorbar_matched(im1, cax1, ax, vmin=vmin, vmax=vmax, shift_x=shift_x, shift_y=shift_y)
            axes_dict[1] = ax

            # Column 2: FunDAPS Error with colorbar
            ax = fig.add_subplot(gs_row[plot_positions[2]])
            im2 = ax.imshow(model1_error, cmap="viridis", norm=get_norm(0, err_max, gamma=gamma), interpolation="none")
            if color_scale == "individual":
                cax2 = fig.add_subplot(gs_row[cbar_positions[2]])  # Last colorbar position for individual mode
            else:
                cax2 = fig.add_subplot(gs_row[cbar_positions[1]])  # Last colorbar position for shared scale mode
            add_colorbar_matched(im2, cax2, ax, vmin=0, vmax=err_max, shift_x=shift_x, shift_y=shift_y)
            axes_dict[2] = ax

            # Remove ticks from all plots
            for ax in axes_dict.values():
                ax.set_xticks([])
                ax.set_yticks([])

            # Add row title to first plot
            row_label = f"{row_titles[row_idx]}"
            axes_dict[0].set_ylabel(row_label, fontsize=20)

            # Add titles only to first pair and first row (inverse problem)
            if pair_idx == 0 and row_idx == 0:
                for i, title in enumerate(titles):
                    axes_dict[i].set_title(title, pad=10, fontsize=20)
            
            # Add L2 error percentage below error column for both rows (inverse and forward problems)
            # Always calculate L2 error per channel (inverse vs forward) to ensure correctness
            # Calculate L2 error as percentage: ||pred - gt||_2 / ||gt||_2 * 100
            l2_error_model1 = np.linalg.norm(model1_pred - ground_truth) / np.linalg.norm(ground_truth) * 100
            
            # Only display for first pair to avoid clutter
            if pair_idx == 0:
                # Add text below error column (closer to the plot) with upward arrow
                axes_dict[2].text(0.5, -0.02, f'↑L2 Error: {l2_error_model1:.2f}%', 
                                 ha='center', va='top', transform=axes_dict[2].transAxes, fontsize=16)

        # Plot inverse and forward problems (matching row_titles order)
        plot_row(inverse_data, 0, gs_rows[0])  # Row 0: Inverse Problem
        plot_row(forward_data, 1, gs_rows[1])  # Row 1: Forward Problem

    # Plot all pairs in 2x2 grid
    for i, (forward_data, inverse_data) in enumerate(data_pairs):
        row = i // 2
        col = i % 2
        plot_pair(forward_data, inverse_data, outer_gs[row, col], i)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()

if __name__ == "__main__":
    import torch
    from datasets import load_from_disk
    from training.dataset_hf import PDEDataset
    import os
    import glob

    # ========== USER INPUTS ==========
    dataset_name = "poisson"  # Options: "poisson", "helmholtz", etc.
    exp_dir = "Z_test_plot_poisson"  # Experiment directory name (e.g., "Z_test_plot_poisson", "Z_test_plot_poisson_1pct")
    sample_indices = [2]  # Which samples from the batch to plot (default: [0, 1, 2, 3] for 2x2 grid, max 4)
                                    # The first index (default 0) is the main sample used
    file_suffix = {
        "ddis": None,    # Optional: specify suffix for DDIS files (e.g., "1706" for ddis_scarce500_1706.npy)
        "fundps": None,  # Optional: specify suffix for FunDPS files (e.g., "3674" for fundps_scarce500_3674.npy)
        "fundaps": None  # Optional: specify suffix for FunDAPS files (e.g., "XXXX" for fundaps_scarce500_XXXX.npy)
    }  # If None for any model, uses the first file alphabetically (default behavior)

    # Color scale option: "gt" (GT scale), "ddis" (DDIS scale), "global" (min/max across all), "individual" (each its own), "self-def" (custom)
    # "global" is recommended to better show over-smoothing in FunDPS
    # "ddis" uses DDIS as the standard (all columns use DDIS scale, only DDIS has colorbar)
    # "self-def" allows you to specify custom min/max values
    color_scale = "individual"  # Options: "gt", "ddis", "global", "self-def", "individual"
    custom_vmin = -0.65  # Custom minimum value (required when color_scale="self-def")
    custom_vmax = 0.74  # Custom maximum value (required when color_scale="self-def")
    # Example: color_scale = "self-def", custom_vmin = -1.0, custom_vmax = 1.0
    
    # Scientific notation position offsets (in figure coordinates)
    shift_x = 0.001  # Horizontal offset for scientific notation (default: 0.015, to the right of colorbar)
    shift_y = -0.165  # Vertical offset for scientific notation (default: 0.025, above the colorbar)
    
    # Color contrast enhancement (gamma correction)
    # gamma < 1.0: Enhances contrast in middle ranges (makes differences more visible)
    # gamma = 1.0: Linear normalization (default, no enhancement)
    # gamma > 1.0: Enhances contrast at extremes
    # Recommended: 0.5-0.7 for stronger distinction while keeping viridis colormap
    gamma = 1  # Default: 1.0 (no enhancement), try 0.5-0.7 for stronger contrast
    # ==================================
    
    # Derive all paths from inputs
    base_exp_path = f"exps/{exp_dir}"
    gt_dataset_path = f"data/DiffPDE/{dataset_name}_test_hf"
    
    # Auto-detect prediction files in the experiment directory
    ddis_files = sorted(glob.glob(f"{base_exp_path}/ddis*.npy"))  # Sort for consistent default
    fundps_files = sorted(glob.glob(f"{base_exp_path}/fundps*.npy"))
    fundaps_files = sorted(glob.glob(f"{base_exp_path}/fundaps*.npy"))
    
    if not ddis_files:
        raise FileNotFoundError(f"No DDIS prediction files found in {base_exp_path}/ddis*.npy")
    if not fundps_files:
        raise FileNotFoundError(f"No FunDPS prediction files found in {base_exp_path}/fundps*.npy")
    
    # Select files based on suffix if provided, otherwise use first file alphabetically
    def select_file(files, prefix, suffix):
        """Select file matching suffix pattern or return first file if suffix is None.
        
        Pattern: {prefix}_scarce500_{suffix}.npy (e.g., fundps_scarce500_3674.npy)
        If suffix is None, uses first file alphabetically (default).
        """
        if suffix is None:
            # Default: use first file alphabetically (files without suffix come first)
            return files[0]
        else:
            # Look for file matching pattern: *scarce500_{suffix}.npy
            # This matches files like: ddis_scarce500_3674.npy, fundps_scarce500_3674.npy
            matching = [f for f in files if f"scarce500_{suffix}.npy" in f]
            if not matching:
                available = [os.path.basename(f) for f in files]
                raise FileNotFoundError(
                    f"No {prefix} file matching pattern '*scarce500_{suffix}.npy' found. "
                    f"Available files: {available}"
                )
            return matching[0]
    
    ddis_pred_path = select_file(ddis_files, "ddis", file_suffix.get("ddis"))
    fundps_pred_path = select_file(fundps_files, "fundps", file_suffix.get("fundps"))
    fundaps_pred_path = select_file(fundaps_files, "fundaps", file_suffix.get("fundaps")) if fundaps_files else None
    
    print(f"Loading predictions from {base_exp_path}:")
    print(f"  File selection:")
    print(f"    DDIS suffix: {file_suffix.get('ddis') or 'default (first alphabetically)'}")
    print(f"    FunDPS suffix: {file_suffix.get('fundps') or 'default (first alphabetically)'}")
    print(f"    FunDAPS suffix: {file_suffix.get('fundaps') or 'default (first alphabetically)'}")
    print(f"  Selected files:")
    print(f"    DDIS: {os.path.basename(ddis_pred_path)}")
    print(f"    FunDPS: {os.path.basename(fundps_pred_path)}")
    if fundaps_pred_path:
        print(f"    FunDAPS: {os.path.basename(fundaps_pred_path)}")
    else:
        print(f"    FunDAPS: Not found (will use dummy data for create_comparison_plots_no_fundaps)")
    
    # Load model predictions
    ddis_pred = np.load(ddis_pred_path)  # Shape: (N, 2, 128, 128)
    fundps_pred = np.load(fundps_pred_path)  # Shape: (N, 2, 128, 128)
    
    # Load FunDAPS predictions if available (optional - only needed for create_error_only_plots and create_multiple_comparison_plots)
    fundaps_pred = None
    if fundaps_pred_path:
        fundaps_pred = np.load(fundaps_pred_path)  # Shape: (N, 2, 128, 128)
    
    # Load ground truth data and create normalizer for denormalization
    gt_dataset_raw = load_from_disk(gt_dataset_path)
    dataset_obj = PDEDataset(path=gt_dataset_path, max_size=len(gt_dataset_raw))
    normalizer = dataset_obj.create_normalizer()    

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
    
    # Validate and limit sample_indices
    sample_indices = [idx for idx in sample_indices if idx < batch_size]  # Filter out invalid indices
    if not sample_indices:
        raise ValueError(f"All sample_indices are out of range. Batch size is {batch_size}.")
    sample_indices = sample_indices[:4]  # Limit to max 4 samples for 2x2 grid
    
    print(f"Plotting samples from batch: {sample_indices}")
    print(f"Main sample (first): {sample_indices[0]}")
    
    for i in sample_indices:  # Plot selected samples (up to 4 in 2x2 grid)
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
        fundps_inverse = fundps_pred[i, 0]
        fundps_forward = fundps_pred[i, 1]
        
        # Extract FunDAPS data if available, otherwise use dummy data (only for create_comparison_plots_no_fundaps)
        if fundaps_pred is not None:
            fundaps_inverse = fundaps_pred[i, 0]
            fundaps_forward = fundaps_pred[i, 1]
        else:
            fundaps_inverse = np.zeros_like(gt_inverse)  # Dummy data, not used
            fundaps_forward = np.zeros_like(gt_forward)  # Dummy data, not used
        
        # Create forward problem data with clear naming (swapped: model1=FunDAPS, model2=FunDPS)
        forward_data = {
            "ground_truth": gt_forward,     # True ground truth from dataset
            "model1": fundaps_forward,      # FunDAPS predictions (model1) - dummy, not used
            "model2": fundps_forward,       # FunDPS predictions (model2)
            "model3": ddis_forward,         # DDIS predictions (ours, model3)
            "mask": np.ones_like(gt_forward)  # Full mask (all observed)
        }
        
        # Create inverse problem data with clear naming (swapped: model1=FunDAPS, model2=FunDPS)
        inverse_data = {
            "ground_truth": gt_inverse,     # True ground truth from dataset
            "model1": fundaps_inverse,      # FunDAPS predictions (model1) - dummy, not used
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
    
    # Derive save paths
    save_path = f"{base_exp_path}/comparison_plots.png"
    save_path_errors_only = f"{base_exp_path}/comparison_plots_errors_only.png"
    
    # Generate plots
    if fundaps_pred is not None:
        create_multiple_comparison_plots(data_pairs, save_path=save_path, l2_errors=l2_errors, shift_x=shift_x, shift_y=shift_y, gamma=gamma)
        create_error_only_plots(data_pairs, save_path=save_path_errors_only, l2_errors=l2_errors, shift_x=shift_x, shift_y=shift_y, gamma=gamma)
    else:
        print("Skipping create_multiple_comparison_plots() and create_error_only_plots() - fundaps_pred not available")
    
    # create_comparison_plots_no_fundaps(data_pairs, save_path=save_path, l2_errors=l2_errors, 
    #                                    color_scale=color_scale, custom_vmin=custom_vmin, custom_vmax=custom_vmax,
    #                                    shift_x=shift_x, shift_y=shift_y, gamma=gamma)
    
    create_comparison_plots_fundaps_only(data_pairs, save_path=save_path, l2_errors=l2_errors, 
                                    color_scale=color_scale, custom_vmin=custom_vmin, custom_vmax=custom_vmax,
                                    shift_x=shift_x, shift_y=shift_y, gamma=gamma)
