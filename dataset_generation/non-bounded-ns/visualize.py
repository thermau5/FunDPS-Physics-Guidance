import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Visualize Navier-Stokes PDE evolution')
parser.add_argument('--sample', type=int, default=0, 
                    help='Sample index to visualize (0-9, default: 0)')
parser.add_argument('--output', type=str, default='pde_evolution.gif',
                    help='Output GIF filename (default: pde_evolution.gif)')
args = parser.parse_args()

# Load the visualization mat file (generated with SAVE_ALL_TIMESTEPS_FOR_VIS=True)
mat_file_path = '/home/thomaslin/FunDPS-Physics-dev-v2/dataset_generation/non-bounded-ns/ns-nonbounded-vis.mat'
mat_file = scipy.io.loadmat(mat_file_path)

u = mat_file['u']
t = mat_file['t']

print(f"u.shape: {u.shape}")
print(f"t.shape: {t.shape}")

# Verify we have the full timestep data
if u.shape[-1] != 50:
    raise ValueError(f"Expected 50 timesteps in last dimension, but got {u.shape[-1]}. "
                     f"Please generate visualization data by running ns_2d_test.py")

# u has shape (N, 128, 128, 50) - N samples, 128x128 grid, 50 timesteps
n_samples = u.shape[0]
sample_idx = args.sample

# Validate sample index
if sample_idx < 0 or sample_idx >= n_samples:
    raise ValueError(f"Sample index must be between 0 and {n_samples-1}, got {sample_idx}")

n_frames = u.shape[-1]  # 50 timesteps

print(f"\nVisualizing sample {sample_idx} with {n_frames} timesteps")
print(f"Time range: {t[0, 0]:.3f} to {t[0, -1]:.3f}")

frames = []
temp_image = "temp_frame.png"

# Get the initial condition (frame 0) and frame 25
init_frame = u[sample_idx, :, :, 0]
frame_25 = u[sample_idx, :, :, 24]  # 25th frame (index 24)
time_25 = t[0, 24]

# Create animation showing initial condition, frame 25, and evolution
for i in range(n_frames):
    current_frame = u[sample_idx, :, :, i]
    current_time = t[0, i]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: Initial condition
    axes[0].imshow(init_frame, cmap='jet', vmin=u[sample_idx].min(), vmax=u[sample_idx].max())
    axes[0].set_title('Initial Condition (t=0)', fontsize=14)
    axes[0].axis('off')

    # Middle: Frame 25
    axes[1].imshow(frame_25, cmap='jet', vmin=u[sample_idx].min(), vmax=u[sample_idx].max())
    axes[1].set_title(f'Frame 25 (t={time_25:.3f})', fontsize=14)
    axes[1].axis('off')

    # Right: Current timestep
    im = axes[2].imshow(current_frame, cmap='jet', vmin=u[sample_idx].min(), vmax=u[sample_idx].max())
    axes[2].set_title(f'PDE Evolution (Frame {i+1}/{n_frames}, t={current_time:.3f})', fontsize=14)
    axes[2].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(temp_image, dpi=100)
    plt.close()
    frames.append(imageio.imread(temp_image))

# Save the animation
output_filename = args.output
imageio.mimsave(output_filename, frames, duration=0.1)
os.remove(temp_image)

print(f"\n✓ Animation saved as: {output_filename}")
print(f"  Sample index: {sample_idx}")
print(f"  Total frames: {len(frames)}")
print(f"  Duration: {len(frames) * 0.1:.1f}s")