#!/usr/bin/env python3
"""
Test PINO surrogate model on HuggingFace dataset format.
Equivalent to test_fno_surrogate.py but adapted for 3D PINO model.

Usage:
    python test_pino_surrogate.py \
        --model-path artifacts/models/legacy/pino_trained_forward_ns-temp0main_256.pth \
        --data-path data/DiffPDE/ns-temp0main_test_hf \
        --output-dir pino_predictions \
        --save-examples 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from neuralop.models.fno import FNO
from training.dataset_hf import PDEDataset
from training.dataset_utils import DatasetNormalizer

# Add path for positional encoding
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_scripts_path = os.path.join(script_dir, '..', 'archive', 'temp_0-main', 'scripts')
sys.path.insert(0, temp_scripts_path)
from data0.positional_encoding import get_grid_positional_encoding
import math

# Import config machinery
from configmypy import ConfigPipeline, YamlConfig
from neuralop import get_model


# ----------------------------
# Helper functions
# ----------------------------
def save_prediction_comparison(inp, gt, pred, save_dir, sample_idx, l2_error, l1_error):
    """Save enhanced prediction comparison with metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())
    vrange = max(abs(vmin), abs(vmax))
    vmin, vmax = -vrange, vrange
    
    titles = [
        'Input (t₀)',
        f'Ground Truth (t₀+0.5s)',
        f'Prediction (t₀+0.5s)\nL2={l2_error:.4f}, L1={l1_error:.4f}'
    ]
    
    for ax, img, title in zip(axes, [inp, gt, pred], titles):
        im = ax.pcolormesh(img, cmap='bwr', shading='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    comparison_filename = f"comparison_sample_{sample_idx:03d}.png"
    plt.savefig(os.path.join(save_dir, comparison_filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_filename}")


# ----------------------------
# Argument parsing
# ----------------------------
parser = argparse.ArgumentParser(description="Test PINO surrogate model on HuggingFace dataset")
parser.add_argument(
    "--model-path", "-m", required=True,
    help=".pth file of trained PINO model")
parser.add_argument(
    "--data-path", "-d", default="data/DiffPDE/ns-temp0main_test_hf",
    help="Path to test dataset")
parser.add_argument(
    "--batch-size", "-b", type=int, default=1,
    help="Batch size for DataLoader")
parser.add_argument(
    "--output-dir", "-o", default="pino_predictions",
    help="Directory to save prediction visualizations")
parser.add_argument(
    "--save-examples", "-s", type=int, default=10,
    help="Number of prediction examples to save as images")
# Compute default config path dynamically
default_config_path = os.path.join(script_dir, '..', 'archive', 'temp_0-main', 'config', 'ns1w_plot0.yaml')
parser.add_argument(
    "--config-path", "-c", default=default_config_path,
    help="Path to PINO config file")
args = parser.parse_args()


# ----------------------------
# Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load config
print(f"Loading config...")
original_cwd = os.getcwd()
os.chdir(temp_scripts_path)
try:
    config_name = "default"
    pipe = ConfigPipeline([
        YamlConfig("./ns1w_plot0.yaml", config_name="default", config_folder="../config"),
        YamlConfig(config_folder="../config"),
    ])
    config = pipe.read_conf()
finally:
    os.chdir(original_cwd)

arch = config["arch"].lower()
config_arch = config.get(arch)

# PDE setting
pde_info = {
    'kf': {
        'domain': [[0, 2*math.pi], [0, 2*math.pi]],
        'pde_dim': 2,
        'pde_dim_pino': 3,
        'function_dim': 1,
        'L': [2*math.pi, 2*math.pi]
    }
}
pde_case = pde_info[config.wandb.pde]

# Configure for PINO
pino_t_tag = 0
if hasattr(config, 'wandb') and hasattr(config.wandb, 'pino') and config.wandb.pino:
    config_arch.data_channels = pde_case['pde_dim_pino'] + pde_case['function_dim']
    config_arch.domain = [[0, config.data.t_predict]] + pde_case['domain']
    if pde_case['pde_dim_pino'] != pde_case['pde_dim']:
        pino_t_tag = 1
else:
    config_arch.data_channels = pde_case['pde_dim'] + pde_case['function_dim']
    config_arch.domain = pde_case['domain']

# Store these BEFORE get_model (which pops data_channels)
data_channels_value = config_arch.data_channels
domain_value = config_arch.domain

# Build model
print("Building PINO model...")
config['model'] = config_arch
model = get_model(config)
model = model.to(device)

# Load weights
print(f"Loading model from {args.model_path}")
model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
model.eval()
print("Model loaded successfully")


# ----------------------------
# Dataset & loader
# ----------------------------
print(f"Loading dataset from {args.data_path}")
dataset = PDEDataset(path=args.data_path, resolution=256)
normalizer = dataset.create_normalizer()
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
print(f"Loaded {len(dataset)} samples")

# Create output directory
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


# ----------------------------
# Evaluation loop
# ----------------------------
all_mse = []
all_l2_rel = []
all_l1_error = []
max_err = 0.0
examples_saved = 0

print("\nStarting evaluation...")
with torch.no_grad():
    for i, (data, _) in enumerate(loader):
        # Split input and target
        inp_hf = data[:, 0:1]  # Input vorticity: [batch, 1, 256, 256] (NORMALIZED)
        targ_hf = data[:, 1:2]  # Target vorticity: [batch, 1, 256, 256] (NORMALIZED)
        
        # CRITICAL: Denormalize to raw values (model expects raw)
        inp_raw = normalizer.denormalize(inp_hf.float(), channel=0)
        targ_raw = normalizer.denormalize(targ_hf.float(), channel=1)
        
        # Prepare PINO input: [batch, 4, 32, 256, 256]
        inp_hf_5d = inp_raw.unsqueeze(dim=2).float()  # Add temporal dim
        inp_repeated = inp_hf_5d.repeat(1, 1, config.data.repeat_ini, 1, 1).float()  # Repeat 32×
        
        # Get positional encoding
        data_channels = data_channels_value
        dim_pde = data_channels - pde_case['function_dim']
        domain = domain_value
        gridd = get_grid_positional_encoding(inp_repeated[0], grid_boundaries=domain, 
                                            dim_pde=dim_pde, channel_dim=0)
        
        # Concatenate with positional encoding
        gridd_tem = [g.repeat(inp_repeated.shape[0], 1, 1, 1, 1) for g in gridd]
        inp_pino = torch.cat([inp_repeated] + gridd_tem, dim=1)
        
        # Run model
        inp_pino = inp_pino.to(device)
        pred_raw = model(inp_pino)
        
        # Extract final timestep
        if len(pred_raw.shape) == 5:
            pred = pred_raw[..., -1:, :, :].squeeze(dim=2)
        else:
            pred = pred_raw
        
        # Compute per-sample metrics
        batch_size = pred.shape[0]
        for j in range(batch_size):
            pred_j = pred[j:j+1]
            targ_j = targ_raw[j:j+1].to(device)
            
            # MSE
            mse = F.mse_loss(pred_j, targ_j).item()
            
            # L2 relative error
            # NOTE: For near-zero target fields, L2 relative error becomes ill-defined
            # We compute it anyway to match evaluate_model_0.5sec.py behavior
            norm_targ = torch.norm(targ_j)
            l2_rel = (torch.norm(pred_j - targ_j) / norm_targ).item()
            
            # L1 error
            l1_error = F.l1_loss(pred_j, targ_j).item()
            
            # Max error
            max_err_sample = torch.max(torch.abs(pred_j - targ_j)).item()
            
            all_mse.append(mse)
            all_l2_rel.append(l2_rel)
            all_l1_error.append(l1_error)
            max_err = max(max_err, max_err_sample)
            
            # Save visualization for first few samples
            if examples_saved < args.save_examples:
                print(f"Saving visualization for sample {examples_saved + 1}...")
                inp_np = inp_raw[j, 0].cpu().numpy()
                targ_np = targ_raw[j, 0].cpu().numpy()
                pred_np = pred_j[0, 0].cpu().numpy()
                
                save_prediction_comparison(inp_np, targ_np, pred_np, 
                                          output_dir, examples_saved + 1, 
                                          l2_rel, l1_error)
                examples_saved += 1
        
        # Progress report
        if (i + 1) % 100 == 0:
            recent_l2 = np.array(all_l2_rel[-100:])
            recent_mse = np.array(all_mse[-100:])
            valid_l2 = recent_l2[np.isfinite(recent_l2)]
            print(f"[Batch {i+1:3d}/{len(loader)}] MSE={np.mean(recent_mse):.3e} L2_rel={np.mean(valid_l2):.3e}")


# ----------------------------
# Print summary
# ----------------------------
print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

# Check for any inf/nan values
all_l2_array = np.array(all_l2_rel)
finite_l2 = all_l2_array[np.isfinite(all_l2_array)]
has_nan_inf = len(finite_l2) < len(all_l2_array)

print(f"Test samples:    {len(all_mse)}")

print(f"\nMetrics:")
print(f"  Avg MSE:       {np.mean(all_mse):.6e}")
print(f"  Avg L2 rel err: {np.mean(all_l2_array):.4f} ± {np.std(all_l2_array):.4f}")
print(f"  Avg L1 err:    {np.mean(all_l1_error):.4f}")
print(f"  Max err:       {max_err:.4f}")

print(f"\nRange:")
print(f"  Best L2:       {np.min(all_l2_array):.4f}")
print(f"  Worst L2:      {np.max(all_l2_array):.4f}")

if has_nan_inf:
    print(f"\n  WARNING: {len(all_l2_array) - len(finite_l2)} samples had inf/nan L2 errors")
    print(f"    (likely due to near-zero target fields)")

print("="*70)
print(f"\nVisualizations saved to: {output_dir}/")
