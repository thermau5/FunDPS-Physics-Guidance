"""
Quantitative evaluation of PINO model for 0.5-second prediction (what it was trained for).
Tests the model and saves visualizations to temp_save.
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neuralop
from neuralop import get_model
from neuralop.training import setup
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

# Import utilities from station.py
import My_TOOL as myt
from data0.positional_encoding import get_grid_positional_encoding
import math as mt

# Configuration
config_name = "default"
pipe = ConfigPipeline([
    YamlConfig("./ns1w_plot0.yaml", config_name="default", config_folder="../config"),
    ArgparseConfig(infer_types=True, config_name=None, config_file=None),
    YamlConfig(config_folder="../config"),
])
config = pipe.read_conf()

# Setup
device, is_logger = setup(config)
arch = config["arch"].lower()
config_arch = config.get(arch)

# PDE setting
pde_info = {}
pde_kf = {
    'domain': [[0, 2*mt.pi], [0, 2*mt.pi]],
    'pde_dim': 2,
    'pde_dim_pino': 3,
    'function_dim': 1,
    'L': [2*mt.pi, 2*mt.pi]
}
pde_info['kf'] = pde_kf
pde_case = pde_info[config.wandb.pde]

# Configure for PINO (must be before using config_arch.data_channels)
pino_t_tag = 0
# config_arch already has data_channels from config file
# Override if needed for PINO
if hasattr(config, 'wandb') and hasattr(config.wandb, 'pino') and config.wandb.pino:
    setattr(config_arch, 'data_channels', pde_case['pde_dim_pino'] + pde_case['function_dim'])
    setattr(config_arch, 'domain', [[0, config.data.t_predict]] + pde_case['domain'])
    if pde_case['pde_dim_pino'] != pde_case['pde_dim']:
        pino_t_tag = 1
else:
    setattr(config_arch, 'data_channels', pde_case['pde_dim'] + pde_case['function_dim'])
    setattr(config_arch, 'domain', pde_case['domain'])

# Load dataset
rel_path = '/home/thomaslin/FunDPS-Physics-dev-v2/temp_0-main/dataset_and_model/'
dataset = {
    'link': rel_path + "re1000_grid=256_1_N=40_dt=4.0_Ttj=200-320(stat).pt",
    'N': 40,
    'dtsave': 1 / 4,
    'T': 320,
    'res': 256
}
vorticity = torch.load(dataset['link'])

# Load model
config['model'] = config_arch
model = get_model(config)
model = model.to(device)

model_path = rel_path + 'model_[8, 48, 48]_28_32_8064(ep31)_cvt_2.pt'
cpt = torch.load(model_path, map_location=device)
model.load_state_dict(cpt["model"])
del cpt
model.eval()

# Evaluation parameters - MODEL WAS TRAINED FOR 0.5 SECONDS
PREDICTION_TIME = 0.5  # What the model was trained for!
PREDICTION_TIMESTEPS = int(PREDICTION_TIME / dataset['dtsave'])  # 2 timesteps

print("=" * 70)
print("QUANTITATIVE EVALUATION: 0.5-Second Prediction")
print("=" * 70)
print(f"Model was trained for: {PREDICTION_TIME} seconds")
print(f"Testing on: {PREDICTION_TIME} seconds ({PREDICTION_TIMESTEPS} timesteps)")
print()

# Test on multiple samples
dsp = config.data.dsp
results = []

# Sample different trajectories and time points
test_samples = []
# Test on all trajectories and various time points
for traj_id in range(0, vorticity.shape[0], 2):  # Every other trajectory
    # Sample different time regions
    for time_id in [50, 100, 150, 200, 250, 300, 350]:
        if time_id + PREDICTION_TIMESTEPS < vorticity.shape[1]:
            test_samples.append((traj_id, time_id))

print(f"Testing on {len(test_samples)} samples")

errors = []
correlations = []
relative_errors = []

# Create output directory
os.makedirs('temp_save', exist_ok=True)

for idx, (traj_id, time_id) in enumerate(test_samples):
    # Extract input and target
    x = vorticity[traj_id:traj_id+1, time_id:time_id+1, ::dsp, ::dsp].unsqueeze(dim=1)
    y_true = vorticity[traj_id:traj_id+1, time_id+PREDICTION_TIMESTEPS:time_id+PREDICTION_TIMESTEPS+1, ::dsp, ::dsp].unsqueeze(dim=1)
    
    # Prepare input (same as station.py)
    x_input = x.clone()
    x_input = x_input.repeat(1, 1, config.data.repeat_ini, 1, 1).float()
    
    # Get positional encoding
    # Use data_channels - try different ways to access it
    try:
        data_channels = getattr(config_arch, 'data_channels', None)
        if data_channels is None:
            data_channels = config_arch.get('data_channels', 4)
    except:
        data_channels = 4  # Default from config file
    dim_pde = data_channels - pde_case['function_dim']
    gridd = get_grid_positional_encoding(
        x_input[0],
        grid_boundaries=config_arch.domain,
        dim_pde=dim_pde,
        channel_dim=0
    )
    
    # Prepare model input
    ss = x_input.shape
    gridd_tem = [g.repeat(ss[0], 1, 1, 1, 1) for g in gridd]
    x_model = torch.cat([x_input] + gridd_tem, dim=1)
    
    # Run model
    with torch.no_grad():
        x_model = x_model.to(device)
        output = model(x_model)
        
        # Extract final timestep from output
        if len(output.shape) == 5:  # [B, C, T, X, Y]
            y_pred = output[..., -1:, :, :]  # Take last timestep
        else:  # [B, C, X, Y]
            y_pred = output.unsqueeze(dim=2)  # Add time dimension
    
    # Compute metrics
    x_np = x[0, 0, 0].cpu().numpy()
    y_true_np = y_true[0, 0, 0].cpu().numpy()
    y_pred_np = y_pred[0, 0, 0].cpu().numpy()
    
    # L2 relative error
    l2_error = np.linalg.norm(y_pred_np - y_true_np) / np.linalg.norm(y_true_np)
    relative_errors.append(l2_error)
    
    # Correlation
    corr = np.corrcoef(y_pred_np.flatten(), y_true_np.flatten())[0, 1]
    correlations.append(corr)
    
    # Absolute error
    abs_error = np.mean(np.abs(y_pred_np - y_true_np))
    errors.append(abs_error)
    
    # Save visualization for first few samples
    if idx < 5:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        vmin = min(y_true_np.min(), y_pred_np.min())
        vmax = max(y_true_np.max(), y_pred_np.max())
        vrange = max(abs(vmin), abs(vmax))
        vmin, vmax = -vrange, vrange
        
        # Input
        im1 = axes[0].pcolormesh(x_np, cmap='bwr', shading='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Input (t={time_id})', fontsize=14)
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('y', fontsize=12)
        plt.colorbar(im1, ax=axes[0])
        
        # Prediction
        im2 = axes[1].pcolormesh(y_pred_np, cmap='bwr', shading='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Prediction\nL2={l2_error:.3f}, Corr={corr:.3f}', fontsize=14)
        axes[1].set_xlabel('x', fontsize=12)
        axes[1].set_ylabel('y', fontsize=12)
        plt.colorbar(im2, ax=axes[1])
        
        # Truth
        im3 = axes[2].pcolormesh(y_true_np, cmap='bwr', shading='auto', vmin=vmin, vmax=vmax)
        axes[2].set_title(f'Truth (t={time_id+PREDICTION_TIMESTEPS})', fontsize=14)
        axes[2].set_xlabel('x', fontsize=12)
        axes[2].set_ylabel('y', fontsize=12)
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f'temp_save/eval_sample_{idx}_traj{traj_id}_t{time_id}.png', dpi=150, bbox_inches='tight')
        plt.close()

print()
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Tested on: {len(errors)} samples")
print()
print("Metrics:")
print(f"  L2 Relative Error:  {np.mean(relative_errors):.4f} ± {np.std(relative_errors):.4f}")
print(f"  Correlation:        {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
print(f"  Mean Absolute Error: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
print()
print(f"  Best L2 Error:      {np.min(relative_errors):.4f}")
print(f"  Worst L2 Error:     {np.max(relative_errors):.4f}")
print(f"  Best Correlation:   {np.max(correlations):.4f}")
print(f"  Worst Correlation:  {np.min(correlations):.4f}")
print()
print("Visualizations saved to: temp_save/")
print("=" * 70)

