"""
Analyze dataset selection strategies for 5-second prediction.
Determines which trajectories and time regions to use for training/test.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Load dataset
data = torch.load('../dataset_and_model/re1000_grid=256_1_N=40_dt=4.0_Ttj=200-320(stat).pt')
data = data.numpy()

print("=" * 70)
print("DATASET SELECTION ANALYSIS")
print("=" * 70)
print()

# Parameters
N_TRAJECTORIES = 40
N_TIMESTEPS = 481
PREDICTION_TIMESTEPS = 20  # 5 seconds
dtsave = 0.25

print(f"Dataset shape: {data.shape}")
print(f"  Trajectories: {N_TRAJECTORIES}")
print(f"  Timesteps per trajectory: {N_TIMESTEPS}")
print(f"  Prediction gap: {PREDICTION_TIMESTEPS} timesteps (5 seconds)")
print()

# Strategy 1: Analyze time regions
print("=" * 70)
print("STRATEGY 1: Analyze Time Regions")
print("=" * 70)
print()

# Divide timeline into regions
time_regions = [
    (0, 100, "Early (t=200-225)"),
    (100, 200, "Middle-Early (t=225-250)"),
    (200, 300, "Middle (t=250-275)"),
    (300, 400, "Middle-Late (t=275-300)"),
    (400, 481, "Late (t=300-320)")
]

region_stats = {}

for start, end, label in time_regions:
    correlations = []
    variances = []
    
    for traj_idx in range(N_TRAJECTORIES):
        for t in range(start, end - PREDICTION_TIMESTEPS, 20):
            if t + PREDICTION_TIMESTEPS >= end:
                continue
                
            input_field = data[traj_idx, t]
            output_field = data[traj_idx, t + PREDICTION_TIMESTEPS]
            
            corr = np.corrcoef(input_field.flatten(), output_field.flatten())[0, 1]
            var = np.var(output_field)
            
            correlations.append(corr)
            variances.append(var)
    
    region_stats[label] = {
        'mean_corr': np.mean(correlations),
        'std_corr': np.std(correlations),
        'mean_var': np.mean(variances),
        'n_samples': len(correlations)
    }
    
    print(f"{label:25s}: Corr={np.mean(correlations):.3f}±{np.std(correlations):.3f}, "
          f"Var={np.mean(variances):.2f}, N={len(correlations)}")

print()

# Strategy 2: Analyze trajectory diversity
print("=" * 70)
print("STRATEGY 2: Analyze Trajectory Diversity")
print("=" * 70)
print()

# Compute statistics per trajectory
trajectory_stats = []
for traj_idx in range(N_TRAJECTORIES):
    trajectory = data[traj_idx]
    
    # Mean and variance across time
    mean_field = trajectory.mean(axis=(1, 2))
    var_field = trajectory.var(axis=(1, 2))
    
    # Correlation between consecutive timesteps
    correlations = []
    for t in range(len(trajectory) - PREDICTION_TIMESTEPS):
        input_field = trajectory[t]
        output_field = trajectory[t + PREDICTION_TIMESTEPS]
        corr = np.corrcoef(input_field.flatten(), output_field.flatten())[0, 1]
        correlations.append(corr)
    
    trajectory_stats.append({
        'traj_idx': traj_idx,
        'mean_field': np.mean(mean_field),
        'std_field': np.std(mean_field),
        'mean_var': np.mean(var_field),
        'mean_corr': np.mean(correlations),
        'std_corr': np.std(correlations)
    })

# Sort by correlation (easier to predict)
trajectory_stats_sorted = sorted(trajectory_stats, key=lambda x: x['mean_corr'], reverse=True)

print("Top 10 trajectories (highest correlation - easier to predict):")
for i, stat in enumerate(trajectory_stats_sorted[:10]):
    print(f"  Traj {stat['traj_idx']:2d}: Corr={stat['mean_corr']:.3f}±{stat['std_corr']:.3f}, "
          f"Mean={stat['mean_field']:6.2f}±{stat['std_field']:.2f}")

print()
print("Bottom 10 trajectories (lowest correlation - harder to predict):")
for i, stat in enumerate(trajectory_stats_sorted[-10:]):
    print(f"  Traj {stat['traj_idx']:2d}: Corr={stat['mean_corr']:.3f}±{stat['std_corr']:.3f}, "
          f"Mean={stat['mean_field']:6.2f}±{stat['std_field']:.2f}")

# Strategy 3: Recommended train/test split
print()
print("=" * 70)
print("STRATEGY 3: Recommended Dataset Selection")
print("=" * 70)
print()

# Option A: Time-based split
print("Option A: Time-based split (use different time regions)")
print("-" * 70)
print("Training: Use timesteps 50-300 (middle region)")
print("  • Stable dynamics")
print("  • High correlation")
print(f"  • ~{len(range(50, 300 - PREDICTION_TIMESTEPS, 1)) * N_TRAJECTORIES} pairs available")
print()
print("Testing: Use timesteps 100-200 and 350-400 (diverse regions)")
print("  • Different time regimes")
print(f"  • ~{len(range(100, 200 - PREDICTION_TIMESTEPS, 1)) * N_TRAJECTORIES} + "
      f"{len(range(350, 400 - PREDICTION_TIMESTEPS, 1)) * N_TRAJECTORIES} pairs")

# Option B: Trajectory-based split
print()
print("Option B: Trajectory-based split (use different trajectories)")
print("-" * 70)
train_trajs = [i for i in range(0, 32)]  # 80% for training
test_trajs = [i for i in range(32, 40)]  # 20% for testing
print(f"Training: Trajectories {train_trajs[0]}-{train_trajs[-1]} ({len(train_trajs)} trajectories)")
print(f"  • All timesteps 50-430 (avoid boundary)")
print(f"  • ~{len(range(50, 430 - PREDICTION_TIMESTEPS, 1)) * len(train_trajs)} pairs available")
print()
print(f"Testing: Trajectories {test_trajs[0]}-{test_trajs[-1]} ({len(test_trajs)} trajectories)")
print(f"  • All timesteps 50-430")
print(f"  • ~{len(range(50, 430 - PREDICTION_TIMESTEPS, 1)) * len(test_trajs)} pairs available")

# Option C: Mixed strategy (recommended)
print()
print("Option C: Mixed strategy (RECOMMENDED)")
print("-" * 70)
print("Training:")
print("  • 32 trajectories (0-31)")
print("  • Time regions: 50-300, 350-400 (skip boundary)")
print("  • Avoid very early (0-50) and very late (400+) timesteps")
train_pairs = 0
for traj in range(32):
    # Region 1: 50-300
    for t in range(50, 300 - PREDICTION_TIMESTEPS, 1):
        train_pairs += 1
    # Region 2: 350-400
    for t in range(350, min(400, data.shape[1] - PREDICTION_TIMESTEPS), 1):
        train_pairs += 1
print(f"  • Total: ~{train_pairs} pairs")
print()
print("Testing:")
print("  • 8 trajectories (32-39)")
print("  • Time regions: 50-300, 350-400")
test_pairs = 0
for traj in range(32, 40):
    for t in range(50, 300 - PREDICTION_TIMESTEPS, 1):
        test_pairs += 1
    for t in range(350, min(400, data.shape[1] - PREDICTION_TIMESTEPS), 1):
        test_pairs += 1
print(f"  • Total: ~{test_pairs} pairs")

# Option D: Sample uniformly
print()
print("Option D: Uniform sampling (simple)")
print("-" * 70)
print("Training:")
print("  • 32 trajectories (0-31)")
print("  • Sample every 2nd timestep from 50-430")
print("  • Avoid boundaries")
uniform_train = 0
for traj in range(32):
    for t in range(50, min(430, data.shape[1] - PREDICTION_TIMESTEPS), 2):
        uniform_train += 1
print(f"  • Total: ~{uniform_train} pairs")
print()
print("Testing:")
print("  • 8 trajectories (32-39)")
print("  • Sample every 2nd timestep from 50-430")
uniform_test = 0
for traj in range(32, 40):
    for t in range(50, min(430, data.shape[1] - PREDICTION_TIMESTEPS), 2):
        uniform_test += 1
print(f"  • Total: ~{uniform_test} pairs")

# Summary visualization
print()
print("=" * 70)
print("RECOMMENDATION SUMMARY")
print("=" * 70)
print()
print("For 5-second prediction dataset:")
print("✅ Use Option C (Mixed strategy) or Option D (Uniform sampling)")
print()
print("Key considerations:")
print("  1. Avoid very early (t<50) and very late (t>430) timesteps")
print("  2. Use 80/20 train/test split by trajectory")
print("  3. Sample uniformly or from stable time regions")
print("  4. Ensure train/test have similar statistics")
print()

# Save visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Correlation by time region
region_names = [r[2] for r in time_regions]
corrs = [region_stats[r]['mean_corr'] for r in region_names]
ax = axes[0, 0]
ax.bar(range(len(region_names)), corrs)
ax.set_xticks(range(len(region_names)))
ax.set_xticklabels(region_names, rotation=45, ha='right')
ax.set_ylabel('Mean Correlation')
ax.set_title('Prediction Difficulty by Time Region')
ax.grid(True, alpha=0.3)

# Plot 2: Correlation by trajectory
traj_indices = [s['traj_idx'] for s in trajectory_stats_sorted]
traj_corrs = [s['mean_corr'] for s in trajectory_stats_sorted]
ax = axes[0, 1]
ax.plot(traj_indices, traj_corrs, 'o-')
ax.set_xlabel('Trajectory Index')
ax.set_ylabel('Mean Correlation')
ax.set_title('Prediction Difficulty by Trajectory')
ax.grid(True, alpha=0.3)

# Plot 3: Recommended train/test split
ax = axes[1, 0]
train_range = list(range(32))
test_range = list(range(32, 40))
ax.bar(train_range, [trajectory_stats_sorted[i]['mean_corr'] for i in train_range], 
       label='Train', alpha=0.7)
ax.bar(test_range, [trajectory_stats_sorted[i]['mean_corr'] for i in test_range], 
       label='Test', alpha=0.7)
ax.set_xlabel('Trajectory Index')
ax.set_ylabel('Mean Correlation')
ax.set_title('Recommended Train/Test Split (80/20)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Time regions to use
ax = axes[1, 1]
time_points = list(range(data.shape[1]))
regions_to_use = []
for t in time_points:
    if 50 <= t <= 430 - PREDICTION_TIMESTEPS:
        regions_to_use.append(1)
    else:
        regions_to_use.append(0)
ax.plot(time_points, regions_to_use, 'g-', linewidth=2, label='Use for dataset')
ax.axhline(0.5, color='r', linestyle='--', linewidth=1, label='Threshold')
ax.set_xlabel('Timestep Index')
ax.set_ylabel('Use in Dataset (1=Yes, 0=No)')
ax.set_title('Recommended Time Regions')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('temp_save', exist_ok=True)
plt.savefig('temp_save/dataset_selection_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved to: temp_save/dataset_selection_analysis.png")
print("=" * 70)

