#!/usr/bin/env python3
"""
Minimal script to convert temp_0-main PyTorch dataset to HuggingFace format.
Memory-efficient version using globals.
"""

import torch
import numpy as np
import os
import json
from datasets import Dataset, Features, Array3D, Value

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = 'temp_0-main/dataset_and_model/re1000_grid=256_1_N=40_dt=4.0_Ttj=200-320(stat).pt'
PREDICTION_TIMESTEP = 2  # Predict 2 timesteps ahead (0.5 physical time)
RESOLUTION = 256  # Keep your full resolution, no downsampling needed!
OUTPUT_DIR = 'data/DiffPDE'

# Globals for generator access (avoids pickle issues)
_data = None
_mean_input = None
_std_input = None
_mean_output = None
_std_output = None

print("=" * 70)
print("CONVERTING temp_0-main TO HUGGINGFACE FORMAT")
print("=" * 70)

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("\n1. Loading PyTorch data...")
data = torch.load(INPUT_FILE)
data = data.numpy()
print(f"   Original shape: {data.shape}")
# Drop last timestep which is padded zeros
data = data[:, :-1, :, :]
print(f"   After removing padded timestep: {data.shape}")
print(f"   Dtype: {data.dtype}")

# ==============================================================================
# EXTRACT PAIRS AND COMPUTE STATS (MEMORY EFFICIENT)
# ==============================================================================
print("\n2. Computing statistics (streaming)...")
pairs_count = 0
sum_input = 0
sum_output = 0
sum_sq_input = 0
sum_sq_output = 0
min_input = float('inf')
max_input = float('-inf')
min_output = float('inf')
max_output = float('-inf')

for traj_idx in range(data.shape[0]):  # For each trajectory
    trajectory = data[traj_idx]  # [481, 256, 256]
    for t in range(trajectory.shape[0] - PREDICTION_TIMESTEP):
        input_field = trajectory[t]
        output_field = trajectory[t + PREDICTION_TIMESTEP]
        
        # Accumulate stats
        sum_input += input_field.sum()
        sum_output += output_field.sum()
        sum_sq_input += (input_field ** 2).sum()
        sum_sq_output += (output_field ** 2).sum()
        min_input = min(min_input, input_field.min())
        max_input = max(max_input, input_field.max())
        min_output = min(min_output, output_field.min())
        max_output = max(max_output, output_field.max())
        
        pairs_count += 1

# Compute mean and std
total_elements = pairs_count * RESOLUTION * RESOLUTION
mean_input = sum_input / total_elements
mean_output = sum_output / total_elements
var_input = (sum_sq_input / total_elements) - (mean_input ** 2)
var_output = (sum_sq_output / total_elements) - (mean_output ** 2)
std_input = np.sqrt(var_input)
std_output = np.sqrt(var_output)

print(f"   Extracted {pairs_count} pairs")
print(f"   Input:  mean={mean_input:.4f}, std={std_input:.4f}")
print(f"   Output: mean={mean_output:.4f}, std={std_output:.4f}")

# Store in globals for generator
_data = data
_mean_input = mean_input
_std_input = std_input
_mean_output = mean_output
_std_output = std_output

# ==============================================================================
# GENERATOR FUNCTIONS
# ==============================================================================
print("\n3. Creating HuggingFace datasets...")

features = Features({
    'id': Value('int32'),
    'data': Array3D(shape=(2, RESOLUTION, RESOLUTION), dtype='float64')
})

def generator_train():
    """Generator for training data."""
    idx = 0
    for traj_idx in range(32):  # First 32 trajectories
        trajectory = _data[traj_idx]
        for t in range(trajectory.shape[0] - PREDICTION_TIMESTEP):
            input_field = trajectory[t]
            output_field = trajectory[t + PREDICTION_TIMESTEP]
            
            # Normalize on-the-fly
            norm_input = (input_field - _mean_input) * (0.5 / _std_input)
            norm_output = (output_field - _mean_output) * (0.5 / _std_output)
            stacked = np.stack((norm_input, norm_output), axis=0).astype(np.float64)
            
            yield {'id': idx, 'data': stacked}
            idx += 1

def generator_test():
    """Generator for testing data."""
    idx = 0
    for traj_idx in range(32, 40):  # Last 8 trajectories
        trajectory = _data[traj_idx]
        for t in range(trajectory.shape[0] - PREDICTION_TIMESTEP):
            input_field = trajectory[t]
            output_field = trajectory[t + PREDICTION_TIMESTEP]
            
            # Normalize on-the-fly
            norm_input = (input_field - _mean_input) * (0.5 / _std_input)
            norm_output = (output_field - _mean_output) * (0.5 / _std_output)
            stacked = np.stack((norm_input, norm_output), axis=0).astype(np.float64)
            
            yield {'id': idx, 'data': stacked}
            idx += 1

# Create datasets
train_samples = 32 * (data.shape[1] - PREDICTION_TIMESTEP)
test_samples = 8 * (data.shape[1] - PREDICTION_TIMESTEP)

print(f"\n   Processing train set ({train_samples} samples)...")
dataset_train = Dataset.from_generator(generator_train, features=features)
output_dir_train = os.path.join(OUTPUT_DIR, 'ns-temp0main_train_hf')
dataset_train.save_to_disk(output_dir_train)

print(f"   Processing test set ({test_samples} samples)...")
dataset_test = Dataset.from_generator(generator_test, features=features)
output_dir_test = os.path.join(OUTPUT_DIR, 'ns-temp0main_test_hf')
dataset_test.save_to_disk(output_dir_test)

# Save metadata
for split_name, num_samples, output_dir in [
    ('train', train_samples, output_dir_train),
    ('test', test_samples, output_dir_test)
]:
    metadata = {
        'name': 'ns-temp0main',
        'stats': {
            'mean': [float(mean_input), float(mean_output)],
            'std': [float(std_input), float(std_output)],
            'min': [float(min_input), float(min_output)],
            'max': [float(max_input), float(max_output)]
        },
        'shape': [2, RESOLUTION, RESOLUTION],
        'num_samples': num_samples,
        '__version__': '2.0'
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Saved {split_name} to {output_dir}")

# Free memory
del data, _data

# ==============================================================================
# VERIFICATION
# ==============================================================================
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

print("\nTesting dataset loading...")
from training.dataset_hf import PDEDataset

# Test loading
train_dataset = PDEDataset(path='data/DiffPDE/ns-temp0main_train_hf')
test_dataset = PDEDataset(path='data/DiffPDE/ns-temp0main_test_hf')

print(f"✓ Train dataset: {len(train_dataset)} samples")
print(f"✓ Test dataset: {len(test_dataset)} samples")

# Test normalizer
normalizer = train_dataset.create_normalizer()
sample_data, _ = train_dataset[0]
sample_data_torch = torch.from_numpy(sample_data[np.newaxis, ...])
denorm = normalizer.denormalize(sample_data_torch)[0].numpy()

print(f"✓ Sample shape: {sample_data.shape}")
print(f"✓ Denormalization works: {denorm.shape}")

print("\n" + "=" * 70)
print("✓ CONVERSION COMPLETE!")
print("=" * 70)
print(f"\nOutput directories:")
print(f"  - data/DiffPDE/ns-temp0main_train_hf/")
print(f"  - data/DiffPDE/ns-temp0main_test_hf/")
