# Navier-Stokes Visualization Guide

## Overview
This directory contains scripts to generate and visualize 2D Navier-Stokes equation solutions.

## Files
- `ns_2d_test.py` - Data generation script (generates both test and visualization data)
- `visualize.py` - Creates animated GIF from full timestep data
- `ns_2d.py` - Training data generation (saves only 2 timesteps)

## Quick Start for Visualization

### Step 1: Generate Data

Simply run the test data generation script:
```bash
python ns_2d_test.py
```

This will generate **both files in a single run**:
- **`ns-nonbounded.mat`**: 1000 samples × 128×128 × 2 timesteps (~0.8 MB)
  - For model training/testing
  - Contains timesteps 25 and 50
  
- **`ns-nonbounded-vis.mat`**: 10 samples × 128×128 × 50 timesteps (~20 MB)
  - For visualization
  - Contains all 50 timesteps
  - **First 10 samples are IDENTICAL to the first 10 in ns-nonbounded.mat**

**Time**: ~1-2 minutes on GPU

### Step 2: Create Visualization

Run the visualization script:
```bash
# Visualize sample 0 (default)
python visualize.py

# Visualize a specific sample (0-9)
python visualize.py --sample 3

# Specify custom output filename
python visualize.py --sample 5 --output sample_5_evolution.gif

# Generate all 10 samples
for i in {0..9}; do
    python visualize.py --sample $i --output pde_evolution_sample_$i.gif
done
```

This will create:
- **File**: `pde_evolution.gif` (or custom name via `--output`)
- **Shows**: Three-column comparison:
  - Left: Initial condition (frame 0)
  - Middle: Frame 25
  - Right: Evolution through all 50 timesteps

### Command-Line Options

```bash
python visualize.py --help
```

Options:
- `--sample N`: Sample index to visualize (0-9, default: 0)
- `--output FILE`: Output GIF filename (default: pde_evolution.gif)

## Data Files

### Test Data: `ns-nonbounded.mat`
- **Samples**: 1000
- **Timesteps**: 2 (25th and 50th)
- **Shape**: `[1000, 128, 128, 2]`
- **Size**: ~0.8 MB
- **Use**: Move to `data/DiffPDE/testing/`

### Visualization Data: `ns-nonbounded-vis.mat`
- **Samples**: 10 (same as first 10 in test data)
- **Timesteps**: 50 (full evolution)
- **Shape**: `[10, 128, 128, 50]`
- **Size**: ~20 MB
- **Use**: With `visualize.py` for creating animations

## Simulation Parameters
- **Grid size**: 128×128
- **Timesteps**: 50
- **Physical time**: 0 to 5
- **Viscosity**: 1e-3 (Reynolds number = 1000)
- **Time step**: 1e-3

