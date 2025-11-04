# Dataset Information

## Overview

The dataset contains **40 Navier-Stokes vorticity field trajectories** at Reynolds number 1000.

**File**: `re1000_grid=256_1_N=40_dt=4.0_Ttj=200-320(stat).pt`

## Dataset Structure

### Tensor Shape
```python
Shape: [40, 481, 256, 256]
  Axis 0: 40 different trajectories (initial conditions)
  Axis 1: 481 timesteps (time evolution)
  Axis 2-3: 256x256 spatial grid
```

### Data Type & Statistics
- **Type**: PyTorch float32 tensor
- **Min value**: -69.98
- **Max value**: 69.02
- **Mean**: 0.00
- **Std**: 5.97

## Parameters from Filename

Decoding `re1000_grid=256_1_N=40_dt=4.0_Ttj=200-320(stat).pt`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `re1000` | 1000 | Reynolds number |
| `grid=256` | 256x256 | Spatial resolution |
| `N=40` | 40 | Number of trajectories |
| `dt=4.0` | 4.0 | Physical time unit (in filename) |
| `Ttj=200-320` | 200-320 | Physical time range |
| Actual `dtsave` | 0.25 | Time step between snapshots |

## Time Domain

### Physical Time
- **Start**: t = 200
- **End**: t = 320
- **Duration**: 120 time units
- **Time step**: 0.25 (so 1/dt = 4.0 from filename)

### Timestep Mapping
```
Physical time t = 200 + timestep_index × 0.25

Examples:
  timestep 0   → t = 200.00
  timestep 100 → t = 225.00
  timestep 200 → t = 250.00
  timestep 481 → t = 320.25 (but array only has 481 steps, so last is t=320.00)
```

## Spatial Domain

### Physical Space
- **Domain**: [0, 2π] × [0, 2π]
- **Grid points**: 256 × 256
- **Grid spacing**: 2π/256 ≈ 0.0245 per grid point

### What's Represented
Each grid point stores the **vorticity** (ω) at that spatial location:
- Vorticity = curl of velocity field
- Units: dimensionless (after normalization)
- Range: approximately [-70, +70]

## Example: How the Model Uses the Data

From `station.py`:

```python
# Select one trajectory
traj_id = 2  # Use trajectory #2
time_id = 5  # Start at timestep 5

# Calculate physical time
t_input = 200 + (time_id - 1) × 0.25 = 201.00

# Extract input and target
x = data[traj_id, time_id, :, :]      # Input: vorticity at t=201.00
y = data[traj_id, time_id+2, :, :]    # Target: vorticity at t=201.50 (0.5 later)

# Model task: predict y from x
```

### Model Input/Output
- **Input**: Vorticity field at time t (256×256)
- **Output**: Vorticity field at time t + 0.5 (256×256)
- **Task**: Short-term time evolution prediction

## Physics: Navier-Stokes Equations

The data comes from solving the 2D incompressible Navier-Stokes equations:

```
∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u + f
∇·u = 0

Where:
  u = velocity field
  p = pressure
  Re = Reynolds number = 1000
  f = forcing term
```

The vorticity ω = ∇×u evolves according to:
```
∂ω/∂t + (u·∇)ω = (1/Re)∇²ω + F
```

This dataset contains ω(x,y,t) for 40 different initial conditions and forcing patterns.

## Key Features

1. **Multiple trajectories**: 40 different flow patterns
2. **Temporal evolution**: 481 snapshots covering 120 time units
3. **High resolution**: 256×256 spatial grid
4. **Turbulent flow**: Re=1000 (moderately turbulent)
5. **Stationary statistics**: "stat" in filename suggests statistical stationarity reached

## Usage in Training

For PINO training:
- Each timestep can be an input snapshot
- Future timesteps are targets
- Model learns the evolution operator: ω(t) → ω(t+Δt)
- Physics-informed loss enforces Navier-Stokes constraints

## Visualization

The `station.py` script generates three plots:
1. **input.png**: Initial vorticity field
2. **output.png**: Model prediction
3. **truth.png**: Actual future state

Each plot shows the vorticity field as a heatmap on the 2π×2π domain.

