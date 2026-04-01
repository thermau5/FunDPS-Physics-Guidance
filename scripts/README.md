# Scripts Organization

This folder contains runnable entrypoints grouped by purpose.

## Folder guide

- `scripts/train/`: Training entrypoints and training variants.
- `scripts/generate/`: PDE generation / inference entrypoints.
- `scripts/plot/`: Plotting and analysis scripts used after runs.
- `scripts/slurm/`: Cluster submission scripts.

## Current files

- `scripts/train/train.py`: Main diffusion training launcher.
- `scripts/train/training_fno.py`: FNO no-surrogate training entrypoint (YAML config, W&B, checkpoints). Run from repo root: `python scripts/train/training_fno.py -c <config.yml>`.
- `scripts/generate/generate_pde.py`: PDE generation/sampling entrypoint.
- `scripts/plot/plot_ground_truth.py`: Ground-truth visualization helper.
- `scripts/plot/plot_l2_comparison.py`: L2 metric comparison plotting helper.
- `scripts/slurm/run_fno.slurm`: SLURM job script for FNO-related runs.

If new scripts are added, place them in one of these folders and update this file.
