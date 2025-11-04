# temp_0-main: PINO Model for Navier-Stokes

## What You Have

A **complete PINO (Physics-Informed Neural Operator)** package for solving 2D Navier-Stokes equations:
- ✅ Trained model: `model_save/model_[8,48,48]_28_32_8064(ep31)_cvt_2.pt`
- ✅ Dataset: `model_save/re1000_grid=256_1_N=40_dt=4.0_Ttj=200-320(stat).pt` (4.7 GB)
- ✅ Inference script: `scripts/station.py`

**Why a folder instead of a `.pth` file?** PINO models include the model architecture, physics constraints, data processing pipeline, and configurations - not just weights.

## Quick Start

```bash
cd /home/thomaslin/FunDPS-Physics-dev-v2/temp_0-main/scripts
python station.py
```

**Output**: Visualizations saved to `temp_save/visual_*.png` (input, output, truth)

## Dataset & Model Details

**Dataset**: 40 Navier-Stokes trajectories, 256×256 resolution, Re=1000
- Format: PyTorch tensor `[N=40, T=481, X=256, Y=256]`
- Each trajectory has 481 timesteps
- See **[DATASET_INFO.md](DATASET_INFO.md)** for complete details

**Model**: FNO-based PINO 
- Input: Vorticity at time t
- Output: Vorticity at time t + 0.5
- Modes: [8, 48, 48], 28 hidden channels, 4 layers

## Training Your Own Model

The main project has training scripts for newer dataset format:

```bash
cd /home/thomaslin/FunDPS-Physics-dev-v2

# Generate dataset (128×128, 50K samples)
cd dataset_generation/non-bounded-ns
python ns_2d.py              # ~10-20 hours
python ns_2d_test.py         # ~15-30 minutes
./move_files.sh
cd ../..
python utils/dataset_process.py ns-nonbounded_new

# Train model
python training_fno_v2.py
```

Note: This uses a different format (128×128 HuggingFace) than temp_0-main (256×256 PyTorch).

## Troubleshooting

**"Module not found: neuralop"**
```bash
pip install neuralop
```

