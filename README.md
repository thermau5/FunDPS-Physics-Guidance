# FunctionSpaceDiffusion

## Setup

We support two tested setup paths depending on your hardware:

- **x86_64 Linux with NVIDIA GPU** (e.g., RTX 4090): use `environment.yml`.
- **VISTA aarch64/ARM64 GPU nodes**: use `environment_platform.yml` plus a small PyTorch post-step.

### 1. x86_64 (e.g., RTX 4090)

```shell
conda env create -f environment.yml
conda activate edm_no

python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
# Expected (tested) output on an RTX 4090 machine:
# 2.7.1+cu126 12.6
# CUDA available: True
```

If your output differs significantly, you may need to adapt CUDA / PyTorch versions for your particular GPU, but this configuration is known to work on an RTX 4090 machine.

### 2. VISTA aarch64/ARM64 GPU nodes

From a VISTA interactive job:

```shell
idev -p ...
module load gcc cuda

conda env create -f environment_platform.yml
conda activate edm_no

# One-time setup per VISTA machine: always use the env's libstdc++ (fixes GLIBCXX_* issues)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat >> "$CONDA_PREFIX/etc/conda/activate.d/edm_no_libstdcxx.sh" << 'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF

# Install a tested CUDA 12.4 PyTorch stack
python -m pip install --force-reinstall --no-deps \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
# Expected (tested) output on VISTA:
# 2.5.1 12.4
# CUDA available: True
```

This keeps `environment_platform.yml` portable across architectures, while the post-step installs a matching GPU-capable PyTorch build on VISTA aarch64 and ensures the environment's C++ runtime is consistently used.

### Install `neuraloperator` (spectral_fix branch)

After the environment is created and activated (either path above), install the `neuraloperator` fork used in this project:

```shell
git clone -b spectral_fix https://github.com/i207M/neuraloperator.git
cd neuraloperator
pip install -e .
cd ..
```

### Download data

Please follow the instructions in [DiffusionPDE](https://github.com/jhhuangchloe/DiffusionPDE) to download the data and place it in the `data/DiffPDE` directory.

To generate the data, run:

```shell
python utils/dataset_process.py all
```

To initialize the wandb environment, run:

```shell
wandb init
```

## Usage

```shell
# Train a new diffusion model on the Darcy Flow dataset.
python train.py -c=configs/training/darcy.yml --name=darcy-test

# Recover both spaces with observation on both sides
python generate_pde.py --config configs/generation/darcy.yaml
```

## Contributing

Please apply the `black` code formatter with the `--line-length=200` option. We typically wouldn't opt for such a long line length, but due to numerous existing lengthy lines, this will help reduce discrepancies.
