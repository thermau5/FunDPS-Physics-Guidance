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

We support both an **automatic Hugging Face download** (recommended) and the **original manual download**.

- **Option A (recommended, automatic from Hugging Face)**

  This repository includes `utils/download_dataset.py`, which pulls the normalized PDE datasets from the Hugging Face dataset `jcy20/DiffusionPDE-normalized` and places them under `data/DiffPDE`:

  ```shell
  # From the project root
  python utils/download_dataset.py all --output-dir data/DiffPDE
  ```

  - By default this downloads the **training** splits for all supported datasets (`darcy`, `helmholtz`, `ns-bounded`, `ns-nonbounded`, `poisson`).
  - To download the **test** splits instead, add `--test`:

    ```shell
    python utils/download_dataset.py all --output-dir data/DiffPDE --test
    ```

  - You can also download a **single dataset** (train or test) by name, e.g.:

    ```shell
    # Single dataset (Darcy, train split)
    python utils/download_dataset.py darcy --output-dir data/DiffPDE

    # Single dataset (Darcy, test split)
    python utils/download_dataset.py darcy --output-dir data/DiffPDE --test
    ```

  After this step, you should have a structure like:

  - `data/DiffPDE/darcy_hf`, `data/DiffPDE/darcy_test_hf`
  - `data/DiffPDE/helmholtz_hf`, `data/DiffPDE/helmholtz_test_hf`
  - `data/DiffPDE/ns-bounded_hf`, `data/DiffPDE/ns-bounded_test_hf`
  - `data/DiffPDE/ns-nonbounded_hf`, `data/DiffPDE/ns-nonbounded_test_hf`
  - `data/DiffPDE/poisson_hf`, `data/DiffPDE/poisson_test_hf`

- **Option B (manual, original DiffusionPDE instructions)**

  You can alternatively follow the original data download instructions in the DiffusionPDE repository and then copy the data into `data/DiffPDE`:

  ```text
  https://github.com/jhhuangchloe/DiffusionPDE
  ```

To generate the processed data used by this repo, run:

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
