# Decoupled Diffusion Inverse Solver

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

### Training

**Diffusion model** (backbone for generation):

```shell
python scripts/train/train.py -c configs/training/poisson.yml --name poisson-run
```

Configs for each PDE are in `configs/training/` (e.g. `poisson.yml`, `helmholtz.yml`, `ns-nonbounded.yml`).

**FNO surrogate** (physics guidance model, trained separately):

```shell
python scripts/train/training_fno.py --config configs/training/no_surrogate/fno_pad_poisson_forward.yml
```

Surrogate configs are in `configs/training/no_surrogate/`, one per PDE and direction (forward/backward).

### Inference

```shell
python scripts/generate/generate_pde.py --config configs/generation/poisson_backward.yaml
```

Generation configs are in `configs/generation/`. Key options to set in the config:
- `surrogate_type`: `fno_pad` (learned surrogate) or `numerical_poisson` (exact solver, Poisson only)
- `surrogate_path`: path to trained FNO `.pth` file
- `guidance.type`: `daps` (recommended) or `dps`

## Unit Tests

Validation scripts live in `unit_tests/`. They test individual components without running the full generation pipeline.

```shell
# Test NumericalPoissonWrapper logic (no data required)
python unit_tests/test_poisson_solver.py

# Validate NumericalPoissonWrapper against real Poisson dataset
python unit_tests/test_poisson_solver.py --data-path data/DiffPDE/poisson_test_hf --save-dir exps/tests/poisson_solver

# Validate a trained FNO/FNO_pad surrogate (architecture auto-detected from checkpoint)
python unit_tests/test_fno_surrogate.py --model-path <path-to-model.pth> --data-path data/DiffPDE/helmholtz_test_hf

# Validate a trained PINO surrogate
python unit_tests/test_pino_surrogate.py --model-path <path-to-model.pth>
```

Outputs (visualizations, metrics) are saved under `exps/tests/`.

## Visualization

Analysis and plotting scripts live in `visualization/scripts/`.

```shell
# Aggregate and plot sweep results
python visualization/scripts/aggregate_sweep_results.py

# Power spectrum comparison across methods
python visualization/scripts/power_spectrum.py
```

## Contributing

Please apply the `black` code formatter with the `--line-length=200` option. We typically wouldn't opt for such a long line length, but due to numerous existing lengthy lines, this will help reduce discrepancies.
