# Visualization Workspace

This folder centralizes sweep/evaluation visualization assets so the project root stays clean.

## Structure

- `scripts/`: orchestration and aggregation scripts
  - `submit_sweep_array.sh`
  - `run_sweep_array.sh`
  - `aggregate_sweep_results.py`
- `sweep_config.yaml`: central hyperparameter/config file for both scripts
  - PDE list
  - generation config paths
  - checkpoint directories
  - epoch list
  - `sweep_exps_outdir` (default `exps/surrogate_sweep`): root for organized sweep outputs
  - `sweep_session_subdir` (default `auto`): one folder per batch, e.g. `G0320_631919` = `G{MMDD}_{SLURM_ARRAY_JOB_ID}` (same spirit as the `G03…` experiment names)
  - `sweep_group_by_pde` (default `false`): optional extra `/<pde>/` under the session folder
- `outputs/`: generated visualization artifacts
  - `sweep_results.csv`
  - `fno_test_loss_by_pde.json`
  - `surrogate_convergence_with_fno_train.png`
- `logs/`: SLURM and sweep logs
  - `sweep_logs/` (array job `out/err` logs)

## Inputs

The visualization pipeline depends on these project-level inputs:

- Trained surrogate checkpoints:
  - `exps/no_surrogate/fno_pad/.../checkpoint_epoch_{EPOCH}.pth`
  - used by `run_sweep_array.sh` and `aggregate_sweep_results.py`
- Generation configs:
  - `configs/generation/poisson_backward.yaml`
  - `configs/generation/helmholtz_backward.yaml`
  - `configs/generation/ns-nonbounded_backward.yaml`
- Experiment results with `metrics.json` + `config.json`:
  - top-level `exps/<run>/` (legacy sweeps and other runs)
  - and/or under `sweep_exps_outdir` with session subfolders, e.g. `exps/surrogate_sweep/G0320_631919/G…sweep_ep…/`
  - consumed by `aggregate_sweep_results.py` (scans top-level `exps/` one deep, and the sweep root up to a few levels)
- Data/model code used by `aggregate_sweep_results.py` when computing FNO test-loss curves
  - e.g. `training/`, `data/`, `neuralop/`

## Workflow

Run from project root:

```bash
# 0) Edit visualization/sweep_config.yaml to choose PDEs/checkpoints/epochs.

# 1) Evaluate all (PDE, epoch) combinations in the SLURM array.
bash visualization/scripts/submit_sweep_array.sh

# 2) Aggregate metrics and regenerate CSV/plot artifacts.
python visualization/scripts/aggregate_sweep_results.py
```

### Optional adaptive submission controls

`submit_sweep_array.sh` supports environment-variable knobs:

```bash
# Default behavior (recommended):
# - splits into chunks of 50 tasks
# - retries on QOS submit-limit errors
CHUNK_SIZE=40 SUBMIT_MODE=independent RETRY_SECONDS=60 MAX_RETRIES=30 \
  bash visualization/scripts/submit_sweep_array.sh
```

- `CHUNK_SIZE` (default `50`): max tasks per submitted array chunk
- `SUBMIT_MODE` (default `independent`):
  - `independent`: chunks run as soon as scheduler allows
  - `chain`: each chunk depends on previous chunk success (`afterok`)
- `RETRY_SECONDS` (default `30`): sleep between retries when hitting `QOSMaxSubmitJobPerUserLimit`
- `MAX_RETRIES` (default `20`): max retries per chunk for submit-limit failures

## Workflow -> Outputs Mapping

This section shows exactly which action causes which result.

### Step 1: `bash visualization/scripts/submit_sweep_array.sh`

Purpose:
- evaluate all `(PDE, epoch)` combinations (`3 PDEs x 9 epochs`)
- reads PDE/checkpoint/config/epoch definitions from `visualization/sweep_config.yaml`
- computes correct `--array` range from config and submits `run_sweep_array.sh`

Primary outputs created/updated:
- `visualization/logs/sweep_logs/<SLURM_JOB_ID>/sweep_ckpts_<JOB_ID>_<TASK_ID>.out`
- `visualization/logs/sweep_logs/<SLURM_JOB_ID>/sweep_ckpts_<JOB_ID>_<TASK_ID>.err`
- experiment result folders under `exps/surrogate_sweep/<session>/` (see `sweep_session_subdir` in `sweep_config.yaml`; used in Step 2)

### Step 2: `python visualization/scripts/aggregate_sweep_results.py`

Purpose:
- aggregate evaluation results and produce final visualization artifacts
- reads PDE/checkpoint/epoch definitions from `visualization/sweep_config.yaml`

Primary outputs created/updated:
- `visualization/outputs/sweep_results.csv`
  - one row per `(pde, epoch)` with relative/spectral metrics and source experiment dir
- `visualization/outputs/surrogate_convergence_with_fno_train.png`
  - combined convergence figure (DDIS error + FNO test-loss overlay)
- `visualization/outputs/fno_test_loss_by_pde.json`
  - cached/recorded FNO test-loss by PDE and epoch for reuse/comparison

## Notes

- **Submit sweep jobs from the project root** (`cd` to `FunDPS-Physics-Guidance` before `bash visualization/scripts/submit_sweep_array.sh`). Batch tasks use `SLURM_SUBMIT_DIR` as the repo root; if you submit from elsewhere, set `FUN_DPS_ROOT` to the absolute path to the repo in your Slurm environment, or tasks may fail before writing `exps/` (e.g. `mkdir: cannot create directory 'visualization': Permission denied` in `.err` logs).
- All outputs are written under `visualization/outputs`.
- All sweep logs are written under `visualization/logs`.
- Use `visualization/scripts/submit_sweep_array.sh` instead of calling `sbatch visualization/scripts/run_sweep_array.sh` directly, so array size stays synchronized with `visualization/sweep_config.yaml`.

### No new folders under `exps/` after a sweep?

1. Check a task log: `visualization/logs/sweep_logs/<JOBID>/sweep_ckpts_<JOBID>_0.err`
2. If you see permission errors on `visualization`, re-submit from project root (see above) after pulling the `run_sweep_array.sh` fix that prefers `SLURM_SUBMIT_DIR`.

### Organizing many sweep folders

- **New runs:** each Slurm array batch writes under **`{sweep_exps_outdir}/{session}/`**, with `sweep_session_subdir: auto` → **`G{MMDD}_{SLURM_ARRAY_JOB_ID}`**, so all epochs/PDEs from that submit share one dated folder; the individual `generate_pde` runs still create `G…` leaf dirs inside it.
- Set `sweep_session_subdir` to a fixed string (e.g. `G0320_ablation`) if you want a human label instead of the job id.
- Optional: `sweep_group_by_pde: true` adds **`/<pde>/`** under the session folder.
- **Chunked `submit_sweep_array.sh`:** each chunk is a new Slurm array job → a **new** `G{MMDD}_{ARRAY_JOB_ID}` folder. To keep one folder across chunks, set a fixed `sweep_session_subdir` (e.g. `G0320_fullgrid`) for that campaign.
