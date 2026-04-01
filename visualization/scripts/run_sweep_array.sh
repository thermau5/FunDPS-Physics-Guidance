#!/usr/bin/env bash
#SBATCH -J sweep_ckpts
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-NAIRR240304
#SBATCH --array=0-26
# One subfolder per array job ID for logs
# (e.g. visualization/logs/sweep_logs/626470/sweep_ckpts_626470_0.out)
#SBATCH -o visualization/logs/sweep_logs/%A/sweep_ckpts_%A_%a.out
#SBATCH -e visualization/logs/sweep_logs/%A/sweep_ckpts_%A_%a.err

set -euo pipefail

# Project root: on batch nodes ${BASH_SOURCE[0]} often points at a Slurm spool copy, so
# $(dirname)/../.. is wrong and mkdir/visualization fails with "Permission denied".
# Always submit from the repo root; then SLURM_SUBMIT_DIR is correct.
if [[ -n "${FUN_DPS_ROOT:-}" ]]; then
  WORKDIR="$(cd "${FUN_DPS_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/visualization/sweep_config.yaml" ]]; then
  WORKDIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$WORKDIR"
CONFIG_PATH="visualization/sweep_config.yaml"

IFS=$'\t' read -r SWEEP_EXPS_OUTDIR SWEEP_GROUP_BY_PDE SWEEP_SESSION_SUBDIR <<EOF
$(python - <<'PY'
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("visualization/sweep_config.yaml").read_text()) or {}
vis = cfg.get("visualization", {})
out = vis.get("sweep_exps_outdir") or "exps/surrogate_sweep"
grp = 1 if vis.get("sweep_group_by_pde", False) else 0
sess = vis.get("sweep_session_subdir", "auto")
if sess is None or str(sess).strip() == "":
    sess = "auto"
print(f"{out}\t{grp}\t{sess}")
PY
)
EOF

# Create per-job log directory (fallback to 'manual' if not under SLURM)
LOG_JOB_ID="${SLURM_ARRAY_JOB_ID:-manual}"
mkdir -p "visualization/logs/sweep_logs/${LOG_JOB_ID}"

# Load PDE names, generation configs, checkpoint dirs, and epochs from YAML.
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config file: $CONFIG_PATH"
  exit 1
fi

mapfile -t MAPPING_ROWS < <(
  python - <<'PY'
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("visualization/sweep_config.yaml").read_text()) or {}
vis = cfg.get("visualization", {})
for item in vis.get("pdes", []):
    print(f"{item['name']}\t{item['generation_config']}\t{item['checkpoint_dir']}")
PY
)

mapfile -t EPOCHS < <(
  python - <<'PY'
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("visualization/sweep_config.yaml").read_text()) or {}
for e in (cfg.get("visualization", {}).get("epochs", [])):
    print(int(e))
PY
)

if [[ "${#MAPPING_ROWS[@]}" -eq 0 || "${#EPOCHS[@]}" -eq 0 ]]; then
  echo "ERROR: invalid or empty visualization config in $CONFIG_PATH"
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_EPOCHS=${#EPOCHS[@]}  # 9
NUM_PDES=${#MAPPING_ROWS[@]}
TOTAL_TASKS=$(( NUM_PDES * NUM_EPOCHS ))

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  echo "ERROR: TASK_ID=${TASK_ID} out of range for ${NUM_PDES} PDEs x ${NUM_EPOCHS} epochs (${TOTAL_TASKS} tasks)."
  echo "Update #SBATCH --array to 0-$((TOTAL_TASKS - 1)) or adjust visualization/sweep_config.yaml."
  exit 1
fi

PDE_IDX=$(( TASK_ID / NUM_EPOCHS ))
EPOCH_IDX=$(( TASK_ID % NUM_EPOCHS ))

IFS=$'\t' read -r PDE CONFIG CKPT_DIR <<< "${MAPPING_ROWS[$PDE_IDX]}"
EPOCH="${EPOCHS[$EPOCH_IDX]}"

CKPT_PATH="${CKPT_DIR}/checkpoint_epoch_${EPOCH}.pth"

echo "[$(date)] Task ${TASK_ID}: PDE=${PDE}, epoch=${EPOCH}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CKPT_PATH}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH"
  exit 1
fi

# Environment:
# We rely on the submit-time environment (e.g. submit from within `edm_no`),
# and avoid sourcing ~/.bashrc or re-activating conda here, since TACC's
# batch ~/.bashrc adds unbound debug vars that trip `set -u`.

# Run evaluation
NAME="sweep_ep${EPOCH}_${PDE}"
echo "[$(date)] Running generate_pde.py with name=${NAME}"

if [[ "${SWEEP_SESSION_SUBDIR}" == "auto" ]]; then
  SESSION="G$(date +%m%d)_${SLURM_ARRAY_JOB_ID:-local}"
else
  SESSION="${SWEEP_SESSION_SUBDIR}"
fi
TASK_OUTDIR="${SWEEP_EXPS_OUTDIR}/${SESSION}"
if [[ "${SWEEP_GROUP_BY_PDE}" == "1" ]]; then
  TASK_OUTDIR="${TASK_OUTDIR}/${PDE}"
fi
mkdir -p "$TASK_OUTDIR"
echo "Output directory: ${TASK_OUTDIR}"

python generate_pde.py \
  --config "$CONFIG" \
  --surrogate_path "$CKPT_PATH" \
  --name "$NAME" \
  --outdir "$TASK_OUTDIR"

echo "[$(date)] Done task ${TASK_ID}"
