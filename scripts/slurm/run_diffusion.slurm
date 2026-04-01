#!/bin/bash
# Diffusion training (scripts/train/train.py) under SLURM.
#
# The training YAML must set a concrete "name" (W&B / run id). Replace
#   name: __required__
# with e.g. name: poisson-128-001 — do not pass a second argument here.
#
# Usage (auto-submits with SLURM job name derived from that "name" field):
#   ./run_diffusion.slurm <config.yml>
#
# Examples:
#   ./run_diffusion.slurm configs/training/poisson.yml
#
# Multi-GPU on one node:
#   NPROC_PER_NODE=4 ./run_diffusion.slurm configs/training/darcy.yml
#
# Optional env:
#   NPROC_PER_NODE   - torchrun --nproc_per_node (default: 1)
#   TRAIN_EXTRA_ARGS - extra args for train.py, e.g. '--duration=1' or '--name=override'

#SBATCH -J dm-run
#SBATCH -o logs/dm-run.%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A TG-NAIRR240304
## Uncomment on clusters that use explicit GPU requests:
##SBATCH --gres=gpu:1

CONFIG="${1:?Usage: $0 <config.yml>}"

_slurm_job_name_from_yaml() {
  export _RUN_DIFFUSION_SLURM_CONFIG="$1"
  python <<'PY'
import os, re, sys, yaml
from pathlib import Path

path = Path(os.environ["_RUN_DIFFUSION_SLURM_CONFIG"])
if not path.is_file():
    print(f"Config not found: {path}", file=sys.stderr)
    sys.exit(1)
cfg = yaml.safe_load(path.read_text())
n = cfg.get("name")
if not n or n == "__required__":
    sys.stderr.write(
        "run_diffusion.slurm: set a concrete 'name' in the training YAML "
        "(replace name: __required__ with your run name).\n"
    )
    sys.exit(1)

def safe(s: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(s)).strip("-")
    return t[:56] if t else "run"

print(f"dm-{safe(n)}"[:64])
PY
  unset _RUN_DIFFUSION_SLURM_CONFIG
}

# When not running under SLURM: derive job name from YAML and submit with -J/-o
if [ -z "${SLURM_JOB_ID:-}" ]; then
  SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  source ~/.bashrc
  conda activate edm_no
  cd "${WORK}/FunDPS-Physics-Guidance" || exit 1
  mkdir -p logs
  RUNNAME="$(_slurm_job_name_from_yaml "$CONFIG")" || exit 1
  exec sbatch -J "$RUNNAME" -o "logs/${RUNNAME}.%j.log" "$SELF" "$CONFIG"
fi

source ~/.bashrc
conda activate edm_no

cd "$WORK/FunDPS-Physics-Guidance/" || exit 1

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

if [ "${NPROC_PER_NODE}" -eq 1 ]; then
  # shellcheck disable=SC2086
  exec python scripts/train/train.py -c "$CONFIG" ${TRAIN_EXTRA_ARGS}
else
  # shellcheck disable=SC2086
  exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/train/train.py -c "$CONFIG" ${TRAIN_EXTRA_ARGS}
fi
