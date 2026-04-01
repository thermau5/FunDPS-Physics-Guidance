#!/bin/bash
# Usage: ./run_fno.slurm <config.yml>
#   Always pass the config YAML as the first argument. Job/log names are derived from config:
#   fno-<dataset>-bs<bs>-lr<lr>-ep<ep>_m<m>
#
#SBATCH -J fno-run
#SBATCH -o logs/fno-run.%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A TG-NAIRR240304

CONFIG="${1:?Usage: $0 <config.yml>}"

# When not running under SLURM: compute run name from config and submit ourselves with -J/-o
if [ -z "${SLURM_JOB_ID:-}" ]; then
  SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  source ~/.bashrc
  conda activate edm_no
  cd "${WORK}/FunDPS-Physics-Guidance" || exit 1
  mkdir -p logs
  RUNNAME="$(python configs/training/no_surrogate/get_fno_runname.py "$CONFIG")"
  exec sbatch -J "$RUNNAME" -o "logs/${RUNNAME}.%j.log" "$SELF" "$CONFIG"
fi

# 1. Activate your environment (module load not needed on Vista)
source ~/.bashrc
conda activate edm_no

# 2. Navigate to the project folder
cd "$WORK/FunDPS-Physics-Guidance/" || exit 1

# 3. Run the training script
python scripts/train/training_fno.py -c "$CONFIG"
