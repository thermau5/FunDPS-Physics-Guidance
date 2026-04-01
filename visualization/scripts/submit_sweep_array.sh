#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"

CONFIG_PATH="visualization/sweep_config.yaml"
TARGET_SCRIPT="visualization/scripts/run_sweep_array.sh"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config file: $CONFIG_PATH"
  exit 1
fi

if [[ ! -f "$TARGET_SCRIPT" ]]; then
  echo "ERROR: missing target script: $TARGET_SCRIPT"
  exit 1
fi

# Adaptable knobs (override via env vars):
# - CHUNK_SIZE: max tasks per submitted array job (default: 50)
# - SUBMIT_MODE: independent | chain (default: independent)
# - RETRY_SECONDS: wait between retries on submit-limit failures (default: 30)
# - MAX_RETRIES: retries per chunk for submit-limit failures (default: 20)
CHUNK_SIZE="${CHUNK_SIZE:-35}"
SUBMIT_MODE="${SUBMIT_MODE:-independent}"
RETRY_SECONDS="${RETRY_SECONDS:-30}"
MAX_RETRIES="${MAX_RETRIES:-1000}"
current_start=120

if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || (( CHUNK_SIZE <= 0 )); then
  echo "ERROR: CHUNK_SIZE must be a positive integer (got: ${CHUNK_SIZE})"
  exit 1
fi

if [[ "$SUBMIT_MODE" != "independent" && "$SUBMIT_MODE" != "chain" ]]; then
  echo "ERROR: SUBMIT_MODE must be 'independent' or 'chain' (got: ${SUBMIT_MODE})"
  exit 1
fi

if ! [[ "$RETRY_SECONDS" =~ ^[0-9]+$ ]] || (( RETRY_SECONDS < 1 )); then
  echo "ERROR: RETRY_SECONDS must be an integer >= 1 (got: ${RETRY_SECONDS})"
  exit 1
fi

if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ ]] || (( MAX_RETRIES < 0 )); then
  echo "ERROR: MAX_RETRIES must be an integer >= 0 (got: ${MAX_RETRIES})"
  exit 1
fi

read -r NUM_PDES NUM_EPOCHS < <(
  python - <<'PY'
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("visualization/sweep_config.yaml").read_text()) or {}
vis = cfg.get("visualization", {})
pdes = vis.get("pdes", [])
epochs = vis.get("epochs", [])

if not pdes or not epochs:
    print("ERROR ERROR")
    sys.exit(0)

print(len(pdes), len(epochs))
PY
)

if [[ "$NUM_PDES" == "ERROR" || "$NUM_EPOCHS" == "ERROR" ]]; then
  echo "ERROR: visualization/sweep_config.yaml must define non-empty visualization.pdes and visualization.epochs"
  exit 1
fi

TOTAL_TASKS=$(( NUM_PDES * NUM_EPOCHS ))
MAX_INDEX=$(( TOTAL_TASKS - 1 ))

echo "Submitting sweep with ${NUM_PDES} PDEs x ${NUM_EPOCHS} epochs = ${TOTAL_TASKS} tasks"
echo "Chunk size: ${CHUNK_SIZE}, mode: ${SUBMIT_MODE}"

submit_chunk() {
  local range="$1"
  local dep_job_id="${2:-}"
  local try_idx=0
  local out
  local rc

  while true; do
    if [[ -n "$dep_job_id" ]]; then
      set +e
      out="$(sbatch --dependency="afterok:${dep_job_id}" --array="$range" "$TARGET_SCRIPT" "$@" 2>&1)"
      rc=$?
      set -e
    else
      set +e
      out="$(sbatch --array="$range" "$TARGET_SCRIPT" "$@" 2>&1)"
      rc=$?
      set -e
    fi

    if (( rc == 0 )); then
      LAST_SUBMIT_OUTPUT="$out"
      echo "$out"
      return 0
    fi

    if [[ "$out" == *"QOSMaxSubmitJobPerUserLimit"* ]] && (( try_idx < MAX_RETRIES )); then
      try_idx=$((try_idx + 1))
      echo "Submit limit hit for range ${range} (retry ${try_idx}/${MAX_RETRIES}); sleeping ${RETRY_SECONDS}s..." >&2
      sleep "$RETRY_SECONDS"
      continue
    fi

    echo "$out"
    return "$rc"
  done
}


last_job_id=""
LAST_SUBMIT_OUTPUT=""
while (( current_start <= MAX_INDEX )); do
  current_end=$(( current_start + CHUNK_SIZE - 1 ))
  if (( current_end > MAX_INDEX )); then
    current_end=$MAX_INDEX
  fi
  range="${current_start}-${current_end}"

  echo "Submitting chunk array range: ${range}"
  if [[ "$SUBMIT_MODE" == "chain" && -n "$last_job_id" ]]; then
    submit_chunk "$range" "$last_job_id"
  else
    submit_chunk "$range"
  fi

  # Parse SLURM job id from standard output: "Submitted batch job <id>"
  job_id="$(awk '/Submitted batch job/ {print $4}' <<< "$LAST_SUBMIT_OUTPUT" | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "ERROR: could not parse submitted job id for range ${range}"
    exit 1
  fi
  last_job_id="$job_id"
  current_start=$(( current_end + 1 ))
done

echo "All chunks submitted successfully."
