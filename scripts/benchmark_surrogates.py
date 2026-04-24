"""Benchmark all available surrogate checkpoints for Poisson and Helmholtz inference.

Runs generate_pde.py for each (pde, surrogate) combo and collects metrics.json,
then prints a summary table sorted by relative error.

Usage:
    python scripts/benchmark_surrogates.py [--max_size N] [--pde poisson|helmholtz|both]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from glob import glob
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Surrogate catalog
# ---------------------------------------------------------------------------

POISSON_SURROGATES = [
    {
        "name": "fno_pad_400_legacy_4090",
        "surrogate_type": "fno_pad",
        "surrogate_path": "generation/fno_pad_trained_forward_poisson_128_400.pth",
        "note": "400-epoch, legacy 4090",
    },
    {
        "name": "fno_pad_500_legacy_4090",
        "surrogate_type": "fno_pad",
        "surrogate_path": "generation/fno_pad_trained_forward_poisson_128_500.pth",
        "note": "500-epoch, legacy 4090",
    },
    {
        "name": "fno_pad_500_oa5hfuiw",
        "surrogate_type": "fno_pad",
        "surrogate_path": "model/no_surrogate/fno_pad/forward_poisson/128/500/fno_trained_500_oa5hfuiw.pth",
        "note": "500-epoch, TACC run oa5hfuiw",
    },
    {
        "name": "fno_pad_1000_i409rm1v",
        "surrogate_type": "fno_pad",
        "surrogate_path": "model/no_surrogate/fno_pad/forward_poisson/128/1000/fno_trained_1000_i409rm1v.pth",
        "note": "1000-epoch, TACC run i409rm1v",
    },
    {
        "name": "fno_pad_1600_xudsngyh",
        "surrogate_type": "fno_pad",
        "surrogate_path": "model/no_surrogate/fno_pad/forward_poisson/128/1600/fno_trained_1600_xudsngyh.pth",
        "note": "1600-epoch, TACC run xudsngyh",
    },
    {
        "name": "numerical_poisson",
        "surrogate_type": "numerical_poisson",
        "surrogate_path": None,
        "note": "Numerical Poisson solver, no learned surrogate",
    },
]

HELMHOLTZ_SURROGATES = [
    {
        "name": "fno_pad_400_legacy_4090",
        "surrogate_type": "fno_pad",
        "surrogate_path": "generation/fno_pad_trained_forward_helmholtz_128_400.pth",
        "note": "400-epoch, legacy 4090",
    },
    {
        "name": "fno_pad_500_legacy_4090",
        "surrogate_type": "fno_pad",
        "surrogate_path": "generation/fno_pad_trained_forward_helmholtz_128_500.pth",
        "note": "500-epoch, legacy 4090",
    },
    {
        "name": "fno_pad_500_elhusgf4",
        "surrogate_type": "fno_pad",
        "surrogate_path": "model/no_surrogate/fno_pad/forward_helmholtz/128/500/fno_trained_500_elhusgf4.pth",
        "note": "500-epoch, TACC run elhusgf4",
    },
    {
        "name": "fno_pad_1000_bs160",
        "surrogate_type": "fno_pad",
        "surrogate_path": "model/no_surrogate/fno_pad/forward_helmholtz/128/1000/fno_trained_1000_bs160.pth",
        "note": "1000-epoch, TACC run bs160",
    },
    {
        "name": "fno_pad_1600_jcjrczob",
        "surrogate_type": "fno_pad",
        "surrogate_path": "model/no_surrogate/fno_pad/forward_helmholtz/128/1600/fno_trained_1600_jcjrczob.pth",
        "note": "1600-epoch, TACC run jcjrczob",
    },
]

BASE_CONFIGS = {
    "poisson": "configs/generation/poisson_backward.yaml",
    "helmholtz": "configs/generation/helmholtz_backward.yaml",
}
SURROGATES = {
    "poisson": POISSON_SURROGATES,
    "helmholtz": HELMHOLTZ_SURROGATES,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_base_config(pde: str) -> dict:
    path = PROJECT_ROOT / BASE_CONFIGS[pde]
    with open(path) as f:
        return yaml.safe_load(f)


def make_temp_config(base: dict, surrogate: dict, max_size: int) -> Path:
    cfg = deepcopy(base)
    cfg["max_size"] = max_size
    cfg["batch_size"] = min(base.get("batch_size", max_size), max_size)
    cfg["name"] = surrogate["name"]
    cfg["wandb"] = "offline"

    # Update surrogate settings under guidance
    cfg["guidance"]["surrogate_type"] = surrogate["surrogate_type"]
    # Remove any old guidance.surrogate_path to avoid stale overrides
    cfg["guidance"].pop("surrogate_path", None)

    # Set top-level surrogate_path (daps.py reads this as fallback)
    if surrogate["surrogate_path"] is not None:
        cfg["surrogate_path"] = surrogate["surrogate_path"]
    else:
        cfg["surrogate_path"] = None

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"bench_{surrogate['name']}_", delete=False
    )
    yaml.dump(cfg, tmp)
    tmp.flush()
    return Path(tmp.name)


def find_metrics_json(outdir: str, run_name_fragment: str) -> Path | None:
    """Find the metrics.json for the most recent run matching the name fragment."""
    pattern = os.path.join(outdir, f"G*-{run_name_fragment}", "metrics.json")
    matches = sorted(glob(pattern), key=os.path.getmtime)
    if matches:
        return Path(matches[-1])
    # broader search
    pattern2 = os.path.join(outdir, f"G*{run_name_fragment}*", "metrics.json")
    matches2 = sorted(glob(pattern2), key=os.path.getmtime)
    return Path(matches2[-1]) if matches2 else None


def parse_metrics(metrics_path: Path) -> dict:
    with open(metrics_path) as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        if k.endswith("_mean") or k.endswith("_std"):
            result[k] = v
    return result


def run_single(pde: str, surrogate: dict, max_size: int, dry_run: bool) -> dict:
    base = load_base_config(pde)
    tmp_cfg = make_temp_config(base, surrogate, max_size)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/generate/generate_pde.py"),
        "--config", str(tmp_cfg),
    ]

    label = f"{pde}/{surrogate['name']}"
    print(f"\n{'='*70}")
    print(f"  Running: {label}")
    print(f"  Surrogate path: {surrogate['surrogate_path']}")
    print(f"  Note: {surrogate['note']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        tmp_cfg.unlink(missing_ok=True)
        return {"label": label, "status": "dry_run", "metrics": {}}

    # Check surrogate file exists (skip for numerical)
    if surrogate["surrogate_path"] is not None:
        full_path = PROJECT_ROOT / surrogate["surrogate_path"]
        if not full_path.exists():
            print(f"  WARNING: surrogate file not found: {full_path}. Skipping.")
            tmp_cfg.unlink(missing_ok=True)
            return {"label": label, "status": "skipped_missing_file", "metrics": {}}

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,  # stream to terminal
        text=True,
    )
    elapsed = time.time() - t0
    tmp_cfg.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"\n  ERROR: run failed (exit code {result.returncode})")
        return {"label": label, "status": "failed", "elapsed": elapsed, "metrics": {}}

    # Locate metrics.json
    outdir = str(PROJECT_ROOT / base.get("outdir", "exps"))
    metrics_path = find_metrics_json(outdir, surrogate["name"])
    if metrics_path is None:
        print(f"  WARNING: metrics.json not found for {label}")
        return {"label": label, "status": "no_metrics", "elapsed": elapsed, "metrics": {}}

    metrics = parse_metrics(metrics_path)
    return {
        "label": label,
        "status": "ok",
        "elapsed": elapsed,
        "metrics": metrics,
        "metrics_path": str(metrics_path),
    }


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)

    # Collect all metric keys
    all_metric_keys = set()
    for r in results:
        all_metric_keys.update(r.get("metrics", {}).keys())
    mean_keys = sorted(k for k in all_metric_keys if k.endswith("_mean"))

    col_width = 30
    header = f"{'Run':<{col_width}}"
    for k in mean_keys:
        short = k.replace("_mean", "").replace("rel_error_channel", "re_ch").replace("error_rate_channel", "er_ch")
        header += f"  {short:>12}"
    header += f"  {'status':>10}  {'time(s)':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        row = f"{r['label']:<{col_width}}"
        for k in mean_keys:
            val = r.get("metrics", {}).get(k)
            row += f"  {val:>12.4f}" if val is not None else f"  {'N/A':>12}"
        row += f"  {r['status']:>10}  {r.get('elapsed', 0):>8.1f}"
        print(row)

    print("=" * 90)

    # Per-PDE ranking by channel-1 rel error (the solution channel)
    for pde in ("poisson", "helmholtz"):
        pde_results = [r for r in results if r["label"].startswith(pde) and r["status"] == "ok"]
        target_key = "rel_error_channel1_mean"
        scored = [(r, r["metrics"].get(target_key)) for r in pde_results if r["metrics"].get(target_key) is not None]
        if not scored:
            continue
        scored.sort(key=lambda x: x[1])
        print(f"\n--- {pde.upper()} ranking by {target_key} ---")
        for rank, (r, val) in enumerate(scored, 1):
            surr_name = r["label"].split("/")[1]
            note = next((s["note"] for s in SURROGATES[pde] if s["name"] == surr_name), "")
            print(f"  {rank}. {surr_name:<30}  {val:.4f}   ({note})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark all surrogate checkpoints")
    parser.add_argument("--max_size", type=int, default=12,
                        help="Max samples per run (default 12 = full test batch; use 4 for quick preview)")
    parser.add_argument("--pde", choices=["poisson", "helmholtz", "both"], default="both",
                        help="Which PDE(s) to benchmark")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    pdes = ["poisson", "helmholtz"] if args.pde == "both" else [args.pde]

    all_results = []
    for pde in pdes:
        for surrogate in SURROGATES[pde]:
            res = run_single(pde, surrogate, args.max_size, args.dry_run)
            res.setdefault("label", f"{pde}/{surrogate['name']}")
            all_results.append(res)

    print_summary(all_results)

    # Save full results to JSON
    out_path = PROJECT_ROOT / "exps" / "benchmark_surrogates_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
