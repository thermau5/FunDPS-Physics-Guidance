#!/usr/bin/env python
"""Collect DDIS sweep results from exps/, write CSV and plot surrogate convergence."""
import json
import os
import re
import csv
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neuralop.models.fno import FNO

from training.dataset_hf import PDEDataset
from training.networks import SongUNO

OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "outputs"))
CONFIG_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "sweep_config.yaml"))

EXPS_DIR = os.path.join(PROJECT_ROOT, "exps")
OUT_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")
OUT_PNG_COMBINED = os.path.join(OUTPUT_DIR, "surrogate_convergence_with_fno_train.png")

# Default values used when YAML config is unavailable/invalid.
DEFAULT_PDES = ["poisson", "helmholtz", "ns-nonbounded"]
DEFAULT_CKPT_DIRS = [
    "exps/no_surrogate/fno_pad/0310_173643-fno_poisson_forward_bs40_ep1000_m64_p35M-ucykqi42",
    "exps/no_surrogate/fno_pad/0310_173644-fno_helmholtz_forward_bs40_ep1000_m64_p35M-ij336nmf",
    "exps/no_surrogate/fno_pad/0310_173643-fno_ns-nonbounded_forward_bs40_ep1000_m64_p35M-qdwh3y7f",
]
DEFAULT_EPOCHS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def load_visualization_config():
    pdes = DEFAULT_PDES
    ckpt_dirs = DEFAULT_CKPT_DIRS
    epochs = DEFAULT_EPOCHS
    sweep_exps_outdir = "exps/surrogate_sweep"

    cfg_file = Path(CONFIG_PATH)
    if not cfg_file.is_file():
        print(f"Config not found at {CONFIG_PATH}; using built-in defaults.")
        return pdes, ckpt_dirs, epochs, sweep_exps_outdir

    try:
        import yaml

        with cfg_file.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        vis_cfg = raw.get("visualization", {})
        if vis_cfg.get("sweep_exps_outdir"):
            sweep_exps_outdir = str(vis_cfg["sweep_exps_outdir"])
        pde_items = vis_cfg.get("pdes", [])
        cfg_epochs = vis_cfg.get("epochs", [])

        loaded_pdes = []
        loaded_ckpts = []
        for item in pde_items:
            name = item.get("name")
            ckpt = item.get("checkpoint_dir")
            if not name or not ckpt:
                continue
            loaded_pdes.append(str(name))
            loaded_ckpts.append(str(ckpt))

        loaded_epochs = [int(e) for e in cfg_epochs]
        if loaded_pdes and loaded_ckpts and len(loaded_pdes) == len(loaded_ckpts) and loaded_epochs:
            return loaded_pdes, loaded_ckpts, loaded_epochs, sweep_exps_outdir
        print(f"Incomplete config in {CONFIG_PATH}; using built-in defaults.")
    except Exception as exc:
        print(f"Failed to parse {CONFIG_PATH} ({exc}); using built-in defaults.")

    return pdes, ckpt_dirs, epochs, sweep_exps_outdir


PDES, CKPT_DIRS, EPOCHS, SWEEP_EXPS_OUTDIR = load_visualization_config()


def _sweep_outdir_abs():
    return (
        SWEEP_EXPS_OUTDIR
        if os.path.isabs(SWEEP_EXPS_OUTDIR)
        else os.path.join(PROJECT_ROOT, SWEEP_EXPS_OUTDIR)
    )


def _collect_experiments_one_level(parent_dir: str):
    """Immediate subdirs of parent_dir that contain metrics.json (legacy top-level exps)."""
    results = []
    if not os.path.isdir(parent_dir):
        return results
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        metrics_path = os.path.join(full, "metrics.json")
        if os.path.isfile(metrics_path):
            is_sweep = "sweep_ep" in name
            results.append((full, is_sweep))
    return results


def _collect_experiments_under_sweep_root(sweep_root: str, max_depth: int = 4):
    """Find experiment dirs (metrics.json) under sweep_exps_outdir.

    Supports flat, session (G0320_jobid/), and optional session/pde/ layouts without
    walking the whole repo.
    """
    results = []
    if not os.path.isdir(sweep_root):
        return results
    sweep_root = os.path.normpath(sweep_root)
    for dirpath, dirnames, filenames in os.walk(sweep_root, topdown=True):
        rel = os.path.relpath(dirpath, sweep_root)
        depth = 0 if rel in (".", "") else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        if "metrics.json" in filenames:
            name = os.path.basename(dirpath)
            results.append((dirpath, "sweep_ep" in name))
            dirnames[:] = []
    return results


def find_experiments():
    """Return list of (exp_dir, is_sweep_named) tuples."""
    exps = []
    exps_root = os.path.join(PROJECT_ROOT, "exps")
    exps.extend(_collect_experiments_one_level(exps_root))

    sweep_path = _sweep_outdir_abs()
    if os.path.normpath(sweep_path) != os.path.normpath(exps_root):
        exps.extend(_collect_experiments_under_sweep_root(sweep_path, max_depth=4))
    return exps


EPOCH_RE = re.compile(r"checkpoint_epoch_(\d+)\.pth")


def extract_epoch_from_config(config_path):
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        return None

    surrogate_path = cfg.get("surrogate_path")
    if not surrogate_path:
        return None

    m = EPOCH_RE.search(surrogate_path)
    if not m:
        return None
    return int(m.group(1))


def extract_pde_from_config(config_path):
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        return None
    return cfg.get("dataset")


def parse_name_for_epoch_pde(dirname):
    """Try to parse from 'sweep_ep{epoch}_{pde}' pattern."""
    base = os.path.basename(dirname)
    m = re.search(r"sweep_ep(\d+)_([A-Za-z0-9\-]+)", base)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def gather_results():
    exps = find_experiments()
    results = []

    if any(is_sweep for _, is_sweep in exps):
        print("Found sweep_ep experiments; using name-based parsing.")
        use_name = True
    else:
        print("No sweep_ep experiments found; falling back to config.json parsing.")
        use_name = False

    for exp_dir, is_sweep in exps:
        metrics_path = os.path.join(exp_dir, "metrics.json")
        config_path = os.path.join(exp_dir, "config.json")
        if not os.path.isfile(metrics_path) or not os.path.isfile(config_path):
            continue
        try:
            mtime = os.path.getmtime(metrics_path)
        except OSError:
            mtime = 0

        # Determine epoch and PDE
        if use_name and is_sweep:
            epoch, pde = parse_name_for_epoch_pde(exp_dir)
            if epoch is None or pde is None:
                epoch = extract_epoch_from_config(config_path)
                pde = extract_pde_from_config(config_path)
        else:
            epoch = extract_epoch_from_config(config_path)
            pde = extract_pde_from_config(config_path)

        if epoch is None or pde is None:
            continue

        with open(metrics_path, "r") as f:
            m = json.load(f)

        ch0_mean = m.get("rel_error_channel0_mean")
        ch0_std = m.get("rel_error_channel0_std")
        ch1_mean = m.get("rel_error_channel1_mean")
        ch1_std = m.get("rel_error_channel1_std")

        # Spectral relative errors (channel 0/1).
        # Try a few possible key variants to be robust to naming.
        spec_ch0 = (
            m.get("spectral_rel_error_channel0_geometric_mean")
            or m.get("spectral_rel_error_channel0_geo_mean")
            or m.get("spectral_rel_error_channel0_mean")
        )
        spec_ch1 = (
            m.get("spectral_rel_error_channel1_geometric_mean")
            or m.get("spectral_rel_error_channel1_geo_mean")
            or m.get("spectral_rel_error_channel1_mean")
        )

        results.append(
            {
                "pde": pde,
                "epoch": epoch,
                "rel_error_ch0_mean": ch0_mean,
                "rel_error_ch0_std": ch0_std,
                "rel_error_ch1_mean": ch1_mean,
                "rel_error_ch1_std": ch1_std,
                "spectral_rel_error_ch0": spec_ch0,
                "spectral_rel_error_ch1": spec_ch1,
                "exp_dir": os.path.relpath(exp_dir, PROJECT_ROOT),
                "_mtime": mtime,
            }
        )

    # Deduplicate (pde, epoch): keep the newest run (by metrics.json mtime)
    by_key = defaultdict(list)
    for r in results:
        by_key[(r["pde"], r["epoch"])].append(r)
    results = []
    for (pde, epoch), group in sorted(by_key.items()):
        best = max(group, key=lambda r: r["_mtime"])
        best = {k: v for k, v in best.items() if k != "_mtime"}
        results.append(best)

    return results


class FNO_pad(FNO):
    def forward(self, x):
        res_2 = x.shape[-1] // 2
        x = F.pad(x, (res_2, res_2, res_2, res_2), mode="reflect")
        ret = super().forward(x)
        ret = ret[:, :, res_2:-res_2, res_2:-res_2]
        return ret


class SongUNOWrapper(torch.nn.Module):
    """Wrapper to make SongUNO compatible with direct forward prediction."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = SongUNO(*args, **kwargs)

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        noise_labels = torch.zeros(batch_size, device=device)
        return self.model(x, noise_labels, None)


class L2Loss(object):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
        return torch.sum(diff_norms / y_norms)


def _evaluate_test_loss(model, test_loader, criterion, device) -> float:
    """Evaluate model on test set and return average loss (same as training_fno)."""
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for test_data, _ in test_loader:
            test_data = test_data.to(device).float()
            PDE_DIRECTION = "forward"  # all three runs in CKPT_DIRS are forward
            if PDE_DIRECTION == "forward":
                inputs, ground_truths = test_data[:, 0:1, :, :], test_data[:, 1:2, :, :]
            elif PDE_DIRECTION == "inverse":
                inputs, ground_truths = test_data[:, 1:2, :, :], test_data[:, 0:1, :, :]

            inputs = torch.nn.functional.interpolate(inputs, size=(64, 64), mode="area")
            ground_truths = torch.nn.functional.interpolate(ground_truths, size=(64, 64), mode="area")

            outputs = model(inputs)
            loss = criterion(outputs, ground_truths)
            test_loss += loss.item()

    return test_loss / len(test_loader.dataset)


def _build_data_loader_from_cfg(cfg, batch_size: int, device: torch.device) -> DataLoader:
    """Create the test DataLoader matching training_fno settings for this config."""
    dataset_name = cfg["dataset_name"]
    data_resolution = cfg["data_resolution"]
    test_dataset = PDEDataset(path=f"data/DiffPDE/{dataset_name}_test_hf", resolution=data_resolution)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def _build_model_from_cfg(cfg, device: torch.device) -> torch.nn.Module:
    """Instantiate FNO_pad model consistent with training config."""
    fno_modes = tuple(cfg["fno_modes"])
    in_channels = cfg["in_channels"]
    out_channels = cfg["out_channels"]
    hidden_channels = cfg["hidden_channels"]
    n_layers = cfg["n_layers"]

    model = FNO_pad(
        n_modes=fno_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
    )
    return model.to(device)


def _compute_fno_test_loss() -> dict:
    """Compute FNO TEST loss vs epoch for each PDE by evaluating checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for FNO test loss: {device}")

    criterion = L2Loss()
    result: dict = {}

    for pde, rel_ckpt_dir in zip(PDES, CKPT_DIRS):
        ckpt_dir = os.path.join(PROJECT_ROOT, rel_ckpt_dir)
        print(f"\nProcessing PDE '{pde}' in checkpoint dir: {ckpt_dir}")

        if not os.path.isdir(ckpt_dir):
            print(f"  WARNING: checkpoint directory not found, skipping: {ckpt_dir}")
            continue

        test_losses = []
        epochs_found = []

        base_cfg = None
        test_loader = None
        model = None

        for epoch in EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
            if not os.path.isfile(ckpt_path):
                print(f"  NOTE: checkpoint not found for epoch {epoch}: {ckpt_path}")
                continue

            print(f"  Loading checkpoint for epoch {epoch}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            if base_cfg is None:
                base_cfg = ckpt.get("config")
                if base_cfg is None:
                    raise RuntimeError(f"Config not found in checkpoint: {ckpt_path}")
                test_loader = _build_data_loader_from_cfg(base_cfg, batch_size=base_cfg["batch_size"], device=device)
                model = _build_model_from_cfg(base_cfg, device=device)

            assert model is not None and test_loader is not None

            model.load_state_dict(ckpt["model_state"])

            test_loss = _evaluate_test_loss(model, test_loader, criterion, device)
            print(f"    Epoch {epoch}: test loss = {test_loss:.6f}")

            epochs_found.append(epoch)
            test_losses.append(float(test_loss))

        if not epochs_found:
            print(f"  No checkpoints found for PDE '{pde}', skipping.")
            continue

        import numpy as np

        order = np.argsort(np.asarray(epochs_found))
        epochs_sorted = [int(epochs_found[i]) for i in order]
        losses_sorted = [float(test_losses[i]) for i in order]

        result[pde] = {
            "epochs": epochs_sorted,
            "loss": losses_sorted,
        }

    if not result:
        print("No FNO test loss data collected.")
        return None
    return result


def write_csv(results):
    results = sorted(results, key=lambda r: (r["pde"], r["epoch"]))
    fields = [
        "pde",
        "epoch",
        "rel_error_ch0_mean",
        "rel_error_ch0_std",
        "rel_error_ch1_mean",
        "rel_error_ch1_std",
        "spectral_rel_error_ch0",
        "spectral_rel_error_ch1",
        "exp_dir",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"Wrote {len(results)} rows to {OUT_CSV}")


def print_summary(results):
    by_pde = defaultdict(list)
    for r in results:
        by_pde[r["pde"]].append(r)

    print("\n=== Surrogate convergence summary ===")
    for pde, rows in sorted(by_pde.items()):
        rows = sorted(rows, key=lambda r: r["epoch"])
        print(f"\nPDE: {pde}")
        # Nicely aligned table header and rows: one space between padded columns
        header = (
            f"{'epoch':>6} "
            f"{'ch0_mean':>10} "
            f"{'ch0_std':>10} "
            f"{'ch1_mean':>10} "
            f"{'ch1_std':>10}"
        )
        print(header)
        for r in rows:
            print(
                f"{r['epoch']:6d} "
                f"{r['rel_error_ch0_mean']:10.4f} "
                f"{r['rel_error_ch0_std']:10.4f} "
                f"{r['rel_error_ch1_mean']:10.4f} "
                f"{r['rel_error_ch1_std']:10.4f}"
            )


def _maybe_load_fno_train_loss():
    """Compute FNO TEST loss curves from checkpoints for each PDE.

    Returns dict[pde] -> {"epochs": [...], "loss": [...]} or None if computation fails.
    """
    return _compute_fno_test_loss()


def make_plot(results):
    import numpy as np

    if not results:
        print("No results to plot.")
        return

    by_pde = defaultdict(list)
    for r in results:
        by_pde[r["pde"]].append(r)

    color_map = {
        "poisson": "blue",
        "helmholtz": "red",
        "ns-nonbounded": "green",
    }

    # Load optional FNO TEST-loss curves (same PDE keys and epoch grid).
    fno_train = _maybe_load_fno_train_loss()

    plt.figure(figsize=(8, 5))

    # Exclude epoch 0 so the y-axis isn't dominated by initial high error.
    PLOT_MIN_EPOCH = 10

    ax1 = plt.gca()

    for pde, rows in sorted(by_pde.items()):
        rows = sorted(rows, key=lambda r: r["epoch"])
        rows = [r for r in rows if r["epoch"] >= PLOT_MIN_EPOCH]
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        mean = [r["rel_error_ch0_mean"] * 100.0 for r in rows]  # convert to %
        std = [r["rel_error_ch0_std"] * 100.0 for r in rows]

        color = color_map.get(pde, None)
        label = pde

        epochs_arr = np.array(epochs)
        mean_arr = np.array(mean)
        std_arr = np.array(std)

        ax1.plot(epochs_arr, mean_arr, color=color, label=f"DDIS {pde}")
        ax1.fill_between(
            epochs_arr,
            mean_arr - std_arr,
            mean_arr + std_arr,
            color=color,
            alpha=0.2,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("DDIS relative error channel 0 (%)")
    ax1.set_yscale("log")
    # Focus y-axis on converged range so curves are distinguishable
    plot_means = [
        r["rel_error_ch0_mean"] * 100.0
        for r in results
        if r["epoch"] >= PLOT_MIN_EPOCH and r.get("rel_error_ch0_mean")
    ]
    if plot_means:
        ax1.set_ylim(max(1e-2, min(plot_means) * 0.5), max(plot_means) * 1.5)
    ax1.grid(True, alpha=0.3)

    lines, labels = ax1.get_legend_handles_labels()

    # Optionally overlay FNO test loss on a second y-axis
    if fno_train is not None:
        ax2 = ax1.twinx()
        for pde, series in sorted(fno_train.items()):
            # Apply the same minimum-epoch filter to keep curves comparable.
            epochs_arr = np.array(series["epochs"])
            loss_arr = np.array(series["loss"])
            mask = epochs_arr >= PLOT_MIN_EPOCH
            epochs_arr = epochs_arr[mask]
            loss_arr = loss_arr[mask]
            if epochs_arr.size == 0:
                continue
            color = color_map.get(pde, None)
            ax2.plot(
                epochs_arr,
                loss_arr,
                color=color,
                linestyle="--",
                label=f"FNO test loss {pde}",
            )
        ax2.set_ylabel("FNO test loss")
        ax2.set_yscale("log")

        l2, lab2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += lab2

    plt.title("Surrogate convergence vs. FNO loss")
    plt.legend(lines, labels, loc="best")
    plt.tight_layout()

    # Single combined PNG: DDIS + FNO training loss
    plt.savefig(OUT_PNG_COMBINED, dpi=300)
    print(f"Saved combined plot to {OUT_PNG_COMBINED}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = gather_results()
    if not results:
        print("No experiments with metrics.json found; nothing to collect.")
        return
    write_csv(results)
    print_summary(results)
    make_plot(results)


if __name__ == "__main__":
    main()
