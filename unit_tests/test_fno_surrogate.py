# test_fno_surrogate.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from neuralop.models.fno import FNO
from training.dataset_hf import PDEDataset
from training.networks import SongUNO


# Custom FNO_pad class to match training architecture
class FNO_pad(FNO):
    def forward(self, x):
        res_2 = x.shape[-1] // 2
        x = F.pad(x, (res_2, res_2, res_2, res_2), mode="reflect")
        ret = super().forward(x)
        ret = ret[:, :, res_2:-res_2, res_2:-res_2]
        return ret


# Wrapper class to make SongUNO compatible with direct forward prediction
class SongUNOWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = SongUNO(*args, **kwargs)

    def forward(self, x):
        # Create dummy noise_labels and class_labels for inference
        batch_size = x.shape[0]
        device = x.device

        # Create dummy noise_labels (timestep 0 for deterministic prediction)
        noise_labels = torch.zeros(batch_size, device=device)

        # Create dummy class_labels (unconditional)
        class_labels = torch.zeros(batch_size, 0, device=device)  # Empty class labels

        return self.model(x, noise_labels, class_labels)


# ----------------------------
# Helper functions (unchanged)
# ----------------------------
class L2Loss:
    def __call__(self, x, y):
        n = x.size(0)
        return torch.sum(torch.norm(x.view(n, -1) - y.view(n, -1), 2, 1) / torch.norm(y.view(n, -1), 2, 1))


def compute_metrics(pred, targ):
    mse = F.mse_loss(pred, targ).item()
    l2_rel = (torch.norm(pred - targ) / torch.norm(targ)).item()
    l1_error = F.l1_loss(pred, targ).item()
    max_err = torch.max(torch.abs(pred - targ)).item()
    return dict(mse=mse, l2_rel=l2_rel, l1_error=l1_error, max_error=max_err)


def visualize(inp, gt, pred, idx=0, save_path=None, sample_idx=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, title in zip(ax, [inp, gt, pred], ["Input", "Ground Truth", "Prediction"]):
        im = a.imshow(img[idx, 0].cpu(), cmap="viridis")
        a.set_title(title)
        fig.colorbar(im, ax=a)
    plt.tight_layout()
    plt.show()


def save_prediction_examples(inp, gt, pred, save_dir, sample_idx, mode):
    """Save prediction examples as comparison plots"""
    os.makedirs(save_dir, exist_ok=True)

    # Save comparison plot (all three side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, img, title in zip(axes, [inp, gt, pred], ["Input", "Ground Truth", "Prediction"]):
        im = ax.imshow(img[0, 0].cpu(), cmap="viridis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    comparison_filename = f"comparison_sample_{sample_idx:03d}.png"
    plt.savefig(os.path.join(save_dir, comparison_filename), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison visualization for sample {sample_idx} to {save_dir}")


# ----------------------------
# Argument parsing
# ----------------------------
parser = argparse.ArgumentParser(description="Test an FNO or SongUNO surrogate (forward or inverse).")
parser.add_argument("--model-path", "-m", required=True, help=".pth file of trained FNO or SongUNO model (forward or inverse).")
parser.add_argument("--mode", choices=["forward", "inverse"], default="forward", help="‘forward’: param→solution; ‘inverse’: solution→param.")
parser.add_argument("--data-path", "-d", default="data/DiffPDE/helmholtz_test_hf", help="Path to test dataset.")
parser.add_argument("--batch-size", "-b", type=int, default=25, help="Batch size for DataLoader.")
parser.add_argument("--output-dir", "-o", default="exps/tests/fno_surrogate", help="Directory to save prediction visualization examples.")
parser.add_argument("--save-examples", "-s", type=int, default=5, help="Number of prediction examples to save as images.")
args = parser.parse_args()


# ----------------------------
# Device & model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_state_dict(path):
    """Load checkpoint, returning (state_dict, config_or_None)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("config")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("config")
    # Raw state dict
    return ckpt, None


def _is_uno(state_dict):
    """Return True if state dict looks like SongUNO (has UNet-style keys)."""
    keys = list(state_dict.keys())
    return any(k.startswith("model.enc") or k.startswith("enc.") or "map_noise" in k for k in keys)


def _fno_dims_from_state(state_dict):
    """Infer (n_modes, hidden_channels, n_layers, in_channels, out_channels) from FNO state dict."""
    import re
    hidden_channels = 64
    in_channels = 1
    out_channels = 1
    n_modes = (64, 64)
    n_layers = 4

    # hidden_channels and n_modes from spectral conv weights: shape [hidden, hidden, m0, m1_rfft]
    # where m1_rfft = n_modes[1]//2+1, so n_modes[1] = (m1_rfft-1)*2
    for k, v in state_dict.items():
        if "fno_blocks" in k and "convs" in k and "weight" in k and v.dim() == 4:
            hidden_channels = v.shape[0]
            m0, m1 = v.shape[2], v.shape[3]
            n_modes = (m0, (m1 - 1) * 2)
            break

    # in_channels from lifting MLP first FC: shape [2*hidden, in_channels+2_grid_coords, 1]
    # FNO internally appends 2 grid coordinate channels, so subtract 2
    for k, v in state_dict.items():
        if k.startswith("lifting") and "fcs.0.weight" in k and v.dim() >= 2:
            in_channels = max(1, v.shape[1] - 2)
            break

    # out_channels from the last projection FC layer
    proj_layers = {}
    for k, v in state_dict.items():
        if k.startswith("projection") and "fcs" in k and "weight" in k:
            m = re.search(r"fcs\.(\d+)\.weight", k)
            if m:
                proj_layers[int(m.group(1))] = v
    if proj_layers:
        out_channels = proj_layers[max(proj_layers)].shape[0]

    # n_layers from fno_blocks spectral conv indices
    layer_indices = set()
    for k in state_dict.keys():
        m = re.search(r"fno_blocks\.convs\.(\d+)\.", k)
        if m:
            layer_indices.add(int(m.group(1)))
    if layer_indices:
        n_layers = max(layer_indices) + 1

    return n_modes, hidden_channels, n_layers, in_channels, out_channels


def _build_model(path):
    """Detect architecture from checkpoint and return a ready-to-load model."""
    state_dict, cfg = _load_state_dict(path)

    # 1. Config key present (training checkpoint) — most reliable
    if cfg is not None:
        n_modes = tuple(cfg.get("fno_modes", [64, 64]))
        hidden = cfg.get("hidden_channels", 64)
        n_layers = cfg.get("n_layers", 4)
        in_ch = cfg.get("in_channels", 1)
        out_ch = cfg.get("out_channels", 1)
        print(f"Architecture from checkpoint config: FNO_pad n_modes={n_modes} hidden={hidden} n_layers={n_layers}")
        return FNO_pad(n_modes=n_modes, in_channels=in_ch, out_channels=out_ch,
                       hidden_channels=hidden, n_layers=n_layers).to(device), state_dict

    # 2. Probe state dict keys
    if _is_uno(state_dict):
        print("Detected SongUNO from state dict keys.")
        model = SongUNOWrapper(
            img_resolution=64, in_channels=1, out_channels=1,
            fmult=0.5, rank=0.15, model_channels=64,
            channel_mult=[1, 2, 2], num_blocks=2,
            attn_resolutions=[16], dropout=0.10, cond=False,
        ).to(device)
        return model, state_dict

    n_modes, hidden, n_layers, in_ch, out_ch = _fno_dims_from_state(state_dict)
    print(f"Inferred FNO_pad from state dict: n_modes={n_modes} hidden={hidden} n_layers={n_layers}")
    return FNO_pad(n_modes=n_modes, in_channels=in_ch, out_channels=out_ch,
                   hidden_channels=hidden, n_layers=n_layers).to(device), state_dict


print(f"Loading {args.mode} surrogate from {args.model_path}…")
model, state_dict = _build_model(args.model_path)
model.load_state_dict(state_dict)
print("Model loaded successfully.")
model.eval()


# ----------------------------
# Dataset & loader
# ----------------------------
dataset = PDEDataset(path=args.data_path, resolution=128)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
print(f"Loaded {len(dataset)} samples from {args.data_path}")

# Create output directory for visualizations
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
print(f"Will save prediction visualizations to: {output_dir}")

# ----------------------------
# Evaluation loop
# ----------------------------
tot_mse = tot_l2 = tot_l1 = 0.0
max_err = 0.0
examples_saved = 0

with torch.no_grad():
    for i, (data, _) in enumerate(loader):
        data = data.to(device).float()
        if args.mode == "forward":
            inp, targ = data[:, 0:1], data[:, 1:2]
        else:  # inverse
            inp, targ = data[:, 1:2], data[:, 0:1]

        pred = model(inp)

        m = compute_metrics(pred, targ)
        tot_mse += m["mse"]
        tot_l2 += m["l2_rel"]
        tot_l1 += m["l1_error"]
        max_err = max(max_err, m["max_error"])

        if i % 100 == 0:
            print(f"[Batch {i:3d}/{len(loader)}] MSE={m['mse']:.3e} L2_rel={m['l2_rel']:.3e}")

        # Save visualization examples
        if examples_saved < args.save_examples:
            print(f"Saving visualization examples for sample {examples_saved + 1}...")
            save_prediction_examples(inp, targ, pred, output_dir, examples_saved + 1, args.mode)
            examples_saved += 1

        if i < 3:
            print(f"Visualizing sample {i+1} ({args.mode}):")
            visualize(inp, targ, pred, idx=0)

# ----------------------------
# Print summary
# ----------------------------
n_batches = len(loader)
print("\n--- Performance Summary ---")
print(f"Mode: {args.mode}")
print(f"Avg MSE:       {tot_mse/n_batches}")
print(f"Avg L2 rel err:{tot_l2/n_batches}")
print(f"Avg L1 err:    {tot_l1/n_batches}")
print(f"Max err (any): {max_err}")
print("---------------------------")
