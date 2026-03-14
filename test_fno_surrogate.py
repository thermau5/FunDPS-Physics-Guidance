# test_fno_surrogate.py

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
parser.add_argument("--output-dir", "-o", default="prediction_visualizations", help="Directory to save prediction visualization examples.")
parser.add_argument("--save-examples", "-s", type=int, default=5, help="Number of prediction examples to save as images.")
args = parser.parse_args()


# ----------------------------
# Device & model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Automatically detect model type based on filename and create appropriate architecture
model_path = args.model_path
if "uno_trained" in model_path:
    print("Detected SongUNO model, creating SongUNOWrapper...")
    model = SongUNOWrapper(
        img_resolution=64,
        in_channels=1,
        out_channels=1,
        fmult=0.5,
        rank=0.15,
        model_channels=64,
        channel_mult=[1, 2, 2],
        num_blocks=2,
        attn_resolutions=[16],
        dropout=0.10,
        cond=False,
    ).to(device)
elif "fno_pad_trained" in model_path:
    print("Detected FNO_pad model, creating FNO_pad architecture...")
    model = FNO_pad(n_modes=(32, 32), in_channels=1, out_channels=1, hidden_channels=64, n_layers=4).to(device)
elif "fno_trained" in model_path and "fno_pad_trained" not in model_path:
    # Standard FNO models should use FNO architecture
    print("Detected FNO model, creating standard FNO architecture...")
    model = FNO(n_modes=(64, 64), in_channels=1, out_channels=1, hidden_channels=64, n_layers=4).to(device)
else:
    # Default to FNO_pad for unknown FNO models (safer for PDEs)
    print("Model type unclear, defaulting to FNO_pad architecture...")
    model = FNO_pad(n_modes=(64, 64), in_channels=1, out_channels=1, hidden_channels=64, n_layers=4).to(device)

print(f"Loading {args.mode} surrogate from {model_path}…")
# Final model loading (architecture already determined and tested above)
try:
    # Check if weights_only is supported (PyTorch >= 1.13.0)
    if hasattr(torch, "__version__") and torch.__version__ >= "1.13.0":
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print("Model loaded successfully with weights_only=True")
    else:
        state_dict = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print("Model loaded successfully (PyTorch < 1.13.0, no weights_only)")
except Exception as e:
    print(f"weights_only=True failed: {e}")
    print("Trying with weights_only=False (less secure but compatible)...")
    # Fallback to weights_only=False for older model formats
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    print("Model loaded successfully with weights_only=False")
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
