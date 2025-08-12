# test_fno_surrogate.py

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from neuralop.models.fno import FNO
from training.dataset_hf import PDEDataset

# ----------------------------
# Helper functions (unchanged)
# ----------------------------
class L2Loss:
    def __call__(self, x, y):
        n = x.size(0)
        return torch.sum(torch.norm(x.view(n, -1)-y.view(n, -1),2,1) /
                         torch.norm(y.view(n, -1),2,1))

def compute_metrics(pred, targ):
    mse      = F.mse_loss(pred, targ).item()
    l2_rel   = (torch.norm(pred-targ)/torch.norm(targ)).item()
    l1_error = F.l1_loss(pred, targ).item()
    max_err  = torch.max(torch.abs(pred-targ)).item()
    return dict(mse=mse, l2_rel=l2_rel, l1_error=l1_error, max_error=max_err)

def visualize(inp, gt, pred, idx=0):
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    for a,img,title in zip(ax,
                           [inp, gt, pred],
                           ['Input','Ground Truth','Prediction']):
        im = a.imshow(img[idx,0].cpu(), cmap='viridis')
        a.set_title(title)
        fig.colorbar(im, ax=a)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Argument parsing
# ----------------------------
parser = argparse.ArgumentParser(
    description="Test an FNO surrogate (forward or inverse).")
parser.add_argument(
    "--model-path", "-m", required=True,
    help=".pth file of trained FNO (forward or inverse).")
parser.add_argument(
    "--mode", choices=["forward","inverse"], default="forward",
    help="‘forward’: param→solution; ‘inverse’: solution→param.")
parser.add_argument(
    "--data-path", "-d", default="data/DiffPDE/helmholtz_test_hf",
    help="Path to test dataset.")
parser.add_argument(
    "--batch-size", "-b", type=int, default=8,
    help="Batch size for DataLoader.")
args = parser.parse_args()


# ----------------------------
# Device & model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNO(n_modes=(64,64), in_channels=1, out_channels=1,
            hidden_channels=64, n_layers=4).to(device)
print(f"Loading {args.mode} surrogate from {args.model_path}…")
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()


# ----------------------------
# Dataset & loader
# ----------------------------
dataset = PDEDataset(path=args.data_path, resolution=128)
loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
print(f"Loaded {len(dataset)} samples from {args.data_path}")


# ----------------------------
# Evaluation loop
# ----------------------------
tot_mse = tot_l2 = tot_l1 = 0.0
max_err = 0.0

with torch.no_grad():
    for i,(data,_) in enumerate(loader):
        data = data.to(device).float()
        if args.mode == "forward":
            inp, targ = data[:,0:1], data[:,1:2]
        else:  # inverse
            inp, targ = data[:,1:2], data[:,0:1]

        pred = model(inp)

        m = compute_metrics(pred, targ)
        tot_mse += m['mse'];  tot_l2 += m['l2_rel']
        tot_l1  += m['l1_error']; max_err = max(max_err, m['max_error'])

        if i % 100 == 0:
            print(f"[Batch {i:3d}/{len(loader)}] MSE={m['mse']:.3e} L2_rel={m['l2_rel']:.3e}")

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
