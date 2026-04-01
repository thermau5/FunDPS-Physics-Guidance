import warnings

warnings.filterwarnings("ignore", "Using a non-tuple sequence for multidimensional indexing is deprecated")
warnings.filterwarnings("ignore", "Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9")

import argparse
import os
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neuralop.models.fno import FNO
from training.networks import SongUNO
import wandb
import numpy as np
from scipy.io import loadmat
from training.dataset_hf import PDEDataset
import time
import psutil
from tqdm import tqdm
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNO_pad no-surrogate model.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/training/no_surrogate/fno_pad_helmholtz_forward.yml",
        help="Path to YAML config for FNO training.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional suffix to append to the W&B run name.",
    )
    return parser.parse_args()


args = parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Dataset configuration (from YAML)
PDE_DIRECTION = cfg["pde_direction"]  # 'forward' or 'inverse'
DATASET_NAME = cfg["dataset_name"]  # Dataset name for saving/loading models
DATA_RESOLUTION = cfg["data_resolution"]  # Original data resolution
TRAIN_RESOLUTION = tuple(cfg["train_resolution"])  # Training resolution (downsampled)

# Training configuration
BATCH_SIZE = cfg["batch_size"]
NUM_EPOCHS = cfg["num_epochs"]
LEARNING_RATE = float(cfg["learning_rate"])
EVAL_EVERY = cfg.get("eval_every", 10)
SUMMARY_EVERY = cfg.get("summary_every", EVAL_EVERY)
CHECKPOINT_EVERY = int(cfg.get("checkpoint_every", 100))

# Model configuration
FNO_MODES = tuple(cfg["fno_modes"])
IN_CHANNELS = cfg["in_channels"]
OUT_CHANNELS = cfg["out_channels"]
HIDDEN_CHANNELS = cfg["hidden_channels"]
N_LAYERS = cfg["n_layers"]

# Wandb / logging configuration
WANDB_PROJECT = cfg.get("wandb_project", "ddis_fno_edm_no")
EXPS_OUTDIR = cfg.get("exps_outdir", "exps/no_surrogate/fno_pad")
USE_TQDM = cfg.get("use_tqdm", True)
PROGRESS_EVERY_EPOCHS = cfg.get("progress_every_epochs", 1)
RESUME = cfg.get("resume")  # Path to checkpoint_epoch_*.pth to resume from (optional)

# ==============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FNO_pad(FNO):
    def forward(self, x):
        res_2 = x.shape[-1] // 2
        x = F.pad(x, (res_2, res_2, res_2, res_2), mode="reflect")
        ret = super().forward(x)
        ret = ret[:, :, res_2:-res_2, res_2:-res_2]
        return ret


class SongUNOWrapper(torch.nn.Module):
    """Wrapper to make SongUNO compatible with direct forward prediction"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = SongUNO(*args, **kwargs)

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        noise_labels = torch.zeros(batch_size, device=device)
        return self.model(x, noise_labels, None)


# Data loading
train_dataset = PDEDataset(path=f"data/DiffPDE/{DATASET_NAME}_hf", resolution=DATA_RESOLUTION)  # To change
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = PDEDataset(path=f"data/DiffPDE/{DATASET_NAME}_test_hf", resolution=DATA_RESOLUTION)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Model initialization
model = FNO_pad(n_modes=FNO_MODES, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, n_layers=N_LAYERS)
model = model.to(device)

# Mixed Resolution Training: Load res = 64 model
# try:
#     # PyTorch 2.6+ approach: add safe globals and use weights_only=True
#     import torch.serialization
#     torch.serialization.add_safe_globals(['torch._C._nn.gelu'])
#     model.load_state_dict(torch.load(f"artifacts/models/legacy/fno_pad_trained_{PDE_DIRECTION}_{DATASET_NAME}_128_200.pth", weights_only=True))
#     print("Successfully loaded checkpoint with weights_only=True")
# except Exception as e:
#     print(f"Loading with weights_only=True failed: {e}")
#     print("Trying with weights_only=False (trusted source only)...")
#     # Fallback to weights_only=False for compatibility with older checkpoints
#     model.load_state_dict(torch.load(f"artifacts/models/legacy/fno_pad_trained_{PDE_DIRECTION}_{DATASET_NAME}_128_200.pth", weights_only=False))
#     print("Successfully loaded checkpoint with weights_only=False")

model = model.to(device)


class L2Loss(object):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
        return torch.sum(diff_norms / y_norms)


def evaluate_test_accuracy(model, test_loader, criterion, device):
    """Evaluate model on test set and return average loss."""
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for test_data, _ in test_loader:
            test_data = test_data.to(device).float()
            if PDE_DIRECTION == "forward":
                inputs, ground_truths = test_data[:, 0:1, :, :], test_data[:, 1:2, :, :]
            elif PDE_DIRECTION == "inverse":
                inputs, ground_truths = test_data[:, 1:2, :, :], test_data[:, 0:1, :, :]

            # Downsample to match training resolution
            inputs = torch.nn.functional.interpolate(inputs, size=TRAIN_RESOLUTION, mode="area")
            ground_truths = torch.nn.functional.interpolate(ground_truths, size=TRAIN_RESOLUTION, mode="area")

            outputs = model(inputs)
            loss = criterion(outputs, ground_truths)
            test_loss += loss.item()

    return test_loss / len(test_loader.dataset)


# Training setup (criterion, optimizer) before resume so we can restore optimizer state
criterion = L2Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize wandb
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

run_name = (
    f"fno_{DATASET_NAME}_{PDE_DIRECTION}"
    f"_bs{BATCH_SIZE}"
    f"_ep{NUM_EPOCHS}"
    f"_m{FNO_MODES[0]}"
    f"_p{num_params/1e6:.0f}M"
)

if args.name is not None and args.name != "":
    run_name = f"{run_name}_{args.name}"

wandb.init(
    project=WANDB_PROJECT,
    name=run_name,
    config={
        "dataset": DATASET_NAME,
        "pde_direction": PDE_DIRECTION,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "resolution": TRAIN_RESOLUTION,
        "model_type": "FNO_pad",
        "fno_modes": FNO_MODES,
        "hidden_channels": HIDDEN_CHANNELS,
        "n_layers": N_LAYERS,
        "num_params": num_params,
        "config_path": args.config,
    },
)

# Resume from checkpoint or create new run directory
start_epoch = 0
if RESUME and os.path.isfile(RESUME):
    ckpt = torch.load(RESUME, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"]
    run_dir = os.path.abspath(os.path.dirname(RESUME))
    print(f"Resuming from epoch {start_epoch} (checkpoint: {RESUME})")
    print(f"Will train from epoch {start_epoch + 1} to {NUM_EPOCHS} (total {NUM_EPOCHS - start_epoch} epochs).")
else:
    if RESUME:
        print(f"Resume path not found or not a file: {RESUME}, starting from epoch 0.")
    formatted_time = datetime.fromtimestamp(wandb.run.start_time).strftime("%m%d_%H%M%S")
    desc = f"{formatted_time}-{wandb.run.name}-{wandb.run.id}"
    run_dir = os.path.join(EXPS_OUTDIR, desc)

os.makedirs(run_dir, exist_ok=True)
print(f"Experiment directory (no-surrogate FNO): {run_dir}")


# Training loop
total_batches = len(train_loader)
print(f"Starting training with {NUM_EPOCHS} epochs, {total_batches} batches per epoch")
print(f"Total training samples: {len(train_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")

# Estimate total training time by running a few warmup batches
print("\nWarming up model and estimating training time...")
model.train()
warmup_batches = min(5, total_batches)
warmup_times = []

for i, (warmup_data, _) in enumerate(train_loader):
    if i >= warmup_batches:
        break

    warmup_start = time.time()
    warmup_data = warmup_data.to(device).float()
    if PDE_DIRECTION == "forward":
        inputs, targets = warmup_data[:, 0:1, :, :], warmup_data[:, 1:2, :, :]
    elif PDE_DIRECTION == "inverse":
        inputs, targets = warmup_data[:, 1:2, :, :], warmup_data[:, 0:1, :, :]

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    warmup_time = time.time() - warmup_start
    warmup_times.append(warmup_time)
    print(f"  Warmup batch {i+1}: {warmup_time:.3f}s")

if warmup_times:
    avg_batch_time = sum(warmup_times) / len(warmup_times)
    estimated_epoch_time = avg_batch_time * total_batches
    estimated_total_time = estimated_epoch_time * NUM_EPOCHS
    print("\nEstimated timing:")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Estimated epoch time: {estimated_epoch_time:.1f}s ({estimated_epoch_time / 60:.1f} minutes)")
    print(f"  Estimated total time: {estimated_total_time:.1f}s ({estimated_total_time / 60:.1f} minutes)")

print("-" * 80)

# Save an initial checkpoint at epoch 0 (skip when resuming)
if start_epoch == 0:
    initial_ckpt_path = os.path.join(run_dir, "checkpoint_epoch_0.pth")
    torch.save(
        {
            "epoch": 0,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
        },
        initial_ckpt_path,
    )
    print(f"Saved initial checkpoint to: {initial_ckpt_path}")

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    batch_losses = []

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Starting...")

    # Only use tqdm / detailed batch prints on selected epochs
    use_progress_this_epoch = USE_TQDM and (
        (epoch + 1) % PROGRESS_EVERY_EPOCHS == 0 or epoch == start_epoch or (epoch + 1) == NUM_EPOCHS
    )

    if use_progress_this_epoch:
        data_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False, ncols=120)
    else:
        data_iter = train_loader

    for batch_idx, (train_data, _) in enumerate(data_iter):
        batch_start_time = time.time()

        train_data = train_data.to(device).float()
        if PDE_DIRECTION == "forward":
            inputs, ground_truths = train_data[:, 0:1, :, :], train_data[:, 1:2, :, :]
        elif PDE_DIRECTION == "inverse":
            inputs, ground_truths = train_data[:, 1:2, :, :], train_data[:, 0:1, :, :]

        # downsample the inputs and ground_truths
        inputs = torch.nn.functional.interpolate(inputs, size=TRAIN_RESOLUTION, mode="area")
        ground_truths = torch.nn.functional.interpolate(ground_truths, size=TRAIN_RESOLUTION, mode="area")

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss using the custom L2Loss function
        loss = criterion(outputs, ground_truths)

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        batch_losses.append(loss.item())

        # Update progress bar with current loss and ETA (only when progress is enabled)
        if use_progress_this_epoch and len(batch_losses) > 0:
            current_loss = batch_losses[-1]
            avg_loss = sum(batch_losses) / len(batch_losses)

            # Memory usage for progress bar
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
                memory_str = f"GPU:{gpu_memory:.1f}GB"
            else:
                process = psutil.Process(os.getpid())
                memory_str = f"RAM:{process.memory_info().rss / 1024**3:.1f}GB"

            data_iter.set_postfix({"Loss": f"{current_loss:.4f}", "Avg": f"{avg_loss:.4f}", "Mem": memory_str})

        # Detailed progress update every N batches for console output (very coarse now)
        if use_progress_this_epoch:
            N_batches_to_report = max(total_batches // 4, 1)  # ~4 prints per epoch
            if (batch_idx + 1) % N_batches_to_report == 0 or (batch_idx + 1) == total_batches:
                batch_time = time.time() - batch_start_time
                avg_batch_loss = sum(batch_losses[-20:]) / min(20, len(batch_losses[-20:]))

                progress = (batch_idx + 1) / total_batches * 100
                eta_epoch = batch_time * (total_batches - batch_idx - 1)

                print(
                    f"\nBatch {batch_idx+1:3d} ({progress:3.1f}%) | "
                    f"Loss: {avg_batch_loss:.4f} | "
                    f"Batch time: {batch_time:.3f}s | "
                    f"ETA epoch: {eta_epoch:.1f}s | "
                    f"{memory_str}"
                )

    if use_progress_this_epoch and hasattr(data_iter, "close"):
        data_iter.close()

    # Epoch summary
    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(train_dataset)
    avg_batch_loss = sum(batch_losses) / len(batch_losses)

    # Evaluate on test set every EVAL_EVERY epochs
    test_loss = None
    if (epoch + 1) % EVAL_EVERY == 0:
        print("\nEvaluating on test set...")
        test_loss = evaluate_test_accuracy(model, test_loader, criterion, device)

    # Print a compact summary every SUMMARY_EVERY epochs
    if (epoch + 1) % SUMMARY_EVERY == 0:
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"  Training Loss: {epoch_loss:.6f}")
        print(f"  Average Batch Loss: {avg_batch_loss:.6f}")
        if test_loss is not None:
            print(f"  Test Loss: {test_loss:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print(f"  Time per batch: {epoch_time / total_batches:.3f}s")

    # Log metrics to wandb every epoch (test_loss only when available)
    log_dict = {
        "epoch": epoch + 1,
        "train_loss": epoch_loss,
        "avg_batch_loss": avg_batch_loss,
        "epoch_time": epoch_time,
        "time_per_batch": epoch_time / total_batches,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
    if test_loss is not None:
        log_dict["test_loss"] = test_loss
    wandb.log(log_dict)

    # Periodic checkpoints + always last epoch (for resume when num_epochs is not a multiple of N)
    save_ckpt = (epoch + 1) == NUM_EPOCHS or (
        CHECKPOINT_EVERY > 0 and (epoch + 1) % CHECKPOINT_EVERY == 0
    )
    if save_ckpt:
        ckpt_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to: {ckpt_path}")

    print("-" * 80)

# Save final model to model/no_surrogate/fno_pad/...
model_dir = os.path.join(
    "model",
    "no_surrogate",
    "fno_pad",
    f"{PDE_DIRECTION}_{DATASET_NAME}",
    str(list(TRAIN_RESOLUTION)[0]),
    str(NUM_EPOCHS),
)
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(
    model_dir,
    f"fno_trained_{NUM_EPOCHS}_{wandb.run.id}.pth", # wandb.run.id is the run ID to avoid covering, need to be removed for the final model path
)
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# Log final model artifact to wandb
wandb.save(model_path)

# Final evaluation
print("\nFinal evaluation on test set:")
final_test_loss = evaluate_test_accuracy(model, test_loader, criterion, device)
print(f"Final Test Loss: {final_test_loss:.6f}")

# Log final test results and finish wandb
wandb.log({"final_test_loss": final_test_loss})
wandb.finish()
