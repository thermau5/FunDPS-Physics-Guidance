import warnings

warnings.filterwarnings("ignore", "Using a non-tuple sequence for multidimensional indexing is deprecated")
warnings.filterwarnings("ignore", "Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9")

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
import os
from tqdm import tqdm

# ==============================================
# HYPERPARAMETERS
# ==============================================

# Dataset configuration
PDE_DIRECTION = "forward"  # 'forward' or 'inverse'
DATASET_NAME = "poisson"  # Dataset name for saving/loading models
DATA_RESOLUTION = 128  # Original data resolution
TRAIN_RESOLUTION = (128, 128)  # Training resolution (downsampled)   # To change

# Training configuration
BATCH_SIZE = 25
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Model configuration
FNO_MODES = (32, 32)  # To change
IN_CHANNELS = 1
OUT_CHANNELS = 1
HIDDEN_CHANNELS = 64
N_LAYERS = 4

# Wandb configuration
WANDB_PROJECT = "fundps-physics-guidance"

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
train_dataset = PDEDataset(path=f"data/DiffPDE/{DATASET_NAME}_hf", resolution=DATA_RESOLUTION, max_size=5000)  # To change
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = PDEDataset(path=f"data/DiffPDE/{DATASET_NAME}_test_hf", resolution=DATA_RESOLUTION)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Model initialization
model = FNO_pad(n_modes=FNO_MODES, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, n_layers=N_LAYERS)
model = model.to(device)

# Mixed Resolution Training: Load res = 64 model
try:
    # PyTorch 2.6+ approach: add safe globals and use weights_only=True
    import torch.serialization
    torch.serialization.add_safe_globals(['torch._C._nn.gelu'])
    model.load_state_dict(torch.load(f"generation/fno_pad_trained_{PDE_DIRECTION}_{DATASET_NAME}_64.pth", weights_only=True))
    print("Successfully loaded checkpoint with weights_only=True")
except Exception as e:
    print(f"Loading with weights_only=True failed: {e}")
    print("Trying with weights_only=False (trusted source only)...")
    # Fallback to weights_only=False for compatibility with older checkpoints
    model.load_state_dict(torch.load(f"generation/fno_pad_trained_{PDE_DIRECTION}_{DATASET_NAME}_64.pth", weights_only=False))
    print("Successfully loaded checkpoint with weights_only=False")
model = model.to(device)

# Initialize wandb
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

wandb.init(
    project=WANDB_PROJECT,
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
    },
)


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


# Training setup
criterion = L2Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    batch_losses = []

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Starting...")

    # Create progress bar for this epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False, ncols=120)

    for batch_idx, (train_data, _) in enumerate(pbar):
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

        # Update progress bar with current loss and ETA
        if len(batch_losses) > 0:
            current_loss = batch_losses[-1]
            avg_loss = sum(batch_losses) / len(batch_losses)

            # Memory usage for progress bar
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
                memory_str = f"GPU:{gpu_memory:.1f}GB"
            else:
                process = psutil.Process(os.getpid())
                memory_str = f"RAM:{process.memory_info().rss / 1024**3:.1f}GB"

            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Avg": f"{avg_loss:.4f}", "Mem": memory_str})

        # Detailed progress update every N batches for console output
        N_batches_to_report = total_batches // 5
        if (batch_idx + 1) % N_batches_to_report == 0 or (batch_idx + 1) == total_batches:
            batch_time = time.time() - batch_start_time
            avg_batch_loss = sum(batch_losses[-20:]) / min(20, len(batch_losses[-20:]))

            progress = (batch_idx + 1) / total_batches * 100
            eta_epoch = batch_time * (total_batches - batch_idx - 1)

            print(f"\nBatch {batch_idx+1:3d} ({progress:3.1f}%) | " f"Loss: {avg_batch_loss:.4f} | " f"Batch time: {batch_time:.3f}s | " f"ETA epoch: {eta_epoch:.1f}s | " f"{memory_str}")

    pbar.close()

    # Epoch summary
    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(train_dataset)
    avg_batch_loss = sum(batch_losses) / len(batch_losses)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = evaluate_test_accuracy(model, test_loader, criterion, device)

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
    print(f"  Training Loss: {epoch_loss:.6f}")
    print(f"  Average Batch Loss: {avg_batch_loss:.6f}")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Epoch Time: {epoch_time:.2f}s")
    print(f"  Time per batch: {epoch_time / total_batches:.3f}s")

    # Log metrics to wandb
    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "avg_batch_loss": avg_batch_loss,
            "test_loss": test_loss,
            "epoch_time": epoch_time,
            "time_per_batch": epoch_time / total_batches,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )

    print("-" * 80)

# Save model
model_path = f"generation/fno_pad_trained_{PDE_DIRECTION}_{DATASET_NAME}_{list(TRAIN_RESOLUTION)[0]}.pth"
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
