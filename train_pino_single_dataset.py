import os
from datetime import datetime
import argparse
import json

import torch
import torch.nn.functional as F
from neuralop.models.fno import FNO
from neuralop.losses import LpLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from training.dataset_hf import PDEDataset
from pino_loss import get_pde_loss

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
NUM_ITERATIONS = 10000  # Total number of training iterations
EVAL_INTERVAL = 100  # Evaluate every N iterations
LEARNING_RATE = 1e-4

# Model configuration
ARCHITECTURE = "fno_pad"  # Model architecture
FNO_MODES = (64, 64)  # To change
IN_CHANNELS = 1
OUT_CHANNELS = 1
HIDDEN_CHANNELS = 64
N_LAYERS = 4

# Wandb configuration
WANDB_PROJECT = "fundps-physics-guidance"
extra_config = {}

# ==============================================

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train PINO model")
parser.add_argument("--resume", "-r", type=str, default=None, help="Path to checkpoint to resume from")
args = parser.parse_args()

config = {
    "dataset": DATASET_NAME,
    "pde_direction": PDE_DIRECTION,
    "batch_size": BATCH_SIZE,
    "num_iterations": NUM_ITERATIONS,
    "eval_interval": EVAL_INTERVAL,
    "learning_rate": LEARNING_RATE,
    "resolution": TRAIN_RESOLUTION,
    "architecture": ARCHITECTURE,
    "fno_modes": FNO_MODES,
    "hidden_channels": HIDDEN_CHANNELS,
    "n_layers": N_LAYERS,
    **extra_config,
}

wandb.init(
    project=WANDB_PROJECT,
    name=f"{ARCHITECTURE}_{DATASET_NAME}_{PDE_DIRECTION}_{TRAIN_RESOLUTION[0]}",
    config=config,
)

# Create timestamped folder for this run
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb_id = wandb.run.id
run_folder = f"exps/run_{run_timestamp}_{wandb_id}"
os.makedirs(run_folder, exist_ok=True)
print(f"Run folder: {run_folder}")

# Save config to run folder
config_path = os.path.join(run_folder, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved to: {config_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FNO_Pad(FNO):
    def forward(self, x):
        res_2 = x.shape[-1] // 2
        x = F.pad(x, (res_2, res_2, res_2, res_2), mode="reflect")
        ret = super().forward(x)
        ret = ret[:, :, res_2:-res_2, res_2:-res_2]
        return ret


if ARCHITECTURE == "fno_pad":
    ModelClass = FNO_Pad
else:
    raise ValueError(f"Unsupported architecture: {ARCHITECTURE}")

# Model initialization
model = ModelClass(
    n_modes=FNO_MODES,
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
    n_layers=N_LAYERS,
)
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")


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
            if TRAIN_RESOLUTION != (DATA_RESOLUTION, DATA_RESOLUTION):
                inputs = torch.nn.functional.interpolate(inputs, size=TRAIN_RESOLUTION, mode="area")
                ground_truths = torch.nn.functional.interpolate(ground_truths, size=TRAIN_RESOLUTION, mode="area")

            outputs = model(inputs)
            loss = criterion(outputs, ground_truths)
            test_loss += loss.item()

    return test_loss / len(test_loader.dataset)


# Data loading
train_dataset = PDEDataset(
    path=f"data/DiffPDE/{DATASET_NAME}_hf",
    resolution=DATA_RESOLUTION,
    max_size=500,  # Only 1 percent of the data
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = PDEDataset(path=f"data/DiffPDE/{DATASET_NAME}_test_hf", resolution=DATA_RESOLUTION)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training setup
criterion_l2 = LpLoss(d=2, p=2)
criterion_pde = get_pde_loss(DATASET_NAME, train_dataset)
print(f"Using PDE loss for dataset: {DATASET_NAME}")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Resume from checkpoint if provided
start_iteration = 0
best_test_loss = float("inf")
best_iteration = 0

if args.resume:
    print(f"\nResuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Try to load optimizer state, but handle incompatibility (e.g., different optimizer type)
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded optimizer state from checkpoint")
    except (ValueError, KeyError, RuntimeError) as e:
        print(f"Warning: Could not load optimizer state from checkpoint: {e}")
        print("Continuing with fresh optimizer state")

    start_iteration = checkpoint["iteration"]
    best_test_loss = checkpoint["test_loss"]  # Use checkpoint's test loss as initial best
    best_iteration = start_iteration
    print(f"Resumed from iteration {start_iteration}")
    print(f"Checkpoint test loss: {checkpoint['test_loss']:.6f}")
else:
    print("\nStarting training from scratch")

# Examine PDE loss on test set before training
# total_test_pde_loss = 0.0
# for test_data, _ in test_loader:
#     test_data = test_data.to(device).float()

#     pde_loss = criterion_pde(test_data)
#     total_test_pde_loss += pde_loss.item()

# print(f"Initial PDE loss on test set before training: {total_test_pde_loss / len(test_loader):.6f}")
# exit()

# Training loop
total_batches = len(train_loader)
print(f"Starting training with {NUM_ITERATIONS} iterations")
print(f"Total training samples: {len(train_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Batches per epoch: {total_batches}")
print(f"Evaluation interval: {EVAL_INTERVAL} iterations")
print(f"Learning rate: {LEARNING_RATE}")

# Estimate total training time by running a few warmup batches
# print("\nWarming up model and estimating training time...")
# model.train()
# warmup_batches = total_batches
# warmup_times = []
# for i, (warmup_data, _) in enumerate(train_loader):
#     if i >= warmup_batches:
#         break

#     warmup_start = time.time()
#     warmup_data = warmup_data.to(device).float()
#     if PDE_DIRECTION == "forward":
#         inputs, targets = warmup_data[:, 0:1, :, :], warmup_data[:, 1:2, :, :]
#     elif PDE_DIRECTION == "inverse":
#         inputs, targets = warmup_data[:, 1:2, :, :], warmup_data[:, 0:1, :, :]

#     outputs = model(inputs)
#     loss = criterion_l2(outputs, targets)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

#     warmup_time = time.time() - warmup_start
#     warmup_times.append(warmup_time)

# if warmup_times:
#     avg_batch_time = sum(warmup_times) / len(warmup_times)
#     estimated_epoch_time = avg_batch_time * total_batches
#     estimated_total_time = estimated_epoch_time * NUM_EPOCHS
#     print("\nEstimated timing:")
#     print(f"  Average batch time: {avg_batch_time:.3f}s")
#     print(f"  Estimated epoch time: {estimated_epoch_time:.1f}s ({estimated_epoch_time / 60:.1f} minutes)")
#     print(f"  Estimated total time: {estimated_total_time:.1f}s ({estimated_total_time / 60:.1f} minutes)")

# test_loss_warmup = evaluate_test_accuracy(model, test_loader, criterion_l2, device)
# print(f"Test loss after warmup: {test_loss_warmup:.6f}")

print("-" * 80)

# Create infinite data loader
from itertools import cycle

train_loader_infinite = cycle(train_loader)

# Skip to the correct position in the data loader if resuming
# if start_iteration > 0:
#     print(f"Skipping first {start_iteration} batches to align with checkpoint...")
#     for _ in range(start_iteration):
#         next(train_loader_infinite)
#     print(f"Training will continue from iteration {start_iteration + 1} to {NUM_ITERATIONS}")

model.train()
pbar = tqdm(range(start_iteration, NUM_ITERATIONS), desc="Training", initial=start_iteration, total=NUM_ITERATIONS)

try:
    for iteration in pbar:
        train_data, _ = next(train_loader_infinite)

        train_data = train_data.to(device).float()
        batch_size_actual = train_data.size(0)

        if PDE_DIRECTION == "forward":
            inputs, ground_truths = train_data[:, 0:1, :, :], train_data[:, 1:2, :, :]
        elif PDE_DIRECTION == "inverse":
            inputs, ground_truths = train_data[:, 1:2, :, :], train_data[:, 0:1, :, :]

        # downsample the inputs and ground_truths
        if TRAIN_RESOLUTION != (DATA_RESOLUTION, DATA_RESOLUTION):
            inputs = torch.nn.functional.interpolate(inputs, size=TRAIN_RESOLUTION, mode="area")
            ground_truths = torch.nn.functional.interpolate(ground_truths, size=TRAIN_RESOLUTION, mode="area")

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss - these return batch-averaged losses
        loss_l2_batch = criterion_l2(outputs, ground_truths)
        loss_pde_batch, loss_bc_batch = criterion_pde(outputs)
        loss_batch = loss_l2_batch

        # Backpropagation and optimization step
        loss_batch.backward()
        optimizer.step()

        # Log per-sample losses to wandb
        wandb.log(
            {
                "loss_l2": loss_l2_batch.item() / batch_size_actual,
                "loss_pde": loss_pde_batch.item() / batch_size_actual,
                "loss_bc": loss_bc_batch.item() / batch_size_actual,
                "loss": loss_batch.item() / batch_size_actual,
                "abs_scale": outputs.norm().item() / batch_size_actual,
            },
            step=iteration,
        )

        # Update progress bar
        pbar.set_postfix({"Loss": f"{loss_batch.item() / batch_size_actual:.6f}"})

        # Periodic evaluation
        if (iteration + 1) % EVAL_INTERVAL == 0 or iteration == NUM_ITERATIONS - 1:
            model.eval()
            test_loss = evaluate_test_accuracy(model, test_loader, criterion_l2, device)
            model.train()

            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_iteration = iteration + 1
                best_model_path = f"{run_folder}/best_model.pth"
                checkpoint = {
                    "iteration": iteration + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": test_loss,
                }
                torch.save(checkpoint, best_model_path)
                print(f"\n*** Iteration {iteration+1}/{NUM_ITERATIONS}: New best! Test Loss: {test_loss:.6f} ***")
            else:
                print(f"\nIteration {iteration+1}/{NUM_ITERATIONS}: Test Loss: {test_loss:.6f}, Best: {best_test_loss:.6f} (iter {best_iteration})")

            # Log metrics to wandb
            wandb.log(
                {
                    "test_loss": test_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=iteration + 1,
            )

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user (Ctrl+C)")
    print(f"Saving checkpoint at iteration {iteration}...")

    interrupt_checkpoint_path = f"{run_folder}/interrupted_iter{iteration}.pth"
    interrupt_checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_loss": None,
    }
    torch.save(interrupt_checkpoint, interrupt_checkpoint_path)
    print(f"Checkpoint saved to: {interrupt_checkpoint_path}")
    print(f"You can resume training with: --resume {interrupt_checkpoint_path}")

    pbar.close()
    wandb.finish()
    raise  # Re-raise to exit the script

pbar.close()
print("-" * 80)

# Final evaluation
print("\nFinal evaluation on test set:")
final_test_loss = evaluate_test_accuracy(model, test_loader, criterion_l2, device)
print(f"Final Test Loss: {final_test_loss:.6f}")

# Save final model
final_model_path = f"{run_folder}/final_model.pth"
final_checkpoint = {
    "iteration": NUM_ITERATIONS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "test_loss": final_test_loss,
}
torch.save(final_checkpoint, final_model_path)
print(f"\nFinal model saved to: {final_model_path}")
print(f"Best model (Iteration {best_iteration}) saved to: {run_folder}/best_model_iter{best_iteration}.pth")
print(f"Best test loss: {best_test_loss:.6f}")

# Log final test results and finish wandb
wandb.log({"final_test_loss": final_test_loss}, step=NUM_ITERATIONS)
wandb.finish()
