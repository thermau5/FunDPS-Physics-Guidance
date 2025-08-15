#%%
import warnings

# Suppress PyTorch deprecation warnings about multidimensional indexing
warnings.filterwarnings("ignore", "Using a non-tuple sequence for multidimensional indexing is deprecated")
warnings.filterwarnings("ignore", "Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from neuralop.models.fno import FNO
from training.networks import SongUNO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FNO_pad(FNO):
    def forward(self, x):
        res_2 = x.shape[-1] // 2
        x = F.pad(x, (res_2, res_2, res_2, res_2), mode='reflect')
        ret = super().forward(x)
        ret = ret[:, :, res_2:-res_2, res_2:-res_2]
        return ret

# Wrapper class to make SongUNO compatible with direct forward prediction
class SongUNOWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = SongUNO(*args, **kwargs)
        
    def forward(self, x):
        # Create dummy noise_labels and class_labels for training
        batch_size = x.shape[0]
        device = x.device
        
        # Create dummy noise_labels (timestep 0 for deterministic prediction)
        noise_labels = torch.zeros(batch_size, device=device)
        
        # Create dummy class_labels (unconditional)
        class_labels = torch.zeros(batch_size, 0, device=device)  # Empty class labels
        
        return self.model(x, noise_labels, None)

# ============================================================
# 1. DATA LOADING & VISUALIZATION
# ============================================================
import numpy as np
from scipy.io import loadmat
from training.dataset_hf import PDEDataset

pde_direction = 'forward'  # or 'inverse', depending on the dataset
dataset_name = 'poisson'  # Name of the dataset for saving/loading models

# === Batch size configuration ===
batch_size = 25  # Increased from default 1 to 64 for better GPU utilization

# === Load training data ===
train_dataset = PDEDataset(path=f'data/DiffPDE/{dataset_name}_hf', resolution=128)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# === Load testing data ===
test_dataset = PDEDataset(path=f'data/DiffPDE/{dataset_name}_test_hf', resolution=128)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# def visualize_sample_or_pred(input_field, ground_truth_field, model, device, visualize_pred=False):
#     """
#     Visualizes an input-ground_truth pair from a given dataset at the specified index.
#     Assumes input and ground_truth tensors are in (batch, channels, height, width) format.
#     Optionally includes model predictions if `visualize_pred` is True.
#     If `model` is None, only input and ground_truth will be visualized.
#     """
#     # Convert to PyTorch tensors and move to device
#     input_field = torch.tensor(input_field).to(device).float()
#     ground_truth_field = torch.tensor(ground_truth_field).to(device).float()

#     # Prediction: Get the model's prediction for the input sample if visualize_pred is True
#     if visualize_pred and model is not None:
#         prediction_field = model(input_field.unsqueeze(0)).squeeze(0)  # Add batch dimension
#     else:
#         prediction_field = None

#     # Move to CPU and detach from the computation graph
#     input_field = input_field.detach().cpu()
#     ground_truth_field = ground_truth_field.detach().cpu()

#     if prediction_field is not None:
#         prediction_field = prediction_field.detach().cpu()

#     # Get the number of channels
#     input_channels = input_field.shape[0]
#     ground_truth_channels = ground_truth_field.shape[0]

#     # Set up subplots
#     num_cols = max(input_channels, ground_truth_channels)

#     # Adjust figure size based on whether prediction needs to be visualized
#     figsize = (5 * num_cols, 5 * 3) if visualize_pred else (5 * num_cols, 5 * 2)
#     fig, axs = plt.subplots(3 if visualize_pred else 2, num_cols, figsize=figsize)

#     if num_cols == 1:
#         axs = axs.reshape(3 if visualize_pred else 2, 1)

#     # Plot input channels
#     for c in range(input_channels):
#         img = input_field[c, :, :].numpy()
#         axs[0, c].imshow(img, cmap='viridis')
#         title = f'Input: component {c}' if input_channels > 1 else 'Input Field'
#         axs[0, c].set_title(title)

#     # Plot prediction channels if visualize_pred is True
#     if prediction_field is not None:
#         for c in range(ground_truth_channels):
#             img = prediction_field[c, :, :].numpy()
#             axs[1, c].imshow(img, cmap='viridis')
#             title = f'Predicted: component {c}' if ground_truth_channels > 1 else 'Predicted Field'
#             axs[1, c].set_title(title)

#     # Plot ground_truth channels
#     for c in range(ground_truth_channels):
#         img = ground_truth_field[c, :, :].numpy()
#         axs[-1, c].imshow(img, cmap='viridis')
#         title = f'Ground Truth: component {c}' if ground_truth_channels > 1 else 'Ground Truth'
#         axs[-1, c].set_title(title)

#     plt.tight_layout()
#     plt.show()


# # === Visualize training samples ===
# print("Visualizing training samples:")
# for i in range(1):  # Visualize first sample from the training dataset
#     data, _ = train_dataset[i]
#     if pde_direction == 'forward':
#         input_field = data[0:1, :, :]  # First channel as input
#         ground_truth_field = data[1:2, :, :]  # Second channel as ground_truth
#     elif pde_direction == 'inverse':
#         input_field = data[1:2, :, :]  # First channel as input
#         ground_truth_field = data[0:1, :, :]
#     visualize_sample_or_pred(input_field, ground_truth_field, model=None, device=device, visualize_pred=False)

# # === Visualize testing samples ===
# print("Visualizing testing samples:")
# for i in range(1):  # Visualize first sample from the testing dataset
#     data, _ = test_dataset[i]
#     if pde_direction == 'forward':
#         input_field = data[0:1, :, :]
#         ground_truth_field = data[1:2, :, :]  # Second channel as ground_truth
#     elif pde_direction == 'inverse':
#         input_field = data[1:2, :, :]  # First channel as input
#         ground_truth_field = data[0:1, :, :]  # Second channel
#     visualize_sample_or_pred(input_field, ground_truth_field, model=None, device=device, visualize_pred=False)


#%%
# ============================================================
# 2. MODEL ARCHITECTURE (CNN ENCODER-DECODER) vs. FOURIER NEURAL OPERATOR (FNO) MODEL
# ============================================================

model = FNO_pad(
    n_modes=(64, 64),
    in_channels=1,      # scalar input
    out_channels=1,     # scalar output
    hidden_channels=64,
    n_layers=4
)

# model = SongUNOWrapper(
#     img_resolution=64,
#     in_channels=1,
#     out_channels=1,
#     fmult=0.5,
#     rank=0.15,  # Slightly increased from 0.1 for better expressiveness
#     model_channels=64,  # Increased from 64 to 68 for ~1.5x parameters
#     channel_mult=[1, 2, 2],  # Keep original for controlled growth
#     num_blocks=2,  # Keep original for controlled growth
#     attn_resolutions=[16],
#     dropout=0.10,
#     cond=False,
# )

# Move model to device
model = model.to(device)

# count the number of parameters in the model
print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")

# ============================================================
# 3. LOSS FUNCTION (L2 LOSS)
# ============================================================

class L2Loss(object):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
        return torch.sum(diff_norms / y_norms)

# ============================================================
# 4. TRAINING LOOP
# ============================================================

# Initialize loss function and optimizer
criterion = L2Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
import time
import psutil
import os
from tqdm import tqdm

num_epochs = 50
total_batches = len(train_loader)
print(f"Starting training with {num_epochs} epochs, {total_batches} batches per epoch")
print(f"Total training samples: {len(train_dataset)}")
print(f"Batch size: {train_loader.batch_size}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

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
    if pde_direction == 'forward':
        inputs, targets = warmup_data[:, 0:1, :, :], warmup_data[:, 1:2, :, :]
    elif pde_direction == 'inverse':
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
    estimated_total_time = estimated_epoch_time * num_epochs
    print(f"\nEstimated timing:")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Estimated epoch time: {estimated_epoch_time:.1f}s ({estimated_epoch_time/60:.1f} minutes)")
    print(f"  Estimated total time: {estimated_total_time:.1f}s ({estimated_total_time/60:.1f} minutes)")

print("-" * 80)

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    batch_losses = []
    
    print(f"\nEpoch {epoch+1}/{num_epochs} - Starting...")
    
    # Create progress bar for this epoch
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                leave=False, ncols=120)
    
    for batch_idx, (train_data, _) in enumerate(pbar):
        batch_start_time = time.time()
        
        train_data = train_data.to(device).float()
        if pde_direction == 'forward':
            inputs, ground_truths = train_data[:, 0:1, :, :], train_data[:, 1:2, :, :]
        elif pde_direction == 'inverse':
            inputs, ground_truths = train_data[:, 1:2, :, :], train_data[:, 0:1, :, :]

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
            
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Avg': f'{avg_loss:.4f}',
                'Mem': memory_str
            })
        
        # Detailed progress update every N batches for console output
        N_batches_to_report = total_batches // 5
        if (batch_idx + 1) % N_batches_to_report == 0 or (batch_idx + 1) == total_batches:
            batch_time = time.time() - batch_start_time
            avg_batch_loss = sum(batch_losses[-20:]) / min(20, len(batch_losses[-20:]))
            
            # Memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                memory_info = f"GPU: {gpu_memory:.2f}GB (allocated), {gpu_memory_reserved:.2f}GB (reserved)"
            else:
                process = psutil.Process(os.getpid())
                memory_info = f"RAM: {process.memory_info().rss / 1024**3:.2f}GB"
            
            progress = (batch_idx + 1) / total_batches * 100
            eta_epoch = batch_time * (total_batches - batch_idx - 1)
            
            print(f"\nBatch {batch_idx+1:3d} ({progress:3.1f}%) | "
                  f"Loss: {avg_batch_loss:.4f} | "
                  f"Batch time: {batch_time:.3f}s | "
                  f"ETA epoch: {eta_epoch:.1f}s | "
                  f"{memory_info}")
    
    pbar.close()

    # Epoch summary
    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(train_dataset)
    avg_batch_loss = sum(batch_losses) / len(batch_losses)
    
    print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
    print(f"  Training Loss: {epoch_loss:.6f}")
    print(f"  Average Batch Loss: {avg_batch_loss:.6f}")
    print(f"  Epoch Time: {epoch_time:.2f}s")
    print(f"  Time per batch: {epoch_time/total_batches:.3f}s")
    
    # Memory summary
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"  GPU Memory: {gpu_memory:.2f}GB (allocated), {gpu_memory_reserved:.2f}GB (reserved)")
    else:
        process = psutil.Process(os.getpid())
        print(f"  RAM Usage: {process.memory_info().rss / 1024**3:.2f}GB")
    
    print("-" * 80)

torch.save(model.state_dict(), f"generation/fno_pad_trained_{pde_direction}_{dataset_name}.pth")

#%%
# ============================================================
# 5. EVALUATION ON TEST DATA
# ============================================================

print("\n" + "="*80)
print("EVALUATION ON TEST DATA")
print("="*80)

model.eval()
test_loss = 0.0
test_batches = len(test_loader)
print(f"Testing on {len(test_dataset)} samples in {test_batches} batches")

with torch.no_grad():
    # Create progress bar for testing
    test_pbar = tqdm(test_loader, desc="Testing", leave=False, ncols=120)
    
    for batch_idx, (test_data, _) in enumerate(test_pbar):
        test_data = test_data.to(device).float()
        if pde_direction == 'forward':
            inputs, ground_truths = test_data[:, 0:1, :, :], test_data[:, 1:2, :, :]
        elif pde_direction == 'inverse':
            inputs, ground_truths = test_data[:, 1:2, :, :], test_data[:, 0:1, :, :]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, ground_truths)
        test_loss += loss.item()
        
        # Update progress bar
        test_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    test_pbar.close()

test_loss /= len(test_dataset)
print(f"\nFinal Test Loss: {test_loss:.6f}")
print("="*80)

# === Visualize predictions ===
# print("Visualizing testing predictions:")
# for i in range(1):  # Visualize first sample from the testing dataset with predictions
#     data, _ = test_dataset[i]
#     if pde_direction == 'forward':
#         input_field = data[0:1, :, :]  # First channel as input
#         ground_truth_field = data[1:2, :, :]  # Second channel as ground_truth
#     elif pde_direction == 'inverse':
#         input_field = data[1:2, :, :]  # First channel as input
#         ground_truth_field = data[0:1, :, :]  # Second channel as ground_truth
#     visualize_sample_or_pred(input_field, ground_truth_field, model=model, device=device, visualize_pred=True)

# #%%
# # ============================================================
# # 6. TESTING THE SAVED MODEL
# # ============================================================

# pde_direction = 'inverse'  # or 'inverse', depending on the dataset
# dataset_name = 'darcy'  # Name of the dataset for saving/loading models

# # Instantiate a new model with the same architecture
# loaded_model = FNO(
#     n_modes=(64, 64),
#     in_channels=1,      # scalar input
#     out_channels=1,     # scalar output
#     hidden_channels=64,
#     n_layers=4
# ).to(device)

# # Load the saved weights
# saved_model_path = f"generation/fno_trained_{pde_direction}_{dataset_name}.pth"
# loaded_model.load_state_dict(torch.load(saved_model_path, map_location=device))
# loaded_model.eval()

# # Evaluate loaded model on test set
# loaded_test_loss = 0.0
# with torch.no_grad():
#     for test_data, _ in test_loader:
#         test_data = test_data.to(device).float()
#         if pde_direction == 'forward':
#             inputs, ground_truths = test_data[:, 0:1, :, :], test_data[:, 1:2, :, :]
#         elif pde_direction == 'inverse':
#             inputs, ground_truths = test_data[:, 1:2, :, :], test_data[:, 0:1, :, :]
#         outputs = loaded_model(inputs)
#         loss = criterion(outputs, ground_truths)
#         loaded_test_loss += loss.item()
# loaded_test_loss /= len(test_dataset)
# print(f"[Loaded Model] Test Loss: {loaded_test_loss:.6f}")

# # Visualize predictions from the loaded model
# print("Visualizing predictions from loaded model:")
# for i in range(1):  # Visualize first sample from the testing dataset with predictions
#     data, _ = test_dataset[i]
#     if pde_direction == 'forward':
#         input_field = data[0:1, :, :]
#         ground_truth_field = data[1:2, :, :]
#     elif pde_direction == 'inverse':
#         input_field = data[1:2, :, :]
#         ground_truth_field = data[0:1, :, :]
#     visualize_sample_or_pred(input_field, ground_truth_field, model=loaded_model, device=device, visualize_pred=True)


# # %%
