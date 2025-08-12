#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from neuralop.models.fno import FNO
from training.networks import SongUNO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. DATA LOADING & VISUALIZATION
# ============================================================
import numpy as np
from scipy.io import loadmat
from training.dataset_hf import PDEDataset

pde_direction = 'inverse'  # or 'inverse', depending on the dataset
dataset_name = 'darcy'  # Name of the dataset for saving/loading models

# === Load training data ===
train_dataset = PDEDataset(path=f'data/DiffPDE/{dataset_name}_hf', resolution=128)
train_loader = DataLoader(train_dataset, shuffle=True)

# === Load testing data ===
test_dataset = PDEDataset(path=f'data/DiffPDE/{dataset_name}_test_hf', resolution=128)
test_loader = DataLoader(test_dataset, shuffle=False)


def visualize_sample_or_pred(input_field, ground_truth_field, model, device, visualize_pred=False):
    """
    Visualizes an input-ground_truth pair from a given dataset at the specified index.
    Assumes input and ground_truth tensors are in (batch, channels, height, width) format.
    Optionally includes model predictions if `visualize_pred` is True.
    If `model` is None, only input and ground_truth will be visualized.
    """
    # Convert to PyTorch tensors and move to device
    input_field = torch.tensor(input_field).to(device).float()
    ground_truth_field = torch.tensor(ground_truth_field).to(device).float()

    # Prediction: Get the model's prediction for the input sample if visualize_pred is True
    if visualize_pred and model is not None:
        prediction_field = model(input_field.unsqueeze(0)).squeeze(0)  # Add batch dimension
    else:
        prediction_field = None

    # Move to CPU and detach from the computation graph
    input_field = input_field.detach().cpu()
    ground_truth_field = ground_truth_field.detach().cpu()

    if prediction_field is not None:
        prediction_field = prediction_field.detach().cpu()

    # Get the number of channels
    input_channels = input_field.shape[0]
    ground_truth_channels = ground_truth_field.shape[0]

    # Set up subplots
    num_cols = max(input_channels, ground_truth_channels)

    # Adjust figure size based on whether prediction needs to be visualized
    figsize = (5 * num_cols, 5 * 3) if visualize_pred else (5 * num_cols, 5 * 2)
    fig, axs = plt.subplots(3 if visualize_pred else 2, num_cols, figsize=figsize)

    if num_cols == 1:
        axs = axs.reshape(3 if visualize_pred else 2, 1)

    # Plot input channels
    for c in range(input_channels):
        img = input_field[c, :, :].numpy()
        axs[0, c].imshow(img, cmap='viridis')
        title = f'Input: component {c}' if input_channels > 1 else 'Input Field'
        axs[0, c].set_title(title)

    # Plot prediction channels if visualize_pred is True
    if prediction_field is not None:
        for c in range(ground_truth_channels):
            img = prediction_field[c, :, :].numpy()
            axs[1, c].imshow(img, cmap='viridis')
            title = f'Predicted: component {c}' if ground_truth_channels > 1 else 'Predicted Field'
            axs[1, c].set_title(title)

    # Plot ground_truth channels
    for c in range(ground_truth_channels):
        img = ground_truth_field[c, :, :].numpy()
        axs[-1, c].imshow(img, cmap='viridis')
        title = f'Ground Truth: component {c}' if ground_truth_channels > 1 else 'Ground Truth'
        axs[-1, c].set_title(title)

    plt.tight_layout()
    plt.show()


# === Visualize training samples ===
print("Visualizing training samples:")
for i in range(1):  # Visualize first sample from the training dataset
    data, _ = train_dataset[i]
    if pde_direction == 'forward':
        input_field = data[0:1, :, :]  # First channel as input
        ground_truth_field = data[1:2, :, :]  # Second channel as ground_truth
    elif pde_direction == 'inverse':
        input_field = data[1:2, :, :]  # First channel as input
        ground_truth_field = data[0:1, :, :]
    visualize_sample_or_pred(input_field, ground_truth_field, model=None, device=device, visualize_pred=False)

# === Visualize testing samples ===
print("Visualizing testing samples:")
for i in range(1):  # Visualize first sample from the testing dataset
    data, _ = test_dataset[i]
    if pde_direction == 'forward':
        input_field = data[0:1, :, :]
        ground_truth_field = data[1:2, :, :]  # Second channel as ground_truth
    elif pde_direction == 'inverse':
        input_field = data[1:2, :, :]  # First channel as input
        ground_truth_field = data[0:1, :, :]  # Second channel
    visualize_sample_or_pred(input_field, ground_truth_field, model=None, device=device, visualize_pred=False)

#%%
# ============================================================
# 2. MODEL ARCHITECTURE (CNN ENCODER-DECODER) vs. FOURIER NEURAL OPERATOR (FNO) MODEL
# ============================================================

# class VectorFieldCNN(nn.Module):
#     def __init__(self):
#         super(VectorFieldCNN, self).__init__()
#         # Encoder: gradually increase feature channels
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)

#         # Bottleneck
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

#         # Decoder: upsampling and reducing channels to 2 (the output vector field)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

#     def forward(self, x):
#         # Encoder
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x_enc = self.pool(x)

#         # Bottleneck
#         x_bot = F.relu(self.conv3(x_enc))

#         # Decoder
#         x = self.up(x_bot)
#         x = F.relu(self.conv4(x))
#         x = self.up(x)
#         out = self.conv5(x)
#         return out
# model = VectorFieldCNN().to(device)

# Define FNO model with appropriate parameters

# Toy dataset
# model = FNO(
#     n_modes=(64, 64),  # Number of Fourier modes per spatial dimension
#     in_channels=2,     # Input has 2 channels (x and y components)
#     out_channels=2,    # Output should also have 2 channels
#     hidden_channels=64, # Number of hidden channels
#     n_layers=4         # Number of layers in FNO
# ).to(device)

# model = FNO(
#     n_modes=(64, 64),
#     in_channels=1,      # scalar input
#     out_channels=1,     # scalar output
#     hidden_channels=64,
#     n_layers=4
# )

model = SongUNO(
    img_resolution=64,
    in_channels=1,
    out_channels=1,
    fmult=0.5,
    rank=0.1,
    model_channels=64,
    channel_mult=[1, 2, 2],
    num_blocks=2,
    attn_resolutions=[16],
    dropout=0.10,
    cond=False,
)

# count the number of parameters in the model
print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
exit()

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
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for train_data, _ in train_loader:
        train_data = train_data.to(device).float()
        if pde_direction == 'forward':
            inputs, ground_truths = train_data[:, 0:1, :, :], train_data[:, 1:2, :, :]
        elif pde_direction == 'inverse':
            inputs, ground_truths = train_data[:, 1:2, :, :], train_data[:, 0:1, :, :]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Print the shapes of inputs, outputs, and ground_truths
        # print(f"Input shape: {inputs.shape}, ground_truth shape: {ground_truths.shape}, Output shape: {outputs.shape}")

        # Compute the loss using the custom L2Loss function
        loss = criterion(outputs, ground_truths)

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Average loss for the epoch
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.6f}")

# torch.save(model.state_dict(), f"generation/fno_trained_{pde_direction}_{dataset_name}.pth")

#%%
# ============================================================
# 5. EVALUATION ON TEST DATA
# ============================================================

model.eval()
test_loss = 0.0
with torch.no_grad():
    for test_data, _ in test_loader:
        test_data = test_data.to(device).float()
        if pde_direction == 'forward':
            inputs, ground_truths = test_data[:, 0:1, :, :], test_data[:, 1:2, :, :]
        elif pde_direction == 'inverse':
            inputs, ground_truths = test_data[:, 1:2, :, :], test_data[:, 0:1, :, :]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, ground_truths)
        test_loss += loss.item()

test_loss /= len(test_dataset)
print(f"Test Loss: {test_loss:.6f}")

# === Visualize predictions ===
print("Visualizing testing predictions:")
for i in range(1):  # Visualize first sample from the testing dataset with predictions
    data, _ = test_dataset[i]
    if pde_direction == 'forward':
        input_field = data[0:1, :, :]  # First channel as input
        ground_truth_field = data[1:2, :, :]  # Second channel as ground_truth
    elif pde_direction == 'inverse':
        input_field = data[1:2, :, :]  # First channel as input
        ground_truth_field = data[0:1, :, :]  # Second channel as ground_truth
    visualize_sample_or_pred(input_field, ground_truth_field, model=model, device=device, visualize_pred=True)

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
