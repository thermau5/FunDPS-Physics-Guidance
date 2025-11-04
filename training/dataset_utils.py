import torch


def transform_darcy(x: torch.Tensor) -> torch.Tensor:
    a, u = x[:, 0, :, :], x[:, 1, :, :]
    a = torch.where(a > 7.5, torch.tensor(12.0, device=x.device), torch.tensor(3.0, device=x.device))
    return torch.stack([a, u], dim=1)


class DatasetNormalizer:

    def __init__(self, dataset_name, stats):
        """
        Initialize the normalizer with dataset-specific parameters.
        """
        self.dataset_name = dataset_name
        
        # Support for both list/array stats and .npy file paths
        mean = stats["mean"]
        std = stats["std"]
        
        if isinstance(mean, str):
            # Load from .npy file (for large datasets like JFM)
            import numpy as np
            mean = np.load(mean)
            std = np.load(std)
        
        # Convert to tensor and reshape
        mean_tensor = torch.tensor(mean) if not isinstance(mean, torch.Tensor) else mean
        std_tensor = torch.tensor(std) if not isinstance(std, torch.Tensor) else std
        
        # Handle different stat shapes
        if len(mean_tensor.shape) == 1:
            # Standard case: [channel1_mean, channel2_mean] → [1, C, 1, 1]
            self.mean = mean_tensor.reshape(1, len(mean), 1, 1)
            self.std = std_tensor.reshape(1, len(std), 1, 1)
        elif len(mean_tensor.shape) == 3:
            # JFM case: [2, H, W] → needs special handling
            # Will be broadcast during normalization
            self.mean = mean_tensor.unsqueeze(0)  # [1, 2, H, W]
            self.std = std_tensor.unsqueeze(0)    # [1, 2, H, W]
        else:
            raise ValueError(f"Unexpected stats shape: {mean_tensor.shape}")
        
        self._transform = lambda x: x
        if dataset_name == "darcy":
            self._transform = transform_darcy

    def _check_shape(self, x: torch.Tensor):
        # Assuming x has shape (batch_size, channels, height, width)
        assert len(x.shape) == 4, f"Expected 4D tensor, got {len(x.shape)}D"
        assert x.shape[1] in [1, 2], f"Expected 1 or 2 channels, got {x.shape[1]}"
        return True

    def normalize(self, x: torch.Tensor, channel=None) -> torch.Tensor:
        self._check_shape(x)
        mean = self.mean
        std = self.std
        if channel is not None:
            mean = mean[:, channel, :, :].reshape(1, -1, 1, 1)
            std = std[:, channel, :, :].reshape(1, -1, 1, 1)
        mean = mean.to(x.device)
        std = std.to(x.device)
        x_normalized = (x - mean) * (0.5 / std)
        return x_normalized

    def denormalize(self, x_normalized: torch.Tensor, channel=None) -> torch.Tensor:
        self._check_shape(x_normalized)
        mean = self.mean
        std = self.std
        if channel is not None:
            mean = mean[:, channel, :, :].reshape(1, -1, 1, 1)
            std = std[:, channel, :, :].reshape(1, -1, 1, 1)
        mean = mean.to(x_normalized.device)
        std = std.to(x_normalized.device)
        x = x_normalized / (0.5 / std) + mean
        return x

    def transform(self, x: torch.Tensor, denormalize=False) -> torch.Tensor:
        self._check_shape(x)
        if denormalize:
            x = self.denormalize(x)
        return self._transform(x)
