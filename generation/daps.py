import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os # Added for auto-detection

from training.dataset_utils import DatasetNormalizer

from .base import PDESolver
from generation.observation import FNO  # For FNO surrogate
from training.networks import SongUNO  # For UNO surrogate

# Import numerical Poisson solver
try:
    import sys
    import os
    # Add parent directory to path to import poisson_numerical_solver
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from poisson_numerical_solver import forward_poisson
    HAS_NUMERICAL_POISSON = True
except ImportError as e:
    print(f"Warning: Could not import numerical Poisson solver: {e}")
    HAS_NUMERICAL_POISSON = False

# Custom FNO_pad class to match training architecture
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


# Wrapper class for numerical Poisson solver (GPU-accelerated)
class NumericalPoissonWrapper(torch.nn.Module):
    """GPU-accelerated wrapper for numerical Poisson solver using PyTorch.
    
    Takes input tensor [B, 1, H, W] (source term f) and returns [B, 1, H, W] (solution φ).
    Supports gradients through torch.linalg.solve, allowing backpropagation w.r.t. inputs.
    Parallel to FNO/FNO_pad in terms of forward operator usage and gradient flow.
    """
    def __init__(self):
        super().__init__()
        # Cache sparse and dense matrices for different grid sizes (lazy initialization)
        self._sparse_matrix_cache = {}
        self._dense_matrix_cache = {}
    
    def _build_poisson_matrix(self, S, device, dtype):
        """Build the sparse Poisson matrix A for grid size S on the given device.
        
        The matrix represents: A @ phi_vec = f_vec where Δφ = f
        """
        cache_key = (S, device, dtype)
        if cache_key in self._sparse_matrix_cache:
            return self._sparse_matrix_cache[cache_key]
        
        h = 1.0 / (S - 1)
        N = S * S
        
        # Build indices and values for sparse matrix
        indices = []
        values = []
        
        for i in range(S):
            for j in range(S):
                k = i * S + j  # linear index
                
                # Dirichlet boundary: φ = 0
                if i == 0 or i == S - 1 or j == 0 or j == S - 1:
                    indices.append([k, k])
                    values.append(1.0)
                else:
                    # Interior points: 5-point stencil
                    indices.append([k, k])          # center
                    values.append(-4.0 / h**2)
                    
                    indices.append([k, k - 1])      # left (i, j-1)
                    values.append(1.0 / h**2)
                    
                    indices.append([k, k + 1])      # right (i, j+1)
                    values.append(1.0 / h**2)
                    
                    indices.append([k, k - S])      # down (i-1, j)
                    values.append(1.0 / h**2)
                    
                    indices.append([k, k + S])      # up (i+1, j)
                    values.append(1.0 / h**2)
        
        # Convert to tensors
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device).t()
        values_tensor = torch.tensor(values, dtype=dtype, device=device)
        
        # Create sparse matrix in COO format, then convert to CSR for efficient solving
        A_sparse = torch.sparse_coo_tensor(indices_tensor, values_tensor, (N, N))
        A_sparse = A_sparse.coalesce()  # Sort indices and sum duplicates
        
        # Cache the sparse matrix
        self._sparse_matrix_cache[cache_key] = A_sparse
        
        return A_sparse
    
    def _get_dense_matrix(self, S, device, dtype):
        """Get dense version of Poisson matrix (cached).
        
        Note: The matrix itself is built from constant values (no gradients needed),
        but gradients will flow through torch.linalg.solve w.r.t. the input tensor.
        This is parallel to how FNO/FNO_pad work: fixed weights, gradients flow through inputs.
        """
        cache_key = (S, device, dtype)
        if cache_key in self._dense_matrix_cache:
            return self._dense_matrix_cache[cache_key]
        
        # Get sparse matrix and convert to dense
        A_sparse = self._build_poisson_matrix(S, device, dtype)
        A_dense = A_sparse.to_dense()
        
        # Matrix is built from constants, so naturally doesn't require gradients
        # torch.linalg.solve will still compute gradients w.r.t. the right-hand side (input)
        
        # Cache the dense matrix
        self._dense_matrix_cache[cache_key] = A_dense
        
        return A_dense
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 1, H, W] where B is batch size, H=W is resolution
               Represents the source term f(x,y)
        
        Returns:
            Tensor of shape [B, 1, H, W] representing the solution φ(x,y) = Δ^(-1) f
        """
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape [B, 1, H, W], got {x.shape}")
        
        batch_size, channels, height, width = x.shape
        if height != width:
            raise ValueError(f"Expected square grid, got {height}x{width}")
        
        S = height
        device = x.device
        dtype = x.dtype
        
        # Get cached dense matrix for efficient batched solving
        # For large grids, this uses more memory but allows batched GPU operations
        A_dense = self._get_dense_matrix(S, device, dtype)  # [N, N]
        
        # Reshape input: [B, 1, H, W] -> [B, H*W]
        f_vec = x.view(batch_size, -1)  # [B, N]
        
        # Batch solve: A_dense @ phi_vec = f_vec
        # A_dense: [N, N], f_vec: [B, N] -> phi_vec: [B, N]
        # Transpose f_vec to [N, B], solve, then transpose back
        f_vec_t = f_vec.t()  # [N, B]
        phi_vec_t = torch.linalg.solve(A_dense, f_vec_t)  # [N, B]
        phi_vec = phi_vec_t.t()  # [B, N]
        
        # Reshape: [B, N] -> [B, 1, H, W]
        phi = phi_vec.view(batch_size, 1, height, width)
        
        return phi


def compute_dynamic_langevin_steps(annealing_step, total_annealing_steps, base_langevin_steps, 
                                  decay_type="exponential", decay_rate=0.5, min_steps=1):
    """
    Compute the number of Langevin steps dynamically based on the annealing step.
    
    Args:
        annealing_step (int): Current annealing step (0-indexed)
        total_annealing_steps (int): Total number of annealing steps
        base_langevin_steps (int): Base number of Langevin steps (at step 0)
        decay_type (str): Type of decay function ("exponential", "linear", "polynomial", "step", "sigmoid", "increasing")
        decay_rate (float): Decay rate parameter (0-1) or increase multiplier for "increasing"
        min_steps (int): Minimum number of Langevin steps
    
    Returns:
        int: Number of Langevin steps for the current annealing step
    """
    if annealing_step >= total_annealing_steps:
        return min_steps
    
    # Normalize step to [0, 1]
    step_ratio = annealing_step / total_annealing_steps
    
    if decay_type == "exponential":
        # Exponential decay: steps = base_steps * (decay_rate ^ step_ratio)
        steps = int(base_langevin_steps * (decay_rate ** step_ratio))
    elif decay_type == "linear":
        # Linear decay: steps = base_steps * (1 - step_ratio * (1 - decay_rate))
        steps = int(base_langevin_steps * (1 - step_ratio * (1 - decay_rate)))
    elif decay_type == "polynomial":
        # Polynomial decay: steps = base_steps * (1 - step_ratio^2 * (1 - decay_rate))
        steps = int(base_langevin_steps * (1 - step_ratio**2 * (1 - decay_rate)))
    elif decay_type == "sigmoid":
        # Sigmoid decay: slow at start and end, faster in middle
        # Use sigmoid function: 1 / (1 + exp(-k * (x - 0.5))) where k controls steepness
        k = 6  # Controls steepness of sigmoid (higher = steeper)
        sigmoid = 1 / (1 + np.exp(-k * (step_ratio - 0.5)))
        # Map sigmoid [0,1] to [1, decay_rate]
        steps = int(base_langevin_steps * (1 - sigmoid * (1 - decay_rate)))
    elif decay_type == "increasing":
        # Increasing steps: start with fewer steps, increase as annealing progresses
        # Use sigmoid function to control the increase: 1 / (1 + exp(-k * (x - 0.5)))
        k = 5  # Controls steepness of sigmoid (higher = steeper)
        sigmoid = 1 / (1 + np.exp(-k * (step_ratio - 0.5)))
        # Map sigmoid [0,1] to [min_steps/base_steps, decay_rate] where decay_rate is the max multiplier
        min_ratio = min_steps / base_langevin_steps
        steps = int(base_langevin_steps * (min_ratio + sigmoid * (decay_rate - min_ratio)))
    elif decay_type == "step":
        # Step decay: reduce steps at specific thresholds
        if step_ratio < 0.3:
            steps = base_langevin_steps
        elif step_ratio < 0.6:
            steps = int(base_langevin_steps * 0.7)
        elif step_ratio < 0.8:
            steps = int(base_langevin_steps * 0.4)
        else:
            steps = int(base_langevin_steps * 0.2)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    
    # print(f"Langevin steps: {steps}")
    return max(steps, min_steps)


def compute_dynamic_learning_rate(annealing_step, total_annealing_steps, base_lr, 
                                 lr_schedule="warmup_decay", lr_decay_rate=0.3, 
                                 lr_min_ratio=0.01, lr_warmup_steps=10):
    """
    Compute the learning rate dynamically based on the annealing step.
    
    Args:
        annealing_step (int): Current annealing step (0-indexed)
        total_annealing_steps (int): Total number of annealing steps
        base_lr (float): Base learning rate (at step 0)
        lr_schedule (str): Type of learning rate schedule ("exponential", "linear", "polynomial", "cosine", "sigmoid", "warmup_decay")
        lr_decay_rate (float): Learning rate decay rate (0-1)
        lr_min_ratio (float): Minimum learning rate ratio relative to base_lr
        lr_warmup_steps (int): Number of warmup steps (learning rate increases then decreases)
    
    Returns:
        float: Learning rate for the current annealing step
    """
    if annealing_step >= total_annealing_steps:
        return base_lr * lr_min_ratio
    
    # Normalize step to [0, 1]
    step_ratio = annealing_step / total_annealing_steps
    
    if lr_schedule == "exponential":
        # Exponential decay: lr = base_lr * (lr_decay_rate ^ step_ratio)
        lr = base_lr * (lr_decay_rate ** step_ratio)
    elif lr_schedule == "linear":
        # Linear decay: lr = base_lr * (1 - step_ratio * (1 - lr_decay_rate))
        lr = base_lr * (1 - step_ratio * (1 - lr_decay_rate))
    elif lr_schedule == "polynomial":
        # Polynomial decay: lr = base_lr * (1 - step_ratio^2 * (1 - lr_decay_rate))
        lr = base_lr * (1 - step_ratio**2 * (1 - lr_decay_rate))
    elif lr_schedule == "cosine":
        # Cosine decay: lr = base_lr * (lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + cos(π * step_ratio)))
        lr = base_lr * (lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + np.cos(np.pi * step_ratio)))
    elif lr_schedule == "sigmoid":
        # Sigmoid decay: slow at start and end, faster in middle
        # Use sigmoid function: 1 / (1 + exp(-k * (x - 0.5))) where k controls steepness
        k = 6  # Controls steepness of sigmoid (higher = steeper)
        sigmoid = 1 / (1 + np.exp(-k * (step_ratio - 0.5)))
        # Map sigmoid [0,1] to [1, lr_decay_rate]
        lr = base_lr * (1 - sigmoid * (1 - lr_decay_rate))
    elif lr_schedule == "warmup_decay":
        # Warmup then decay: increase for warmup_steps, then decrease
        if annealing_step < lr_warmup_steps:
            # Warmup phase: increase learning rate
            warmup_ratio = annealing_step / lr_warmup_steps
            lr = base_lr * (lr_min_ratio + (1 - lr_min_ratio) * warmup_ratio)
        else:
            # Decay phase: decrease learning rate
            decay_ratio = (annealing_step - lr_warmup_steps) / (total_annealing_steps - lr_warmup_steps)
            lr = base_lr * (1 - decay_ratio * (1 - lr_decay_rate))
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")
    
    # Ensure learning rate doesn't go below minimum
    print(f"Learning rate: {lr}")
    return max(lr, base_lr * lr_min_ratio)


class Scheduler(nn.Module):
    """Scheduler for diffusion sigma(t) and discretization step size Delta t"""

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, rho=7, schedule="linear"):
        """Initialize the scheduler with the given parameters.

        Args:
            num_steps (int): Number of steps in the schedule
            sigma_max (float): Maximum value of sigma
            sigma_min (float): Minimum value of sigma
            sigma_final (float): Final value of sigma, defaults to sigma_min
            rho (float): Power parameter for sigma schedule
            schedule (str): Type of schedule for sigma ('linear' or 'sqrt')
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final if sigma_final is not None else sigma_min
        self.schedule = schedule
        self.p = rho

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.p, self.sigma_max, self.sigma_min)

        # Generate time steps and sigma values
        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])

        # Calculate factor = 2∇σ(t)σ(t)Δt
        factor_steps = np.array([2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for i in range(num_steps)])

        self.sigma_steps = sigma_steps
        self.time_steps = time_steps
        self.factor_steps = [max(f, 0) for f in factor_steps]

    def get_sigma_fn(self, schedule):
        """Returns the sigma function, its derivative, and inverse based on schedule.

        Args:
            schedule (str): The schedule type ('sqrt' or 'linear')

        Returns:
            tuple: (sigma_fn, sigma_derivative_fn, sigma_inv_fn)
        """
        if schedule == "sqrt":
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / (2 * np.sqrt(t))
            sigma_inv_fn = lambda sigma: sigma**2
        elif schedule == "linear":
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented")
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_time_step_fn(self, p, sigma_max, sigma_min):
        """Returns the time step function based on parameters.

        Args:
            p (float): Power parameter
            sigma_max (float): Maximum sigma value
            sigma_min (float): Minimum sigma value

        Returns:
            callable: Time step function
        """
        return lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p


class PDESolverDAPS(PDESolver):
    """PDESolver implementation using Decoupled Annealing Posterior Sampling (DAPS).
    
    This implementation is now adaptable to FNO, FNO_pad, SongUNO, and numerical Poisson solver.
    
    Configuration options:
        - surrogate_type (str): Type of surrogate model to use. Options:
            - "auto" (default): Automatically detects the best available model
            - "fno": Uses standard FNO surrogate model
            - "fno_pad": Uses FNO_pad surrogate model (recommended for Helmholtz equation)
            - "uno": Uses UNO surrogate model with SongUNOWrapper
            - "pino": Uses PINO surrogate model
            - "numerical_poisson": Uses numerical Poisson solver (no training required, for Poisson dataset only)
        - dataset (str): Dataset name for loading the appropriate trained surrogate model
        
    Auto-detection priority (when surrogate_type="auto"):
        1. SongUNO (uno_trained_forward_{dataset}.pth)
        2. FNO_pad (fno_pad_trained_forward_{dataset}.pth) 
        3. FNO (fno_trained_forward_{dataset}.pth)
        4. Numerical Poisson (if dataset="poisson" and solver available)
        
    Example config:
        config = {
            "surrogate_type": "numerical_poisson",  # Use numerical Poisson solver
            "dataset": "poisson",
            # ... other config options
        }
    """

    def __init__(self, config):

        super().__init__(config)

        self.annealing_config = config["guidance"]["annealing"].to_dict()
        self.diffusion_config = config["guidance"]["diffusion"].to_dict()
        self.langevin_config = config["guidance"]["langevin"].to_dict()

        self.lr = self.langevin_config["lr"]
        self.lr_min_ratio = self.langevin_config["lr_min_ratio"]
        self.langevin_steps = self.langevin_config["num_steps"]
        self.langevin_weights = torch.tensor(self.langevin_config["weights"], device=self.device).view(1, -1)

        # Dynamic Langevin control parameters
        self.dynamic_langevin = self.langevin_config.get("dynamic_control", False)
        if self.dynamic_langevin:
            self.langevin_decay_type = self.langevin_config.get("decay_type", "exponential")
            self.langevin_decay_rate = self.langevin_config.get("decay_rate", 0.5)
            self.langevin_min_steps = self.langevin_config.get("min_steps", 1)
            print(f"Dynamic Langevin control enabled: {self.langevin_decay_type} decay, rate={self.langevin_decay_rate}, min_steps={self.langevin_min_steps}")

        # Dynamic learning rate control parameters
        self.dynamic_lr = self.langevin_config.get("dynamic_lr", False)
        if self.dynamic_lr:
            self.lr_schedule = self.langevin_config.get("lr_schedule", "warmup_decay")
            self.lr_decay_rate = self.langevin_config.get("lr_decay_rate", 0.3)
            self.lr_warmup_steps = self.langevin_config.get("lr_warmup_steps", 10)
            print(f"Dynamic learning rate control enabled: {self.lr_schedule} schedule, decay_rate={self.lr_decay_rate}, warmup_steps={self.lr_warmup_steps}")

        # assert self.num_steps == self.annealing_config["num_steps"]
        self.num_steps = self.annealing_config["num_steps"]
        self.save_indices = np.linspace(0, self.num_steps - 1, self.n_process_steps, dtype=int)

        # ADDED: forward surrogate to synthesize the second channel when DM is one-channel
        # Make it adaptable to FNO, FNO_pad, and SongUNO based on configuration and available models
        # The surrogate_type config option allows switching between different surrogate types
        surrogate_type = config.get("guidance", {}).get("surrogate_type", "auto")  # Look in guidance section, default to auto-detection for backward compatibility
        print(f"Using surrogate_type: {surrogate_type}")

        # Auto-detect the best available model if surrogate_type is "auto"
        if surrogate_type.lower() == "auto":
            # Check for available models in order of preference
            if f"generation/uno_trained_forward_{config['dataset']}.pth" in os.listdir("generation"):
                surrogate_type = "uno"
                print("Auto-detected SongUNO surrogate model")
            elif f"generation/fno_pad_trained_forward_{config['dataset']}.pth" in os.listdir("generation"):
                surrogate_type = "fno_pad"
                print("Auto-detected FNO_pad surrogate model")
            elif f"generation/fno_trained_forward_{config['dataset']}.pth" in os.listdir("generation"):
                surrogate_type = "fno"
                print("Auto-detected FNO surrogate model")
            elif f"generation/pino_trained_forward_{config['dataset']}.pth" in os.listdir("generation"):
                surrogate_type = "pino"
                print("Auto-detected PINO surrogate model")
            elif HAS_NUMERICAL_POISSON and config.get('dataset', '').lower() == 'poisson':
                surrogate_type = "numerical_poisson"
                print("Auto-detected numerical Poisson solver for Poisson dataset")
            else:
                surrogate_type = "fno"  # Default fallback
                print("No specific model found, defaulting to FNO")
        
        if surrogate_type.lower() == "uno":
            # Initialize UNO surrogate with optimal configuration from training_fno.py
            self.surrogate = SongUNOWrapper(
                img_resolution=64,
                in_channels=1,
                out_channels=1,
                fmult=0.5,
                rank=0.15,  # Slightly increased from 0.1 for better expressiveness
                model_channels=64,  # Increased from 64 to 68 for ~1.5x parameters
                channel_mult=[1, 2, 2],  # Keep original for controlled growth
                num_blocks=2,  # Keep original for controlled growth
                attn_resolutions=[16],
                dropout=0.10,
                cond=False,
            )
            model_path = f"generation/uno_trained_forward_{config['dataset']}.pth"
        elif surrogate_type.lower() == "fno_pad":
            # Initialize FNO_pad surrogate (recommended for Helmholtz equation)
            self.surrogate = FNO_pad(
                n_modes=(64, 64),
                in_channels=1,
                out_channels=1,
                hidden_channels=64,
                n_layers=4
            )
            model_path = f"generation/fno_pad_trained_forward_{config['dataset']}_128_500.pth"
        elif surrogate_type.lower() == "fno_pad_scarce":
            self.surrogate = FNO_pad(
                n_modes=(32, 32),
                in_channels=1,
                out_channels=1,
                hidden_channels=64,
                n_layers=4
            )
            model_path = f"generation/fno_pad_trained_forward_{config['dataset']}_128_400_scarce500.pth"
        elif surrogate_type.lower() == "fno_pad_64":
            self.surrogate = FNO_pad(
                n_modes=(32, 32),
                in_channels=1,
                out_channels=1,
                hidden_channels=64,
                n_layers=4
            )
            model_path = f"generation/fno_pad_trained_forward_{config['dataset']}_64.pth"
        elif surrogate_type.lower() == "fno_pad_mix":
            self.surrogate = FNO_pad(
                n_modes=(32, 32),
                in_channels=1,
                out_channels=1,
                hidden_channels=64,
                n_layers=4
            )
            model_path = f"generation/fno_pad_trained_forward_{config['dataset']}_mix.pth"
        elif surrogate_type.lower() == "pino":
            self.surrogate = Pino(
                n_modes=(64, 64),
                in_channels=1,
                out_channels=1,
                hidden_channels=64,
                n_layers=4
            )
            model_path = f"generation/pino_trained_forward_{config['dataset']}.pth"
        elif surrogate_type.lower() == "numerical_poisson":
            # Use numerical Poisson solver (no neural network, no weights to load)
            if not HAS_NUMERICAL_POISSON:
                raise ImportError(
                    "Numerical Poisson solver requested but not available. "
                    "Ensure poisson_numerical_solver.py is accessible and scipy is installed."
                )
            print("Using numerical Poisson solver (no model weights needed)")
            self.surrogate = NumericalPoissonWrapper()
            model_path = None  # No model file needed
        else:
            # Default to FNO surrogate
            print("Using Default FNO surrogate")
            self.surrogate = FNO(
                n_modes=(64, 64),
                in_channels=1,
                out_channels=1,
                hidden_channels=64,
                n_layers=4
            )
            model_path = f"generation/fno_trained_forward_{config['dataset']}.pth"
        
        # Load trained forward surrogate (skip for numerical solver)
        if model_path is not None:
            print(f"Using surrogate path: {model_path}")
            try:
                # Prefer safe loading first (PyTorch >= 1.13 with weights_only)
                state_dict = torch.load(model_path, weights_only=True)
            except TypeError:
                # Older PyTorch without weights_only argument
                state_dict = torch.load(model_path)
            except Exception as e:
                # Safe loading failed (e.g. legacy checkpoints); explicit unsafe fallback
                print(
                    f"Warning: safe torch.load(weights_only=True) failed with "
                    f"{type(e).__name__}: {e}. Falling back to weights_only=False; "
                    "only do this for trusted checkpoints."
                )
                state_dict = torch.load(model_path, weights_only=False)

            self.surrogate.load_state_dict(state_dict)
        
        self.surrogate.to(self.device)
        self.surrogate.eval()
        self.surrogate.requires_grad_(False)  # freeze weights; gradients still flow to inputs

    def load_data(self):
        super().load_data()
        self.normalizer = self.dataset.create_normalizer()

    def generate_single_batch(self, observations):
        """Generate a single batch of samples using DAPS.

        Args:
            observations (list): List of observation objects

        Returns:
            tuple: (predictions, auxiliary_info)
        """
        # Initialize annealing scheduler
        annealing_scheduler = Scheduler(**self.annealing_config)

        # Initialize starting point
        latents = self.generate_latents()
        xt = latents.to(torch.float64) * annealing_scheduler.sigma_max

        # Store intermediates if requested
        intermediates = []
        intermediates_channel_index = None

        for step in tqdm(range(annealing_scheduler.num_steps), unit="step"):
            sigma_t = annealing_scheduler.sigma_steps[step]
            sigma_t_next = annealing_scheduler.sigma_steps[step + 1]

            # 1. Reverse Diffusion Step
            diffusion_scheduler = Scheduler(**self.diffusion_config, sigma_max=sigma_t)
            x0 = self._reverse_diffusion(xt, diffusion_scheduler)

            # 2. Langevin Dynamics Step with dynamic control
            x0y = self._langevin_dynamics(x0, observations, sigma_t, step / annealing_scheduler.num_steps, step)

            # 3. Forward Diffusion Step
            xt = x0y + self.noise_sampler.sample(self.batch_size) * sigma_t_next

            # Save intermediates if requested
            if self.save_indices is not None and step in self.save_indices:
                channel = 0 if x0.shape[1] == 1 else None  # the default one channel is 0
                denorm_x0 = self.normalizer.denormalize(x0, channel=channel)
                denorm_x0y = self.normalizer.denormalize(x0y, channel=channel)
                denorm_xt = self.normalizer.denormalize(xt, channel=channel)
                intermediates.append(torch.cat([denorm_x0, denorm_x0y, denorm_xt], dim=1))
                
                if intermediates_channel_index is None:
                    C1 = denorm_x0.shape[1]
                    C2 = denorm_x0y.shape[1]
                    C3 = denorm_xt.shape[1]
                    intermediates_channel_index = list(range(C1)) + list(range(C2)) + list(range(C3))

            if xt.isnan().any():
                print(f"Step {step}: NaN detected!")
                break

        # Transform final result
        x_final = xt.detach()

        # ADDED: if final has one channel, synthesize the missing channel before transform (mirrors dps.py behavior)
        if x_final.shape[1] == 1:
            device = x_final.device
            self.surrogate = self.surrogate.to(device)
            x_for_sur = x_final.to(dtype=torch.float32, device=device)
            sol_pred = self.surrogate(x_for_sur)
            x_final = torch.cat([x_for_sur, sol_pred], dim=1)

        pred = self.normalizer.transform(x_final, denormalize=True)

        return pred, {"intermediates": intermediates, "intermediates_channel_index": intermediates_channel_index}

    def _reverse_diffusion(self, x_cur, scheduler):
        """Perform reverse diffusion process.

        Args:
            x_cur (torch.Tensor): Current state
            scheduler (Scheduler): Diffusion scheduler

        Returns:
            torch.Tensor: Reversed state
        """
        for step in range(scheduler.num_steps):
            sigma_t = scheduler.sigma_steps[step]
            sigma_t_next = scheduler.sigma_steps[step + 1]
            sigma_t = torch.tensor(sigma_t, dtype=torch.float64, device=x_cur.device)
            sigma_t_next = torch.tensor(sigma_t_next, dtype=torch.float64, device=x_cur.device)

            # Euler step
            x_N = self.net(x_cur, sigma_t).to(torch.float64)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

            # 2nd order correction
            if step < scheduler.num_steps - 1:
                x_N = self.net(x_next, sigma_t_next).to(torch.float64)
                d_prime = (x_next - x_N) / sigma_t_next
                x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

            x_cur = x_next

        return x_cur

    def _langevin_dynamics(self, x0hat: torch.Tensor, observations, sigma, ratio, annealing_step):
        """Perform Langevin dynamics sampling with dynamic step control.

        Args:
            x0hat (torch.Tensor): Initial state
            observations (list): List of observation objects
            sigma (float): Current sigma value
            ratio (float): Current step ratio
            annealing_step (int): Current annealing step for dynamic control

        Returns:
            torch.Tensor: Updated state
        """
        x = x0hat.detach().clone()
        x.requires_grad_(True)

        rho = self.langevin_config["lr_rho"]
        eta = self.langevin_config["eta"]
        tau = self.langevin_config["tau"]

        # Calculate learning rate with dynamic control
        if self.dynamic_lr:
            # Use dynamic learning rate control
            current_lr = compute_dynamic_learning_rate(
                annealing_step, self.num_steps, self.lr,
                self.lr_schedule, self.lr_decay_rate, 
                self.lr_min_ratio, self.lr_warmup_steps
            )
        else:
            # Use original adaptive learning rate
            multiplier = (1 ** (1 / rho) + ratio * (self.lr_min_ratio ** (1 / rho) - 1 ** (1 / rho))) ** rho
            current_lr = multiplier * self.lr

        # Dynamic Langevin step control
        if self.dynamic_langevin:
            current_langevin_steps = compute_dynamic_langevin_steps(
                annealing_step, self.num_steps, self.langevin_steps,
                self.langevin_decay_type, self.langevin_decay_rate, self.langevin_min_steps
            )
        else:
            current_langevin_steps = self.langevin_steps

        optimizer = optim.SGD([x], lr=current_lr)

        for _ in range(current_langevin_steps):
            optimizer.zero_grad()

            # Compute loss terms
            prior_loss = ((x - x0hat.detach()) ** 2).sum()

            # ADDED: if DM state is one-channel, concatenate surrogate-predicted solution channel
            # This forward operator usage is parallel for FNO, FNO_pad, and numerical Poisson solver:
            # - Forward pass: compute solution from source term
            # - Gradients flow through: autograd tracks w.r.t. x (inputs)
            # - Loss computation and backpropagation: same pattern for all surrogates
            if x.shape[1] == 1:
                device = x.device
                self.surrogate = self.surrogate.to(device)
                x_for_sur = x.to(dtype=torch.float32, device=device)
                sol_pred = self.surrogate(x_for_sur)            # autograd tracks w.r.t. x (parallel to FNO/FNO_pad)
                x_for_loss = torch.cat([x_for_sur, sol_pred], dim=1)  # [B,2,H,W]
            else:
                x_for_loss = x

            obs_loss = []
            denorm_x = self.normalizer.denormalize(x_for_loss)
            for obs in observations:
                loss = obs.get_observation_loss(denorm_x)
                obs_loss.append(loss)
            obs_loss = torch.cat(obs_loss, dim=1)
            weighted_obs_loss = (obs_loss * self.langevin_weights).sum()

            loss = weighted_obs_loss / (2 * tau**2) + prior_loss / (2 * sigma**2)

            # Update
            loss.backward()
            optimizer.step()

            # Add noise scaled by learning rate
            with torch.no_grad():
                noise = self.noise_sampler.sample(self.batch_size) * np.sqrt(2 * current_lr) * eta
                x.add_(noise)

            # Check for numerical stability
            if torch.isnan(x).any():
                print("NaN detected in Langevin dynamics")
                return torch.zeros_like(x)

        return x.detach()
