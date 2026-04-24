import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os  # Added for auto-detection

from training.dataset_utils import DatasetNormalizer

from .base import PDESolver
from generation.observation import FNO  # For FNO surrogate
from training.networks import SongUNO  # For UNO surrogate

# NumericalPoissonWrapper is fully self-contained below (no external solver file needed)
HAS_NUMERICAL_POISSON = True


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
        # Create dummy noise_labels and class_labels for training
        batch_size = x.shape[0]
        device = x.device

        # Create dummy noise_labels (timestep 0 for deterministic prediction)
        noise_labels = torch.zeros(batch_size, device=device)

        # Create dummy class_labels (unconditional)
        class_labels = torch.zeros(batch_size, 0, device=device)  # Empty class labels

        return self.model(x, noise_labels, None)

# Wrapper class for numerical Poisson solver (GPU-accelerated, FFT-based)
class NumericalPoissonWrapper(torch.nn.Module):
    """Solves Δφ = f with homogeneous Dirichlet BCs via 2D DST-I (O(N log N)).

    Takes [B, 1, H, W] (source term f) and returns [B, 1, H, W] (solution φ).
    Gradients flow through torch.fft, supporting backprop w.r.t. inputs.

    Algorithm: diagonalise the 5-point stencil Laplacian with the discrete sine
    transform (DST-I).  For interior grid size M = H-2 and h = 1/(H-1):
        A φ_int = h² f_int,   A = T⊗I + I⊗T,  T = tridiag(1,-2,1)
        eigenvalues: λ_{mn} = 2cos(mπ/(M+1)) + 2cos(nπ/(M+1)) - 4
        φ_int = IDST2(DST2(h² f_int) / λ)
    """

    def __init__(self):
        super().__init__()
        self._eigenvalue_cache = {}

    @staticmethod
    def _dst1_1d(x):
        """1D DST-I along last dim via FFT extension.

        Convention: V_k = Σ_{n=1}^{M} v_n sin(nkπ/(M+1)), k=1..M
        Self-inverse up to factor (M+1)/2: DST(DST(v)) = (M+1)/2 · v
        """
        M = x.shape[-1]
        N = 2 * (M + 1)
        z = torch.zeros(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)
        # Anti-symmetric extension: [0, v, 0, -flip(v)]
        x_ext = torch.cat([z, x, z, x.flip(-1).neg()], dim=-1)
        X = torch.fft.rfft(x_ext, n=N, dim=-1)
        # Im(FFT_k) = -2 V_k  =>  V_k = -Im(FFT_k)/2
        return -X.imag[..., 1 : M + 1] / 2.0

    @classmethod
    def _dst2_2d(cls, x):
        """2D DST-I: apply 1D DST-I along rows then columns."""
        return cls._dst1_1d(cls._dst1_1d(x.transpose(-1, -2)).transpose(-1, -2))

    def _get_eigenvalues(self, M, device, dtype):
        key = (M, device, dtype)
        if key not in self._eigenvalue_cache:
            k = torch.arange(1, M + 1, device=device, dtype=dtype)
            eig1d = 2.0 * torch.cos(k * torch.pi / (M + 1)) - 2.0  # [M]
            self._eigenvalue_cache[key] = eig1d[:, None] + eig1d[None, :]  # [M, M]
        return self._eigenvalue_cache[key]

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, H, W], got {x.shape}")
        B, _, S, _ = x.shape
        M = S - 2  # interior grid size
        h = 1.0 / (S - 1)

        f_int = x[:, 0, 1:-1, 1:-1]  # [B, M, M]
        rhs = (h * h) * f_int

        # Forward 2D DST-I
        F = self._dst2_2d(rhs)  # [B, M, M]

        # Divide by eigenvalues in spectral space
        eig = self._get_eigenvalues(M, x.device, x.dtype)  # [M, M]
        Phi_hat = F / eig  # [B, M, M]

        # Inverse 2D DST-I: IDST2 = 4/(M+1)² · DST2
        phi_int = self._dst2_2d(Phi_hat) * (4.0 / (M + 1) ** 2)  # [B, M, M]

        # Embed interior solution into full grid (boundary stays zero = Dirichlet BC)
        phi = torch.zeros_like(x[:, 0])  # [B, S, S]
        phi[:, 1:-1, 1:-1] = phi_int
        return phi.unsqueeze(1)  # [B, 1, S, S]


def compute_dynamic_langevin_steps(annealing_step, total_annealing_steps, base_langevin_steps, decay_type="exponential", decay_rate=0.5, min_steps=1):
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
        steps = int(base_langevin_steps * (decay_rate**step_ratio))
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


def compute_dynamic_learning_rate(annealing_step, total_annealing_steps, base_lr, lr_schedule="warmup_decay", lr_decay_rate=0.3, lr_min_ratio=0.01, lr_warmup_steps=10):
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
        lr = base_lr * (lr_decay_rate**step_ratio)
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
        1. SongUNO (artifacts/models/legacy/uno_trained_forward_{dataset}.pth)
        2. FNO_pad (artifacts/models/legacy/fno_pad_trained_forward_{dataset}.pth)
        3. FNO (artifacts/models/legacy/fno_trained_forward_{dataset}.pth)
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

        self.task = (config.get("task", "inverse") or "inverse").lower()
        if self.task not in {"forward", "inverse", "forward_inverse"}:
            print(f"Warning: unknown task '{self.task}', defaulting to 'inverse'")
            self.task = "inverse"

        # Optionally report which diffusion model checkpoint is being used
        diffusion_path = config.get("pkl_path", None)
        if diffusion_path is not None:
            print(f"Using diffusion path: {diffusion_path}")

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
        # Make it adaptable to FNO, FNO_pad, SongUNO, PINO, and numerical Poisson based on configuration and available models.
        # The surrogate_type config option allows switching between different surrogate types.
        guidance_cfg = config.get("guidance", {})
        # Config.get supports a default argument; always pass one to avoid ConfigKeyError
        surrogate_type = guidance_cfg.get("surrogate_type", "auto")  # default to auto-detection for backward compatibility
        # Allow users to specify an explicit surrogate checkpoint path (recommended)
        # Prefer path under guidance.surrogate_path, but also support a top-level surrogate_path for convenience.
        surrogate_path = guidance_cfg.get("surrogate_path", None) or config.get("surrogate_path", None)
        print(f"Using surrogate_type: {surrogate_type}")

        # Auto-detect the best available model if surrogate_type is "auto" and
        # no explicit surrogate_path was provided.
        if surrogate_type.lower() == "auto" and surrogate_path is None:
            # Check for available models in order of preference
            legacy_models_dir = os.path.join("artifacts", "models", "legacy")
            generation_files = set(os.listdir(legacy_models_dir)) if os.path.isdir(legacy_models_dir) else set()
            if f"uno_trained_forward_{config['dataset']}.pth" in generation_files:
                surrogate_type = "uno"
                print("Auto-detected SongUNO surrogate model")
            elif f"fno_pad_trained_forward_{config['dataset']}_128_500.pth" in generation_files:
                surrogate_type = "fno_pad"
                print("Auto-detected FNO_pad surrogate model")
            elif f"fno_trained_forward_{config['dataset']}.pth" in generation_files:
                surrogate_type = "fno"
                print("Auto-detected FNO surrogate model")
            elif f"pino_trained_forward_{config['dataset']}.pth" in generation_files:
                surrogate_type = "pino"
                print("Auto-detected PINO surrogate model")
            elif HAS_NUMERICAL_POISSON and config.get("dataset", "").lower() == "poisson":
                surrogate_type = "numerical_poisson"
                print("Auto-detected numerical Poisson solver for Poisson dataset")
            else:
                surrogate_type = "fno"  # Default fallback
                print("No specific model found, defaulting to FNO")

        # Surrogate architecture: read from config['surrogate_config'] with per-type defaults.
        surr_cfg_raw = config.get("surrogate_config", None)
        if surr_cfg_raw is not None and hasattr(surr_cfg_raw, "to_dict"):
            surr_cfg = surr_cfg_raw.to_dict()
        elif isinstance(surr_cfg_raw, dict):
            surr_cfg = surr_cfg_raw
        else:
            surr_cfg = {}

        def _merge_fno_kwargs(defaults):
            out = dict(defaults)
            for k, v in surr_cfg.items():
                if k in out:
                    if k == "n_modes" and isinstance(v, list):
                        out[k] = tuple(v)
                    else:
                        out[k] = v
            return out

        # Default model_path templates (used only when surrogate_path is not provided).
        default_model_path = None
        fno_defaults = {"n_modes": (64, 64), "in_channels": 1, "out_channels": 1, "hidden_channels": 64, "n_layers": 4}
        fno_small_defaults = {"n_modes": (32, 32), "in_channels": 1, "out_channels": 1, "hidden_channels": 64, "n_layers": 4}
        uno_defaults = {"img_resolution": 64, "in_channels": 1, "out_channels": 1, "fmult": 0.5, "rank": 0.15, "model_channels": 64, "channel_mult": [1, 2, 2], "num_blocks": 2, "attn_resolutions": [16], "dropout": 0.10, "cond": False}

        self.uses_numerical_poisson = surrogate_type.lower() == "numerical_poisson"

        if surrogate_type.lower() == "uno":
            kwargs = _merge_fno_kwargs(uno_defaults)
            self.surrogate = SongUNOWrapper(**kwargs)
            default_model_path = f"artifacts/models/legacy/uno_trained_forward_{config['dataset']}.pth"
        elif surrogate_type.lower() == "fno_pad":
            self.surrogate = FNO_pad(**_merge_fno_kwargs(fno_defaults))
            default_model_path = f"artifacts/models/legacy/fno_pad_trained_forward_{config['dataset']}_128_500.pth"
        elif surrogate_type.lower() == "fno_pad_scarce":
            self.surrogate = FNO_pad(**_merge_fno_kwargs(fno_small_defaults))
            default_model_path = f"artifacts/models/legacy/fno_pad_trained_forward_{config['dataset']}_128_400_scarce500.pth"
        elif surrogate_type.lower() == "fno_pad_64":
            self.surrogate = FNO_pad(**_merge_fno_kwargs(fno_small_defaults))
            default_model_path = f"artifacts/models/legacy/fno_pad_trained_forward_{config['dataset']}_64.pth"
        elif surrogate_type.lower() == "fno_pad_mix":
            self.surrogate = FNO_pad(**_merge_fno_kwargs(fno_small_defaults))
            default_model_path = f"artifacts/models/legacy/fno_pad_trained_forward_{config['dataset']}_mix.pth"
        elif surrogate_type.lower() == "pino":
            self.surrogate = Pino(**_merge_fno_kwargs(fno_defaults))
            default_model_path = f"artifacts/models/legacy/pino_trained_forward_{config['dataset']}.pth"
        elif surrogate_type.lower() == "numerical_poisson":
            if not HAS_NUMERICAL_POISSON:
                raise ImportError("Numerical Poisson solver requested but not available. " "Ensure poisson_numerical_solver.py is accessible and scipy is installed.")
            print("Using numerical Poisson solver (no model weights needed)")
            self.surrogate = NumericalPoissonWrapper()
            model_path = None  # No model file needed
        else:
            print("Using Default FNO surrogate")
            self.surrogate = FNO(**_merge_fno_kwargs(fno_defaults))
            default_model_path = f"artifacts/models/legacy/fno_trained_forward_{config['dataset']}.pth"

        # Decide which path to use for loading weights (if any).
        # Priority:
        #   1) Explicit surrogate_path from config (recommended, model-agnostic)
        #   2) Legacy default_model_path based on dataset/surrogate_type (for backward compatibility)
        if surrogate_type.lower() != "numerical_poisson":
            model_path = surrogate_path or default_model_path
        else:
            model_path = None

        # Load trained forward surrogate (skip for numerical solver or when no path is provided)
        if model_path is not None:
            print(f"Using surrogate path: {model_path}")
            try:
                loaded = torch.load(model_path, weights_only=True)
            except TypeError:
                loaded = torch.load(model_path)
            except Exception as e:
                print(
                    f"Warning: safe torch.load(weights_only=True) failed with "
                    f"{type(e).__name__}: {e}. Falling back to weights_only=False; "
                    "only do this for trusted checkpoints."
                )
                loaded = torch.load(model_path, weights_only=False)
            # Support both raw state dict and training checkpoint format {'epoch', 'model_state', 'optimizer_state', 'config'}
            if isinstance(loaded, dict) and "model_state" in loaded:
                state_dict = loaded["model_state"]
                print("Checkpoint format: training checkpoint (using 'model_state').")
            else:
                state_dict = loaded
                print("Checkpoint format: state dict.")
            self.surrogate.load_state_dict(state_dict)
        self.surrogate.to(self.device)
        self.surrogate.eval()
        self.surrogate.requires_grad_(False)  # freeze weights; gradients still flow to inputs

    def _predict_solution_from_normalized_source(self, x_source_normalized):
        """Predict normalized solution channel from normalized source channel.

        Numerical Poisson surrogate expects/returns physical units, while learned
        surrogates expect/return normalized units.
        """
        if self.uses_numerical_poisson:
            # Channel 0 is source term f; channel 1 is solution phi.
            x_source_physical = self.normalizer.denormalize(x_source_normalized, channel=0)
            sol_physical = self.surrogate(x_source_physical)
            return self.normalizer.normalize(sol_physical, channel=1)
        return self.surrogate(x_source_normalized)

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
            sol_pred = self._predict_solution_from_normalized_source(x_for_sur)
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
            current_lr = compute_dynamic_learning_rate(annealing_step, self.num_steps, self.lr, self.lr_schedule, self.lr_decay_rate, self.lr_min_ratio, self.lr_warmup_steps)
        else:
            # Use original adaptive learning rate
            multiplier = (1 ** (1 / rho) + ratio * (self.lr_min_ratio ** (1 / rho) - 1 ** (1 / rho))) ** rho
            current_lr = multiplier * self.lr

        # Dynamic Langevin step control
        if self.dynamic_langevin:
            current_langevin_steps = compute_dynamic_langevin_steps(annealing_step, self.num_steps, self.langevin_steps, self.langevin_decay_type, self.langevin_decay_rate, self.langevin_min_steps)
        else:
            current_langevin_steps = self.langevin_steps

        optimizer = optim.SGD([x], lr=current_lr)

        for _ in range(current_langevin_steps):
            optimizer.zero_grad()

            # Compute loss terms
            prior_loss = ((x - x0hat.detach()) ** 2).sum()

            # If DM state is one-channel, we may need a two-channel tensor for observation losses.
            #
            # - inverse / forward_inverse: use surrogate inside Langevin so observation can involve solution channel.
            # - forward: only guide on coefficient/channel-0 observations; avoid surrogate calls inside Langevin.
            if x.shape[1] == 1:
                device = x.device
                x_for_sur = x.to(dtype=torch.float32, device=device)
                if self.task in {"inverse", "forward_inverse"}:
                    self.surrogate = self.surrogate.to(device)
                    sol_pred = self._predict_solution_from_normalized_source(x_for_sur)
                    x_for_loss = torch.cat([x_for_sur, sol_pred], dim=1)  # [B,2,H,W]
                else:
                    # Dummy second channel (ignored when observation masks/weights for channel-1 are zero).
                    x_for_loss = torch.cat([x_for_sur, torch.zeros_like(x_for_sur)], dim=1)
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
