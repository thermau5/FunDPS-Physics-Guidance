import torch
import torch.nn.functional as F
from tqdm import tqdm
import os  # Added for auto-detection

from .base import PDESolver
from generation.observation import FNO  # For FNO surrogate
from training.networks import SongUNO  # For UNO surrogate

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


class PDESolverDPS(PDESolver):
    def __init__(self, config):
        super().__init__(config)

        self.sigma_min = config["sigma_min"]
        self.sigma_max = config["sigma_max"]
        self.rho = config["rho"]

        self.weights = config["guidance"]["weights"]

        # ADDED: forward surrogate to synthesize the second channel when DM is one-channel
        # Make it adaptable to FNO, FNO_pad, SongUNO, and related variants based on configuration and available models.
        # The surrogate_type config option allows switching between different surrogate types.
        guidance_cfg = config.get("guidance", {})
        # Config.get supports a default argument; always pass one to avoid ConfigKeyError
        surrogate_type = guidance_cfg.get("surrogate_type", "auto")  # default to auto-detection for backward compatibility
        # Allow users to specify an explicit surrogate checkpoint path (recommended)
        # Prefer path under guidance.surrogate_path, but also support a top-level surrogate_path for convenience.
        surrogate_path = guidance_cfg.get("surrogate_path", None) or config.get("surrogate_path", None)
        print(f"Using surrogate_type: {surrogate_type}")
        if surrogate_path is not None:
            print(f"Using surrogate_path from config: {surrogate_path}")

        # Auto-detect the best available model if surrogate_type is "auto" and
        # no explicit surrogate_path was provided.
        if surrogate_type.lower() == "auto" and surrogate_path is None:
            # Check for available models in order of preference
            generation_files = set(os.listdir("generation")) if os.path.isdir("generation") else set()
            if f"uno_trained_forward_{config['dataset']}.pth" in generation_files:
                surrogate_type = "uno"
                print("Auto-detected SongUNO surrogate model")
            elif f"fno_pad_trained_forward_{config['dataset']}.pth" in generation_files:
                surrogate_type = "fno_pad"
                print("Auto-detected FNO_pad surrogate model")
            elif f"fno_trained_forward_{config['dataset']}.pth" in generation_files:
                surrogate_type = "fno"
                print("Auto-detected FNO surrogate model")
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

        if surrogate_type.lower() == "uno":
            self.surrogate = SongUNOWrapper(**_merge_fno_kwargs(uno_defaults))
            default_model_path = f"generation/uno_trained_forward_{config['dataset']}.pth"
        elif surrogate_type.lower() == "fno_pad":
            self.surrogate = FNO_pad(**_merge_fno_kwargs(fno_defaults))
            default_model_path = f"generation/fno_pad_trained_forward_{config['dataset']}.pth"
        elif surrogate_type.lower() == "fno_pad_scarce":
            self.surrogate = FNO_pad(**_merge_fno_kwargs(fno_small_defaults))
            default_model_path = f"generation/fno_pad_trained_forward_{config['dataset']}_128_400_scarce500.pth"
        elif surrogate_type.lower() == "fno_pad_64":
            self.surrogate = FNO_pad(**_merge_fno_kwargs(fno_small_defaults))
            default_model_path = f"generation/fno_pad_trained_forward_{config['dataset']}_64.pth"
        elif surrogate_type.lower() == "fno_pad_mix":
            self.surrogate = FNO_pad(**_merge_fno_kwargs(fno_small_defaults))
            default_model_path = f"generation/fno_pad_trained_forward_{config['dataset']}_mix.pth"
        else:
            # Surrogate Not Specified/Used
            self.surrogate = None
            default_model_path = None
            print(f"Warning: surrogate_type '{surrogate_type}' not recognized. Surrogate will not be used.")
            return

        # Decide which path to use for loading weights (if any).
        # Priority:
        #   1) Explicit surrogate_path from config (recommended, model-agnostic)
        #   2) Legacy default_model_path based on dataset/surrogate_type (for backward compatibility)
        model_path = surrogate_path or default_model_path

        # Load trained forward surrogate (skip if surrogate is None or no path is provided)
        if self.surrogate is not None and model_path is not None:
            print(f'Using surrogate path: {model_path}')
            try:
                if hasattr(torch, '__version__') and torch.__version__ >= '1.13.0':
                    loaded = torch.load(model_path, weights_only=True)
                else:
                    loaded = torch.load(model_path)
            except Exception:
                loaded = torch.load(model_path)
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

    def load_data(self):
        super().load_data()
        self.normalizer = self.dataset.create_normalizer()

    def generate_single_batch(self, observations, class_labels=None):
        """Generate a single batch of samples"""
        # Initialize latents
        latents = self.generate_latents()

        # Set up sigma schedule
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=self.device)
        sigma_t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        sigma_t_steps = torch.cat([self.net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])
        self.sigma_t_steps = sigma_t_steps

        # Initial state
        x_next = latents.to(torch.float64) * sigma_t_steps[0]
        intermediates = []
        intermediates_channel_index = None
        loss_history = []

        for i, (sigma_t_cur, sigma_t_next) in enumerate(tqdm(zip(sigma_t_steps[:-1], sigma_t_steps[1:]), total=self.num_steps)):
            x_cur = x_next.detach().clone()
            if any(w != 0 for w in self.weights):  # Only enable gradient tracking if any guidance weight is non-zero
                x_cur.requires_grad_(True)
            else:
                # When all guidance weights are 0, don't enable gradient tracking to save memory
                x_cur.requires_grad_(False)
            sigma_t = self.net.round_sigma(sigma_t_cur)
            # print("x_cur.requires_grad:", x_cur.requires_grad)
            # print("x_cur.grad_fn:", x_cur.grad_fn)

            # Euler step
            x_N = self.net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

            # 2nd order correction
            if i < self.num_steps - 1:
                x_N = self.net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
                d_prime = (x_next - x_N) / sigma_t_next
                x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

            # print('Before Concatenate...')
            # print(f"x_N.shape: {x_N.shape}")
            # print("x_N.requires_grad:", x_N.requires_grad)
            # print("x_N.grad_fn:", x_N.grad_fn)

            # Concatenate into two channel
            if x_N.shape[1] == 1:
                if self.surrogate is None:
                    raise ValueError("Cannot concatenate channels: surrogate model is None. Please specify a valid surrogate_type in the config.")
                # Predict the second channel using the surrogate
                # Make sure x_N is the correct dtype and device for the surrogate
                device = x_N.device
                self.surrogate = self.surrogate.to(device)
                x_N = x_N.to(dtype=torch.float32, device=device)
                # Call surrogate WITHOUT torch.no_grad()
                sol_pred = self.surrogate(x_N)  # This will be tracked by autograd
                # Concatenate along channel dimension
                x_N = torch.cat([x_N, sol_pred], dim=1)  # [batch, 2, H, W]
                
            # print('After Concatenate...')
            # print(f"x_N.shape: {x_N.shape}")
            # print("x_N.requires_grad:", x_N.requires_grad)
            # print("x_N.grad_fn:", x_N.grad_fn)
                    
            # Denormalization
            denorm_x_N = self.normalizer.denormalize(x_N) # always two channels
            channel = 0 if x_next.shape[1] == 1 else None  # the default one channel is 0
            denorm_x_next = self.normalizer.denormalize(x_next.detach(), channel=channel)
            # print(f"denorm_x_N.shape: {denorm_x_N.shape}")
            # print(f"denorm_x_next.shape: {denorm_x_next.shape}")

            update = torch.zeros_like(x_cur)
            if i < self.num_steps - 1:
                weight_ptr = 0
                step_losses = []
                active_losses = []

                for obs in observations:
                    loss = obs.get_observation_loss(denorm_x_N)
                    step_losses.append(loss.detach())

                    coef = self.get_coef(cur_step=i, obs_type=obs.type)
                    for c in range(obs.n_channels):
                                                
                        if weight_ptr >= len(self.weights):
                            raise IndexError(f"weight_ptr ({weight_ptr}) exceeded length of weights ({len(self.weights)}). Check that 'guidance.weights' matches the total number of observation channels.")

                        if coef != 0 and self.weights[weight_ptr] != 0:
                            active_losses.append((loss[:, c].sum(), coef * self.weights[weight_ptr]))  # NOTE: for numerical stability, we don't weight the loss here
                        weight_ptr += 1

                for idx, (loss, weight) in enumerate(active_losses):
                    flag_retain_graph = idx < len(active_losses) - 1
                    # print(f'loss: {loss}')
                    # print("loss.requires_grad:", loss.requires_grad)
                    # print("loss.grad_fn:", loss.grad_fn if hasattr(loss, 'grad_fn') else None)
                    grad = torch.autograd.grad(loss, x_cur, retain_graph=flag_retain_graph)[0]
                    update = update + weight * grad

                loss_history.append(torch.cat(step_losses, dim=1))

            # Project update
            if getattr(self, "project_gradient", None) is not None:
                update = self.project_gradient(x_next, update)

            # Apply updates
            x_next = x_next - update

            if x_next.isnan().any():
                print(f"Step {i}: NaN detected!")
                break

            # Save intermediate results if requested
            if self.save_indices is not None and i in self.save_indices:
                denorm_x_updated = self.normalizer.denormalize(x_next.detach(), channel=channel)
                # print("denorm_x_N.detach().shape:", denorm_x_N.detach().shape)    # torch.Size([8, 2, 128, 128])
                # print("denorm_x_next.shape:", denorm_x_next.shape)                # torch.Size([8, 1, 128, 128])
                # print("denorm_x_updated.shape:", denorm_x_updated.shape)          # torch.Size([8, 1, 128, 128])
                intermediates.append(torch.cat([denorm_x_N.detach(), denorm_x_next, denorm_x_updated], dim=1))
                
                if intermediates_channel_index is None:
                    C1 = denorm_x_N.shape[1]
                    C2 = denorm_x_next.shape[1]
                    C3 = denorm_x_updated.shape[1]
                    intermediates_channel_index = list(range(C1)) + list(range(C2)) + list(range(C3))

        x_final = x_next.detach()

        if x_final.shape[1] == 1:
            if self.surrogate is None:
                raise ValueError("Cannot concatenate channels: surrogate model is None. Please specify a valid surrogate_type in the config.")
            # Predict the second channel using the surrogate
            device = x_final.device
            x_final = x_final.to(dtype=torch.float32, device=device)
            self.surrogate = self.surrogate.to(device)
            sol_pred = self.surrogate(x_final)
            x_final = torch.cat([x_final, sol_pred], dim=1)  # [batch, 2, H, W]
            
        pred = self.normalizer.transform(x_final, denormalize=True)
        aux = {"intermediates": intermediates, "intermediates_channel_index": intermediates_channel_index, "loss_history": loss_history} #3
        return pred, aux

    def get_coef(self, cur_step, obs_type):
        if self.sigma_t_steps[cur_step] > 1.0:
            if obs_type == "pde":
                return 0
            else:
                return 1
        else:
            return self.sigma_t_steps[cur_step].item()
