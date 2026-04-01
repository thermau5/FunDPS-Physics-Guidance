import torch
from tqdm import tqdm

from .daps import PDESolverDAPS, Scheduler


class PDESolverDAPS_MultiRes(PDESolverDAPS):
    def __init__(self, config):
        super().__init__(config)

        self.sigma_max_1 = config["sigma_max"]
        self.sigma_min_1 = config["guidance"]["sigma_min_1"]
        self.sigma_max_2 = config["guidance"]["sigma_max_2"]
        self.sigma_min_2 = config["sigma_min"]
        self.rho = config["rho"]

        # Dynamic weights support - similar to sigma_min_1/sigma_max_2 pattern
        if "weights_1" in config["guidance"]["langevin"] and "weights_2" in config["guidance"]["langevin"]:
            self.weights_1 = config["guidance"]["langevin"]["weights_1"]  # Low resolution phase
            self.weights_2 = config["guidance"]["langevin"]["weights_2"]  # High resolution phase
            self.use_dynamic_weights = True
            print(f"Using dynamic weights: {self.weights_1} and {self.weights_2}")
        else:
            self.weights = config["guidance"]["langevin"]["weights"]
            self.use_dynamic_weights = False
            print(f"Using static weights: {self.weights}")
            
        self.init_resolution = config["guidance"]["init_resolution"]
        self.upsampling_mode = config["guidance"]["upsampling_mode"]
        self.upsampling_step = config["guidance"]["upsampling_step"]

    def generate_latents(self, resolution=None):
        if resolution is None:
            resolution = self.resolution

        if self.config["init_latents"] == "white_noise":
            return torch.randn([self.batch_size, self.n_channels, resolution, resolution], device=self.device)
        elif self.config["init_latents"] == "rbf":
            if self.noise_sampler.Ln1 != resolution:
                from training.noise_samplers import RBFKernel

                self.noise_sampler = RBFKernel(self.n_channels, resolution, resolution, scale=self.config["rbf_scale"], device=self.device)
            return self.noise_sampler.sample(self.batch_size)
        else:
            raise ValueError("Invalid init_latents value")

    def interpolate_sample(self, x, target_res):
        # batch_size, channels, height, width = x.shape

        # Perform upsampling with specific size
        x_upsampled = torch.nn.functional.interpolate(
            x,
            size=(target_res, target_res),
            mode=self.upsampling_mode,
            align_corners=True,
        )
        return x_upsampled

    def generate_single_batch(self, observations, class_labels=None):
        """Generate a single batch of samples with multi-resolution support"""
        # Set up observations for multi-resolution
        for obs in observations:
            obs.interpolation_mode = self.upsampling_mode

        # Initialize annealing scheduler
        annealing_scheduler = Scheduler(**self.annealing_config)

        # Initialize starting point at low resolution
        latents = self.generate_latents(resolution=self.init_resolution)
        xt = latents.to(torch.float64) * annealing_scheduler.sigma_max

        # Store intermediates if requested
        intermediates = []
        intermediates_channel_index = None

        for step in tqdm(range(annealing_scheduler.num_steps), unit="step"):
            sigma_t = annealing_scheduler.sigma_steps[step]
            sigma_t_next = annealing_scheduler.sigma_steps[step + 1]

            # Multi-resolution upsampling at specified step
            if step == self.upsampling_step:
                print(f"\nUpsampling at step {step}: {xt.shape[-1]} x 2")
                target_res = 2 * xt.shape[-1]
                xt = self.interpolate_sample(xt, target_res=target_res)
                # Add noise at the new resolution
                noise = self.generate_latents(resolution=target_res) * sigma_t_next
                xt = xt + noise

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

        # ADDED: if final has one channel, synthesize the missing channel before transform (mirrors daps.py behavior)
        if xt.shape[1] == 1:
            device = xt.device
            self.surrogate = self.surrogate.to(device)
            x_for_sur = xt.to(dtype=torch.float32, device=device)
            sol_pred = self._predict_solution_from_normalized_source(x_for_sur)
            xt = torch.cat([x_for_sur, sol_pred], dim=1)

        # Final downsampling if needed
        if xt.shape[-1] != self.resolution:
            assert xt.shape[-1] > self.resolution
            print(f"Final downsampling from {xt.shape[-1]} to {self.resolution}")
            xt = self.interpolate_sample(xt, target_res=self.resolution)

        pred = self.normalizer.transform(xt, denormalize=True)
        
        return pred, {"intermediates": intermediates, "intermediates_channel_index": intermediates_channel_index}

    def _langevin_dynamics(self, x0hat: torch.Tensor, observations, sigma, ratio, annealing_step):
        """Override Langevin dynamics to support dynamic weights based on resolution phase."""
        # Determine which weights to use based on current step
        if self.use_dynamic_weights:
            if annealing_step < self.upsampling_step:
                # Low resolution phase - use weights_1
                current_weights = torch.tensor(self.weights_1, device=self.device).view(1, -1)
            else:
                # High resolution phase - use weights_2  
                current_weights = torch.tensor(self.weights_2, device=self.device).view(1, -1)
        else:
            # Use static weights
            current_weights = self.langevin_weights

        # Temporarily replace the weights for this call
        original_weights = self.langevin_weights
        self.langevin_weights = current_weights
        
        # Call parent method
        result = super()._langevin_dynamics(x0hat, observations, sigma, ratio, annealing_step)
        
        # Restore original weights
        self.langevin_weights = original_weights
        
        return result

