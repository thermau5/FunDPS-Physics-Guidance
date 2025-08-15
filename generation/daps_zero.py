import torch
import torch.optim as optim

from .daps import PDESolverDAPS


class PDESolverDAPSZero(PDESolverDAPS):
    def _langevin_dynamics(self, x0hat, observations, sigma, ratio):
        x = x0hat.detach().clone()
        tau = self.langevin_config["tau"]

        # Calculate adaptive learning rate
        multiplier = 1
        if sigma < 1:
            ratio = 1 - sigma / 2
            multiplier = 1 + ratio * (self.lr_min_ratio - 1)
        current_lr = multiplier * self.lr

        current_sigma = sigma
        if current_sigma > 0.1:
            current_sigma = 0.1

        sigma_noise = torch.tensor(current_sigma, dtype=torch.float64, device=x.device)
        noise = self.noise_sampler.sample(self.batch_size) * sigma_noise
        noise_0 = noise.clone()
        noise.requires_grad_(True)

        # Optimizer: SGD, Adam, LBFGS, etc.
        optimizer = optim.SGD([noise], lr=current_lr)

        for _ in range(self.langevin_steps):
            optimizer.zero_grad()

            prior_loss = (noise**2).sum()

            # Learn the noise
            # x_0 = self.net(x + noise, sigma_noise).to(torch.float64)
            # denorm_x_0 = self.normalizer.denormalize(x_0)

            # Convert input to float32 before calling FNO/UNO/FNO_pad surrogate to match model weights
            input_to_surrogate = (x + noise).to(dtype=torch.float32)
            surrogate_x0 = self.surrogate(input_to_surrogate)
            x12 = torch.cat([x, surrogate_x0], dim=1)
            denorm_x12 = self.normalizer.denormalize(x12)

            obs_loss = []
            for obs in observations:
                loss = obs.get_observation_loss(denorm_x12)
                obs_loss.append(loss)
            obs_loss = torch.cat(obs_loss, dim=1)
            weighted_obs_loss = (obs_loss * self.langevin_weights).sum()

            loss = weighted_obs_loss / (2 * tau**2) + prior_loss / (2 * sigma**2)

            loss.backward()
            optimizer.step()

            if torch.isnan(noise).any():
                print("NaN detected in Langevin dynamics")
                return x

        # Update term
        x = x + (noise.detach() - noise_0)
        # or, second option:
        # x = x + noise.detach()
        return x
