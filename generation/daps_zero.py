import torch
import torch.optim as optim

from .daps import PDESolverDAPS


class PDESolverDAPSZero(PDESolverDAPS):
    def _langevin_dynamics(self, x0hat, observations, sigma, ratio):
        x = x0hat.detach().clone()
        tau = self.langevin_config["tau"]

        # Calculate adaptive learning rate
        if sigma > 1:
            multiplier = 1
        else:
            ratio = 1 - sigma / 2
            multiplier = 1 + ratio * (self.lr_min_ratio - 1)
        current_lr = multiplier * self.lr

        sigma_noise = torch.tensor(sigma, dtype=torch.float64, device=x.device)
        noise = self.noise_sampler.sample(self.batch_size) * sigma_noise
        noise_0 = noise.clone()
        noise.requires_grad_(True)

        optimizer = optim.SGD([noise], lr=current_lr)

        for _ in range(self.langevin_steps):
            optimizer.zero_grad()

            prior_loss = (noise**2).sum()

            x_0 = self.net(x + noise, sigma_noise).to(torch.float64)
            denorm_x_0 = self.normalizer.denormalize(x_0)

            obs_loss = []
            for obs in observations:
                loss = obs.get_observation_loss(denorm_x_0)
                obs_loss.append(loss)
            obs_loss = torch.cat(obs_loss, dim=1)
            weighted_obs_loss = (obs_loss * self.langevin_weights).sum()

            loss = weighted_obs_loss / (2 * tau**2) + prior_loss / (2 * sigma**2)

            loss.backward()
            optimizer.step()

            if torch.isnan(noise).any():
                print("NaN detected in Langevin dynamics")
                return x

        x = x + (noise.detach() - noise_0)
        return x
