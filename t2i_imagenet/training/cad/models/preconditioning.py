import torch
import transformers
from torch import nn

# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative networking through Stochastic
# Differential Equations".


class VPPrecond(torch.nn.Module):
    def __init__(
        self,
        network,
        label_dim=0,  # Number of class labels, 0 = unconditional.
        beta_d=19.9,  # Extent of the noise level schedule.
        beta_min=0.1,  # Initial slope of the noise level schedule.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        epsilon_t=1e-5,  # Minimum t-value used during training.
    ):
        super().__init__()
        self.label_dim = label_dim
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.network = network

    def forward(self, x, sigma, conditioning=None, force_fp32=False, **network_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        conditioning = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if conditioning is None
            else conditioning.to(torch.float32).reshape(-1, self.label_dim)
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.network(
            (c_in * x),
            c_noise.flatten(),
            conditioning=conditioning,
            **network_kwargs,
        )
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class VEPrecond(torch.nn.Module):
    def __init__(
        self,
        network,
        label_dim=0,  # Number of class labels, 0 = unconditional.
        sigma_min=0.02,  # Minimum supported noise level.
        sigma_max=100,  # Maximum supported noise level.
    ):
        super().__init__()
        self.label_dim = label_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.network = network

    def forward(self, x, sigma, conditioning=None, **network_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        conditioning = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if conditioning is None
            else conditioning.to(torch.float32).reshape(-1, self.label_dim)
        )

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.network(
            (c_in * x),
            c_noise.flatten(),
            conditioning=conditioning,
            **network_kwargs,
        )
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative networks" (EDM).


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        network,
        label_dim=0,  # Number of class labels, 0 = unconditional.
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.label_dim = label_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.network = network

    def forward(self, x, sigma, conditioning=None, **network_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        conditioning = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if conditioning is None
            else conditioning.to(torch.float32)
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.network(
            (c_in * x),
            c_noise.flatten(),
            conditioning=conditioning,
            **network_kwargs,
        )
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class DDPMPrecond(nn.Module):
    def __init__(self, num_latents=256, latents_dim=512):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latents_dim

    def forward(self, network, batch):
        if "previous_latents" not in batch or batch["previous_latents"] is None:
            batch["previous_latents"] = torch.zeros(
                batch["y"].shape[0],
                self.num_latents,
                self.latent_dim,
                device=batch["y"].device,
                dtype=torch.float32,
            )
        F_x, z = network(batch)
        return F_x, z
