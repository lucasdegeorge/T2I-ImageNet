# Code chaged from https://github.com/NVlabs/edm

import random

import torch

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


def loss_reco(D_y, y, weight, loss_type="l2", hinge_margin=1.0):
    if loss_type == "l2":
        return weight * ((D_y - y) ** 2)
    elif loss_type == "l1":
        return weight * torch.abs(D_y - y)
    elif loss_type == "hinge_l2":
        loss = weight * torch.nn.functional.relu((D_y - y) ** 2 - hinge_margin)
        reweigth_zeros = (loss != 0).sum(
            [i for i in range(loss.dim())], keepdim=True
        ) / loss.numel()
        return loss / (reweigth_zeros + 1e-8)


class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5, loss_type="l2"):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.loss_type = loss_type

    def __call__(self, net, images, conditioning, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0]], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = (1 / sigma**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        D_yn = net(y + n, sigma, conditioning, augment_labels=augment_labels)
        loss = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, loss_type="l2"):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.loss_type = loss_type

    def __call__(self, net, images, conditioning, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0]], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = (1 / sigma**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        D_yn = net(y + n, sigma, conditioning, augment_labels=augment_labels)
        loss = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, loss_type="l2"):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.loss_type = loss_type

    def __call__(self, net, images, conditioning=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0]], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (
            ((sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        D_yn = net(y + n, sigma, conditioning, augment_labels=augment_labels)
        loss = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * ((D_yn - y) ** 2)
        return loss


class DDPMLoss:
    def __init__(
        self,
        scheduler,
        uncond_conditioning,
        self_cond_rate=0.9,
        cond_drop_rate=0.0,
        conditioning_key="label",
        resample_by_coherence=False,
        sample_random_when_drop=False,
        do_offset_noise=False,
    ):
        self.scheduler = scheduler
        self.self_cond_rate = self_cond_rate
        self.cond_drop_rate = cond_drop_rate
        self.uncond_conditioning = uncond_conditioning
        self.conditioning_key = conditioning_key
        self.resample_by_coherence = resample_by_coherence
        self.sample_random_when_drop = sample_random_when_drop
        self.do_offset_noise = do_offset_noise

    def __call__(self, preconditioning, network, batch, t=None, generator=None):
        x_0 = batch["x_0"]
        batch_size = x_0.shape[0]
        device = x_0.device
        if t is None:
            t = torch.rand(batch_size, device=device, dtype=x_0.dtype, generator=generator)
        assert isinstance(t, torch.Tensor) and t.shape[0] == batch_size and t.dim() == 1
        gamma = self.scheduler(t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        n = torch.randn(x_0.shape, device=device, dtype=x_0.dtype, generator=generator)
        if self.do_offset_noise:
            n = n + 0.1 * torch.randn(
                x_0.shape[0], x_0.shape[1], 1, 1, device=device, dtype=x_0.dtype, generator=generator
            )
        y = torch.sqrt(gamma) * x_0 + torch.sqrt(1 - gamma) * n
        batch["y"] = y
        if f"{self.conditioning_key}_embeddings" in batch:
            conditioning = batch[f"{self.conditioning_key}_embeddings"]
            if f"{self.conditioning_key}_mask" in batch:
                conditioning_mask = batch[f"{self.conditioning_key}_mask"]
        elif self.conditioning_key in batch:
            conditioning = batch[self.conditioning_key]
        if conditioning is not None and self.cond_drop_rate > 0:
            drop_mask = (
                torch.rand(batch_size, device=device, generator=generator)
                < self.cond_drop_rate
            )
            if self.sample_random_when_drop:
                num_samples_to_drop = drop_mask.sum().item()
                if num_samples_to_drop > 0:
                    new_samples = randint_except(
                        (num_samples_to_drop,),
                        0,
                        batch_size,
                        torch.tensor(
                            [i for i in range(batch_size) if drop_mask[i]],
                            device=device,
                        ),
                        generator=generator,
                    )
                    conditioning[drop_mask] = conditioning[new_samples]
                    if f"{self.conditioning_key}_mask" in batch:
                        conditioning_mask[drop_mask] = conditioning_mask[new_samples]
            else:
                if isinstance(self.uncond_conditioning, str):
                    for i, drop_i in enumerate(drop_mask):
                        if drop_i:
                            conditioning[i] = self.uncond_conditioning
                elif isinstance(self.uncond_conditioning, float) or isinstance(
                    self.uncond_conditioning, int
                ):
                    for i, drop_i in enumerate(drop_mask):
                        if drop_i:
                            conditioning[i] = (
                                torch.ones_like(conditioning[i])
                                * self.uncond_conditioning
                            )
                else:
                    uncond = self.uncond_conditioning[: conditioning.shape[1]]
                    conditioning[drop_mask] = uncond
                if f"{self.conditioning_key}_mask" in batch:
                    conditioning_mask[drop_mask] = torch.tensor(
                        [1] + [0] * (conditioning_mask.shape[-1] - 1),
                        device=device,
                        dtype=torch.bool,
                    )
            batch[self.conditioning_key] = conditioning.detach()
            if f"{self.conditioning_key}_mask" in batch:
                batch[f"{self.conditioning_key}_mask"] = conditioning_mask
            if "coherence" in batch:
                batch["coherence"] = (batch["coherence"]) * (
                    1 - drop_mask.to(batch["coherence"].dtype)
                )
        batch["gamma"] = gamma.squeeze(-1).squeeze(-1).squeeze(-1)
        batch["previous_latents"] = self.latent_self_cond(
            preconditioning,
            network,
            batch,
            generator=generator,
        )
        D_n, _ = preconditioning(network, batch)
        loss = (D_n - n) ** 2
        if self.resample_by_coherence:
            assert "coherence" in batch
            weight = batch["coherence"].to(loss.dtype).unsqueeze(-1).unsqueeze(
                -1
            ).unsqueeze(-1) / (batch["coherence"].to(loss.dtype).mean() + 1e-8)
            loss = loss * weight.to(loss.dtype)
        return loss

    def latent_self_cond(self, preconditioning, network, batch, generator=None):
        # with torch.no_grad(): # Cannot because of torch.amp bug
        _, latents = preconditioning(network, batch)
        drop_mask = (
            torch.rand(
                latents.shape[0],
                *[1 for _ in range(latents.dim() - 1)],
                device=latents.device,
                generator=generator,
            )
            > 1 - self.self_cond_rate
        ).to(latents.dtype)
        latents = latents * drop_mask
        latents = latents.detach()
        network.zero_grad()
        return latents


def randint_except(shape, low, high, except_array, generator=None):
    samples = torch.randint(
        low, high - 1, size=shape, device=except_array.device, generator=generator
    )
    samples = torch.where(samples >= except_array, samples + 1, samples)
    return samples
