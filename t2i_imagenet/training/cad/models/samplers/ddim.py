import torch

from .base_sampler import Sampler


class DDIMSampler(Sampler):
    def compute_next_step(
        self,
        x_cur,
        gamma_now,
        gamma_next,
        latents_cond,
        latents_uncond,
        process_step_kwargs,
        thresholding_kwargs,
        generator,
    ):
        denoised, latents_cond, latents_uncond = self.process_step(
            x_cur,
            gamma_now,
            latents_cond,
            latents_uncond,
            **process_step_kwargs,
        )

        x_pred = (x_cur - torch.sqrt(1 - gamma_now) * denoised) / torch.sqrt(gamma_now)
        x_pred = self.apply_thresholding(x_pred, **thresholding_kwargs)
        noise_pred = (x_cur - torch.sqrt(gamma_now) * x_pred) / torch.sqrt(
            1 - gamma_now
        )

        x_next = (
            torch.sqrt(gamma_next) * x_pred + torch.sqrt(1 - gamma_next) * noise_pred
        )
        return x_next, latents_cond, latents_uncond


def ddim_sampler(net, batch, num_steps=250, scheduler=None, **kwargs):
    sampler = DDIMSampler(net, scheduler)
    return sampler.sample(batch, num_steps=num_steps, **kwargs)
