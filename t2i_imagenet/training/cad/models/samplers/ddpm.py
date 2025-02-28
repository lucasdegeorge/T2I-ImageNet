import torch

from .base_sampler import Sampler


class DDPMSampler(Sampler):
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

        log_alpha_t = torch.log(gamma_now) - torch.log(gamma_next)
        alpha_t = torch.clip(torch.exp(log_alpha_t), 0, 1)
        x_mean = torch.rsqrt(alpha_t) * (
            x_cur - torch.rsqrt(1 - gamma_now) * (1 - alpha_t) * noise_pred
        )
        var_t = 1 - alpha_t
        eps = torch.randn(x_cur.shape, device=x_cur.device, generator=generator)
        return x_mean + torch.sqrt(var_t) * eps, latents_cond, latents_uncond


def ddpm_sampler(net, batch, num_steps=1000, scheduler=None, **kwargs):
    sampler = DDPMSampler(net, scheduler)
    return sampler.sample(batch, num_steps=num_steps, **kwargs)
