import torch

from .base_sampler import Sampler


class DPMEulerSampler(Sampler):
    def __init__(self, net, scheduler):
        super().__init__(net, scheduler)

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
        if not hasattr(self, "noise_prev"):
            self.noise_prev = None
            self.gamma_prev = None

        if self.noise_prev is None:
            x_next, noise_now = self._step1(gamma_now, gamma_next, noise_pred, x_cur)
            self.gamma_prev = gamma_now
            self.noise_prev = noise_now
        else:
            x_next, noise_now = self._step2(
                gamma_now,
                gamma_next,
                self.gamma_prev,
                self.noise_prev,
                noise_pred,
                x_cur,
            )
            self.gamma_prev = gamma_now
            self.noise_prev = noise_now

        return x_next, latents_cond, latents_uncond

    def _step1(self, gamma_now, gamma_next, noise_pred, x_cur):
        sigma_now = torch.sqrt(1 - gamma_now)
        alpha_now = torch.sqrt(gamma_now)
        lambda_now = torch.log(alpha_now / sigma_now)
        sigma_next = torch.sqrt(1 - gamma_next)
        alpha_next = torch.sqrt(gamma_next)
        lambda_next = torch.log(alpha_next / sigma_next)
        h = lambda_next - lambda_now

        x_next = (
            alpha_next / alpha_now * x_cur - sigma_now * torch.expm1(h) * noise_pred
        )

        return x_next, noise_pred

    def _step2(self, gamma_now, gamma_next, gamma_prev, noise_prev, noise_now, x_cur):
        sigma_now = torch.sqrt(1 - gamma_now)
        alpha_now = torch.sqrt(gamma_now)
        lambda_now = torch.log(alpha_now / sigma_now)

        sigma_prev = torch.sqrt(1 - gamma_prev)
        alpha_prev = torch.sqrt(gamma_prev)
        lambda_prev = torch.log(alpha_prev / sigma_prev)

        sigma_next = torch.sqrt(1 - gamma_next)
        alpha_next = torch.sqrt(gamma_next)
        lambda_next = torch.log(alpha_next / sigma_next)

        h = lambda_next - lambda_now
        h_prev = lambda_now - lambda_prev
        r0 = h_prev / h

        D0 = noise_now
        D1 = 1.0 / r0 * (noise_now - noise_prev)

        x_next = (
            alpha_next / alpha_now * x_cur
            - sigma_next * torch.expm1(h) * D0
            - 0.5 * sigma_next * torch.expm1(h) * D1
        )

        return x_next, noise_now


def dpm_euler_sampler(net, batch, num_steps=50, scheduler=None, **kwargs):
    sampler = DPMEulerSampler(net, scheduler)
    return sampler.sample(batch, num_steps=num_steps, **kwargs)
