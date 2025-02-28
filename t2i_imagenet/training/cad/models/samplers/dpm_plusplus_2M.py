import torch

from .base_sampler import Sampler


class DPMPlusPlus2MSampler(Sampler):
    def __init__(self, net, scheduler):
        super().__init__(net, scheduler)
        self.x_buffer = None
        self.h_buffer = None

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

        sigma_now = torch.sqrt(1 - gamma_now)
        alpha_now = torch.sqrt(gamma_now)
        lambda_now = torch.log(alpha_now / sigma_now)

        sigma_next = torch.sqrt(1 - gamma_next)
        alpha_next = torch.sqrt(gamma_next)
        lambda_next = torch.log(alpha_next / sigma_next)

        h = lambda_next - lambda_now

        denoised, latents_cond, latents_uncond = self.process_step(
            x_cur,
            gamma_now,
            latents_cond,
            latents_uncond,
            **process_step_kwargs,
        )
        x_pred_now = (x_cur - torch.sqrt(1 - gamma_now) * denoised) / torch.sqrt(
            gamma_now
        )
        x_pred_now = self.apply_thresholding(x_pred_now, **thresholding_kwargs)

        if self.x_buffer is None:
            x_next = (
                sigma_next / sigma_now * x_cur
                - alpha_next * torch.expm1(-h) * x_pred_now
            )
            self.x_buffer = x_next
            self.h_buffer = h

        else:
            r = self.h_buffer / h
            x_next = sigma_next / sigma_now * x_cur - alpha_next * torch.expm1(-h) * (
                (1 + 1 / (2 * r)) * x_pred_now - (1 / (2 * r)) * self.x_buffer
            )

        return x_next, latents_cond, latents_uncond

    def inverse_lambda(self, lambda_now):
        return torch.sigmoid(2 * lambda_now)


def dpm_plusplus_2M_sampler(net, batch, num_steps=50, scheduler=None, **kwargs):
    sampler = DPMPlusPlus2MSampler(net, scheduler)
    return sampler.sample(batch, num_steps=num_steps, **kwargs)
