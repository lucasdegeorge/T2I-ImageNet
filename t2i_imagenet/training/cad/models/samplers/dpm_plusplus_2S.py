import torch

from .base_sampler import Sampler


class DPMPlusPlus2SSampler(Sampler):
    def __init__(self, net, scheduler, r_1=0.5):
        super().__init__(net, scheduler)
        self.r_1 = r_1

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
        s = self.inverse_lambda(lambda_now + self.r_1 * h)
        sigma_s = torch.sqrt(1 - s)
        alpha_s = torch.sqrt(s)
        denoised, latents_cond_now, latents_uncond_now = self.process_step(
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

        u = (sigma_s / sigma_now) * x_cur - alpha_s * torch.expm1(
            -self.r_1 * h
        ) * x_pred_now
        denoised, latents_cond_s, latents_uncond_s = self.process_step(
            u,
            s,
            latents_cond_now,
            latents_uncond_now,
            **process_step_kwargs,
        )
        x_pred_s = (x_cur - torch.sqrt(1 - gamma_now) * denoised) / torch.sqrt(
            gamma_now
        )
        x_pred_s = self.apply_thresholding(x_pred_s, **thresholding_kwargs)

        x_next = (
            sigma_next / sigma_now * x_cur
            - alpha_next * torch.expm1(-h) * x_pred_s
            - (0.5 / self.r_1)
            * alpha_next
            * (torch.expm1(-h) / h + 1.0)
            * (x_pred_s - x_pred_now)
        )

        return x_next, latents_cond_s, latents_uncond_s

    def inverse_lambda(self, lambda_now):
        return torch.sigmoid(2 * lambda_now)


def dpm_plusplus_2S_sampler(net, batch, num_steps=50, scheduler=None, **kwargs):
    sampler = DPMPlusPlus2SSampler(net, scheduler)
    return sampler.sample(batch, num_steps=num_steps, **kwargs)
