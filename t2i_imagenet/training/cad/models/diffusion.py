import copy
import logging
from functools import partial
from typing import Any, List

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate


class DiffusionModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.strict_loading = False
        self.cfg = cfg
        self.network = instantiate(cfg.network)
        # self.network = torch.compile(
        #     self.network,
        #     mode="default",
        # )

        self.train_noise_scheduler = instantiate(cfg.train_noise_scheduler)
        self.inference_noise_scheduler = instantiate(cfg.inference_noise_scheduler)
        self.data_preprocessing = instantiate(cfg.data_preprocessing)
        self.cond_preprocessing = instantiate(cfg.cond_preprocessing)
        self.preconditioning = instantiate(cfg.preconditioning)

        self.ema_network = copy.deepcopy(self.network).requires_grad_(False)
        self.ema_network.eval()
        self.postprocessing = instantiate(cfg.postprocessing)
        self.val_sampler = instantiate(cfg.val_sampler)
        self.test_sampler = instantiate(cfg.test_sampler)

        uncond_conditioning = instantiate(cfg.uncond_conditioning)
        if isinstance(uncond_conditioning, np.ndarray):
            self.uncond_conditioning = nn.Parameter(
                torch.from_numpy(uncond_conditioning), requires_grad=False
            )
        else:
            self.uncond_conditioning = uncond_conditioning

        self.loss = instantiate(cfg.loss)(
            self.train_noise_scheduler, self.uncond_conditioning
        )

    def training_step(self, batch, batch_idx):
        if self.cfg.cutmix.do_cutmix:
            batch, timestep = self._cutmix_pre_training_step(batch, batch_idx)
        else:
            timestep = None
            with torch.no_grad():
                batch = self.data_preprocessing(batch)
                batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(self.preconditioning, self.network, batch, t=timestep).mean()
        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss
    
    def _cutmix_pre_training_step(self, batch, batch_idx):
        raw_batch = batch[0]
        cutmix_batch = batch[1]
        with torch.no_grad():
            raw_batch = self.data_preprocessing(raw_batch)
            raw_batch = self.cond_preprocessing(raw_batch)

            cutmix_batch = self.data_preprocessing(cutmix_batch)
            cutmix_batch = self.cond_preprocessing(cutmix_batch)
        
        batch_size = raw_batch["x_0"].shape[0]
        dtype = raw_batch["x_0"].dtype

        timesteps = torch.rand(batch_size, device=self.device, dtype=dtype)

        ## less efficient
        # change_to_cutmix = torch.where(
        #     timesteps > self.cfg.cutmix.cutmix_noise_threshold_ratio,
        #     (
        #         torch.tensor(True, device=self.device)
        #         if (random.random() > self.cfg.cutmix.do_cutmix_above_threshold_prob)
        #         else torch.tensor(False, device=self.device)
        #     ),
        #     torch.tensor(False, device=self.device),
        # )

        ## more efficient
        change_to_cutmix = (timesteps > self.cfg.cutmix.cutmix_noise_threshold_ratio) & (
            torch.rand(timesteps.shape[0], device=self.device)
            < self.cfg.cutmix.do_cutmix_above_threshold_prob
        )

        # batch = copy.deepcopy(raw_batch)
        batch = {}
        try:
            batch["x_0"] = torch.where(
                change_to_cutmix.view(-1,1,1,1), 
                cutmix_batch["x_0"], 
                raw_batch["x_0"]
            )
            batch[f"{self.cfg.cond_preprocessing.output_key_root}_embeddings"] = (
                torch.where(
                    change_to_cutmix.view(-1,1,1), 
                    cutmix_batch[f"{self.cfg.cond_preprocessing.input_key}_embeddings"],
                    raw_batch[f"{self.cfg.cond_preprocessing.input_key}_embeddings"]
                    )
            )
            batch[f"{self.cfg.cond_preprocessing.output_key_root}_mask"] = (
                torch.where(
                    change_to_cutmix.view(-1,1), 
                    cutmix_batch[f"{self.cfg.cond_preprocessing.input_key}_mask"],
                    raw_batch[f"{self.cfg.cond_preprocessing.input_key}_mask"]
                    )
            )
        except Exception as e:
            print(e)
            del cutmix_batch
            torch.cuda.empty_cache()
            return raw_batch, timesteps

        del raw_batch, cutmix_batch
        torch.cuda.empty_cache()  
        return batch, timesteps



    def on_before_optimizer_step(self, optimizer):
        if self.global_step == 0:
            no_grad = []
            for name, param in self.network.named_parameters():
                if param.grad is None and "dummy" not in name:
                    no_grad.append(name)
            if len(no_grad) > 0:
                print("Parameters without grad:")
                print(no_grad)

    def on_validation_start(self):
        self.validation_generator = torch.Generator(device=self.device).manual_seed(
            3407
        )
        self.validation_generator_ema = torch.Generator(device=self.device).manual_seed(
            3407
        )

    def validation_step(self, batch, batch_idx):
        batch = self.data_preprocessing(batch)
        batch = self.cond_preprocessing(batch)
        batch_size = batch["x_0"].shape[0]
        loss = self.loss(
            self.preconditioning,
            self.network,
            batch,
            generator=self.validation_generator,
        ).mean()
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if hasattr(self, "ema_model"):
            loss_ema = self.loss(
                self.preconditioning,
                self.ema_network,
                batch,
                generator=self.validation_generator_ema,
            ).mean()
            self.log(
                "val/loss_ema",
                loss_ema,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.network, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,
                },
                {
                    "params": [
                        p
                        for n, p in self.network.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.network.parameters())
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

    def sample(
        self,
        batch_size=None,
        shape=None,
        cond=None,
        x_N=None,
        latents=None,
        num_steps=None,
        sampler=None,
        scheduler=None,
        stage="test",
        cfg=0,
        guidance_type="constant",
        guidance_start_step=0,
        latents_cfg_rate=0,
        generator=None,
        coherence_value=1.0,
        uncoherence_value=0.0,
        unconfident_prompt=None,
        thresholding_type="clamp",
        clamp_value=1.0,
        thresholding_percentile=0.995,
    ):
        batch = {"previous_latents": latents}
        if x_N is None and (shape is None or batch_size is None):
            raise ValueError("Shape must be specified if x_N are not provided")
        if x_N is None:
            assert shape is not None
            assert batch_size is not None
            x_N = torch.randn(batch_size, *shape, device=self.device)
        else:
            if x_N.ndim == 3:
                x_N = x_N.unsqueeze(0)
            batch_size = x_N.shape[0]
            shape = x_N.shape[1:]
        batch["y"] = x_N
        if sampler is None:
            if stage == "val":
                sampler = self.val_sampler
            elif stage == "test":
                sampler = self.test_sampler
            else:
                raise ValueError(f"Unknown stage {stage}")
        if scheduler is None:
            scheduler = self.inference_noise_scheduler
        if unconfident_prompt is not None:
            uncond_conditioning = unconfident_prompt
        else:
            uncond_conditioning = self.uncond_conditioning
        if cond is not None:
            if isinstance(cond, dict):
                for key, value in cond.items():
                    batch[key] = value
                    uncond_tokens = uncond_conditioning.repeat(
                        1 if batch_size is None else batch_size,
                        *[1 for _ in uncond_conditioning.shape],
                    )
                    uncond_tokens_mask = torch.ones(
                        uncond_tokens.shape[0],
                        uncond_tokens.shape[1],
                        dtype=torch.bool,
                        device=uncond_tokens.device,
                    )
                    uncond_tokens_batch = {
                        f"{self.cfg.cond_preprocessing.input_key}_embeddings": uncond_tokens,
                        f"{self.cfg.cond_preprocessing.input_key}_mask": uncond_tokens_mask,
                    }

            else:
                if isinstance(cond, str):
                    uncond_tokens = [uncond_conditioning] * (
                        1 if batch_size is None else batch_size
                    )
                    cond = [cond] * (1 if batch_size is None else batch_size)
                elif isinstance(cond, list) and all(isinstance(i, str) for i in cond):
                    uncond_tokens = (
                        [uncond_conditioning] * len(cond)
                        if isinstance(uncond_conditioning, str)
                        else uncond_conditioning
                    )
                    assert len(cond) == batch_size or batch_size is None
                elif isinstance(cond, torch.Tensor) and isinstance(
                    self.uncond_conditioning, float
                ):
                    uncond_tokens = (
                        torch.ones_like(cond, device=self.device) * uncond_conditioning
                    )
                else:
                    if len(uncond_conditioning.shape) < len(cond.shape):
                        uncond_tokens = uncond_conditioning.repeat(
                            1 if batch_size is None else batch_size,
                            *[1 for _ in uncond_conditioning.shape],
                        )
                    else:
                        uncond_tokens = uncond_conditioning
                batch[self.cfg.cond_preprocessing.input_key] = cond
                uncond_tokens_batch = {
                    self.cfg.cond_preprocessing.input_key: uncond_tokens
                }
            uncond_tokens = self.cond_preprocessing(
                uncond_tokens_batch,
                device=self.device,
            )
        else:
            uncond_tokens = None
        batch = self.cond_preprocessing(batch, device=self.device)
        if num_steps is None:
            image = sampler(
                self.ema_model,
                batch,
                conditioning_keys=[self.cfg.cond_preprocessing.output_key_root],
                scheduler=scheduler,
                uncond_tokens=uncond_tokens,
                cfg_rate=cfg,
                guidance_type=guidance_type,
                guidance_start_step=guidance_start_step,
                latents_cfg_rate=latents_cfg_rate,
                generator=generator,
                use_coherence_sampling=(
                    self.cfg.use_coherence_sampling
                    if "use_coherence_sampling" in self.cfg
                    else False
                ),
                coherence_value=coherence_value,
                uncoherence_value=uncoherence_value,
                thresholding_type=thresholding_type,
                clamp_value=clamp_value,
                thresholding_percentile=thresholding_percentile,
            )
        else:
            image = sampler(
                self.ema_model,
                batch,
                conditioning_keys=[self.cfg.cond_preprocessing.output_key_root],
                scheduler=scheduler,
                uncond_tokens=uncond_tokens,
                num_steps=num_steps,
                cfg_rate=cfg,
                guidance_type=guidance_type,
                guidance_start_step=guidance_start_step,
                latents_cfg_rate=latents_cfg_rate,
                generator=generator,
                use_coherence_sampling=(
                    self.cfg.use_coherence_sampling
                    if "use_coherence_sampling" in self.cfg
                    else False
                ),
                coherence_value=coherence_value,
                uncoherence_value=uncoherence_value,
                thresholding_type=thresholding_type,
                clamp_value=clamp_value,
                thresholding_percentile=thresholding_percentile,
            )
        return self.postprocessing(image)

    def model(self, *args, **kwargs):
        return self.preconditioning(self.network, *args, **kwargs)

    def ema_model(self, *args, **kwargs):
        return self.preconditioning(self.ema_network, *args, **kwargs)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
