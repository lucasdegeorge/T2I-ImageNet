import torch


class Sampler:
    def __init__(self, net, scheduler):
        """
        Initialize the Sampler with a neural network and scheduler.

        Args:
            net: The neural network model
            scheduler: The scheduling algorithm for sampling
        """
        self.net = net
        self.scheduler = scheduler

    def sample(
        self,
        batch,
        conditioning_keys=None,
        uncond_tokens=None,
        num_steps=1000,
        cfg_rate=0,
        guidance_type="constant",
        guidance_start_step=0,
        latents_cfg_rate=0,
        generator=None,
        use_coherence_sampling=False,
        use_uncond_token=True,
        coherence_value=1.0,
        uncoherence_value=0.0,
        thresholding_type="clamp",
        clamp_value=1.0,
        thresholding_percentile=0.995,
    ):
        """
        Perform the sampling process.

        Args:
            batch: Input data batch
            conditioning_keys: Keys for conditional inputs
            uncond_tokens: Unconditional tokens
            num_steps: Number of sampling steps
            cfg_rate: Classifier-free guidance rate
            guidance_type: Type of guidance ('constant' or 'linear')
            guidance_start_step: Step at which guidance starts
            latents_cfg_rate: Rate for latent guidance
            generator: Random number generator
            use_coherence_sampling: Whether to use coherence sampling
            use_uncond_token: Whether to use unconditional tokens
            coherence_value: Value for coherence sampling
            uncoherence_value: Value for uncoherence sampling
            thresholding_type: Type of thresholding to apply
            clamp_value: Value for clamping
            thresholding_percentile: Percentile for dynamic thresholding

        Returns:
            The final sampled output
        """
        if self.scheduler is None:
            raise ValueError("Scheduler must be provided")

        x_cur = batch["y"].to(torch.float32)
        latents = batch["previous_latents"]
        if use_coherence_sampling:
            batch["coherence"] = (
                torch.ones(x_cur.shape[0], device=x_cur.device) * coherence_value
            )

        step_indices = torch.arange(
            num_steps + 1, dtype=torch.float32, device=x_cur.device
        )
        steps = 1 - step_indices / num_steps
        gammas = self.scheduler(steps)
        latents_cond = latents_uncond = latents
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        stacked_batch = self.prepare_stacked_batch(
            batch,
            uncond_tokens,
            conditioning_keys,
            use_coherence_sampling,
            use_uncond_token,
            coherence_value,
            uncoherence_value,
            x_cur,
            latents_cfg_rate,
        )

        for step, (gamma_now, gamma_next) in enumerate(zip(gammas[:-1], gammas[1:])):
            process_step_kwargs = {
                "batch": batch,
                "stacked_batch": stacked_batch,
                "cfg_rate": (
                    cfg_rate
                    if guidance_type == "constant"
                    else 2 * cfg_rate * (step / num_steps)
                ),
                "conditioning_keys": conditioning_keys,
                "step": step,
                "guidance_start_step": guidance_start_step,
                "latents_cfg_rate": latents_cfg_rate,
            }
            thresholding_kwargs = {
                "thresholding_type": thresholding_type,
                "clamp_value": clamp_value,
                "thresholding_percentile": thresholding_percentile,
            }

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                x_cur, latents_cond, latents_uncond = self.compute_next_step(
                    x_cur,
                    gamma_now,
                    gamma_next,
                    latents_cond,
                    latents_uncond,
                    process_step_kwargs,
                    thresholding_kwargs,
                    generator,
                )

        return x_cur.to(torch.float32)

    def process_step(
        self,
        x_cur,
        gamma_now,
        latents_cond,
        latents_uncond,
        batch,
        stacked_batch,
        cfg_rate,
        conditioning_keys,
        step,
        guidance_start_step,
        latents_cfg_rate,
    ):
        """
        Process a single step in the sampling loop.

        Args:
            x_cur: Current state
            gamma_now: Current gamma value
            latents_cond: Conditional latents
            latents_uncond: Unconditional latents
            batch: Input data batch
            stacked_batch: Prepared stacked batch
            cfg_rate: Classifier-free guidance rate
            conditioning_keys: Keys for conditional inputs
            step: Current step
            guidance_start_step: Step at which guidance starts
            latents_cfg_rate: Rate for latent guidance
            guidance_type: Type of guidance ('constant' or 'linear')
            num_steps: Total number of sampling steps

        Returns:
            Tuple of (denoised output, updated conditional latents, updated unconditional latents)
        """
        if (
            cfg_rate > 0
            and conditioning_keys is not None
            and step > guidance_start_step
        ):
            if latents_cfg_rate > 0:
                return self._process_with_latents_cfg(
                    x_cur,
                    gamma_now,
                    latents_cond,
                    latents_uncond,
                    stacked_batch,
                    cfg_rate,
                    latents_cfg_rate,
                )
            else:
                return self._process_without_latents_cfg(
                    x_cur,
                    gamma_now,
                    latents_cond,
                    latents_uncond,
                    stacked_batch,
                    cfg_rate,
                )
        else:
            return self._process_without_guidance(x_cur, gamma_now, latents_cond, batch)

    def _process_with_latents_cfg(
        self,
        x_cur,
        gamma_now,
        latents_cond,
        latents_uncond,
        stacked_batch,
        cfg_rate,
        latents_cfg_rate,
    ):
        stacked_batch["y"] = torch.cat([x_cur, x_cur, x_cur], dim=0)
        stacked_batch["gamma"] = gamma_now.expand(x_cur.shape[0] * 3)
        stacked_batch["previous_latents"] = (
            torch.cat(
                [latents_cond, latents_uncond, torch.zeros_like(latents_uncond)], dim=0
            )
            if latents_cond is not None and latents_uncond is not None
            else None
        )
        denoised_all, latents_all = self.net(stacked_batch)
        denoised_cond, denoised_uncond, denoised_uncond_no_latent = denoised_all.chunk(
            3, dim=0
        )
        latents_cond, latents_uncond, _ = latents_all.chunk(3, dim=0)
        denoised = (
            denoised_cond * (1 + cfg_rate)
            + denoised_uncond * (latents_cfg_rate - cfg_rate)
            - latents_cfg_rate * denoised_uncond_no_latent
        )
        return denoised, latents_cond, latents_uncond

    def _process_without_latents_cfg(
        self,
        x_cur,
        gamma_now,
        latents_cond,
        latents_uncond,
        stacked_batch,
        cfg_rate,
    ):
        stacked_batch["y"] = torch.cat([x_cur, x_cur], dim=0)
        stacked_batch["gamma"] = gamma_now.expand(x_cur.shape[0] * 2)
        stacked_batch["previous_latents"] = (
            torch.cat([latents_cond, latents_uncond], dim=0)
            if latents_cond is not None and latents_uncond is not None
            else None
        )
        denoised_all, latents_all = self.net(stacked_batch)
        denoised_cond, denoised_uncond = denoised_all.chunk(2, dim=0)
        latents_cond, latents_uncond = latents_all.chunk(2, dim=0)
        denoised = denoised_cond * (1 + cfg_rate) - denoised_uncond * cfg_rate
        return denoised, latents_cond, latents_uncond

    def _process_without_guidance(self, x_cur, gamma_now, latents, batch):
        batch["y"] = x_cur
        batch["gamma"] = gamma_now.expand(x_cur.shape[0])
        batch["previous_latents"] = latents
        denoised, latents = self.net(batch)
        return denoised, latents, latents  # Return latents for both cond and uncond

    def prepare_stacked_batch(
        self,
        batch,
        uncond_tokens,
        conditioning_keys,
        use_coherence_sampling,
        use_uncond_token,
        coherence_value,
        uncoherence_value,
        x_cur,
        latents_cfg_rate,
    ):
        """
        Prepare the stacked batch for processing.

        Args:
            batch: Input data batch
            uncond_tokens: Unconditional tokens
            conditioning_keys: Keys for conditional inputs
            use_coherence_sampling: Whether to use coherence sampling
            use_uncond_token: Whether to use unconditional tokens
            coherence_value: Value for coherence sampling
            uncoherence_value: Value for uncoherence sampling
            x_cur: Current state
            latents_cfg_rate: Rate for latent guidance

        Returns:
            Prepared stacked batch
        """
        stacked_batch = {}
        for key in conditioning_keys:
            if f"{key}_mask" in batch:
                stacked_batch[f"{key}_mask"] = self._prepare_mask(
                    batch, uncond_tokens, key, use_coherence_sampling, use_uncond_token
                )
            if f"{key}_embeddings" in batch:
                stacked_batch[f"{key}_embeddings"] = self._prepare_embeddings(
                    batch, uncond_tokens, key, use_coherence_sampling, use_uncond_token
                )
            elif key not in batch:
                raise ValueError(f"Key {key} not in batch")
            else:
                stacked_batch[key] = self._prepare_other(
                    batch, uncond_tokens, key, use_coherence_sampling, use_uncond_token
                )

        if use_coherence_sampling:
            stacked_batch["coherence"] = self._prepare_coherence(
                x_cur, coherence_value, uncoherence_value
            )

        if latents_cfg_rate > 0:
            stacked_batch = self._extend_for_latents_cfg(
                stacked_batch,
                conditioning_keys,
                x_cur,
                use_coherence_sampling,
                coherence_value,
                uncoherence_value,
            )

        return stacked_batch

    def apply_thresholding(
        self, x_pred, thresholding_type, clamp_value, thresholding_percentile
    ):
        """
        Apply thresholding to the predicted output.

        Args:
            x_pred: Predicted output
            thresholding_type: Type of thresholding to apply
            clamp_value: Value for clamping
            thresholding_percentile: Percentile for dynamic thresholding

        Returns:
            Thresholded output
        """
        if thresholding_type == "clamp":
            return torch.clamp(x_pred, -clamp_value, clamp_value)
        elif thresholding_type == "dynamic_thresholding":
            return self._apply_dynamic_thresholding(x_pred, thresholding_percentile)
        elif thresholding_type == "per_channel_dynamic_thresholding":
            return self._apply_per_channel_dynamic_thresholding(
                x_pred, thresholding_percentile
            )
        else:
            raise ValueError(f"{thresholding_type} not supported")

    def compute_next_step(
        self, x_cur, x_pred, noise_pred, gamma_now, gamma_next, generator
    ):
        """
        Compute the next step in the sampling process.

        This method should be implemented by subclasses.

        Args:
            x_cur: Current state
            x_pred: Predicted state
            noise_pred: Predicted noise
            gamma_now: Current gamma value
            gamma_next: Next gamma value
            generator: Random number generator

        Raises:
            NotImplementedError: This method should be implemented by subclasses
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    # Helper methods
    def _prepare_mask(
        self, batch, uncond_tokens, key, use_coherence_sampling, use_uncond_token
    ):
        if use_coherence_sampling and not use_uncond_token:
            return torch.cat([batch[f"{key}_mask"], batch[f"{key}_mask"]], dim=0)
        else:
            if batch[f"{key}_mask"].shape[1] > uncond_tokens[f"{key}_mask"].shape[1]:
                uncond_mask = (
                    torch.zeros_like(batch[f"{key}_mask"])
                    if batch[f"{key}_mask"].dtype == torch.bool
                    else torch.ones_like(batch[f"{key}_mask"]) * -torch.inf
                )
                uncond_mask[:, : uncond_tokens[f"{key}_mask"].shape[1]] = uncond_tokens[
                    f"{key}_mask"
                ]
            else:
                uncond_mask = uncond_tokens[f"{key}_mask"]
                batch[f"{key}_mask"] = torch.cat(
                    [
                        batch[f"{key}_mask"],
                        torch.zeros(
                            batch[f"{key}_mask"].shape[0],
                            uncond_tokens[f"{key}_embeddings"].shape[1]
                            - batch[f"{key}_mask"].shape[1],
                            device=batch[f"{key}_mask"].device,
                            dtype=batch[f"{key}_mask"].dtype,
                        ),
                    ],
                    dim=1,
                )
            return torch.cat([batch[f"{key}_mask"], uncond_mask], dim=0)

    def _prepare_embeddings(
        self, batch, uncond_tokens, key, use_coherence_sampling, use_uncond_token
    ):
        if use_coherence_sampling and not use_uncond_token:
            return torch.cat(
                [batch[f"{key}_embeddings"], batch[f"{key}_embeddings"]], dim=0
            )
        else:
            if (
                batch[f"{key}_embeddings"].shape[1]
                > uncond_tokens[f"{key}_embeddings"].shape[1]
            ):
                uncond_tokens[f"{key}_embeddings"] = torch.cat(
                    [
                        uncond_tokens[f"{key}_embeddings"],
                        torch.zeros(
                            uncond_tokens[f"{key}_embeddings"].shape[0],
                            batch[f"{key}_embeddings"].shape[1]
                            - uncond_tokens[f"{key}_embeddings"].shape[1],
                            uncond_tokens[f"{key}_embeddings"].shape[2],
                            device=uncond_tokens[f"{key}_embeddings"].device,
                        ),
                    ],
                    dim=1,
                )
            elif (
                batch[f"{key}_embeddings"].shape[1]
                < uncond_tokens[f"{key}_embeddings"].shape[1]
            ):
                batch[f"{key}_embeddings"] = torch.cat(
                    [
                        batch[f"{key}_embeddings"],
                        torch.zeros(
                            batch[f"{key}_embeddings"].shape[0],
                            uncond_tokens[f"{key}_embeddings"].shape[1]
                            - batch[f"{key}_embeddings"].shape[1],
                            batch[f"{key}_embeddings"].shape[2],
                            device=batch[f"{key}_embeddings"].device,
                        ),
                    ],
                    dim=1,
                )
            return torch.cat(
                [batch[f"{key}_embeddings"], uncond_tokens[f"{key}_embeddings"]], dim=0
            )

    def _prepare_other(
        self, batch, uncond_tokens, key, use_coherence_sampling, use_uncond_token
    ):
        if isinstance(batch[key], torch.Tensor):
            if use_coherence_sampling and not use_uncond_token:
                return torch.cat([batch[key], batch[key]], dim=0)
            else:
                return torch.cat([batch[key], uncond_tokens[key]], dim=0)
        elif isinstance(batch[key], list):
            if use_coherence_sampling and not use_uncond_token:
                return [*batch[key], *batch[key]]
            else:
                return [*batch[key], *uncond_tokens[key]]
        else:
            raise ValueError("Conditioning must be a tensor or a list of tensors")

    def _prepare_coherence(self, x_cur, coherence_value, uncoherence_value):
        return torch.cat(
            [
                torch.ones(x_cur.shape[0], device=x_cur.device) * coherence_value,
                torch.ones(x_cur.shape[0], device=x_cur.device) * uncoherence_value,
            ],
            dim=0,
        )

    def _extend_for_latents_cfg(
        self,
        stacked_batch,
        conditioning_keys,
        x_cur,
        use_coherence_sampling,
        coherence_value,
        uncoherence_value,
    ):
        for key in conditioning_keys:
            if f"{key}_mask" in stacked_batch:
                stacked_batch[f"{key}_mask"] = torch.cat(
                    [
                        stacked_batch[f"{key}_mask"],
                        stacked_batch[f"{key}_mask"][
                            stacked_batch[f"{key}_mask"].shape[0] // 2 :
                        ],
                    ]
                )
            if f"{key}_embeddings" in stacked_batch:
                stacked_batch[f"{key}_embeddings"] = torch.cat(
                    [
                        stacked_batch[f"{key}_embeddings"],
                        stacked_batch[f"{key}_embeddings"][
                            stacked_batch[f"{key}_embeddings"].shape[0] // 2 :
                        ],
                    ]
                )
            elif key in stacked_batch:
                if isinstance(stacked_batch[key], torch.Tensor):
                    stacked_batch[key] = torch.cat(
                        [
                            stacked_batch[key],
                            stacked_batch[key][stacked_batch[key].shape[0] // 2 :],
                        ],
                        dim=0,
                    )
                elif isinstance(stacked_batch[key], list):
                    stacked_batch[key] = [
                        *stacked_batch[key],
                        *stacked_batch[key][len(stacked_batch[key]) // 2 :],
                    ]
        if use_coherence_sampling:
            stacked_batch["coherence"] = torch.cat(
                [
                    stacked_batch["coherence"],
                    torch.ones(x_cur.shape[0], device=x_cur.device) * uncoherence_value,
                ],
                dim=0,
            )
        return stacked_batch

    def _apply_dynamic_thresholding(self, x_pred, thresholding_percentile):
        s = torch.quantile(
            x_pred.reshape(x_pred.shape[0], -1).abs(), thresholding_percentile, dim=-1
        )
        s = s.view(-1, 1, 1, 1).expand_as(x_pred)
        s = torch.maximum(s, torch.ones_like(s))
        return torch.clamp(x_pred, -s, s) / s

    def _apply_per_channel_dynamic_thresholding(self, x_pred, thresholding_percentile):
        s = torch.quantile(
            x_pred.reshape(x_pred.shape[0], x_pred.shape[1], -1).abs(),
            thresholding_percentile,
            dim=-1,
        )
        s = s.view(x_pred.shape[0], x_pred.shape[1], 1, 1).expand_as(x_pred)
        s = torch.maximum(s, torch.ones_like(s))
        return torch.clamp(x_pred, -s, s) / s
