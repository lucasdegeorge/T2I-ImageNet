from typing import Tuple

import torch
import transformers
from torch import nn


class TextPreprocessing(nn.Module):
    def __init__(
        self,
        tokenizer,
        model,
        drop_labels=False,
        input_key="text",
        output_key_root="text",
    ):
        super().__init__()
        self.cond_encoding = TextTokenizerPreprocess(
            tokenizer, input_key, output_key_root
        )
        self.cond_preprocess = TextEncoderPreprocess(
            model,
            input_ids_key=f"{output_key_root}_ids",
            input_mask_key=f"{output_key_root}_mask",
            output_key=f"{output_key_root}_embeddings",
        )
        self.drop_labels = drop_labels
        self.input_key = input_key
        self.output_key_root = output_key_root

    def forward(self, batch, training=True, device=None):
        if self.drop_labels:
            batch[f"{self.output_key_root}_embeddings"] = None
            batch[f"{self.output_key_root}_mask"] = None
            return batch
        batch = self.cond_encoding(batch)
        has_to_device = callable(getattr(self.cond_encoding, "to_device", None))
        if has_to_device:
            batch = self.cond_encoding.to_device(batch, device)
        batch = self.cond_preprocess(batch)
        return batch


class PrecomputedTextConditioning:
    def __init__(
        self,
        input_key="flan_t5_xl",
        output_key_root="text",
        drop_labels=False,
    ):
        self.input_key = input_key
        self.output_key_root = output_key_root
        self.drop_labels = drop_labels

    def __call__(self, batch, device=None):
        if self.drop_labels:
            batch[f"{self.output_key_root}_embeddings"] = None
            batch[f"{self.output_key_root}_mask"] = None
            return batch
        batch[f"{self.output_key_root}_embeddings"] = batch[
            f"{self.input_key}_embeddings"
        ]
        batch[f"{self.output_key_root}_mask"] = batch[f"{self.input_key}_mask"]
        return batch


class TextTokenizerPreprocess:
    def __init__(self, tokenizer, input_key="text", output_key_root="text"):
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key_root = output_key_root

    def __call__(self, batch):
        conditioning = batch[self.input_key]
        conditioning = self.tokenizer(
            conditioning,
            truncation=True,
            padding="longest",  # "max_length
            max_length=128,
            return_tensors="pt",
        )
        batch[f"{self.output_key_root}_mask"] = conditioning["attention_mask"].to(
            torch.bool
        )
        batch[f"{self.output_key_root}_ids"] = conditioning["input_ids"]
        return batch

    def to_device(self, batch, device):
        batch[f"{self.output_key_root}_ids"] = batch[f"{self.output_key_root}_ids"].to(
            device
        )
        batch[f"{self.output_key_root}_mask"] = batch[
            f"{self.output_key_root}_mask"
        ].to(device)
        return batch


class TextEncoderPreprocess(nn.Module):
    def __init__(
        self,
        text_model,
        input_ids_key="text_ids",
        input_mask_key="text_mask",
        output_key="text_embeddings",
    ):
        super().__init__()
        self.text_model = text_model
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.input_ids_key = input_ids_key
        self.input_mask_key = input_mask_key
        self.output_key = output_key

    def forward(self, batch):
        conditioning = {
            "input_ids": batch[f"{self.input_ids_key}"],
            "attention_mask": batch[f"{self.input_mask_key}"],
        }
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if isinstance(self.text_model, transformers.T5EncoderModel):
                    conditioning = self.text_model(
                        **conditioning
                    ).last_hidden_state.detach()
                elif isinstance(self.text_model, transformers.CLIPTextModel):
                    conditioning = (
                        self.text_model(**conditioning, output_hidden_states=True)
                        .hidden_states[-2]
                        .detach()
                    )
        batch[self.output_key] = conditioning
        return batch


class SDLatentPreconditioning(nn.Module):
    def __init__(
        self,
        vae,
        input_key="image",
        output_key="x_0",
        channel_wise_normalisation=False,
        vae_sample=False,
    ):
        super().__init__()
        self.vae = vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.input_key = input_key
        self.output_key = output_key
        self.vae_sample = vae_sample

        if channel_wise_normalisation:
            scale = 0.5 / torch.tensor([4.17, 4.62, 3.71, 3.28])
            bias = -torch.tensor([5.81, 3.25, 0.12, -2.15]) * scale
        else:
            scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215])
            bias = torch.tensor([0.0, 0.0, 0.0, 0.0])
        scale = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("scale", nn.Parameter(scale))
        self.register_buffer("bias", nn.Parameter(bias))

    def forward(self, batch):
        x = batch[self.input_key]
        latents = self.vae.encode(x).latent_dist
        if self.vae_sample and self.training:
            latents_mean = latents.mean
            latents_std = latents.std
            latents = torch.randn(latents.shape) * latents_std + latents_mean
        else:
            latents = latents.mean
        latents = latents * self.scale + self.bias
        batch[self.output_key] = latents
        return batch


class PrecomputedSDLatentPreconditioning(nn.Module):
    def __init__(
        self,
        input_key_mean="vae_embeddings_mean",
        input_key_std="vae_embeddings_std",
        output_key_root="x_0",
        vae_sample=False,
        channel_wise_normalisation=False,
    ):
        super().__init__()
        self.input_key_mean = input_key_mean
        self.input_key_std = input_key_std
        self.output_key_root = output_key_root
        self.vae_sample = vae_sample
        if channel_wise_normalisation:
            scale = 0.5 / torch.tensor([4.17, 4.62, 3.71, 3.28])
            bias = -torch.tensor([5.81, 3.25, 0.12, -2.15]) * scale
        else:
            scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215])
            bias = torch.tensor([0.0, 0.0, 0.0, 0.0])
        scale = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("scale", nn.Parameter(scale))
        self.register_buffer("bias", nn.Parameter(bias))

    def forward(self, batch):
        if self.vae_sample:
            mean = batch[self.input_key_mean]
            std = batch[self.input_key_std]
            latents = torch.randn_like(mean) * std + mean
        else:
            latents = batch[self.input_key_mean]
        latents = latents * self.scale + self.bias
        batch[self.output_key_root] = latents
        return batch
    

class PrecomputedAELatentPreconditioning(nn.Module):
    def __init__(
        self,
        input_key_mean="vae_embeddings",
        input_key_std=None,
        output_key_root="x_0",
        vae_sample=False,
        channel_wise_normalisation=True,
    ):
        super().__init__()
        self.input_key_mean = input_key_mean
        self.input_key_std = input_key_std
        self.output_key_root = output_key_root
        self.vae_sample = vae_sample
        if channel_wise_normalisation:
            scale = 0.5 / torch.tensor(
                [
                    2.01,1.99,1.88,1.89,2.01,2.02,2.61,2.65,
                    1.89,2.32,2.05,1.94,2.07,1.85,2.21,2.06,
                    2.13,2.10,2.02,1.93,2.29,2.04,2.02,2.07,
                    4.66,2.34,2.18,1.95,1.98,1.84,2.01,1.87,
                ]
            )
            bias = -torch.tensor(
                [
                    -0.12,0.05,0.14,0.07,-0.15, 0.62,0.37,1.67,
                    0.11,-0.85,0.53,-0.04,-0.32,0.18,-0.14,0.14,
                    0.15,0.32,0.47,-0.01,-0.15,0.17,-0.39,-0.05,
                    -0.29,0.30,0.64,0.01,0.15,-0.26,0.53,0.11,
                ]
            ) * scale
        else:
            raise NotImplementedError(
                "No normalisation for AE without channel-wise normalisation implemented"
            )
        scale = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("scale", nn.Parameter(scale))
        self.register_buffer("bias", nn.Parameter(bias))

    def forward(self, batch):
        if self.vae_sample:
            assert self.input_key_std is not None 
            mean = batch[self.input_key_mean]
            std = batch[self.input_key_std]
            latents = torch.randn_like(mean) * std + mean
        else:
            latents = batch[self.input_key_mean]
        latents = latents * self.scale + self.bias
        batch[self.output_key_root] = latents
        return batch


class PrecomputedPreconditioning:
    def __init__(
        self, input_key="vae_embeddings", output_key_root="x_0", drop_labels=False
    ):
        self.input_key = input_key
        self.output_key_root = output_key_root
        self.drop_labels = drop_labels

    def __call__(self, batch, device=None):
        if self.drop_labels:
            batch[self.output_key_root] = None
            return batch
        batch[self.output_key_root] = batch[self.input_key]
        return batch


def add_padding_to_embedding(
    mask: torch.Tensor, embeddings: torch.Tensor, num_padding_tokens=5
) -> Tuple[torch.Tensor, torch.Tensor]:
    last_1 = mask.sum(dim=-1).long()
    if last_1.max() > mask.shape[-1] - num_padding_tokens:
        mask = torch.cat(
            [
                mask,
                torch.zeros(
                    mask.shape[0],
                    last_1.max() - mask.shape[1] + num_padding_tokens,
                    device=mask.device,
                    dtype=mask.dtype,
                ),
            ],
            dim=1,
        )
        embeddings = torch.cat(
            [
                embeddings,
                torch.zeros(
                    embeddings.shape[0],
                    last_1.max() - embeddings.shape[1] + num_padding_tokens,
                    embeddings.shape[2],
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                ),
            ],
            dim=1,
        )
    mask = torch.masked_fill(mask.float(), mask == 0, -torch.inf)
    for i in range(num_padding_tokens):
        mask[:, last_1 + i] = 0
    return embeddings, mask
