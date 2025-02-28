import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from models.positional_embeddings import FourierEmbedding, PositionalEmbedding

torch.fx.wrap("rearrange")
from typing import Optional, Tuple

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

allow_ops_in_compiled_graph()

from models.networks.transformers import (
    CrossAttentionBlock,
    FusedMLP,
    LayerNorm16Bits,
    SelfAttentionBlock,
)


class TimeEmbedder(nn.Module):
    def __init__(
        self,
        noise_embedding_type: str,
        dim: int,
        time_scaling: float,
        expansion: int = 4,
    ):
        super().__init__()
        self.encode_time = (
            PositionalEmbedding(num_channels=dim, endpoint=True)
            if noise_embedding_type == "positional"
            else FourierEmbedding(num_channels=dim)
        )
        self.time_scaling = time_scaling
        self.map_time = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim * expansion),
        )

    def forward(self, t: Tensor) -> Tensor:
        time = self.encode_time(t * self.time_scaling)
        time_mean = time.mean(dim=-1, keepdim=True)
        time_std = time.std(dim=-1, keepdim=True)
        time = (time - time_mean) / time_std
        return self.map_time(time)


class RINBlock(nn.Module):
    def __init__(
        self,
        data_dim: int,
        latents_dim: int,
        num_processing_layers: int,
        read_write_heads: int = 16,
        compute_heads: int = 16,
        latent_mlp_multiplier: int = 4,
        data_mlp_multiplier: int = 4,
        rw_dropout: float = 0.0,
        compute_dropout: float = 0.0,
        rw_stochastic_depth: float = 0.0,
        compute_stochastic_depth: float = 0.0,
        use_biases: bool = True,
        retrieve_attention_scores: bool = False,
        use_16_bits_layer_norm: bool = False,
    ):
        super().__init__()

        self.retriever_ca = CrossAttentionBlock(
            dim_q=latents_dim,
            dim_kv=data_dim,
            num_heads=read_write_heads,
            mlp_multiplier=latent_mlp_multiplier,
            dropout=rw_dropout,
            stochastic_depth=rw_stochastic_depth,
            use_biases=use_biases,
            retrieve_attention_scores=retrieve_attention_scores,
            use_16_bits_layer_norm=use_16_bits_layer_norm,
        )
        self.processer_sa = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=latents_dim,
                    num_heads=compute_heads,
                    mlp_multiplier=latent_mlp_multiplier,
                    dropout=compute_dropout,
                    stochastic_depth=compute_stochastic_depth,
                    use_biases=use_biases,
                    retrieve_attention_scores=retrieve_attention_scores,
                    use_16_bits_layer_norm=use_16_bits_layer_norm,
                )
                for _ in range(num_processing_layers)
            ]
        )
        self.writer_ca = CrossAttentionBlock(
            dim_q=data_dim,
            dim_kv=latents_dim,
            num_heads=read_write_heads,
            mlp_multiplier=data_mlp_multiplier,
            dropout=rw_dropout,
            stochastic_depth=rw_stochastic_depth,
            use_biases=use_biases,
            retrieve_attention_scores=retrieve_attention_scores,
            use_16_bits_layer_norm=use_16_bits_layer_norm,
        )

    def forward(
        self,
        data: torch.Tensor,
        latents: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Retrieve the latent representations from the data
        latents = self.retriever_ca(latents, data)
        # Process the latent representations
        for sa in self.processer_sa:
            latents = sa(latents)
        # Write the latent representations into the data
        data = self.writer_ca(data, latents)
        return data, latents


class RINBlockCond(RINBlock):
    def __init__(
        self,
        data_dim: int,
        latents_dim: int,
        num_processing_layers: int,
        read_write_heads: int = 16,
        compute_heads: int = 16,
        latent_mlp_multiplier: int = 4,
        data_mlp_multiplier: int = 4,
        rw_dropout: float = 0.0,
        compute_dropout: float = 0.0,
        rw_stochastic_depth: float = 0.0,
        compute_stochastic_depth: float = 0.0,
        use_biases: bool = True,
        retrieve_attention_scores: bool = False,
        use_16_bits_layer_norm: bool = False,
    ):
        super().__init__(
            data_dim,
            latents_dim,
            num_processing_layers,
            read_write_heads,
            compute_heads,
            latent_mlp_multiplier,
            data_mlp_multiplier,
            rw_dropout,
            compute_dropout,
            rw_stochastic_depth,
            compute_stochastic_depth,
            use_biases,
            retrieve_attention_scores,
            use_16_bits_layer_norm,
        )
        self.retrieve_cond = CrossAttentionBlock(
            dim_q=latents_dim,
            dim_kv=latents_dim,
            num_heads=read_write_heads,
            mlp_multiplier=latent_mlp_multiplier,
            dropout=rw_dropout,
            stochastic_depth=rw_stochastic_depth,
            use_biases=use_biases,
            retrieve_attention_scores=retrieve_attention_scores,
            use_16_bits_layer_norm=use_16_bits_layer_norm,
        )

    def forward(
        self,
        data: torch.Tensor,
        latents: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.retrieve_cond(latents, cond, from_token_mask=cond_attention_mask)
        return super().forward(data, latents, cond)


class RINBackbone(nn.Module):
    def __init__(
        self,
        data_size: int,
        data_dim: int,
        num_input_channels: int,
        num_latents: int,
        latents_dim: int,
        label_dim: int,
        num_processing_layers: int,
        num_blocks: int,
        path_size: int,
        num_cond_tokens: int,
        read_write_heads: int = 16,
        compute_heads: int = 16,
        latent_mlp_multiplier: int = 4,
        data_mlp_multiplier: int = 4,
        rw_dropout: float = 0.0,
        compute_dropout: float = 0.0,
        rw_stochastic_depth: float = 0.0,
        compute_stochastic_depth: float = 0.0,
        time_scaling: float = 1000.0,
        noise_embedding_type: str = "positional",
        data_positional_embedding_type: str = "learned",
        weight_init: str = "xavier_uniform",
        bias_init: str = "zeros",
        use_cond_token: bool = True,
        use_biases: bool = True,
        concat_cond_token_to_latents: bool = True,
        use_cond_rin_block: bool = False,
        retrieve_attention_scores: bool = False,
        use_16_bits_layer_norm: bool = False,
    ):
        super().__init__()
        if use_16_bits_layer_norm and not retrieve_attention_scores:
            LayerNorm = nn.LayerNorm
        else:
            LayerNorm = LayerNorm16Bits
        self.latents_dim = latents_dim
        self.num_latents = num_latents
        self.concat_cond_token_to_latents = concat_cond_token_to_latents
        self.retrieve_attention_scores = retrieve_attention_scores
        if concat_cond_token_to_latents:
            self.num_learned_latents = num_latents - num_cond_tokens - 1
        else:
            self.num_learned_latents = num_latents
        # Patch encoding
        self.path_size = path_size
        self.patch_extractor = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=data_dim,
            kernel_size=path_size,
            stride=path_size,
            padding=0,
        )
        if data_positional_embedding_type == "learned":
            self.data_pos_embedding = nn.Parameter(
                torch.randn((data_size // path_size) ** 2, data_dim),
                requires_grad=True,
            )
            nn.init.trunc_normal_(
                self.data_pos_embedding, std=0.02, a=-2 * 0.02, b=2 * 0.02
            )
        elif data_positional_embedding_type == "positional":
            self.data_pos_embedding = PositionalEmbedding(
                num_channels=data_dim, endpoint=True
            )(torch.arange(data_size**2))

        self.data_ln = LayerNorm(data_dim, eps=1e-6)

        # Latents

        self.latents = nn.Parameter(
            torch.randn(self.num_learned_latents, latents_dim),
            requires_grad=True,
        )
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2 * 0.02, b=2 * 0.02)

        self.time_embedder = TimeEmbedder(
            noise_embedding_type, latents_dim // 4, time_scaling, expansion=4
        )
        if use_cond_token:
            self.init_cond_mapping(label_dim, latents_dim, num_cond_tokens)

        # Previous latents encoding
        self.linear_previous = nn.Sequential(
            LayerNorm(latents_dim, eps=1e-6),
            FusedMLP(
                dim_model=latents_dim,
                dropout=0.0,
                activation=nn.GELU,
                hidden_layer_multiplier=latent_mlp_multiplier,
            ),
        )

        self.ln_previous = LayerNorm(latents_dim, eps=1e-6)

        self.ln_previous.weight.data.fill_(0.0)
        self.ln_previous.bias.data.fill_(0.0)

        # RIN blocks
        rin_block_kwargs = {
            "data_dim": data_dim,
            "latents_dim": latents_dim,
            "num_processing_layers": num_processing_layers,
            "read_write_heads": read_write_heads,
            "compute_heads": compute_heads,
            "latent_mlp_multiplier": latent_mlp_multiplier,
            "data_mlp_multiplier": data_mlp_multiplier,
            "rw_dropout": rw_dropout,
            "compute_dropout": compute_dropout,
            "rw_stochastic_depth": rw_stochastic_depth,
            "compute_stochastic_depth": compute_stochastic_depth,
            "use_biases": use_biases,
            "retrieve_attention_scores": retrieve_attention_scores,
            "use_16_bits_layer_norm": use_16_bits_layer_norm,
        }
        if use_cond_rin_block:
            self.rin_blocks = nn.ModuleList(
                [RINBlockCond(**rin_block_kwargs) for _ in range(num_blocks)]
            )

        else:
            self.rin_blocks = nn.ModuleList(
                [RINBlock(**rin_block_kwargs) for _ in range(num_blocks)]
            )

        self.map_tokens_to_patches = nn.Sequential(
            LayerNorm(data_dim, eps=1e-6),
            nn.Linear(data_dim, num_input_channels * path_size * path_size),
        )
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.init_weights()

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create patches
        x = batch["y"]
        gamma = batch["gamma"]
        previous_latents = batch["previous_latents"]

        b, _, h, w = x.shape
        x = self.patch_extractor(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        # Add positional embeddings
        x = self.data_ln(x) + self.data_pos_embedding.unsqueeze(0)

        # Cat latent tokens, conditioning tokens and timestep token
        encoded_noise = self.time_embedder(gamma)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        cond = [
            encoded_noise.unsqueeze(1),
        ]
        mapped_conditioning = self.cond_mapping(batch)
        if mapped_conditioning is not None:
            cond.append(mapped_conditioning)
        if self.concat_cond_token_to_latents:
            token_list = [
                latents,
                *cond,
            ]

            z = torch.cat(token_list, dim=1)
        else:
            z = latents + encoded_noise.unsqueeze(1)

        cond = torch.cat(cond, dim=1)

        z = z + self.ln_previous(
            previous_latents + self.linear_previous(previous_latents)
        )

        for rin_block in self.rin_blocks:
            x, z = rin_block(
                x,
                z,
                cond=cond,
                cond_attention_mask=self.return_cond_mask(cond.shape[1], batch),
            )

        # Map tokens to patches
        x = self.map_tokens_to_patches(x)

        # Reshape to image
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.path_size,
            p2=self.path_size,
            h=h // self.path_size,
            w=w // self.path_size,
        )

        return x, z

    def init_weights_(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            match self.weight_init:
                case "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                case "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                case "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight)
                case "torch_default":
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                case "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight)
                case "orthogonal":
                    nn.init.orthogonal_(m.weight)
                case "normal":
                    nn.init.normal_(m.weight)
                case "zeros":
                    nn.init.zeros_(m.weight)
                case _:
                    raise ValueError(f"Invalid weight init {self.weight_init}")
            if m.bias is not None:
                match self.bias_init:
                    case "zeros":
                        nn.init.zeros_(m.bias)
                    case "uniform" | "torch_default":
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)
                    case _:
                        raise ValueError(f"Invalid bias init {self.bias_init}")

    def init_weights(self):
        self.apply(self.init_weights_)

    def init_cond_mapping(self, label_dim: int, latents_dim: int, num_cond_tokens: int):
        raise NotImplementedError

    def cond_mapping(self, cond: Tensor) -> Tensor:
        raise NotImplementedError

    def return_cond_mask(self, num_cond, batch) -> Tensor:
        raise NotImplementedError


class RINClassCond(RINBackbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_cond_mapping(self, label_dim: int, latents_dim: int, num_cond_tokens: int):
        self.map_class = nn.Linear(label_dim, latents_dim)

    def cond_mapping(self, batch) -> Tensor:
        cond = batch["label"]
        if cond is None:
            return None
        else:
            return self.map_class(cond).unsqueeze(1)

    def return_cond_mask(self, num_cond, batch) -> Tensor:
        return None


class RINTextCond(RINBackbone):
    def __init__(self, *args, num_text_registers=16, **kwargs):
        self.num_text_registers = num_text_registers
        super().__init__(*args, **kwargs)

    def init_cond_mapping(self, label_dim: int, latents_dim: int, num_cond_tokens: int):
        self.transformers_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=latents_dim,
                    num_heads=16,
                    mlp_multiplier=4,
                    dropout=0.0,
                    stochastic_depth=0.0,
                    use_biases=True,
                    use_layer_scale=True,
                    layer_scale_value=0.1,
                )
                for _ in range(2)
            ]
        )
        self.map_text_tokens = nn.Linear(label_dim, latents_dim)
        self.text_registers = nn.Parameter(
            torch.randn(self.num_text_registers, latents_dim),
            requires_grad=True,
        )
        nn.init.trunc_normal_(self.text_registers, std=0.02, a=-2 * 0.02, b=2 * 0.02)

    def cond_mapping(self, batch) -> Tensor:
        embeddings = batch["text_tokens_embeddings"]
        mask = batch["text_tokens_mask"]
        embeddings = self.map_text_tokens(embeddings)
        embeddings = torch.cat(
            [
                self.text_registers.unsqueeze(0).expand(embeddings.shape[0], -1, -1),
                embeddings,
            ],
            dim=1,
        )
        mask = torch.cat(
            [
                torch.ones(
                    mask.shape[0],
                    self.num_text_registers,
                    device=mask.device,
                    dtype=mask.dtype,
                ),
                mask,
            ],
            dim=1,
        )
        for block in self.transformers_blocks:
            embeddings = block(embeddings, token_mask=mask)
        return embeddings

    def return_cond_mask(self, num_cond, batch) -> Tensor:
        text_mask = batch["text_tokens_mask"]
        text_mask = torch.cat(
            [
                torch.ones(
                    text_mask.shape[0],
                    self.num_text_registers,
                    device=text_mask.device,
                    dtype=text_mask.dtype,
                ),
                text_mask,
            ],
            dim=1,
        )
        mask = torch.zeros(
            text_mask.shape[0], num_cond, device=text_mask.device, dtype=text_mask.dtype
        )
        num_text_tokens = text_mask.shape[-1]
        num_other_tokens = num_cond - num_text_tokens
        mask[:, :num_other_tokens] = 1
        mask[:, num_other_tokens:] = text_mask
        return mask


class CADRINTextCond(RINBackbone):
    def __init__(
        self, *args, num_text_registers=16, modulate_conditioning=False, **kwargs
    ):
        self.num_text_registers = num_text_registers
        self.modulate_conditioning = modulate_conditioning
        super().__init__(*args, **kwargs)

        #### This is necessary to make it work with torch.fx
        self.register_parameter(
            "registers_coherence_mask",
            nn.Parameter(
                torch.ones(self.num_text_registers + 1, dtype=torch.bool),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "cond_mask",
            nn.Parameter(
                torch.ones(self.num_text_registers + 2, dtype=torch.bool),
                requires_grad=False,
            ),
        )

    def init_cond_mapping(self, label_dim: int, latents_dim: int, num_cond_tokens: int):
        self.transformers_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=latents_dim,
                    num_heads=16,
                    mlp_multiplier=4,
                    dropout=0.0,
                    stochastic_depth=0.0,
                    use_biases=True,
                    use_layer_scale=True,
                    layer_scale_value=0.1,
                    retrieve_attention_scores=self.retrieve_attention_scores,
                )
                for _ in range(2)
            ]
        )
        self.coherence_embedder = TimeEmbedder(
            "positional", latents_dim // 4, time_scaling=1000.0, expansion=4
        )
        self.coherence_positional_embedding = nn.Parameter(
            torch.randn(latents_dim),
            requires_grad=True,
        )
        nn.init.trunc_normal_(
            self.coherence_positional_embedding, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )

        self.map_text_tokens = nn.Linear(label_dim, latents_dim)
        self.text_registers = nn.Parameter(
            torch.randn(self.num_text_registers, latents_dim),
            requires_grad=True,
        )
        nn.init.trunc_normal_(self.text_registers, std=0.02, a=-2 * 0.02, b=2 * 0.02)
        if self.modulate_conditioning:
            self.modulation = nn.Sequential(
                nn.LayerNorm(latents_dim, eps=1e-6),
                nn.SiLU(),
                nn.Linear(latents_dim, 2 * latents_dim),
            )

    def init_weights(self):
        super().init_weights()
        if self.modulate_conditioning:
            self.modulation[-1].weight.data.fill_(0.0)
            self.modulation[-1].bias.data.fill_(0.0)

    def cond_mapping(self, batch) -> Tensor:
        embeddings = batch["text_tokens_embeddings"]
        mask = batch["text_tokens_mask"]

        batch_size, _, _ = embeddings.shape
        embeddings = self.map_text_tokens(embeddings)
        coherence = self.coherence_embedder(
            batch["coherence"].to(embeddings.dtype)
        ).unsqueeze(1)
        if self.modulate_conditioning:
            scale_coherence, shift_coherence = torch.chunk(
                self.modulation(coherence).unsqueeze(1), 2, dim=-1
            )
            embeddings = embeddings * (1 + scale_coherence) + shift_coherence
        embeddings = torch.cat(
            [
                coherence
                + self.coherence_positional_embedding.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, -1, -1),
                self.text_registers.unsqueeze(0).expand(batch_size, -1, -1),
                embeddings,
            ],
            dim=1,
        )
        registers_mask = self.registers_coherence_mask.unsqueeze(0).expand(
            batch_size, -1
        )
        mask = torch.cat(
            [
                registers_mask,
                mask,
            ],
            dim=1,
        )
        for block in self.transformers_blocks:
            embeddings = block(embeddings, token_mask=mask)
        return embeddings

    def return_cond_mask(self, num_cond, batch) -> Tensor:
        text_mask = batch["text_tokens_mask"]
        batch_size = text_mask.shape[0]
        mask = self.cond_mask.unsqueeze(0).expand(batch_size, -1)
        mask = torch.cat(
            [
                mask,
                text_mask,
            ],
            dim=1,
        )
        return mask
