import torch
import torch.nn as nn


class Resample(nn.Module):
    """
    Adapted from https://github.com/NVlabs/edm
    """

    def __init__(self, in_channels, up=False, down=False, resample_filter=[1, 1]):
        super().__init__()
        self.up = up
        self.down = down
        self.in_channels = in_channels
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        if self.up or self.down:
            f = (
                self.resample_filter.to(x.dtype)
                if self.resample_filter is not None
                else None
            )
            f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        if self.up:
            x = nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=f_pad,
            )
        if self.down:
            x = nn.functional.conv2d(
                x,
                f.tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=f_pad,
            )
        return x


def xavier_uniform(shape, fan_in, fan_out, gain=1.0):
    std = gain * (6.0 / (fan_in + fan_out)) ** 0.5
    return (torch.rand(*shape) * 2 - 1) * std


def init_xavier_uniform(module, weight_gain=1.0, bias_gain=0.0):
    weight_shape = module.weight.shape
    if len(weight_shape) == 2:
        fan_in, fan_out = weight_shape
    elif len(weight_shape) == 4:
        fan_in = weight_shape[1] * weight_shape[2] * weight_shape[3]
        fan_out = weight_shape[0] * weight_shape[2] * weight_shape[3]
    module.weight = nn.Parameter(
        xavier_uniform(module.weight.shape, fan_in, fan_out, weight_gain)
    )
    if hasattr(module, "bias"):
        module.bias = nn.Parameter(
            xavier_uniform(module.bias.shape, fan_in, fan_out, bias_gain)
        )
    return module


def init_xavier_uniform_mha(
    module, qkv_weight_gain=1.0, qkv_bias_gain=0, out_weight_gain=1.0, out_bias_gain=0.0
):
    assert type(module) is nn.MultiheadAttention

    if module._qkv_same_embed_dim:
        nn.init.xavier_uniform_(module.in_proj_weight, gain=qkv_weight_gain)
    else:
        nn.init.xavier_uniform_(module.q_proj_weight, gain=qkv_weight_gain)
        nn.init.xavier_uniform_(module.k_proj_weight, gain=qkv_weight_gain)
        nn.init.xavier_uniform_(module.v_proj_weight, gain=qkv_weight_gain)

    if hasattr(module, "in_proj_bias"):
        module.in_proj_bias = nn.Parameter(
            xavier_uniform(
                module.in_proj_bias.shape,
                module.in_proj_weight.shape[1],
                module.in_proj_weight.shape[0],
                gain=qkv_bias_gain,
            )
        )
    if hasattr(module, "k_bias"):
        module.k_bias = xavier_uniform(
            module.k_bias.shape,
            module.k_proj_weight.shape[1],
            module.k_proj_weight.shape[0],
            gain=qkv_bias_gain,
        )
    if hasattr(module, "v_bias"):
        module.v_bias = nn.Parameter(
            xavier_uniform(
                module.v_bias.shape,
                module.v_proj_weight.shape[1],
                module.v_proj_weight.shape[0],
                gain=qkv_bias_gain,
            )
        )
    if hasattr(module, "out_proj.weight"):
        nn.init.xavier_uniform_(module.out_proj.weight, gain=out_weight_gain)
    if hasattr(module, "out_proj.bias"):
        module.out_proj.bias = nn.Parameter(
            xavier_uniform(
                module.out_proj.bias.shape,
                module.out_proj.weight.shape[1],
                module.out_proj.weight.shape[0],
                gain=out_bias_gain,
            )
        )
    return module
