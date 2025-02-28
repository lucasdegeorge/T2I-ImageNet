import torch
import torch.nn as nn

from utils.image_processing import remap_image_torch


class SD1_5VAEPostProcessing(nn.Module):
    def __init__(self, channel_wise_normalisation=False):
        super().__init__()
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

    def forward(self, x):
        x = (x - self.bias) / self.scale
        return x


class SD1_5VAEDecoderPostProcessing(SD1_5VAEPostProcessing):
    def __init__(self, vae, channel_wise_normalisation=False):
        super().__init__(channel_wise_normalisation=channel_wise_normalisation)
        self.vae = vae
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = super().forward(x)
        with torch.no_grad():
            return remap_image_torch(self.vae.decode(x).sample.detach())
        

class AEPostProcessing(nn.Module):
    def __init__(self, channel_wise_normalisation=False):
        super().__init__()
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
            scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215])
            bias = torch.tensor([0.0, 0.0, 0.0, 0.0])
        scale = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("scale", nn.Parameter(scale))
        self.register_buffer("bias", nn.Parameter(bias))

    def forward(self, x):
        x = (x - self.bias) / self.scale
        return x
    

class AEDecoderPostProcessing(AEPostProcessing):
    def __init__(self, ae, channel_wise_normalisation=False):
        super().__init__(channel_wise_normalisation=channel_wise_normalisation)
        self.ae = ae
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = super().forward(x)
        with torch.no_grad():
            return remap_image_torch(self.ae.decode(x).sample.detach())

