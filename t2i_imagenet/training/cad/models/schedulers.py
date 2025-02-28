import torch


class SigmoidScheduler:
    def __init__(self, start=-3, end=3, tau=1, clip_min=1e-9, clip_max=1 - 1e-4):
        self.start = start
        self.end = end
        self.tau = tau
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.v_start = torch.sigmoid(torch.tensor(self.start / self.tau))
        self.v_end = torch.sigmoid(torch.tensor(self.end / self.tau))

    def __call__(self, t):
        output = (
            -torch.sigmoid((t * (self.end - self.start) + self.start) / self.tau)
            + self.v_end
        ) / (self.v_end - self.v_start)
        return torch.clamp(output, min=self.clip_min, max=self.clip_max)


class LinearScheduler:
    def __init__(self, start=1, end=0, clip_min=1e-9, clip_max=1 - 1e-4):
        self.start = start
        self.end = end
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, t):
        output = (self.end - self.start) * t + self.start
        return torch.clamp(output, min=self.clip_min)


class CosineScheduler:
    def __init__(
        self,
        start: float = 1,
        end: float = 0,
        tau: float = 1.0,
        clip_min: float = 1e-9,
        clip_max: float = 1 - 1e-4,
    ):
        self.start = start
        self.end = end
        self.tau = tau
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.v_start = torch.cos(torch.tensor(self.start) * torch.pi / 2) ** (
            2 * self.tau
        )
        self.v_end = torch.cos(torch.tensor(self.end) * torch.pi / 2) ** (2 * self.tau)

    def __call__(self, t: float) -> float:
        output = (
            torch.cos((t * (self.end - self.start) + self.start) * torch.pi / 2)
            ** (2 * self.tau)
            - self.v_end
        ) / (self.v_start - self.v_end)
        return torch.clamp(output, min=self.clip_min, max=self.clip_max)


class CosineSchedulerSimple:
    def __init__(self, ns: float = 0.0002, ds: float = 0.00025):
        self.ns = ns
        self.ds = ds

    def __call__(self, t: float) -> float:
        return torch.cos(((t + self.ns) / (1 + self.ds)) * torch.pi / 2) ** 2
