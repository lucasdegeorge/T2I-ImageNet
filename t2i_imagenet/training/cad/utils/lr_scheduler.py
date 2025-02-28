import math


class WarmupLR:
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = None

    def get_lr(self, lr, step):
        return lr * min(step / max(self.warmup_steps, 1), 1.0)

    def step(self, step):
        if self.base_lr is None:
            self.base_lr = [
                param_group["lr"] for param_group in self.optimizer.param_groups
            ]
        for param_group, base_lr_group in zip(
            self.optimizer.param_groups, self.base_lr
        ):
            param_group["lr"] = self.get_lr(base_lr_group, step)

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class WarmupCosineDecayLR:
    def __init__(self, optimizer, warmup_steps, total_steps, rate=1.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = None
        self.total_steps = total_steps
        self.rate = rate

    def get_lr(self, lr, step):
        if step < self.warmup_steps:
            return lr * min(step / max(self.warmup_steps, 1), 1.0)
        else:
            return (
                0.5
                * lr
                * (
                    1
                    + math.cos(
                        self.rate
                        * math.pi
                        * (step - self.warmup_steps)
                        / (self.total_steps - self.warmup_steps)
                    )
                )
            )

    def step(self, step):
        if self.base_lr is None:
            self.base_lr = [
                param_group["lr"] for param_group in self.optimizer.param_groups
            ]
        for param_group, base_lr_group in zip(
            self.optimizer.param_groups, self.base_lr
        ):
            param_group["lr"] = self.get_lr(base_lr_group, step)

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class WarmupSqrtDecayLR:
    def __init__(self, optimizer, warmup_steps, decay_steps, total_steps):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.base_lr = None

    def get_lr(self, lr, step):
        if step < self.warmup_steps:
            return lr * min(step / max(self.warmup_steps, 1), 1.0)
        elif step > self.total_steps - self.decay_steps:
            return lr * (
                1
                - min(
                    math.sqrt(
                        (step - self.total_steps + self.decay_steps)
                        / max(self.decay_steps, 1)
                    ),
                    1.0,
                )
            )
        else:
            return lr

    def step(self, step):
        if self.base_lr is None:
            self.base_lr = [
                param_group["lr"] for param_group in self.optimizer.param_groups
            ]
        for param_group, base_lr_group in zip(
            self.optimizer.param_groups, self.base_lr
        ):
            param_group["lr"] = self.get_lr(base_lr_group, step)

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
