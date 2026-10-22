import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class DummyScheduler:
    def __init__(self, optimizer):
        self.lr = [group["lr"] for group in optimizer.param_groups]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        pass

    def get_last_lr(self):
        return self.lr


class CosineScheduler(LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, max_epochs: int, min_factor: float):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_factor = min_factor
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / float(self.max_epochs - self.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine


class WarmupLinearScheduler(LambdaLR):
    """
    Warmup linear scheduler.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_ratio: float):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_ratio = min_ratio
        super().__init__(optimizer, self.schedule)

    def schedule(self, step) -> float:
        if step < self.warmup_steps:
            # Warmup: linearly increase from 0.0 to 1.0
            return float(step) / float(max(1, self.warmup_steps))

        # Decay: linearly decrease from 1.0 back down to min_ratio
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        return 1.0 - progress * (1.0 - self.min_ratio)
