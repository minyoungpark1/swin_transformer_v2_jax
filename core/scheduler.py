import torch
import optax
import jax
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
def build_scheduler(config, optimizer, epoch_iters):
    num_steps = int(config.TRAIN.EPOCHS * epoch_iters)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * epoch_iters)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * epoch_iters)

    lr_scheduler = None

    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

def build_scheduler_jax(config, epoch_iters):
    num_steps = int(config.TRAIN.EPOCHS * epoch_iters)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * epoch_iters)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * epoch_iters)

    lr_scheduler = None

    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=config.TRAIN.WARMUP_LR,
            peak_value=config.TRAIN.BASE_LR,
            warmup_steps=warmup_steps,
            decay_steps=num_steps,
            end_value=config.TRAIN.MIN_LR
            )

    elif config.TRAIN.LR_SCHEDULER.NAME == 'decay':
        lr_scheduler = optax.warmup_exponential_decay_schedule(
            init_value=config.TRAIN.WARMUP_LR,
            peak_value=config.TRAIN.BASE_LR,
            warmup_steps=warmup_steps,
            transition_steps =num_steps,
            decay_rate=0.975,
            end_value=config.TRAIN.MIN_LR
            )

    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        boundaries_and_scales = {int(i):config.TRAIN.LR_SCHEDULER.DECAY_RATE \
        for i in np.arange(decay_steps, decay_steps, num_steps)}
        lr_scheduler = optax.piecewise_constant_schedule(
            init_value=config.TRAIN.BASE_LR,
            boundaries_and_scales=boundaries_and_scales
        )

    return lr_scheduler
