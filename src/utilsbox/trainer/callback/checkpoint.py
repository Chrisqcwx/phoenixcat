import abc
import os
from typing import Literal

import torch

from .base import Callback
from ..context import TrainerContext


class CheckpointCallback(Callback):

    def __init__(self, interval: int, level: Literal['epoch', 'step'] = 'epoch'):
        super().__init__()
        self.interval = interval

        self.level = level
        if level == 'epoch':
            self.on_train_epoch_end = self._epoch_end
        elif level == 'step':
            self.on_train_step_end = self._step_end
        else:
            raise RuntimeError(f'The level must be epoch or step')

    def get_value(self, run_context: TrainerContext):
        return (
            run_context.flag.epoch if self.level == 'epoch' else run_context.flag.step
        )

    @abc.abstractmethod
    def save_checkpoint(self, run_context: TrainerContext):
        pass

    def _step_end(self, run_context: TrainerContext):
        if run_context.flag.step % self.interval == 0:
            self.save_checkpoint(run_context)

    def _epoch_end(self, run_context: TrainerContext):
        if run_context.flag.epoch % self.interval == 0:
            self.save_checkpoint(run_context)

    def on_train_end(self, run_context: TrainerContext):
        self.save_checkpoint(run_context)


class ModelCheckpointCallback(CheckpointCallback):

    def __init__(
        self, interval: int, level: Literal['epoch'] | Literal['step'] = 'epoch'
    ):
        super().__init__(interval, level)

    def save_checkpoint(self, run_context: TrainerContext):
        save_dir = os.path.join(
            run_context.output_dir,
            'checkpoints',
            f'{self.level}_{self.get_value(run_context)}',
        )
        for name, model in run_context.models.items():
            save_path = os.path.join(save_dir, name)
            os.makedirs(save_path, exist_ok=True)
            model.model.save_pretrained(
                save_path,
            )
