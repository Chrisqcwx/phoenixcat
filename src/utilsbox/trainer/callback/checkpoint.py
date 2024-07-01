import abc
import os
from typing import Literal

import torch
from diffusers.configuration_utils import register_to_config

from .base import CallbackMixin, register_callback
from ..trainer_utils import TrainerMixin
from ..constant import TRAINER_STATUS_NAME


class CheckpointCallbackMixin(CallbackMixin):

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

    def get_value(self, trainer: TrainerMixin):
        return trainer.flag.epoch if self.level == 'epoch' else trainer.flag.step

    @abc.abstractmethod
    def save_checkpoint(self, trainer: TrainerMixin):
        pass

    def _step_end(self, trainer: TrainerMixin):
        if trainer.is_local_main_process and trainer.flag.step % self.interval == 0:
            self.save_checkpoint(trainer)

    def _epoch_end(self, trainer: TrainerMixin):
        if trainer.is_local_main_process and trainer.flag.epoch % self.interval == 0:
            self.save_checkpoint(trainer)

    def on_train_end(self, trainer: TrainerMixin):
        if trainer.is_local_main_process:
            self.save_checkpoint(trainer)


@register_callback
class ModelCheckpointCallback(CheckpointCallbackMixin):

    @register_to_config
    def __init__(
        self, interval: int, level: Literal['epoch'] | Literal['step'] = 'epoch'
    ):
        super().__init__(interval, level)

    def save_checkpoint(self, trainer: TrainerMixin):
        save_dir = os.path.join(
            trainer.output_dir,
            'checkpoints',
            f'{self.level}_{self.get_value(trainer)}',
        )
        for name, model in trainer.models.items():
            save_path = os.path.join(save_dir, name)
            os.makedirs(save_path, exist_ok=True)
            model.model.save_pretrained(
                save_path, is_main_process=trainer.is_local_main_process
            )

    def on_train_end(self, trainer: TrainerMixin):
        self.save_checkpoint(trainer)


@register_callback
class TrainStatusCheckpointCallback(CheckpointCallbackMixin):

    @register_to_config
    def __init__(
        self, interval: int, level: Literal['epoch'] | Literal['step'] = 'epoch'
    ):
        super().__init__(interval, level)

    def save_checkpoint(self, trainer: TrainerMixin):
        return torch.save(
            trainer, os.path.join(trainer.output_dir, TRAINER_STATUS_NAME)
        )
