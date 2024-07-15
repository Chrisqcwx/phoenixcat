import logging

import torch
import numpy as np
from diffusers.configuration_utils import register_to_config

from ..trainer_utils import TrainerMixin
from .base import CallbackMixin, register_callback

logger = logging.getLogger(__name__)


class AvgDataCallback(CallbackMixin):

    def __init__(self, fullname: str):
        super().__init__()
        self.fullname = fullname
        self.fullnames = fullname.split('.')
        self.name = self.fullnames[-1]
        self.data = 0
        self.cnt = 0

    def epoch_begin(self, trainer: TrainerMixin):
        self.data = 0
        self.cnt = 0

    def epoch_end(self, trainer: TrainerMixin):
        avg_data = 0 if self.cnt == 0 else self.data / self.cnt
        logger.info(f'{self.name}: {avg_data:.6f}')

    def step_end(self, trainer: TrainerMixin):
        data = trainer
        for n in self.fullnames:
            if not hasattr(data, n):
                raise RuntimeError(f'`trainer` do not has attribute `{self.fullname}`')
            data = getattr(data, n)

        self.cnt += 1
        self.data += float(torch.mean(data))


@register_callback
class AvgLossCallback(AvgDataCallback):

    @register_to_config
    def __init__(self):
        super().__init__('train_temp_values.loss')
