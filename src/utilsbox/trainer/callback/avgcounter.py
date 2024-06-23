import logging

import torch
import numpy as np

from utilsbox.trainer.context import TrainerContext

from .base import Callback

logger = logging.getLogger(__name__)


class AvgDataCallback(Callback):

    def __init__(self, fullname: str):
        super().__init__()
        self.fullname = fullname
        self.fullnames = fullname.split('.')
        self.name = self.fullnames[-1]
        self.data = 0
        self.cnt = 0

    def epoch_begin(self, run_context: TrainerContext):
        self.data = 0
        self.cnt = 0

    def epoch_end(self, run_context: TrainerContext):
        avg_data = 0 if self.cnt == 0 else self.data / self.cnt
        logger.info(f'{self.name}: {avg_data:.6f}')

    def step_end(self, run_context: TrainerContext):
        data = run_context
        for n in self.fullnames:
            if not hasattr(data, n):
                raise RuntimeError(
                    f'`run_context` do not has attribute `{self.fullname}`'
                )
            data = getattr(data, n)

        self.cnt += 1
        self.data += float(torch.mean(data))


class AvgLossCallback(AvgDataCallback):

    def __init__(self):
        super().__init__('train_temp_values.loss')
