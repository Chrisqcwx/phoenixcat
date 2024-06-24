import time
import logging

from .base import Callback
from ..base import TrainerMixin
from ...format import format_time_invterval


logger = logging.getLogger(__name__)


class TimeCallback(Callback):

    def epoch_begin(self, trainer: TrainerMixin):
        self.t = time.time()

    def epoch_end(self, trainer: TrainerMixin):
        t = time.time() - self.t
        t_format = format_time_invterval(t)
        logger.info(f'time: {t_format}')


class TrainFlagCallback(Callback):

    def epoch_end(self, trainer: TrainerMixin):
        logger.info(f'epoch: {trainer.flag.epoch} step: {trainer.flag.step}')
