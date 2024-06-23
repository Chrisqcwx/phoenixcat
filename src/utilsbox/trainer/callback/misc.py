import time
import logging

from .base import Callback
from ..context import TrainerContext
from ...format import format_time_invterval


logger = logging.getLogger(__name__)


class TimeCallback(Callback):

    def epoch_begin(self, run_context: TrainerContext):
        self.t = time.time()

    def epoch_end(self, run_context: TrainerContext):
        t = time.time() - self.t
        t_format = format_time_invterval(t)
        logger.info(f'time: {t_format}')


class TrainFlagCallback(Callback):

    def epoch_end(self, run_context: TrainerContext):
        logger.info(f'epoch: {run_context.flag.epoch} step: {run_context.flag.step}')
