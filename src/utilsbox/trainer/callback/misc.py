import time
import logging

from diffusers.configuration_utils import register_to_config

from .base import CallbackMixin, register_callback
from ..base import TrainerMixin
from ...format import format_time_invterval

logger = logging.getLogger(__name__)


@register_callback
class TimeCallback(CallbackMixin):

    @register_to_config
    def __init__(self):
        super().__init__()

    def epoch_begin(self, trainer: TrainerMixin):
        self.t = time.time()

    def epoch_end(self, trainer: TrainerMixin):
        t = time.time() - self.t
        t_format = format_time_invterval(t)
        logger.info(f'time: {t_format}')


@register_callback
class TrainFlagCallback(CallbackMixin):

    @register_to_config
    def __init__(self):
        super().__init__()

    def epoch_end(self, trainer: TrainerMixin):
        logger.info(f'epoch: {trainer.flag.epoch} step: {trainer.flag.step}')
