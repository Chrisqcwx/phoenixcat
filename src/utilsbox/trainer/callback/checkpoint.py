from typing import Literal
from .base import Callback
from ..context import TrainerContext


class TrainCheckpointCallback(Callback):

    def __init__(self, interval: int, level: Literal['epoch', 'step'] = 'epoch'):
        super().__init__()
        self.interval = interval

        if level == 'epoch':
            self.on_train_epoch_end = self._epoch_end
        elif level == 'step':
            self.on_train_step_end = self._step_end
        else:
            raise RuntimeError(f'The level must be epoch or step')

    def _save_checkpoint(self, run_context: TrainerContext):
        # TODO
        pass

    def _step_end(self, run_context: TrainerContext):
        if run_context.flag.step % self.interval == 0:
            self._save_checkpoint(run_context)

    def _epoch_end(self, run_context: TrainerContext):
        if run_context.flag.epoch % self.interval == 0:
            self._save_checkpoint(run_context)

    def on_train_end(self, run_context: TrainerContext):
        self._save_checkpoint(run_context)
