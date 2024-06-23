import logging

from ..context import TrainerContext

logger = logging.getLogger(__name__)


class Callback:

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *err):
        """Release resources here if have any."""

    def begin(self, run_context: TrainerContext):
        pass

    def step_begin(self, run_context: TrainerContext):
        pass

    def step_end(self, run_context: TrainerContext):
        pass

    def epoch_begin(self, run_context: TrainerContext):
        pass

    def epoch_end(self, run_context: TrainerContext):
        pass

    def end(self, run_context: TrainerContext):
        pass

    def on_train_begin(self, run_context: TrainerContext):
        """
        Called once before the network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.begin(run_context)

    def on_train_epoch_begin(self, run_context: TrainerContext):
        """
        Called before each training epoch begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_begin(run_context)

    def on_train_epoch_end(self, run_context: TrainerContext):
        """
        Called after each training epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_end(run_context)

    def on_train_step_begin(self, run_context: TrainerContext):
        """
        Called before each training step begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_begin(run_context)

    def on_train_step_end(self, run_context: TrainerContext):
        """
        Called after each training step end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_end(run_context)

    def on_train_end(self, run_context: TrainerContext):
        """
        Called after training end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.end(run_context)

    def on_eval_begin(self, run_context: TrainerContext):
        """
        Called before eval begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.begin(run_context)

    def on_eval_epoch_begin(self, run_context: TrainerContext):
        """
        Called before eval epoch begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_begin(run_context)

    def on_eval_epoch_end(self, run_context: TrainerContext):
        """
        Called after eval epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_end(run_context)

    def on_eval_step_begin(self, run_context: TrainerContext):
        """
        Called before each eval step begin.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_begin(run_context)

    def on_eval_step_end(self, run_context: TrainerContext):
        """
        Called after each eval step end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_end(run_context)

    def on_eval_end(self, run_context: TrainerContext):
        """
        Called after eval end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.end(run_context)


class ComposeCallback(Callback):
    """
    Sequential execution of callback functions.

    Execute Callback functions at certain points.

    Args:
        callbacks (Optional[list[Callback], Callback]): None, callback, or callbacks list.
    """

    def __init__(self, callbacks):
        self._callbacks = []
        if isinstance(callbacks, Callback):
            self._callbacks.append(callbacks)
        elif isinstance(callbacks, list):
            for cb in callbacks:
                if not isinstance(cb, Callback):
                    raise TypeError(
                        "When the 'callbacks' is a list, the elements in "
                        "'callbacks' must be Callback functions."
                    )
                self._callbacks.append(cb)
        elif callbacks is not None:
            raise TypeError("The 'callbacks' is not a Callback or a list of Callback.")

    def begin(self, run_context):
        """Called once before network train or eval."""
        for cb in self._callbacks:
            cb.begin(run_context)

    def epoch_begin(self, run_context):
        """Called before each epoch begin."""
        for cb in self._callbacks:
            cb.epoch_begin(run_context)

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        for cb in self._callbacks:
            cb.epoch_end(run_context)

    def step_begin(self, run_context):
        """Called before each step begin."""
        for cb in self._callbacks:
            cb.step_begin(run_context)

    def step_end(self, run_context):
        """Called after each step finished."""
        for cb in self._callbacks:
            cb.step_end(run_context)

    def end(self, run_context):
        """Called once after network train or eval."""
        for cb in self._callbacks:
            cb.end(run_context)

    def on_train_begin(self, run_context):
        """Called before network train."""
        for cb in self._callbacks:
            cb.on_train_begin(run_context)

    def on_train_epoch_begin(self, run_context):
        """Called before each train epoch begin."""
        for cb in self._callbacks:
            cb.on_train_epoch_begin(run_context)

    def on_train_epoch_end(self, run_context):
        """Called after each train epoch finished."""
        for cb in self._callbacks:
            cb.on_train_epoch_end(run_context)

    def on_train_step_begin(self, run_context):
        """Called before each train step begin."""
        for cb in self._callbacks:
            cb.on_train_step_begin(run_context)

    def on_train_step_end(self, run_context):
        """Called after each train step finished."""
        for cb in self._callbacks:
            cb.on_train_step_end(run_context)

    def on_train_end(self, run_context):
        """Called after network train end."""
        for cb in self._callbacks:
            cb.on_train_end(run_context)

    def on_eval_begin(self, run_context):
        """Called before network eval."""
        for cb in self._callbacks:
            cb.on_eval_begin(run_context)

    def on_eval_epoch_begin(self, run_context):
        """Called before eval epoch begin."""
        for cb in self._callbacks:
            cb.on_eval_epoch_begin(run_context)

    def on_eval_epoch_end(self, run_context):
        """Called after eval epoch finished."""
        for cb in self._callbacks:
            cb.on_eval_epoch_end(run_context)

    def on_eval_step_begin(self, run_context):
        """Called before each eval step begin."""
        for cb in self._callbacks:
            cb.on_eval_step_begin(run_context)

    def on_eval_step_end(self, run_context):
        """Called after each eval step finished."""
        for cb in self._callbacks:
            cb.on_eval_step_end(run_context)

    def on_eval_end(self, run_context):
        """Called after network eval end."""
        for cb in self._callbacks:
            cb.on_eval_end(run_context)
