import logging
from typing import Dict, List, Any, Literal

from diffusers.configuration_utils import register_to_config

from ..base import TrainerMixin
from ...decorators import Register
from ...configuration import ConfigMixin, extract_init_dict, auto_cls_from_pretrained


logger = logging.getLogger(__name__)

_callback_register = Register('callback')

register_callback = _callback_register.register()


def list_callbacks():
    return list(_callback_register.keys())


def get_callback_builder(name: str):
    return _callback_register[name]


class CallbackMixin(ConfigMixin):

    config_name = 'callback.json'

    def __init__(self):
        pass

    # def __enter__(self):
    #     pass

    # def __exit__(self, *err):
    #     """Release resources here if have any."""

    def begin(self, trainer: TrainerMixin):
        pass

    def step_begin(self, trainer: TrainerMixin):
        pass

    def step_end(self, trainer: TrainerMixin):
        pass

    def epoch_begin(self, trainer: TrainerMixin):
        pass

    def epoch_end(self, trainer: TrainerMixin):
        pass

    def end(self, trainer: TrainerMixin):
        pass

    def on_train_begin(self, trainer: TrainerMixin):
        """
        Called once before the network training.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.begin(trainer)

    def on_train_epoch_begin(self, trainer: TrainerMixin):
        """
        Called before each training epoch begin.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.epoch_begin(trainer)

    def on_train_epoch_end(self, trainer: TrainerMixin):
        """
        Called after each training epoch end.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.epoch_end(trainer)

    def on_train_step_begin(self, trainer: TrainerMixin):
        """
        Called before each training step begin.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.step_begin(trainer)

    def on_train_step_end(self, trainer: TrainerMixin):
        """
        Called after each training step end.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.step_end(trainer)

    def on_train_end(self, trainer: TrainerMixin):
        """
        Called after training end.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.end(trainer)

    def on_eval_begin(self, trainer: TrainerMixin):
        """
        Called before eval begin.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.begin(trainer)

    def on_eval_epoch_begin(self, trainer: TrainerMixin):
        """
        Called before eval epoch begin.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.epoch_begin(trainer)

    def on_eval_epoch_end(self, trainer: TrainerMixin):
        """
        Called after eval epoch end.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.epoch_end(trainer)

    def on_eval_step_begin(self, trainer: TrainerMixin):
        """
        Called before each eval step begin.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.step_begin(trainer)

    def on_eval_step_end(self, trainer: TrainerMixin):
        """
        Called after each eval step end.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.step_end(trainer)

    def on_eval_end(self, trainer: TrainerMixin):
        """
        Called after eval end.

        Args:
            trainer (TrainerMixin): Include some information of the model.
        """
        self.end(trainer)


@register_callback
class ComposeCallback(CallbackMixin):
    """
    Sequential execution of callback functions.

    Execute Callback functions at certain points.
    """

    @register_to_config
    def __init__(self, callbacks_init_list: List[Dict[Literal['name', 'kwargs'], Any]]):
        self._callbacks = []
        for cb_args in callbacks_init_list:
            if 'name' not in cb_args and 'kwargs' not in cb_args:
                raise RuntimeError(f'Parameters of `callbacks_init_list` error!')
            factory = get_callback_builder(cb_args['name'])
            cb = factory(**cb_args['kwargs'])
            self._callbacks.append(cb)

    @classmethod
    def create_from_callbacks(cls, callbacks: CallbackMixin, **kwargs):

        _callbacks = []
        if isinstance(callbacks, CallbackMixin):
            _callbacks.append(callbacks)
        elif isinstance(callbacks, list):
            for cb in callbacks:
                if not isinstance(cb, CallbackMixin):
                    raise TypeError(
                        "When the 'callbacks' is a list, the elements in "
                        "'callbacks' must be Callback functions."
                    )
                _callbacks.append(cb)
        elif callbacks is not None:
            raise TypeError("The 'callbacks' is not a Callback or a list of Callback.")

        cb_list = []
        for inner_cb in _callbacks:
            inner_cb_name = inner_cb.__class__.__name__
            inner_cb_kwargs = extract_init_dict(inner_cb)[0]
            cb_list.append({'name': inner_cb_name, 'kwargs': inner_cb_kwargs})

        return cls(cb_list)

    def begin(self, trainer):
        """Called once before network train or eval."""
        for cb in self._callbacks:
            cb.begin(trainer)

    def epoch_begin(self, trainer):
        """Called before each epoch begin."""
        for cb in self._callbacks:
            cb.epoch_begin(trainer)

    def epoch_end(self, trainer):
        """Called after each epoch finished."""
        for cb in self._callbacks:
            cb.epoch_end(trainer)

    def step_begin(self, trainer):
        """Called before each step begin."""
        for cb in self._callbacks:
            cb.step_begin(trainer)

    def step_end(self, trainer):
        """Called after each step finished."""
        for cb in self._callbacks:
            cb.step_end(trainer)

    def end(self, trainer):
        """Called once after network train or eval."""
        for cb in self._callbacks:
            cb.end(trainer)

    def on_train_begin(self, trainer):
        """Called before network train."""
        for cb in self._callbacks:
            cb.on_train_begin(trainer)

    def on_train_epoch_begin(self, trainer):
        """Called before each train epoch begin."""
        for cb in self._callbacks:
            cb.on_train_epoch_begin(trainer)

    def on_train_epoch_end(self, trainer):
        """Called after each train epoch finished."""
        for cb in self._callbacks:
            cb.on_train_epoch_end(trainer)

    def on_train_step_begin(self, trainer):
        """Called before each train step begin."""
        for cb in self._callbacks:
            cb.on_train_step_begin(trainer)

    def on_train_step_end(self, trainer):
        """Called after each train step finished."""
        for cb in self._callbacks:
            cb.on_train_step_end(trainer)

    def on_train_end(self, trainer):
        """Called after network train end."""
        for cb in self._callbacks:
            cb.on_train_end(trainer)

    def on_eval_begin(self, trainer):
        """Called before network eval."""
        for cb in self._callbacks:
            cb.on_eval_begin(trainer)

    def on_eval_epoch_begin(self, trainer):
        """Called before eval epoch begin."""
        for cb in self._callbacks:
            cb.on_eval_epoch_begin(trainer)

    def on_eval_epoch_end(self, trainer):
        """Called after eval epoch finished."""
        for cb in self._callbacks:
            cb.on_eval_epoch_end(trainer)

    def on_eval_step_begin(self, trainer):
        """Called before each eval step begin."""
        for cb in self._callbacks:
            cb.on_eval_step_begin(trainer)

    def on_eval_step_end(self, trainer):
        """Called after each eval step finished."""
        for cb in self._callbacks:
            cb.on_eval_step_end(trainer)

    def on_eval_end(self, trainer):
        """Called after network eval end."""
        for cb in self._callbacks:
            cb.on_eval_end(trainer)


def auto_trainer_from_pretrained(path: str, **kwargs):

    return auto_cls_from_pretrained(_callback_register, CallbackMixin, path, **kwargs)
