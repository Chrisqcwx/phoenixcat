import functools
import logging
from typing import Dict

import diffusers
from diffusers.utils import is_accelerate_available

logger = logging.getLogger(__name__)

if is_accelerate_available():
    import accelerate
    from accelerate import Accelerator
else:
    accelerate = None


def only_local_main_process(fn):

    @functools.wraps(fn)
    def inner_fn(self: AccelerateMixin, *args, **kwargs):
        self.wait_for_everyone()
        if self.is_local_main_process:
            return fn(self, *args, **kwargs)
        self.wait_for_everyone()

    return inner_fn


def only_main_process(fn):

    @functools.wraps(fn)
    def inner_fn(self: AccelerateMixin, *args, **kwargs):
        self.wait_for_everyone()
        if self.is_main_process:
            return fn(self, *args, **kwargs)
        self.wait_for_everyone()

    return inner_fn


class AccelerateMixin:

    _use_ddp: bool = False
    _accelerator: "Accelerator" = None

    def register_accelerator(self, accelerator_config: Dict = None) -> None:
        if accelerate is None or accelerator_config is None:
            if accelerator_config is not None:
                logger.warn(
                    "accelerate is not installed, so the accelerator_config will be ignored."
                )
            self._accelerator = None
            self._use_ddp = False
        else:
            self._accelerator = Accelerator(**accelerator_config)
            self._use_ddp = True

    @property
    def accelerator(self) -> "Accelerator":
        return self._accelerator

    @property
    def use_ddp(self) -> bool:
        return self._use_ddp

    @property
    def is_local_main_process(self) -> bool:
        if self.accelerator is None:
            return True
        return self.accelerator.is_local_main_process

    @property
    def is_main_process(self) -> bool:
        if self.accelerator is None:
            return True
        return self.accelerator.is_main_process

    def wait_for_everyone(self):
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
