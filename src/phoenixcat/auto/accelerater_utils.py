# Copyright 2024 Hongyao Yu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
from typing import Dict

import torch
from torch import compile
from torch.nn.parallel import DataParallel, DistributedDataParallel

import diffusers
from diffusers.utils import is_accelerate_available, is_torch_version

from .autosave_utils import auto_register_save_load

logger = logging.getLogger(__name__)

if is_accelerate_available():
    import accelerate
    from accelerate import Accelerator, DataLoaderConfiguration

    auto_register_save_load(DataLoaderConfiguration)

    auto_register_save_load(
        Accelerator,
        ignore_list=[
            'dispatch_batches',
            'split_batches',
            'even_batches',
            'use_seedable_sampler',
        ],
    )
else:
    accelerate = None


def is_compiled_module(module):
    """
    Check whether the module was compiled with torch.compile()
    """
    if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


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

    def register_accelerator(
        self, accelerator_or_config: Dict | "Accelerator" = None
    ) -> None:
        if accelerate is None or accelerator_or_config is None:
            if accelerator_or_config is not None:
                logger.warning(
                    "accelerate is not installed, so the accelerator_config will be ignored."
                )
            self._accelerator = None
            self._use_ddp = False
        else:
            if not isinstance(accelerator_or_config, Accelerator):
                self._accelerator = Accelerator(**accelerator_or_config)
            else:
                self._accelerator = accelerator_or_config
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

    def accelerator_prepare(self, *args, device_placement=None):
        if self.accelerator is not None:
            self.accelerator.prepare(*args, device_placement=device_placement)

    def unwrap_model(self, model):
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)

        if is_compiled_module(model):
            model = model._orig_mod

        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module

        return model

    def accelerator_log(self, values: dict, step: int | None = None):
        if self.accelerator is not None:
            self.accelerator.log(values, step=step)
