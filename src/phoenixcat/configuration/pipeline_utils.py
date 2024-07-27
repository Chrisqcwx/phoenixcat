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

import os
import json
import copy
import functools
import inspect
import importlib
import logging
from collections import ChainMap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers.utils import is_accelerate_available

if is_accelerate_available():
    import accelerate
    from accelerate import Accelerator
else:
    accelerate = None

from .configuration_utils import ConfigMixin
from .autosave_utils import AutoSaver, split_init_other_parameters
from .dataclass_utils import config_dataclass_wrapper
from .version import VersionInfo
from .accelerater_utils import (
    only_local_main_process,
    only_main_process,
    AccelerateMixin,
)
from .order_utils import ExecuteOrderMixin

logger = logging.getLogger(__name__)


def register_to_pipeline_init(init):
    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):

        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        if not isinstance(self, PipelineMixin):
            raise RuntimeError(
                f"`@register_to_pipeline_init` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `PipelineMixin`."
            )

        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default
            for i, (name, p) in enumerate(signature.parameters.items())
            if i > 0
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in new_kwargs
            }
        )

        new_kwargs = {**config_init_kwargs, **new_kwargs}
        init(self, *args, **init_kwargs)

        for name, value in new_kwargs.items():
            self.register_save_values(**{name: value})

    return inner_init


@config_dataclass_wrapper(config_name='outputfiles.json')
@dataclass
class OutputFilesManager:
    logging_file: str | os.PathLike = "debug.log"
    logging_dir: str | os.PathLike = "logs"
    tensorboard_dir: str | os.PathLike = "tensorboard"
    wandb_dir: str | os.PathLike = "wandb"


class PipelineMixin(ConfigMixin, AccelerateMixin, ExecuteOrderMixin):

    config_name = 'pipeline_config.json'
    # record_folder: str = 'record'
    ignore_for_pipeline = set()
    output_files_manager: OutputFilesManager = OutputFilesManager()

    def __init__(self) -> None:
        super().__init__()
        self._pipeline_record = AutoSaver()
        # self.register_modules(pipeline_record=PipelineRecord())

        self.register_version()
        self.register_save_values(execute_counts=self.execute_counts)

    def register_version(self):
        self.register_save_values(_version=VersionInfo.create(clear_package=True))

    def register_logger(self, logger_config: Dict = None):
        if logger_config is None:
            logger_config = {}

    @only_main_process
    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        # safe_serialization: bool = True,
        # variant: str | None = None,
        # push_to_hub: bool = False,
        # **kwargs,
    ):
        # TODO: add these params

        # super().save_pretrained(
        #     save_directory, safe_serialization, variant, push_to_hub, **kwargs
        # )
        # record_path = os.path.join(save_directory, self.record_folder)
        record_path = save_directory
        self.register_version()
        self._pipeline_record.save_pretrained(record_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        # record_path = os.path.join(pretrained_model_name_or_path, cls.record_folder)
        record_path = pretrained_model_name_or_path
        try:
            records = AutoSaver.load(record_path)
        except Exception as e:
            records = {}

        kwargs = {**records, **kwargs}

        # init_parameters = inspect.signature(cls.__init__).parameters.keys()
        # init_kwargs = {k: v for k, v in kwargs.items() if k in init_parameters}
        # other_kwargs = {k: v for k, v in kwargs.items() if k not in init_parameters}
        init_kwargs, other_kwargs = split_init_other_parameters(cls, kwargs)
        # self = super().from_pretrained(pretrained_model_name_or_path, **init_kwargs)
        self = cls(**init_kwargs)

        self.register_save_values(**other_kwargs)

        return self

    # def __setattr__(self, name: str, value):

    #     super().__setattr__(name, value)
    # if name.startswith('_'):
    #     super().__setattr__(name, value)
    #     return

    # self.register_save_values(**{name: value})

    def register_save_values(self, **kwargs):

        for name, value in kwargs.items():
            if not name in self.ignore_for_pipeline:

                update_config_dict = self._pipeline_record.set(name, value)
                self.register_to_config(**update_config_dict)

            ConfigMixin.__setattr__(self, name, value)

    def to(self, *args, **kwargs):
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        # silence_dtype_warnings = kwargs.pop("silence_dtype_warnings", False)

        dtype_arg = None
        device_arg = None
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_arg = args[0]
            else:
                device_arg = torch.device(args[0]) if args[0] is not None else None
        elif len(args) == 2:
            if isinstance(args[0], torch.dtype):
                raise ValueError(
                    "When passing two arguments, make sure the first corresponds to `device` and the second to `dtype`."
                )
            device_arg = torch.device(args[0]) if args[0] is not None else None
            dtype_arg = args[1]
        elif len(args) > 2:
            raise ValueError(
                "Please make sure to pass at most two arguments (`device` and `dtype`) `.to(...)`"
            )

        if dtype is not None and dtype_arg is not None:
            raise ValueError(
                "You have passed `dtype` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        dtype = dtype or dtype_arg

        if device is not None and device_arg is not None:
            raise ValueError(
                "You have passed `device` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        device = device or device_arg

        for module in self.modules:
            is_loaded_in_8bit = (
                hasattr(module, "is_loaded_in_8bit") and module.is_loaded_in_8bit
            )

            if is_loaded_in_8bit and dtype is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in 8bit and conversion to {dtype} is not yet supported. Module is still in 8bit precision."
                )

            if is_loaded_in_8bit and device is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in 8bit and moving it to {dtype} via `.to()` is not yet supported. Module is still on {module.device}."
                )
            else:
                module.to(device, dtype)

        return self

    @property
    def modules(self):
        return [
            m
            for m in self._pipeline_record._auto_save_modules.values()
            if isinstance(m, torch.nn.Module)
        ]

    @property
    def device(self):

        for module in self.modules:
            if hasattr(module, "device"):
                return module.device
            for param in module.parameters():
                return param.device

        return torch.device("cpu")

    @property
    def dtype(self):

        for module in self.modules:
            if hasattr(module, "dtype"):
                return module.dtype
            for param in module.parameters():
                return param.dtype

        return torch.float32
