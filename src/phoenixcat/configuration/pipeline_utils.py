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

from ..files import load_json, safe_save_as_json
from ..conversion import get_obj_from_str
from .configuration_utils import ConfigMixin
from .autosave_utils import is_json_serializable
from .dataclass_utils import config_dataclass_wrapper
from .version import VersionInfo

logger = logging.getLogger(__name__)


class PipelineRecord:

    config_name = "pipeline_config.json"
    _auto_save_name = "_auto_save_modules"
    _pt_save_name = "_pt_save_modules"

    def __init__(self, **kwargs):
        self._constant = {}
        self._auto_save_modules = {}
        self._pt_save_modules = {}
        for name, value in kwargs.items():
            self.set(name, value)

    def set(self, key, value):
        # self._record[key] = value
        if hasattr(value, 'from_pretrained') and hasattr(value, 'save_pretrained'):
            self._auto_save_modules[key] = value
            return {self._auto_save_name: list(self._auto_save_modules.keys())}
        elif is_json_serializable(value):
            self._constant[key] = value
            return {key: value}
        else:
            self._pt_save_modules[key] = value
            return {self._pt_save_name: list(self._pt_save_modules.keys())}

    def get(self, key):
        return ChainMap(
            self._constant, self._auto_save_modules, self._pt_save_modules
        ).get(key, None)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        init_kwargs = cls.load(pretrained_model_name_or_path)
        return cls(**init_kwargs)

    @staticmethod
    def load(pretrained_model_name_or_path: str):
        config_path = os.path.join(
            pretrained_model_name_or_path, PipelineRecord.config_name
        )

        config = load_json(config_path)

        _pt_save_module = {
            key: torch.load(os.path.join(pretrained_model_name_or_path, f'{key}.pt'))
            for key in config.pop(PipelineRecord._pt_save_name, [])
            if not key.startswith('_')
        }

        _auto_save_module = {}
        for name, cls_name in config.pop(PipelineRecord._auto_save_name, {}).items():
            if name.startswith('_'):
                continue
            builder = get_obj_from_str(cls_name)
            module = builder.from_pretrained(
                os.path.join(pretrained_model_name_or_path, name)
            )
            _auto_save_module[name] = module

        config = {k: v for k, v in config.items() if not k.startswith("_")}

        init_kwargs = {**config, **_pt_save_module, **_auto_save_module}

        return init_kwargs

    def save_pretrained(self, path: str):
        # print(self._record)
        config_path = os.path.join(path, self.config_name)

        save_constant = copy.deepcopy(self._constant)
        save_constant[self._pt_save_name] = list(self._pt_save_modules.keys())

        save_constant[self._auto_save_name] = {}
        for name, value in self._auto_save_modules.items():
            save_constant[self._auto_save_name][
                name
            ] = f'{value.__class__.__module__}.{value.__class__.__name__}'
            name = name.lstrip('_')
            value.save_pretrained(os.path.join(path, name))

        safe_save_as_json(save_constant, config_path)

        for name, value in self._pt_save_modules.items():
            torch.save(value, os.path.join(path, f'{name}.pt'))

    @property
    def config(self):
        _config = copy.deepcopy(self._constant)
        _config[self._pt_save_name] = list(self._pt_save_modules.keys())
        _config[self._auto_save_name] = list(self._auto_save_modules.keys())
        return _config


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

        # Take note of the parameters that were not present in the loaded config
        # if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
        #     new_kwargs["_use_default_values"] = list(
        #         set(new_kwargs.keys()) - set(init_kwargs)
        #     )

        new_kwargs = {**config_init_kwargs, **new_kwargs}
        # getattr(self, "register_to_config")(**new_kwargs)
        # self.register_to_status(**new_kwargs)
        init(self, *args, **init_kwargs)

        for name, value in new_kwargs.items():
            # if is_json_serializable(value):
            #     self.register_constants(**{name: value})
            #     # print(f'>> {name} {value.__class__.__name__}')
            # else:
            #     # print(f'>>> {name} {value}')
            #     self.register_modules(**{name: value})
            # if hasattr(value, 'from_pretrained') and hasattr(value, 'save_pretrained'):
            #     self.register_modules(**{name: value})
            # else:
            self.register_save_values(**{name: value})

    return inner_init


@config_dataclass_wrapper(config_name='outputfiles.json')
@dataclass
class OutputFilesManager:
    logging_file: str | os.PathLike = "debug.log"
    version_file: str | os.PathLike = "version.json"
    logging_dir: str | os.PathLike = "logs"
    tensorboard_dir: str | os.PathLike = "tensorboard"
    wandb_dir: str | os.PathLike = "wandb"


def only_local_main_process(fn):

    @functools.wraps(fn)
    def inner_fn(self: PipelineMixin, *args, **kwargs):
        self.wait_for_everyone()
        if self.is_local_main_process:
            return fn(self, *args, **kwargs)
        self.wait_for_everyone()

    return inner_fn


def only_main_process(fn):

    @functools.wraps(fn)
    def inner_fn(self: PipelineMixin, *args, **kwargs):
        self.wait_for_everyone()
        if self.is_main_process:
            return fn(self, *args, **kwargs)
        self.wait_for_everyone()

    return inner_fn


class PipelineMixin(ConfigMixin):

    config_name = 'pipeline_config.json'
    # record_folder: str = 'record'
    ignore_for_pipeline = set()
    output_files_manager: OutputFilesManager = OutputFilesManager()

    def __init__(self) -> None:
        super().__init__()
        self._pipeline_record = PipelineRecord()
        # self.register_modules(pipeline_record=PipelineRecord())

        self.register_version()

    def register_accelerator(self, accelerator_config: Dict = None) -> None:
        if accelerate is None or accelerator_config is None:
            if accelerator_config is not None:
                logger.warn(
                    "accelerate is not installed, so the accelerator_config will be ignored."
                )
            self._accelerator = None
            self.use_ddp = False
        else:
            self._accelerator = Accelerator(**accelerator_config)
            self.use_ddp = True

    @property
    def accelerator(self) -> "Accelerator" | None:
        return getattr(self, "_accelerator", None)

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

    def register_version(self):
        self.register_save_values(_version=VersionInfo.create())

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
        self._pipeline_record.save_pretrained(record_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        # record_path = os.path.join(pretrained_model_name_or_path, cls.record_folder)
        record_path = pretrained_model_name_or_path
        try:
            records = PipelineRecord.load(record_path)
        except Exception as e:
            records = {}

        kwargs = {**records, **kwargs}

        init_parameters = inspect.signature(cls.__init__).parameters.keys()
        init_kwargs = {k: v for k, v in kwargs.items() if k in init_parameters}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in init_parameters}
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

            super().__setattr__(name, value)

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
