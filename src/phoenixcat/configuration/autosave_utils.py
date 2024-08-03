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
import abc
import copy
import json
import inspect
import logging
import functools
from typing import Callable
from collections import ChainMap
from typing import List

import torch

from ..files import safe_save_as_json, load_json
from ..conversion import get_obj_from_str

logger = logging.getLogger(__name__)


def get_init_parameters(cls, pop_kwargs=False):
    result = dict(inspect.signature(cls.__init__).parameters)
    result.pop("self", None)
    if pop_kwargs:
        result.pop("kwargs", None)
    return result


def split_init_other_parameters(cls, parameters):
    init_parameters = get_init_parameters(cls, pop_kwargs=True).keys()
    init_results = {k: v for k, v in parameters.items() if k in init_parameters}
    other_results = {k: v for k, v in parameters.items() if k not in init_parameters}
    return init_results, other_results


def auto_create_cls(cls, config, **kwargs):
    kwargs = {**{k: v for k, v in config.items() if k not in kwargs}, **kwargs}
    init_kwargs, _ = split_init_other_parameters(cls, kwargs)
    return cls(**init_kwargs)


def _register_fn(_method_name, fn: str | Callable, cls=None):

    def _inner_wrapper(cls):
        nonlocal fn
        if isinstance(fn, str):
            fn = getattr(cls, fn, None)
            if fn is None:
                raise ValueError(f"Cannot find method {fn} in {cls}")

        if hasattr(fn, _method_name):
            logger.warning(
                f"Method {fn.__name__} already has a {cls.__name__} method. Overwriting..."
            )

        setattr(cls, _method_name, fn)

    if cls is None:
        return _inner_wrapper
    return _inner_wrapper(cls)


def register_from_pretrained(load_method: str | Callable, cls=None):
    _register_fn('from_pretrained', load_method, cls)


def register_save_pretrained(save_method: str | Callable, cls=None):
    _register_fn('save_pretrained', save_method, cls)


# class ConvertToJsonMixin(abc.ABC):

#     @abc.abstractmethod
#     def to_json(self):
#         """Convert the model to a JSON object."""
#         raise NotImplementedError("This method should be implemented in the subclass.")

#     @abc.abstractmethod
#     @classmethod
#     def from_json(cls, json_obj: dict):
#         raise NotImplementedError("This method should be implemented in the subclass.")


def is_json_serializable(obj):
    # if hasattr(obj, 'to_json') and hasattr(obj, 'from_json'):
    #     return True
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


class AutoSaver:

    config_name = "config.json"
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
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        init_kwargs = cls.load(pretrained_model_name_or_path)
        return cls(**init_kwargs, **kwargs)

    @staticmethod
    def load(pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, AutoSaver.config_name)

        config = load_json(config_path)

        _pt_save_module = {
            key: torch.load(os.path.join(pretrained_model_name_or_path, f'{key}.pt'))
            for key in config.pop(AutoSaver._pt_save_name, [])
            if not key.startswith('_')
        }

        _auto_save_module = {}
        for name, cls_name in config.pop(AutoSaver._auto_save_name, {}).items():
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


def _register_to_autosave_init(init, ignore_list: List[str] = None):

    if ignore_list is None:
        ignore_list = []

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):

        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}

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

        self._phoenixcat_autosave_kwargs = {
            k: v for k, v in new_kwargs.items() if k not in ignore_list
        }

    return inner_init


def auto_register_save_load(
    cls,
    ignore_list: List[str] = None,
):
    cls.__init__ = _register_to_autosave_init(cls.__init__, ignore_list)

    @classmethod
    def from_pretrained(cls, pretrain_name_or_path, **kwargs):
        save_kwargs = AutoSaver.load(pretrain_name_or_path)
        # save_kwargs = {**save_kwargs, **kwargs}
        # init_kwargs, other_kwargs = split_init_other_parameters(cls, save_kwargs)
        init_kwargs = save_kwargs
        residule_kwargs = {}
        for key, value in kwargs.items():
            if key in init_kwargs:
                init_kwargs[key] = value
            else:
                residule_kwargs[key] = value
        residule_init_kwargs, residule_other_kwargs = split_init_other_parameters(
            cls, residule_kwargs
        )
        self = cls(**init_kwargs, **residule_init_kwargs)
        for key, value in residule_other_kwargs.items():
            setattr(self, key, value)
        return self

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        saver = AutoSaver()
        for name, value in self._phoenixcat_autosave_kwargs.items():
            saver.set(name, value)

        saver.save_pretrained(save_directory)

    cls.from_pretrained = from_pretrained
    cls.save_pretrained = save_pretrained

    return cls
