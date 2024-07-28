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
import logging
from dataclasses import dataclass
from typing import Dict, Union, Any, Optional, AnyStr

import torch
from diffusers.optimization import get_scheduler as diffusers_get_scheduler

from ..conversion import get_obj_from_str
from ..decorators import Register
from ..files import safe_save_as_json, load_json, get_safe_save_path
from ..configuration import config_dataclass_wrapper


logger = logging.getLogger(__name__)

optimizer_register = Register('optimizer')
lr_shceduler_register = Register('lr_scheduler')


def _search(
    register: Register,
    name: Optional[AnyStr] = None,
    first_input: Any = None,
    build_params: Optional[Dict] = None,
    source: Optional[str] = None,
):
    if name is None:
        return None

    if source is not None and source not in register.keys():
        raise RuntimeError(f'Source {source} is not valid')

    if build_params is None:
        build_params = {}

    search_keys = [source] if source is not None else register.keys()
    for src in search_keys:
        cls_builder = register[src]
        instance = cls_builder(name, first_input, build_params)
        if instance is not None:
            logger.info(f'get {register.name} from {src}')
            return instance

    raise RuntimeError(f'Name {name} can not be found for {register.name}.')


@optimizer_register.register('torch')
def _get_torch_optimizer(name: str, params, build_params):
    if not name.startswith('torch.optim.'):
        name = f'torch.optim.{name}'
    optimizer_cls = get_obj_from_str(name)
    return optimizer_cls(params, **build_params)


@lr_shceduler_register.register('torch')
def _get_torch_lr_scheduler(name: str, optimizer, kwargs):
    lr_scheduler_cls = get_obj_from_str(f'torch.optim.lr_scheduler.{name}')
    if lr_scheduler_cls is None:
        return None
    return lr_scheduler_cls(optimizer, **kwargs)


@lr_shceduler_register.register('diffusers')
def _get_diffusers_lr_scheduler(name: str, optimizer, kwargs):
    try:
        lr_scheduler = diffusers_get_scheduler(name, optimizer, **kwargs)
    except:
        return None

    return lr_scheduler


def get_optimizer(name, params, optimizer_params):

    return _search(optimizer_register, name, params, optimizer_params)


def get_lr_scheduler(name, optimizer, lr_scheduler_params):

    return _search(lr_shceduler_register, name, optimizer, lr_scheduler_params)


@config_dataclass_wrapper('optimization_config.json')
@dataclass
class OptimizationConfig:
    optimizer_name: str
    optimizer_params: Dict = None
    lr_scheduler_name: str = None
    lr_scheduler_params: Dict = None


class SingleOptimizationManager:

    save_name = 'optimization.pt'

    def __init__(
        self,
        params,
        config: OptimizationConfig,
    ):
        self.params = params
        self._config = config
        self.optimizer = get_optimizer(
            config.optimizer_name, params, config.optimizer_params
        )
        self.lr_scheduler = get_lr_scheduler(
            config.lr_scheduler_name, self.optimizer, config.lr_scheduler_params
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()

    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def state_dict(self):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
        }

        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.lr_scheduler is not None and 'lr_scheduler' in state_dict:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

    def load_state_dict_from_file(self, save_directory: str):
        if not save_directory.endswith(self.save_name):
            save_path = os.path.join(save_directory, self.save_name)
        else:
            save_path = save_directory
        state_dict = torch.load(save_path, map_location=next(self.params).device)
        self.load_state_dict(state_dict)

    def save_state_dict_to_file(self, save_directory: str):
        if not save_directory.endswith(self.save_name):
            save_path = os.path.join(save_directory, self.save_name)
        else:
            save_path = save_directory
        save_path = get_safe_save_path(save_path)
        torch.save(self.state_dict(), save_path)

    def __iter__(self):
        # 定义迭代器，返回两个值
        yield self.optimizer
        yield self.lr_scheduler

    def accelerator_prepare(self, accelerator: "Accelerator", device_placement=True):
        self.optimizer = accelerator.prepare(
            self.optimizer, device_placement=device_placement
        )
        if self.lr_scheduler is not None:
            self.lr_scheduler = accelerator.prepare(
                self.lr_scheduler, device_placement=device_placement
            )


class OptimizationManager:

    optimization_group: Dict[str, SingleOptimizationManager] = {}

    def __init__(self) -> None:
        pass

    def register_optimization(
        self,
        tag,
        params,
        optimization_config,
    ):
        _optimize_manager = SingleOptimizationManager(params, optimization_config)
        self.optimization_group[tag] = _optimize_manager

        return _optimize_manager

    def get_optimization(self, tag):
        return self.optimization_group[tag]

    def get_optimizer(self, tag):
        return self.optimization_group[tag].optimizer

    def get_lr_scheduler(self, tag):
        return self.optimization_group[tag].lr_scheduler

    def state_dict(self, tag: Optional[str] = None):
        if tag is None:
            return {
                tag: manager.state_dict()
                for tag, manager in self.optimization_group.items()
            }
        else:
            return self.optimization_group[tag].state_dict()

    def load_state_dict_from_file(self, save_directory: str):
        for tag, manager in self.optimization_group.items():
            tag_directory = os.path.join(save_directory, tag)
            if not os.path.exists(tag_directory):
                logger.warning(
                    f'No save directory found for optimization {tag}. Automatically ignore it.'
                )
                continue
            manager.load_state_dict_from_file(tag_directory)

    def save_state_dict_to_file(self, save_directory: str):
        for tag, manager in self.optimization_group.items():
            tag_directory = os.path.join(save_directory, tag)
            manager.save_state_dict_to_file(tag_directory)

    def accelerator_prepare(self, accelerator: "Accelerator", device_placement=True):
        for manager in self.optimization_group.values():
            manager.accelerator_prepare(accelerator, device_placement=device_placement)

    def zero_grad(self, tag: Optional[str] = None):
        if tag is None:
            for manager in self.optimization_group.values():
                manager.optimizer.zero_grad()
        else:
            self.optimization_group[tag].optimizer.zero_grad()

    def optimizer_step(self, tag: Optional[str] = None):
        if tag is None:
            for manager in self.optimization_group.values():
                manager.optimizer.step()
        else:
            self.optimization_group[tag].optimizer.step()

    def lr_scheduler_step(self, tag: Optional[str] = None):
        if tag is None:
            for manager in self.optimization_group.values():
                if manager.lr_scheduler is not None:
                    manager.lr_scheduler.step()
        else:
            if self.optimization_group[tag].lr_scheduler is not None:
                self.optimization_group[tag].lr_scheduler.step()
