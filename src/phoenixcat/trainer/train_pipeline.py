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
import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Literal

import torch


from diffusers.utils import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator

from ..auto import PipelineMixin, config_dataclass_wrapper, only_main_process
from ..random import seed_every_thing
from .optimization import OptimizationManager, SingleOptimizationManager

logger = logging.getLogger(__name__)


def register_evaluate_function(func):

    @functools.wraps(func)
    def wrapper(self: TrainPipelineMixin, *args, **kwargs):
        if self.training is not False:
            self.set_to_eval_mode()
        return func(self, *args, **kwargs)

    return wrapper


def register_train_function(func):

    @functools.wraps(func)
    def wrapper(self: TrainPipelineMixin, *args, **kwargs):
        if self.training is not True:
            self.set_to_train_mode()
        # logger.critical(f'execute {func.__name__} {self.execute_counts}')
        return func(self, *args, **kwargs)

    return wrapper


class TrainPipelineMixin(PipelineMixin):

    optimization_manager: OptimizationManager = OptimizationManager()
    _optimization_save_folder = 'optimization'

    _training = None

    def __init__(
        self,
        output_dir: str,
        seed: int = 0,
        accelerator: Optional["Accelerator"] = None,
    ) -> None:
        super().__init__()
        self.register_accelerator(accelerator)
        self.register_save_values(execute_counts=self.execute_counts)
        self.output_dir = output_dir
        self._set_seed(seed)

        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def logger(self):
        return self._logger

    @property
    def training(self):
        return self._training

    @staticmethod
    def register_training_function(func):
        return register_train_function(func)

    @staticmethod
    def register_evaluate_function(func):
        return register_evaluate_function(func)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        output_dir = kwargs.pop('output_dir', pretrained_model_name_or_path)
        self = super().from_pretrained(
            pretrained_model_name_or_path, output_dir=output_dir, **kwargs
        )
        optimization_save_path = os.path.join(
            self.output_dir, self._optimization_save_folder
        )
        self.optimization_manager.load_state_dict_from_file(optimization_save_path)

        return self

    @only_main_process
    def save_pretrained(self, save_directory: str | os.PathLike):
        optimization_save_path = os.path.join(
            self.output_dir, self._optimization_save_folder
        )
        self.optimization_manager.save_state_dict_to_file(optimization_save_path)
        super().save_pretrained(save_directory)

    def _set_seed(self, seed):
        self.seed = seed
        seed_every_thing(seed)

    def register_optimization(self, tag, params, optimization_config):
        self.optimization_manager.register_optimization(
            tag, params, optimization_config
        )

    # def train(self, training: bool = True):
    #     self._training = training
    #     if training:
    #         self.set_to_train_mode()
    #     else:
    #         self.set_to_eval_mode()

    # def eval(self):
    #     self.train(False)

    @abstractmethod
    def set_to_train_mode(self):
        pass

    @abstractmethod
    def set_to_eval_mode(self):
        pass

    def accelerator_prepare(self, *args, device_placement=None):
        if self.accelerator is None:
            return args

        result = []
        for to_prepare in args:
            if isinstance(to_prepare, (OptimizationManager, SingleOptimizationManager)):
                to_prepare.accelerator_prepare(
                    self.accelerator, device_placement=device_placement
                )
            else:
                to_prepare = self.accelerator.prepare(to_prepare)
            result.append(to_prepare)

        return result

    def backward(self, loss):
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
