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

from ..configuration import PipelineMixin, config_dataclass_wrapper
from ..random import seed_every_thing

logger = logging.getLogger(__name__)


@config_dataclass_wrapper(config_name='training_config.json')
@dataclass
class TrainingConfig:
    batch_size: int
    test_batch_size: int = None
    max_epoches: int = None
    max_steps: int = None

    def __post_init__(self) -> None:
        if self.test_batch_size is None:
            logger.warning(
                f"`test_batch_size` is None, auto set to `batch_size` ({self.batch_size})."
            )
            self.test_batch_size = self.batch_size
        if (self.max_epoches is None) and (self.max_steps is None):
            logger.warning(
                f"Both `max_epochs` and `max_steps` are None. "
                f"`max_epochs` is automatically set to 10000."
            )
            self.max_epoches = 10000
        elif (self.max_epoches is not None) and (self.max_steps is not None):
            logger.warning(
                f"Both `max_epochs` and `max_steps` are given. "
                f"Training will end when either limit is reached."
            )


@config_dataclass_wrapper(config_name='train_outputfiles.json')
@dataclass
class TrainingOutputFilesManager:
    logging_file: str | os.PathLike = "debug.log"
    version_file: str | os.PathLike = "version.json"
    logging_dir: str | os.PathLike = "logs"
    tensorboard_dir: str | os.PathLike = "tensorboard"
    wandb_dir: str | os.PathLike = "wandb"
    checkpoints_dir: str | os.PathLike = "checkpoints"


@config_dataclass_wrapper(config_name='flags.json')
@dataclass
class TrainingFlag:
    step: int = 0
    epoch: int = 0


class TrainPipelineMixin(PipelineMixin):

    output_files_manager: TrainingOutputFilesManager = TrainingOutputFilesManager()
    flag: TrainingFlag = TrainingFlag()

    ignore_for_pipeline = {'accelerator'}

    def __init__(
        self,
        output_dir: str,
        training_config: TrainingConfig,
        seed: int = 0,
        accelerator: Optional["Accelerator" | Dict] = None,
    ) -> None:
        super().__init__()
        self.register_accelerator(accelerator)
        self.output_dir = output_dir
        self.training_config = training_config
        self._set_seed(seed)

    def reset_flag(self):
        self.flag.epoch = 0
        self.flag.step = 0

    def set_training_config(self, training_config: TrainingConfig):
        self.training_config = training_config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        output_dir = kwargs.pop('output_dir', pretrained_model_name_or_path)
        return super().from_pretrained(
            pretrained_model_name_or_path, output_dir=output_dir, **kwargs
        )

    def _set_seed(self, seed):
        self.seed = seed
        seed_every_thing(seed)
