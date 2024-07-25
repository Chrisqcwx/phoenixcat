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
from typing import Optional, Union, List, Dict, Any

import torch

from diffusers.utils import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator

from .data import DataManagerGroup, DataManager
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
    checkpointing_epoches: int = None
    checkpointing_steps: int = None
    validation_epoches: int = None
    validation_steps: int = None
    saving_epoches: int = None
    saving_steps: int = None
    watching_epoches: int = None
    watching_steps: int = None

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
        if (self.checkpointing_epoches is None) and (self.checkpointing_steps is None):
            logger.warning(
                f"Both `checkpointing_epochs` and `checkpointing_steps` are None. "
                f"No checkpoints will be saved during the training."
            )
        elif (self.checkpointing_epoches is not None) and (
            self.checkpointing_steps is not None
        ):
            logger.warning(
                f"Both `checkpointing_epochs` and `checkpointing_steps` are given. "
                f"All checkpoints meeting the criteria will be saved."
            )
        if (self.validation_epoches is None) and (self.validation_steps is None):
            logger.warning(
                f"Both `validation_epochs` and `validation_steps` are None. "
                f"No validation will be performed during the training."
            )
        elif (self.validation_epoches is not None) and (
            self.validation_steps is not None
        ):
            logger.warning(
                f"Both `validation_epochs` and `validation_steps` are given. "
                f"All validation meeting the criteria will be performed."
            )
        if (self.saving_epoches is None) and (self.saving_steps is None):
            logger.warning(
                f"Both `saving_epochs` and `saving_steps` are None. "
                f"No states will be saved during the training."
            )
        elif (self.saving_epoches is not None) and (self.saving_steps) is not None:
            logger.warning(
                f"Both `saving_epochs` and `saving_steps` are given. "
                f"All states meeting the criteria will be saved."
            )
        if (self.watching_epoches is None) and (self.watching_steps is None):
            logger.warning(
                f"Both `saving_epochs` and `saving_steps` are None. "
                f"No variables will be saved during the training."
            )
        elif (self.watching_epoches is not None) and (self.watching_steps is not None):
            logger.warning(
                f"Both `saving_epochs` and `saving_steps` are given. "
                f"all variables meeting the criteria will be saved."
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

    dataset_group: DataManagerGroup

    def __init__(
        self, output_dir: str, seed: int = 0, accelerator: Optional[Accelerator] = None
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.output_dir = output_dir
        self._set_seed(seed)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        output_dir = kwargs.pop('output_dir', pretrained_model_name_or_path)
        return super().from_pretrained(
            pretrained_model_name_or_path, output_dir=output_dir, **kwargs
        )

    def _set_seed(self, seed):
        self.seed = seed
        seed_every_thing(seed)

    def register_dataset(self, name: str, dataset_manager: DataManager):
        self.dataset_group[name] = dataset_manager
