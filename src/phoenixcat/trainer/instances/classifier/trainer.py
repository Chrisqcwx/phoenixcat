from os import PathLike
from typing import Dict

from accelerate.accelerator import Accelerator
import torch

from ...trainer_utils import (
    TrainerMixin,
    TrainingConfig,
    TrainingDatasetManager,
    register_to_run_one_epoch,
    register_to_run_one_iteration,
    register_trainer,
)
from ....models.classifiers import BaseImageClassifier, BaseImageClassifierOutput

import pytorch_lightning


# @register_trainer
class ClassifierTrainerMixin(TrainerMixin):

    def __init__(
        self,
        output_dir: str | PathLike,
        models: BaseImageClassifier,
        seed=0,
    ) -> None:
        super().__init__(output_dir, seed=0)
