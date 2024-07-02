from os import PathLike
from typing import AnyStr

from accelerate.accelerator import Accelerator
import torch

from ...trainer_utils import (
    TrainerMixin,
    TrainModelManager,
    TrainingConfig,
    TrainingDatasetManager,
    register_to_run_one_epoch,
    register_to_run_one_iteration,
    register_trainer,
)
from ....models.classifiers import BaseImageClassifier, BaseImageClassifierOutput


@register_trainer
class ClassifierTrainer(TrainerMixin):

    def __init__(
        self,
        output_dir: str | PathLike,
        models: torch.Dict[AnyStr, TrainModelManager],
        training_config: TrainingConfig,
        dataset_manager: TrainingDatasetManager,
        seed: int = 0,
        accerator: Accelerator | None = None,
    ) -> None:
        super().__init__(
            output_dir, models, training_config, dataset_manager, seed, accerator
        )
