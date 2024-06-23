import os
import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


class TrainingConfig:
    def __init__(
        self,
        max_epoches: int = None,
        max_steps: int = None,
    ) -> None:
        self.max_epoches = max_epoches
        self.max_steps = max_steps
        if (max_epoches is None) and (max_steps is None):
            logger.warning(
                f"Both `max_epochs` and `max_steps` are None. "
                f"`max_epochs` is automatically set to 10000."
            )
            self.max_epoches = 10000
        elif (max_epoches is not None) and (max_steps is not None):
            logger.warning(
                f"Both `max_epochs` and `max_first` are given. "
                f"Training will end when either limit is reached."
            )


@dataclass
class TrainingFlag:
    step: int = 0
    epoch: int = 0


@dataclass
class TrainingDatasetManager:
    training_dataset: torch.utils.data.Dataset = None
    validation_dataset: torch.utils.data.Dataset = None
    training_dataloader: torch.utils.data.DataLoader = None
    validation_dataloader: torch.utils.data.DataLoader = None


@dataclass
class TrainerContext:

    output_dir: str
    output_files_manager: TrainingDatasetManager = field(
        default_factory=TrainingDatasetManager
    )

    flag: TrainingFlag = field(default_factory=TrainingFlag)
    dataset_manager: TrainingDatasetManager = field(
        default_factory=TrainingDatasetManager
    )

    training_config: TrainingConfig = field(default_factory=TrainingConfig)
