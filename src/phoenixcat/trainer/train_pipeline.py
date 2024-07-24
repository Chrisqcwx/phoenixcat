import os
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

import torch
from accelerate import Accelerator

from ..configuration import PipelineMixin, config_dataclass_wrapper
from ..random import seed_every_thing


@config_dataclass_wrapper(config_name='train_outputfiles.json')
@dataclass
class TrainingOutputFilesManager:
    logging_file: str | os.PathLike = "training.log"
    tensorboard_dir: str | os.PathLike = "tensorboard"
    wandb_dir: str | os.PathLike = "wandb"
    checkpoints_dir: str | os.PathLike = "checkpoints"


class TrainPipelineMixin(PipelineMixin):

    output_files_manager: TrainingOutputFilesManager = TrainingOutputFilesManager()

    def __init__(
        self, output_dir: str, seed: int = 0, accelerator: Optional[Accelerator] = None
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.output_dir = output_dir

        seed_every_thing(seed)
