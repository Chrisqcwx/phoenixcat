from os import PathLike
import torch
from diffusers.configuration_utils import register_to_config

from .classifier_utils import BaseImageClassifier, BaseImageClassifierOutput
from ...trainer_utils import TrainerMixin


class ClassifierTrainer(TrainerMixin):

    # @register_to_config
    def __init__(self, output_dir: str | PathLike, seed: int = 0) -> None:
        super().__init__(output_dir, seed)

    # def
