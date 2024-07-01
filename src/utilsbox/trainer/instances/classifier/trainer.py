from os import PathLike
import torch

from ...trainer_utils import TrainerMixin
from ....models.classifiers import BaseImageClassifier, BaseImageClassifierOutput


class ClassifierTrainer(TrainerMixin):

    def __init__(self, output_dir: str | PathLike, seed: int = 0) -> None:
        super().__init__(output_dir, seed)
