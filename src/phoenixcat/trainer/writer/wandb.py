import os

import torch
import wandb

from .base import WriterMixin


class WandbWriter(WriterMixin):
    def __init__(self, project: str, name: str, dir: os.PathLike, **kwargs) -> None:
        super().__init__(project, name, dir)
        wandb.init(project=project, name=name, dir=dir, **kwargs)

    def log_config(self, config: dict):
        wandb.config.update(config)

    def log_dict(self, _dict: dict):
        wandb.log(_dict)

    def watch_model(self, model: torch.nn.Module, **kwargs):
        wandb.watch(model, **kwargs)
