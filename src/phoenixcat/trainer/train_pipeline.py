# from typing import Optional, Union, List, Dict, Any

# import torch
# from accelerate import Accelerator

# from .pipeline_utils import PipelineMixin, register_to_pipeline_init
# from ..random import seed_every_thing


# @config_dataclass_wrapper(config_name='train_outputfiles.json')
# @dataclass
# class TrainingOutputFilesManager:
#     logging_file: str | os.PathLike = "training.log"
#     tensorboard_dir: str | os.PathLike = "tensorboard"
#     wandb_dir: str | os.PathLike = "wandb"
#     checkpoints_dir: str | os.PathLike = "checkpoints"


# class TrainPipelineMixin(PipelineMixin):

#     def __init__(
#         self, output_dir: str, seed: int = 0, accelerator: Optional[Accelerator] = None
#     ) -> None:
#         super().__init__()
#         self.accelerator = accelerator
#         self.seed = seed
#         self.output_dir = output_dir

#     # def
