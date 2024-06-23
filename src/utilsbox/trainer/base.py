import abc
import datetime
import functools
import logging
import os
from dataclasses import dataclass
from typing import Dict

import accelerate

# from accelerate.logging import get_logger
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from . import constant, context
from ..logger.logging import init_logger
from ..random._seeds import seed_every_thing
from ..decorators import Register
from ..configuration import ConfigMixin, auto_cls_from_pretrained

logger = logging.getLogger(__name__)


_trainer_register = Register('trainer')

register_trainer = _trainer_register.register()


def list_trainers():
    return list(_trainer_register.keys())


def get_trainer_builder(name: str):
    return _trainer_register[name]


@dataclass
class TrainingOutputFilesManager:
    logging_file: str | os.PathLike = "training.log"
    tensorboard_dir: str | os.PathLike = "tensorboard"
    wandb_dir: str | os.PathLike = "wandb"
    checkpoints_dir: str | os.PathLike = "checkpoints"


class TrainerMixin(abc.ABC, ConfigMixin):

    config_name = "config.json"
    output_files_manager = TrainingOutputFilesManager()

    def __init__(
        self,
        output_dir: str | os.PathLike,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._set_seed(seed)
        # self.output_dir = output_dir
        # self.flag = TrainingFlag()
        # self.dataset_manager = TrainingDatasetManager()
        self.context = context.TrainerContext(output_dir=output_dir)
        self.register_accelerator(None)

    def _set_seed(self, seed: int):
        self.seed = seed
        seed_every_thing(seed)

    def _set_training_config(self, training_config: Dict):
        self.training_config = context.TrainingConfig(**training_config)

    def _reset_flag(self, epoch: int = 0, step: int = 0):
        self.context.flag.epoch = epoch
        self.context.flag.step = step

    def register_accelerator(self, accelerator: accelerate.Accelerator):
        self.accelerator = accelerator

    @property
    def is_local_main_process(self) -> bool:
        if self.accelerator is None:
            return True
        return self.accelerator.is_local_main_process

    def wait_for_everyone(self):
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    @abc.abstractmethod
    def auto_init(self, reload: bool = False):
        raise NotImplementedError("Please implement the `auto_init` method.")

    @abc.abstractmethod
    def auto_set_to_train_mode(self):
        raise NotImplementedError(
            "Please implement the `auto_set_to_train_mode` method."
        )

    @abc.abstractmethod
    def auto_set_to_eval_mode(self):
        raise NotImplementedError(
            "Please implement the `auto_set_to_eval_mode` method."
        )

    @abc.abstractmethod
    def auto_save_checkpoint(self):
        raise NotImplementedError("Please implement the `auto_save_checkpoint` method.")

    @abc.abstractmethod
    def auto_save_training_status(self):
        raise NotImplementedError(
            "Please implement the `auto_save_training_status` method."
        )

    @abc.abstractmethod
    def auto_validation(self):
        raise NotImplementedError("Please implement the `auto_validation` method.")

    @abc.abstractmethod
    def auto_watching(self):
        raise NotImplementedError("Please implement the `auto_watching` method.")

    @abc.abstractmethod
    def auto_load_status(self, checkpoint_path):
        raise NotImplementedError("Please implement the `auto_load_status` method.")

    @torch.no_grad()
    def _save_checkpoint(self):
        if self.is_local_main_process:
            self.wait_for_everyone()
            self.auto_save_checkpoint()

    @torch.no_grad()
    def _save_training_status(self):
        if self.is_local_main_process:
            self.wait_for_everyone()
            self.auto_save_training_status()

    @torch.no_grad()
    def _validation(self):
        self.wait_for_everyone()
        self.auto_validation()

    @torch.no_grad()
    def _watching(self):
        if self.is_local_main_process:
            self.wait_for_everyone()
            self.auto_watching()

    @classmethod
    def from_yaml_config(cls, config_path: os.PathLike):
        import yaml

        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        return cls.from_config(config)

    @classmethod
    def from_json_config(cls, config_path: os.PathLike):
        import json

        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        return cls.from_config(config)

    @classmethod
    def from_ini_config(cls, config_path: os.PathLike):
        import configparser

        config = configparser.ConfigParser()
        config.read(config_path)
        config = {section: dict(config.items(section)) for section in config.sections()}
        return cls.from_config(config)

    @classmethod
    def from_py_config(
        cls,
        config_path: os.PathLike = None,
        config_module: str = None,
    ):
        # TODO: 完善从 py 文件读取配置文件的功能.
        # TODO: 太恶心了 我写不动 谁爱用这个功能谁写.
        raise NotImplementedError
        if (config_path is None) and (config_module is None):
            raise ValueError(
                "Cannot init form config with both `config_path` and `config_module` are None."
            )
        if (config_path is not None) and (config_module is not None):
            logger.warning(
                f"Both `config_path` and `config_module` are given. "
                f"`config_path` is used by default."
            )
        if config_path is not None:
            pass

    @classmethod
    def from_config_file(cls, config_path: os.PathLike):
        from pathlib import Path

        extension = Path(config_path).suffix.lower()
        if extension in constant.ConfigSuffix.json:
            return cls.from_json_config(config_path)
        if extension in constant.ConfigSuffix.yaml:
            return cls.from_yaml_config(config_path)
        if extension in constant.ConfigSuffix.ini:
            return cls.from_ini_config(config_path)
        raise NotImplementedError(
            f"Unknown suffix '{extension}' in path '{config_path}'."
        )

    @classmethod
    def from_output_dir(cls, dir_path: os.PathLike):
        config_path = os.path.join(dir_path, cls.config_name)
        self = cls.from_config_file(config_path)
        self.auto_load_status()

        logger.info(f"Warm up for epoch={self.flag.epoch} and step={self.flag.step}.")
        if self.flag.step == 0 and self.flag.epoch == 0:
            return self
        # _step = 0
        # _epoch = 0
        # while True:
        #     for _ in self.dataset_manager.training_dataloader:
        #         _step += 1
        #         if _step == self.flag.step:
        #             if _epoch == self.flag.epoch:
        #                 return self
        #             else:
        #                 raise RuntimeError(
        #                     "`current_epoch` and `current_step` mismatch."
        #                 )
        #     _epoch += 1

        expect_epoch = self.flag.step // len(self.dataset_manager.training_dataloader)
        if expect_epoch != self.flag.epoch:
            raise RuntimeError("`current_epoch` and `current_step` mismatch.")

        return self


def register_to_run_one_epoch(only_training: bool = False):

    def one_epoch_func_decorator(one_epoch_func: function):

        @functools.wraps(one_epoch_func)
        def run_one_epoch(self: TrainerMixin, *args, **kwargs):
            result = one_epoch_func(*args, **kwargs)
            self.flag.epoch += 1

            if only_training:
                return result

            self.auto_set_to_eval_mode()

            if self.training_config.checkpointing_epoches is not None:
                if self.flag.epoch % self.training_config.checkpointing_epoches == 0:
                    self._save_checkpoint()

            if self.training_config.saving_epoches is not None:
                if self.flag.epoch % self.training_config.saving_epoches == 0:
                    self._save_training_status()

            if self.training_config.validation_epoches is not None:
                if self.flag.epoch % self.training_config.validation_epoches == 0:
                    self._validation()

            if self.training_config.watching_epoches is not None:
                if self.flag.epoch % self.training_config.watching_epoches == 0:
                    self._watching()

            self.auto_set_to_train_mode()

            return result

        return run_one_epoch

    return one_epoch_func_decorator


def register_to_run_one_iteration(only_training: bool = False):

    def one_iteration_func_decorator(one_iteration_func: function):

        @functools.wraps(one_iteration_func)
        def run_one_iteration(self: TrainerMixin, *args, **kwargs):
            result = one_iteration_func(*args, **kwargs)
            self.flag.step += 1

            if only_training:
                return result

            self.auto_set_to_eval_mode()

            if self.training_config.checkpointing_steps is not None:
                if self.flag.step % self.training_config.checkpointing_steps == 0:
                    self._save_checkpoint()

            if self.training_config.saving_steps is not None:
                if self.flag.step % self.training_config.saving_steps == 0:
                    self._save_training_status()

            if self.training_config.validation_steps is not None:
                if self.flag.step % self.training_config.validation_steps == 0:
                    self._validation()

            if self.training_config.watching_steps is not None:
                if self.flag.step % self.training_config.watching_steps == 0:
                    self._watching()

            self.auto_set_to_train_mode()

            return result

        return run_one_iteration

    return one_iteration_func_decorator


def auto_trainer_from_pretrained(path: str, **kwargs):

    return auto_cls_from_pretrained(_trainer_register, TrainerMixin, path, **kwargs)
