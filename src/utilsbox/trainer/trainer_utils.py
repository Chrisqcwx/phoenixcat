import abc
import json
import datetime
import functools
import logging
import os
import importlib
from dataclasses import dataclass, field
from typing import Dict, AnyStr, Any

import accelerate

# from accelerate.logging import get_logger
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler

from . import constant
from ..logger.logging import init_logger
from ..random._seeds import seed_every_thing
from ..decorators import Register
from ..configuration import ConfigMixin, auto_cls_from_pretrained
from ..models import ModelMixin, auto_model_from_pretrained
from ..conversion import get_obj_from_str
from ..files.save import safe_save_as_json, safe_save_torchobj

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


# @dataclass
class TrainModelManager:
    model: ModelMixin
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    optimizer_config_name = 'optimizer.json'
    optimizer_state_dict_name = 'optimizer.bin'
    lr_scheduler_config_name = 'lr_scheduler.json'
    lr_scheduler_state_dict_name = 'lr_scheduler.bin'

    def __init__(
        self,
        model: ModelMixin,
        optimizer_name: torch.optim.Optimizer,
        optimizer_kwargs: Dict[AnyStr, Any],
        lr_scheduler_name: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Dict[AnyStr, Any] | None = None,
    ) -> None:
        # torch.optim.Adam()
        self.model = model
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        optimizer_cls = get_obj_from_str(f'torch.optim.{optimizer_name}')
        if optimizer_cls is not None:
            raise RuntimeError(f'`optimizer_name` cannot be found in torch.optim')
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

        if lr_scheduler_name is not None:
            lr_scheduler_cls = get_obj_from_str(
                f'torch.optim.lr_scheduler.{lr_scheduler_name}'
            )
            if lr_scheduler_cls is None:
                try:
                    self.lr_scheduler = get_scheduler(
                        lr_scheduler_name, self.optimizer, **lr_scheduler_kwargs
                    )
                except:
                    raise RuntimeError(
                        f'`optimizer_name` cannot be found in torch.optim.lr_scheduler or diffusers.optimization.get_scheduler'
                    )
            else:
                self.lr_scheduler = lr_scheduler_cls(
                    self.optimizer, **lr_scheduler_kwargs
                )

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)
        safe_save_as_json(
            {'name': self.optimizer_name, 'kwargs': self.optimizer_kwargs},
            os.path.join(save_directory, self.optimizer_config_name),
        )
        safe_save_torchobj(self.optimizer.state_dict(), self.optimizer_state_dict_name)
        if self.lr_scheduler is not None:
            safe_save_as_json(
                {'name': self.lr_scheduler_name, 'kwargs': self.lr_scheduler_kwargs},
                os.path.join(save_directory, self.lr_scheduler_config_name),
            )
            safe_save_torchobj(
                self.lr_scheduler.state_dict(), self.lr_scheduler_state_dict_name
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_path, dtype=None, device=None):
        model = auto_model_from_pretrained(pretrained_model_path)
        if dtype is not None:
            model = model.to(dtype)
        if device is not None:
            model = model.to(device)

        with open(os.path.join(pretrained_model_path, cls.optimizer_config_name)) as f:
            optimizer_config = json.load(f)

        lr_scheduler_path = os.path.join(
            pretrained_model_path, cls.lr_scheduler_config_name
        )
        if os.path.exists(lr_scheduler_path):
            with open(lr_scheduler_path) as f:
                lr_scheduler_config = json.load(f)
        else:
            lr_scheduler_config = {'name': None, 'kwargs': None}

        self = cls(
            model=model,
            optimizer_name=optimizer_config['name'],
            optimizer_kwargs=optimizer_config['kwargs'],
            lr_scheduler_name=lr_scheduler_config['name'],
            lr_scheduler_kwargs=lr_scheduler_config['kwargs'],
        )

        self._load_state_dict(
            self.optimizer,
            os.path.join(pretrained_model_path, self.optimizer_state_dict_name),
        )
        if self.lr_scheduler is not None:
            self._load_state_dict(
                self.lr_scheduler,
                os.path.join(pretrained_model_path, self.lr_scheduler_state_dict_name),
            )

    def _load_state_dict(self, dst, path):
        device = next(self.model.parameters()).device
        state_dict = torch.load(path, map_location=device)
        dst.load_state_dict(state_dict)


@dataclass
class TrainTempValues:

    loss: torch.Tensor = 0


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
        self.output_dir = output_dir
        self.register_accelerator(None)
        self.callbacks = []

        self.flag = TrainingFlag()
        self.dataset_manager = TrainingDatasetManager()

        self.training_config = TrainingConfig()

        self.models: Dict[AnyStr, TrainModelManager] = {}

        self.accelerator: accelerate.Accelerator | None = None

        self.train_temp_values = TrainTempValues()

    def _set_seed(self, seed: int):
        self.seed = seed
        seed_every_thing(seed)

    def _set_training_config(self, training_config: Dict):
        self.training_config = TrainingConfig(**training_config)

    def _reset_flag(self, epoch: int = 0, step: int = 0):
        self.flag.epoch = epoch
        self.flag.step = step

    def set_callbacks(self, callbacks):
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

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
    ):
        module_name = os.path.splitext(os.path.basename(config_path))[0]

        spec = importlib.util.spec_from_file_location(module_name, config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        config = {k: v for k, v in module.__dict__.items() if not k.startswith('__')}
        return cls.from_config(config)

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
        if extension in constant.ConfigSuffix.py:
            return cls.from_py_config(config_path)
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

            self.flag.epoch += 1

            for callback in self.callbacks:
                callback.on_train_epoch_begin()

            result = one_epoch_func(*args, **kwargs)

            for callback in self.callbacks:
                callback.on_train_epoch_end()

            if only_training:
                return result

            self.auto_set_to_eval_mode()

            for callback in self.callbacks:
                callback.on_eval_epoch_begin()

            # if self.training_config.checkpointing_epoches is not None:
            #     if self.flag.epoch % self.training_config.checkpointing_epoches == 0:
            #         self._save_checkpoint()

            # if self.training_config.saving_epoches is not None:
            #     if self.flag.epoch % self.training_config.saving_epoches == 0:
            #         self._save_training_status()

            # if self.training_config.validation_epoches is not None:
            #     if self.flag.epoch % self.training_config.validation_epoches == 0:
            #         self._validation()

            # if self.training_config.watching_epoches is not None:
            #     if self.flag.epoch % self.training_config.watching_epoches == 0:
            #         self._watching()

            for callback in self.callbacks:
                callback.on_eval_epoch_end()

            self.auto_set_to_train_mode()

            return result

        return run_one_epoch

    return one_epoch_func_decorator


def register_to_run_one_iteration(only_training: bool = False):

    def one_iteration_func_decorator(one_iteration_func: function):

        @functools.wraps(one_iteration_func)
        def run_one_iteration(self: TrainerMixin, *args, **kwargs):

            self.flag.step += 1

            for callback in self.callbacks:
                callback.on_train_step_begin()

            result = one_iteration_func(*args, **kwargs)

            for callback in self.callbacks:
                callback.on_train_step_end()

            if only_training:
                return result

            self.auto_set_to_eval_mode()

            for callback in self.callbacks:
                callback.on_eval_step_begin()

            # if self.training_config.checkpointing_steps is not None:
            #     if self.flag.step % self.training_config.checkpointing_steps == 0:
            #         self._save_checkpoint()

            # if self.training_config.saving_steps is not None:
            #     if self.flag.step % self.training_config.saving_steps == 0:
            #         self._save_training_status()

            # if self.training_config.validation_steps is not None:
            #     if self.flag.step % self.training_config.validation_steps == 0:
            #         self._validation()

            # if self.training_config.watching_steps is not None:
            #     if self.flag.step % self.training_config.watching_steps == 0:
            #         self._watching()

            for callback in self.callbacks:
                callback.on_eval_step_end()

            self.auto_set_to_train_mode()

            return result

        return run_one_iteration

    return one_iteration_func_decorator


def auto_trainer_from_pretrained(path: str, **kwargs):

    return auto_cls_from_pretrained(_trainer_register, TrainerMixin, path, **kwargs)
