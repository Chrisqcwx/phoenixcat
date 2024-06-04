import abc
import datetime
import functools
import logging
import os
from typing import Dict

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.utils.tensorboard import SummaryWriter

from . import constant
from ..check.check_args import only_one_given
from ..logger.logging import init_logger
from ..random._seeds import seed_every_thing

logger = logging.getLogger(__name__)

class TrainingConfig:
    def __init__(
        self,
        max_epoches: int = None,
        max_steps: int = None,
        checkpointing_epoches: int = None,
        checkpointing_steps: int = None,
        validation_epoches: int = None,
        validation_steps: int = None,
        saving_epoches: int = None,
        saving_steps: int = None,
        watching_epoches: int = None,
        watching_steps: int = None
    ) -> None:
        self.max_epoches = max_epoches
        self.max_steps = max_steps
        if (max_epoches is None) and (max_steps is None):
            logger.warning(f"Both `max_epochs` and `max_steps` are None. "
                           f"`max_epochs` is automatically set to 10000.")
            self.max_epoches = 10000
        elif (max_epoches is not None) and (max_steps is not None):
            logger.warning(f"Both `max_epochs` and `max_first` are given. "
                           f"Training will end when either limit is reached.")
        
        self.checkpointing_epoches = checkpointing_epoches
        self.checkpointing_steps = checkpointing_steps
        if (checkpointing_epoches is None) and (checkpointing_steps is None):
            logger.warning(f"Both `checkpointing_epochs` and `checkpointing_steps` are None. "
                           f"No checkpoints will be saved during the training.")
        elif (checkpointing_epoches is not None) and (checkpointing_steps is not None):
            logger.warning(f"Both `checkpointing_epochs` and `checkpointing_steps` are given. "
                           f"All checkpoints meeting the criteria will be saved.")
        
        self.validation_epoches = validation_epoches
        self.validation_steps = validation_steps
        if (validation_epoches is None) and (validation_steps is None):
            logger.warning(f"Both `validation_epochs` and `validation_steps` are None. "
                           f"No validation will be performed during the training.")
        elif (validation_epoches is not None) and (validation_steps is not None):
            logger.warning(f"Both `validation_epochs` and `validation_steps` are given. "
                           f"All validation meeting the criteria will be performed.")
        
        self.saving_epoches = saving_epoches
        self.saving_steps = saving_steps
        if (saving_epoches is None) and (saving_steps is None):
            logger.warning(f"Both `saving_epochs` and `saving_steps` are None. "
                           f"No states will be saved during the training.")
        elif (saving_epoches is not None) and (saving_steps) is not None:
            logger.warning(f"Both `saving_epochs` and `saving_steps` are given. "
                           f"All states meeting the criteria will be saved.")
        
        self.watching_epoches = watching_epoches
        self.watching_steps = watching_steps
        if (watching_epoches is None) and (watching_steps is None):
            logger.warning(f"Both `saving_epochs` and `saving_steps` are None. "
                           f"No variables will be saved during the training.")
        elif (watching_epoches is not None) and (watching_steps is not None):
            logger.warning(f"Both `saving_epochs` and `saving_steps` are given. "
                           f"all variables meeting the criteria will be saved.")
    

# TODO: 请测试
class TrainerMixin(abc.ABC, ConfigMixin):
    
    config_name = "config.json"
    logger_name = "training.log"
    tensorboard_subfolder = "tensorboard"
    checkpoints_subfolder = "checkpoints"
    last_training_status_subfolder = "last"
    
    ignore_for_config = ["reload"]
    
    @register_to_config
    def __init__(
        self,
        output_dir_config: dict,
        training_config: dict,
        seed: int,
        *,
        _reload: bool = False,
        _output_path: os.PathLike = None,
    ) -> None:
        super().__init__()
        self._set_output_dir(**output_dir_config, _reload=_reload, _output_path=_output_path)
        self._set_seed(seed)
        self._set_training_config(**training_config)
        
        self.current_epoch = 0
        self.current_step = 0
        self.training_dataset = None
        self.training_dataloader = None
        
    
    def _set_output_dir(
        self,
        root: os.PathLike = ".outputs",
        name: os.PathLike = None,
        use_timestamp: bool = True,
        logger_config: Dict[str, str] = None,
        *,
        _reload: bool = False,
        _output_path: os.PathLike = None
    ):
        if _reload:
            self.output_dir = _output_path
        else:
            if name is None:
                path = root
            else:
                path = os.path.join(root, name)
            
            if use_timestamp:
                now = datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S,%f")[:-3]
                path = os.path.join(path, now)
            
            os.makedirs(path, exist_ok=True)
            self.output_dir = path
        
        init_logger(
            file_name=os.path.join(self.output_dir, self.logger_name),
            **logger_config
        )
        
        os.makedirs(os.path.join(self.output_dir, self.tensorboard_subfolder), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.output_dir, self.tensorboard_subfolder))
        
        self.save_config(save_directory=self.output_dir, push_to_hub=False)
    
    
    def _set_seed(
        self,
        seed: int
    ):
        self.seed = seed
        seed_every_thing(seed)
        
    
    def _set_training_config(
        self,
        training_config: Dict
    ):
        self.training_config = TrainingConfig(**training_config)
    
    @abc.abstractmethod
    def auto_init(self, reload: bool = False):
        raise NotImplementedError("Please implement the `auto_init` method.")
    
    @abc.abstractmethod
    def auto_set_to_train_mode(self):
        raise NotImplementedError("Please implement the `auto_set_to_train_mode` method.")
    
    @abc.abstractmethod
    def auto_set_to_eval_mode(self):
        raise NotImplementedError("Please implement the `auto_set_to_eval_mode` method.")
    
    @abc.abstractmethod
    def auto_save_checkpoint(self):
        raise NotImplementedError("Please implement the `auto_save_checkpoint` method.")
    
    @abc.abstractmethod
    def auto_save_training_status(self):
        raise NotImplementedError("Please implement the `auto_save_training_status` method.")
    
    @abc.abstractmethod
    def auto_validation(self):
        raise NotImplementedError("Please implement the `auto_validation` method.")
    
    @abc.abstractmethod
    def auto_watching(self):
        raise NotImplementedError("Please implement the `auto_watching` method.")
    
    @abc.abstractmethod
    def auto_load_status(self, checkpoint_path):
        raise NotImplementedError("Please implement the `auto_load_status` method.")
    
    @classmethod
    def from_yaml_config(cls, config_path: os.PathLike, _reload: bool = False):
        import yaml
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            if _reload:
                config["_reload"] = _reload
                config["_output_path"] = os.path.dirname(config_path)
        return cls.from_config(config)
    
    @classmethod
    def from_json_config(cls, config_path: os.PathLike, _reload: bool = False):
        import json
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            if _reload:
                config["_reload"] = _reload
                config["_output_path"] = os.path.dirname(config_path)
        return cls.from_config(config)
    
    @classmethod
    def from_ini_config(cls, config_path: os.PathLike, _reload: bool = False):
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)
        config = {section: dict(config.items(section)) for section in config.sections()}
        if _reload:
                config["_reload"] = _reload
                config["_output_path"] = os.path.dirname(config_path)
        return cls.from_config(config)
    
    @classmethod
    def from_py_config(
        cls, 
        config_path: os.PathLike = None, 
        config_module: str = None,
        _reload: bool = False
    ):
        # TODO: 完善从 py 文件读取配置文件的功能.
        # TODO: 太恶心了 我写不动 谁爱用这个功能谁写.
        raise NotImplementedError
        if (config_path is None) and (config_module is None):
            raise ValueError("Cannot init form config with both `config_path` and `config_module` are None.")
        if (config_path is not None) and (config_module is not None):
            logger.warning(f"Both `config_path` and `config_module` are given. "
                           f"`config_path` is used by default.")
        if config_path is not None:
            pass
    
    @classmethod
    def from_config_file(cls, config_path: os.PathLike, *, _reload: bool = False):
        from pathlib import Path
        extension = Path(config_path).suffix.lower()
        if extension in constant.ConfigSuffix.json:
            return cls.from_json_config(config_path, _reload=_reload)
        if extension in constant.ConfigSuffix.yaml:
            return cls.from_yaml_config(config_path, _reload=_reload)
        if extension in constant.ConfigSuffix.ini:
            return cls.from_ini_config(config_path, _reload=_reload)
        raise NotImplementedError(f"Unknown suffix '{extension}' in path '{config_path}'.")
            
    @classmethod
    def from_output_dir(cls, dir_path: os.PathLike):
        config_path = os.path.join(dir_path, cls.config_name)
        self = cls.from_config_file(config_path, _reload=True)
        self.auto_load_status()
        
        logger.info(f"Warm up for epoch={self.current_epoch} and step={self.current_step}.")
        if self.current_step == 0 and self.current_epoch == 0:
            return self
        _step = 0
        _epoch = 0
        while True:
            for _ in self.training_dataloader:
                _step += 1
                if _step == self.current_step:
                    if _epoch == self.current_epoch:
                        return self
                    else:
                        raise RuntimeError("`current_epoch` and `current_step` mismatch.")
            _epoch += 1
            

def register_to_run_one_epoch(one_epoch_func, only_training: bool = False):
    
    @functools.wraps(one_epoch_func)
    def run_one_epoch(self: TrainerMixin, *args, **kwargs):
        result = one_epoch_func(*args, **kwargs)
        self.current_epoch += 1
        
        if only_training:
            return result
        
        self.auto_set_to_eval_mode()
        
        with torch.no_grad():
            if self.training_config.checkpointing_epoches is not None:
                if self.current_epoch % self.training_config.checkpointing_epoches == 0:
                    self.auto_save_checkpoint()
            
            if self.training_config.saving_epoches is not None:
                if self.current_epoch % self.training_config.saving_epoches == 0:
                    self.auto_save_training_status()
            
            if self.training_config.validation_epoches is not None:
                if self.current_epoch % self.training_config.validation_epoches == 0:
                    self.auto_validation()
                
            if self.training_config.watching_epoches is not None:
                if self.current_epoch % self.training_config.watching_epoches == 0:
                    self.auto_watching()
        
        self.auto_set_to_train_mode()
        
        return result
    
    return run_one_epoch


def register_to_run_one_iteration(one_iteration_func, only_training: bool = False):
    
    @functools.wraps(one_iteration_func)
    def run_one_iteration(self: TrainerMixin, *args, **kwargs):
        result = one_iteration_func(*args, **kwargs)
        self.current_step += 1
        
        if only_training:
            return result
        
        self.auto_set_to_eval_mode()
        
        with torch.no_grad():
            if self.training_config.checkpointing_steps is not None:
                if self.current_step % self.training_config.checkpointing_steps == 0:
                    self.auto_save_checkpoint()
            
            if self.training_config.saving_steps is not None:
                if self.current_step % self.training_config.saving_steps == 0:
                    self.auto_save_training_status()
            
            if self.training_config.validation_steps is not None:
                if self.current_step % self.training_config.validation_steps == 0:
                    self.auto_validation()
                
            if self.training_config.watching_steps is not None:
                if self.current_step % self.training_config.watching_steps == 0:
                    self.auto_watching()
        
        self.auto_set_to_train_mode()
        
        return result
    
    return run_one_iteration