import os
import copy
import shutil
from typing import Dict, Any, Optional

import torchvision

from . import constant
from .conversion import get_obj_from_str
from .files import safe_save_as_json
from .trainer.optimization import OptimizationConfig


class ConfigParser:
    """
    Config parser class
    """

    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None):
        """
        Initialize the config parser

        :param config_file: Path to the config file
        :param kwargs: Additional config parameters
        """
        self.config = config
        self.config_path = config_path
        self._visit_stack = [('/', config)]

    def save_config(self, save_path):
        """
        Save the config to a file

        :param save_path: Path to save the config file
        """

        if self.config_path is not None:
            shutil.copyfile(self.config_path, save_path)
        else:
            safe_save_as_json(self.config, save_path)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the config

        :param key: Key to get the value for
        :param default: Default value to return if the key is not found
        :return: Value from the config
        """
        try:
            return self._visit_stack[-1][1][key]
        except KeyError:
            return default

    def clone(self):
        """
        Clone the config parser

        :return: ConfigParser object
        """
        return ConfigParser(config=self.config)

    def __repr__(self):
        return f"ConfigParser(config={self.config})"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def cd(self, *keys: str, absolute=False):
        """
        Change directory in the config

        :param key: Key to change directory to
        :return: ConfigParser object
        """
        keys = [k for key in keys for k in key.split("/")]
        if absolute:
            self._visit_stack = [('/', self.config)]
        for key in keys:
            self._visit_stack.append((key, self._visit_stack[-1][1][key]))
        return self

    def visit(self, *keys: str, absolute=False):
        """
        Visit a key in the config

        :param key: Key to visit
        :return: ConfigParser object
        """
        keys = [k for key in keys for k in key.split("/")]
        if absolute:
            top = self.config
        else:
            top = self.top
        for key in keys:
            if key is None:
                continue
            top = top[key]
        return top

    @property
    def current_path(self):
        if len(self._visit_stack) == 1:
            return "/"
        return "/".join([key for key, _ in self._visit_stack[1:]])

    @property
    def top(self):
        """
        Get the top level config

        :return: ConfigParser object
        """
        return self._visit_stack[-1][1]

    @classmethod
    def from_yaml_config(cls, config_path: os.PathLike):
        import yaml

        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        return cls(config=config, config_path=config_path)

    @classmethod
    def from_json_config(cls, config_path: os.PathLike):
        import json

        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        return cls(config=config, config_path=config_path)

    @classmethod
    def from_ini_config(cls, config_path: os.PathLike):
        import configparser

        config = configparser.ConfigParser()
        config.read(config_path)
        config = {section: dict(config.items(section)) for section in config.sections()}
        return cls(config=config, config_path=config_path)

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

    def create_accelerator(self, subfolder="accelerator", absolute=False):
        """
        Create an accelerator config

        :param subfolder: Subfolder to create accelerator config in
        :return: accelerate.Accelerator object
        """
        import accelerate

        if subfolder not in self.config:
            return None

        config = copy.copy(self.visit(subfolder, absolute=absolute))
        if config is None:
            config = {}

        if 'dataloader_config' in config:
            dataloader_config = config['dataloader_config']
            dataloader_config = accelerate.DataLoaderConfiguration(**dataloader_config)
            config['dataloader_config'] = dataloader_config
        accelerator = accelerate.Accelerator(**config)
        return accelerator

    def create_from_name(self, name, kwargs: Dict):
        if kwargs is None:
            kwargs = {}

        builder = get_obj_from_str(name)
        return builder(**kwargs)

    def _create_transform_impl(self, name, kwargs: Dict):
        # return self.create_from_name(f'torchvision.transforms.{name}', kwargs)
        if name == "Compose":
            components = []
            for component, component_kwargs in kwargs.items():
                components.append(
                    self._create_transform_impl(component, component_kwargs)
                )
            return torchvision.transforms.Compose(components)
        else:
            return self.create_from_name(f'torchvision.transforms.{name}', kwargs)

    def create_transform(self, subfolder="transform", absolute=False):
        """
        Create a transform config

        :param subfolder: Subfolder to create transform config in
        :return: torchvision.transforms.Compose object
        """
        config = self.visit(subfolder, absolute=absolute)
        if config is None:
            return None
        return self._create_transform_impl("Compose", config)

    def create_model(self, subfolder="model", absolute=False):
        """
        Create a model config

        :param subfolder: Subfolder to create model config in
        :return: torch.nn.Module object
        """
        config = self.visit(subfolder, absolute=absolute)

        from .models import auto_model_from_pretrained, get_model_builder

        if "pretrained" in config:
            return auto_model_from_pretrained(config["pretrained"])

        builder = get_model_builder(config["name"])

        return builder(**config["kwargs"])

    def create_optimization_config(self, subfolder="optimization", absolute=False):
        """
        Create an optimization config

        :param subfolder: Subfolder to create optimization config in
        :return: torch.optim.Optimizer object
        """
        config = self.visit(subfolder, absolute=absolute)
        return OptimizationConfig(
            optimizer_name=config["optimizer"],
            optimizer_params=config["optimizer_params"],
            lr_scheduler_name=config["lr_scheduler"],
            lr_scheduler_params=config["lr_scheduler_params"],
        )
