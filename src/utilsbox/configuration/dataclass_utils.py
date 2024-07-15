import os
import json

from ..files.save import safe_save_as_json


def config_dataclass_wrapper(config_name='config.json'):

    def _inner_wrapper(cls):

        @classmethod
        def from_config(cls, config_or_path: dict | str):
            if not isinstance(config_or_path, dict):
                config_or_path: str

                if not config_or_path.endswith(config_name):
                    config_or_path = os.path.join(config_or_path, config_name)

                with open(config_or_path, 'r') as f:
                    config_or_path = json.load(f)

            config = config_or_path
            return cls(**config)

        def save_config(self, path: str):
            path = os.path.join(path, config_name)
            safe_save_as_json(self.__dict__, path)

        cls.from_config = from_config
        cls.save_config = save_config

        return cls

    return _inner_wrapper
