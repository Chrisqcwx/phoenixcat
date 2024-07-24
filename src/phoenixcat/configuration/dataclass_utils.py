import os
import json
from typing import get_args, Dict

from ..files.save import safe_save_as_json
from .pipeline_utils import pipeline_loadable


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

        cls.from_config = cls.from_pretrained = from_config
        cls.save_config = cls.save_pretrained = save_config

        cls = pipeline_loadable()(cls)

        return cls

    return _inner_wrapper


def dict2dataclass(_data, _class):
    if isinstance(_data, dict):
        fieldtypes = {f.name: f.type for f in _class.__dataclass_fields__.values()}
        return _class(
            **{f: dict2dataclass(_data.get(f), fieldtypes[f]) for f in fieldtypes}
        )
    elif isinstance(_data, list):
        if hasattr(_class, '__origin__') and _class.__origin__ == list:
            elem_type = get_args(_class)[0]
            return [dict2dataclass(d, elem_type) for d in _data]
        else:
            raise TypeError("Expected a list type annotation.")
    else:
        return _data
