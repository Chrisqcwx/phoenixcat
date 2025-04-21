# Copyright 2025 Hongyao Yu and Sijin Yu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from typing import get_args, Optional
from types import UnionType
from dataclasses import is_dataclass, dataclass
from itertools import chain

from ..files.save import safe_save_as_json


def dict2dataclass(_data, _class):
    if isinstance(_data, dict):
        fieldtypes = {f.name: f.type for f in _class.__dataclass_fields__.values()}
        kwargs = {}
        for f in fieldtypes:
            value = _data.get(f)
            fieldtype = fieldtypes[f]
            if value is None:
                kwargs[f] = None
            else:
                _success = False

                searched_types = chain([fieldtype], get_args(fieldtype))

                for elem_type in searched_types:
                    try:
                        kwargs[f] = dict2dataclass(value, elem_type)
                        _success = True
                        break
                    except:
                        pass

                if not _success:
                    raise TypeError(f"Cannot convert {value} to {fieldtype}")

        return _class(**kwargs)

    elif isinstance(_data, list):
        if hasattr(_class, '__origin__') and _class.__origin__ == list:
            elem_type = get_args(_class)[0]
            return [dict2dataclass(d, elem_type) for d in _data]
        else:
            raise TypeError("Expected a list type annotation.")
    else:
        return _data


# _auto_save_configs_name = "_auto_save_configs"


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

            # config = config_or_path
            # return cls(**config)
            return dict2dataclass(config_or_path, cls)

        def get_config(self):
            config = {}
            for k, v in self.__dict__.items():
                if is_dataclass(v):
                    config[k] = v.get_config()
                elif isinstance(v, list):
                    config[k] = [i.get_config() if is_dataclass(i) else i for i in v]
                else:
                    config[k] = v
            return config

        def save_config(self, path: str):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, config_name)
            # safe_save_as_json(self.__dict__, path)
            save_info = self.get_config()

            safe_save_as_json(save_info, path)

        cls.from_config = cls.from_pretrained = from_config
        cls.save_config = cls.save_pretrained = save_config
        cls.get_config = get_config

        return cls

    return _inner_wrapper
