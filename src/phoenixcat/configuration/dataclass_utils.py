# Copyright 2024 Hongyao Yu and Sijin Yu.
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
from typing import get_args, Dict
from dataclasses import is_dataclass, dataclass

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

        cls.from_config = cls.from_pretrained = from_config
        cls.save_config = cls.save_pretrained = save_config

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
