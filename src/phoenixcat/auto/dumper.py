# Copyright 2025 Hongyao Yu.
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
import abc
import copy
from pathlib import Path
from typing import Literal
from ..files import safe_save_as_json, safe_save_as_yaml
from ..decorators import Register
from ..constant import ConfigSuffix

_dumper_register = Register("dumper")


class Dumper:

    def __init__(self, contents=None):
        self._contents = contents

    @property
    def contents(self):
        return self._contents

    @classmethod
    def create(cls, dtype: Literal['dict', 'list'] = 'dict'):
        cls = _dumper_register[dtype]
        if cls is None:
            raise ValueError(f'Invalid dtype: {dtype}')
        return cls()

    @abc.abstractmethod
    def _remove_dumper(self): ...

    def dump(self, dump_path: str | None = None):
        item = self._remove_dumper()
        if dump_path is None:
            if dump_path.endswith(ConfigSuffix.yaml):
                safe_save_as_yaml(item, dump_path)
            else:
                safe_save_as_json(item, dump_path)
        return item


@_dumper_register.register('dict')
class DictDumper(Dumper):

    def __init__(self, contents=None):
        super().__init__(contents=contents)
        if contents is None:
            self._contents = {}

    def update(self, key, content):
        self._contents[key] = content

    def get_subdumper(self, key, dtype: Literal['dict', 'list'] = 'dict'):
        item = self._contents.get(key)
        if item is None:
            item = self._contents[key] = self.create(dtype)
        return item

    def _remove_dumper(self):
        ret = {
            k: v._remove_dumper() if isinstance(v, Dumper) else v
            for k, v in self._contents.items()
        }
        return ret

    # def pop(self, key):
    #     return self._contents.pop(key)


@_dumper_register.register('list')
class ListDumper(Dumper):

    def __init__(self, contents=None):
        super().__init__(contents=contents)
        if contents is None:
            self._contents = []

    def update(self, content):
        self._contents.append(content)

    # def pop(self):
    #     return self._contents.pop()

    def get_subdumper(self, dtype: Literal['dict', 'list'] = 'dict'):
        item = self.create(dtype)
        self._contents.append(item)
        return item

    def _remove_dumper(self):
        ret = [
            v._remove_dumper() if isinstance(v, Dumper) else v for v in self._contents
        ]
        return ret
