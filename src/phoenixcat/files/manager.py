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
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Literal
from contextlib import contextmanager
from functools import wraps

from .load import load_json, load_yaml
from .save import safe_save_as_json, safe_save_as_yaml
from .walk import walk_dict


class CacheManager:

    cache_info_filename = 'cache.json'
    files_dirname = 'files'

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        # self.root_dir.mkdir(exist_ok=True)
        os.makedirs(self.root_dir, exist_ok=True)
        self.cache_info_file = self.root_dir / self.cache_info_filename

        self._cache_info = None

    @property
    def cache_info(self):
        if self._cache_info is None:
            if not self.cache_info_file.exists():
                with open(self.cache_info_file, 'r') as f:
                    self._cache_info = json.load(f)
            else:
                self._cache_info = {}
        return self._cache_info

    def get_cache_dir(self, name):
        cnt = len(self.cache_info)
        if name in self.cache_info:
            return self.cache_info[name]
        else:
            cache_dir = self.root_dir / self.files_dirname / f'{cnt}'
            # cache_dir.mkdir(exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_info[name] = cache_dir
            self.dump_cache_info()
            return cache_dir

    def dump_cache_info(self):
        with open(self.cache_info_file, 'w') as f:
            json.dump(self.cache_info, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dump_cache_info()


class FolderManager:

    def __init__(self, root, read_only=True):
        self.root = Path(root)
        self.read_only = read_only

        if not self.read_only:
            # self.root.mkdir(exist_ok=True)
            os.makedirs(self.root, exist_ok=True)

        self.ptr = self.root.resolve()

    def get_target_path(self, path: str = None, root=False):
        ptr = self.root if root else self.ptr
        if path is not None:
            ptr = ptr / path
        return ptr.resolve()

    def get_related_to_root(self):
        return self.ptr.relative_to(self.root)

    def cd(self, path: str, root=False):
        self.ptr = self.get_target_path(path, root=root)
        return self.ptr

    def ls(self, path: str = None):
        ptr = self.get_target_path(path)
        return os.listdir(str(ptr))

    def makedirs(self, path: str = None):
        if self.read_only:
            raise PermissionError('Read only mode')
        ptr = self.get_target_path(path)
        # ptr.mkdir(exist_ok=True)
        os.makedirs(ptr, exist_ok=True)
        return ptr

    def rm(self, path: str = None):
        if self.read_only:
            raise PermissionError('Read only mode')
        ptr = self.get_target_path(path)
        shutil.rmtree(ptr)

    def open(self, path: str = None, mode: str = 'r', open_kwargs=None, root=False):
        is_read = mode.startswith('r')

        if self.read_only and not is_read:
            raise PermissionError('Read only mode')

        ptr = self.get_target_path(path, root=root)
        if is_read and not ptr.exists():
            raise FileNotFoundError(f'{ptr} not found')

        if ptr.is_dir():
            raise IsADirectoryError(f'{ptr} is a directory')

        # ptr.parent.mkdir(exist_ok=True)
        os.makedirs(ptr.parent, exist_ok=True)

        if open_kwargs is None:
            open_kwargs = {}
        return ptr.open(mode, **open_kwargs)

    def pwd(self, related=False):
        if related:
            return self.get_related_to_root()
        else:
            return self.ptr

    def is_file(self, path: str = None):
        return self.get_target_path(path).is_file()

    def is_dir(self, path: str = None):
        return self.get_target_path(path).is_dir()

    def parent(self):
        self.ptr = self.ptr.parent
        return self.ptr

    def __str__(self):
        return str(self.ptr)


class DualFolderManager:

    def __init__(self, read_root, write_root):
        self.read_manager = FolderManager(read_root, read_only=True)
        self.write_manager = FolderManager(write_root, read_only=False)

    def get_target_path(self, path: str = None, root=False):
        return (
            self.read_manager.get_target_path(path, root=root),
            self.write_manager.get_target_path(path, root=root),
        )

    def cd(self, path: str, root=False):
        self.read_manager.cd(path, root=root)
        self.write_manager.cd(path, root=root)

    def ls(self, path: str = None):
        return self.read_manager.ls(path)

    def is_file(self, path: str = None):
        return self.read_manager.is_file(path)

    def is_dir(self, path: str = None):
        return self.read_manager.is_dir(path)

    @contextmanager
    def open(
        self,
        path: str = None,
        read_mode: str = 'r',
        write_mode: str = 'w',
        open_kwargs=None,
        root: bool = False,
        write_extension: str = None,
    ):
        read_file = self.read_manager.open(
            path, mode=read_mode, open_kwargs=open_kwargs, root=root
        )

        if write_extension is not None:
            if not write_extension.startswith('.'):
                write_extension = '.' + write_extension
            path = str(path).rsplit('.', 1)[0] + write_extension
        write_file = self.write_manager.open(
            path, mode=write_mode, open_kwargs=open_kwargs, root=root
        )
        try:
            yield read_file, write_file
        finally:
            read_file.close()
            write_file.close()

    def copy(self, path: str = None, root=False):
        src_file = self.read_manager.get_target_path(path, root=root)
        dst_file = self.write_manager.get_target_path(path, root=root)
        # dst_file.parent.mkdir(exist_ok=True)
        os.makedirs(dst_file.parent, exist_ok=True)
        if src_file.is_dir():
            shutil.copytree(src_file, dst_file)
        else:
            shutil.copyfile(src_file, dst_file)

        return src_file, dst_file

    def pwd(self, related=False):
        return self.read_manager.pwd(related=related)

    def parent(self):
        self.read_manager.parent()
        self.write_manager.parent()


class RecordManager:

    def __init__(
        self,
        data: Dict = None,
        file: str | Path = None,
        file_format: Literal['json', 'yaml'] = 'json',
    ):
        if file:
            file = str(file)
        self.file = file
        self.file_format = file_format
        self.load_fn = load_json if file_format == 'json' else load_yaml
        self.save_fn = safe_save_as_json if file_format == 'json' else safe_save_as_yaml
        if data is None:
            if file is None:
                self.data = {}
            else:
                self.data = self.load_fn(file)
        else:
            self.data = data

    def get_dict_value(self, data: Dict, key: str):
        keys = key.split(".")
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]

        last_key = keys[-1]
        if last_key not in data:
            return False, None
        return True, data[keys[-1]]

    def set_dict_value(self, data: Dict, key: str, value):
        keys = key.split(".")
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value

    def save(self, file: str | None = None, raise_error: bool = True):
        if file is None:
            file = self.file
        if file is None:
            if raise_error:
                raise ValueError('No file specified')
            return

        self.save_fn(file, self.data)

    def cache_apply(
        self, key, file: str | None = None, raise_save_error: bool = True, func=None
    ):
        wrapper = self.__call__(key, file, raise_save_error)
        if func is None:
            return wrapper
        return wrapper(func)

    def __call__(self, key, file: str | None = None, raise_save_error: bool = True):

        def _inner_func(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                has_cache, value = self.get_dict_value(self.data, key)
                if has_cache:
                    return value
                return_value = func(value, *args, **kwargs)
                self.set_dict_value(self.data, key, return_value)
                self.save(file, raise_save_error)
                return return_value

            return wrapper

        return _inner_func
