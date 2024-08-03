# Copyright 2024 Hongyao Yu.
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

import functools
from typing import Any, Literal
from collections import defaultdict
from abc import abstractmethod

import torch

from ..conversion import get_attribute_from_obj


def register_execute_order(
    tag: str,
    order=0,
    interval: int | str = 1,
    execute_stage: Literal['before', 'after'] = 'after',
):
    def wrapper(func):

        func._order_info = {
            'tag': tag,
            'order': order,
            'interval': interval,
            'execute_stage': execute_stage,
        }

        return func

    return wrapper


def register_execute_main(tag: str):
    def wrapper(func):
        func._main_tag = tag
        return func

    return wrapper


class ExecuteOrderMixin:

    _execute_main_method = None  # type: ignore
    _execute_order_before = None
    _execute_order_after = None

    execute_counts = None

    @staticmethod
    def register_execute_order(
        tag: str,
        order=0,
        interval: int | str = 1,
        execute_stage: Literal['before', 'after'] = 'after',
    ):
        return register_execute_order(tag, order, interval, execute_stage)

    @staticmethod
    def register_execute_main(tag: str):
        return register_execute_main(tag)

    def _get_interval(self, func):
        interval = func._order_info['interval']
        if isinstance(interval, str):
            interval = get_attribute_from_obj(self, interval)
        return interval

    def reset_execute_flag(self, tag=None):
        if tag is None:
            self.execute_counts = defaultdict(lambda: 0)
        else:
            self.execute_counts[tag] = 0

    def _build_execute_order(self):
        self._execute_main_method = {}
        self._execute_order_before = defaultdict(list)
        self._execute_order_after = defaultdict(list)
        self.execute_counts = defaultdict(lambda: 0)
        for name, func in self.__class__.__dict__.items():
            # print(f'has name: {name}', callable(func))
            if callable(func):
                if hasattr(func, '_main_tag'):
                    if func._main_tag in self._execute_main_method:
                        raise ValueError(
                            f'Already have main method with tag {func._main_tag}'
                        )
                    self._execute_main_method[func._main_tag] = func
                if hasattr(func, '_order_info'):
                    if func._order_info['execute_stage'] == 'before':
                        self._execute_order_before[func._order_info['tag']].append(func)
                    else:
                        self._execute_order_after[func._order_info['tag']].append(func)

        for stage in self._execute_order_before.keys():
            self._execute_order_before[stage].sort(key=lambda x: x._order_info['order'])

        for stage in self._execute_order_after.keys():
            self._execute_order_after[stage].sort(key=lambda x: x._order_info['order'])

        for tag, func in self._execute_main_method.items():

            # func._execute_cnt = 0

            @functools.wraps(func)
            def wrapper(*args, _tag=tag, _func=func, **kwargs):

                for _f in self._execute_order_before[_tag]:
                    intervel = self._get_interval(_f)
                    if self.execute_counts[_tag] % intervel == 0:
                        _f(self)
                ret = _func(self, *args, **kwargs)
                for _f in self._execute_order_after[_tag]:
                    intervel = self._get_interval(_f)
                    if self.execute_counts[_tag] % intervel == 0:
                        _f(self)

                self.execute_counts[_tag] += 1
                return ret

            # print(f'set name {name}')
            self.__setattr__(func.__name__, wrapper)

    def __init__(self) -> None:
        # print('Execute Mixin')
        self._build_execute_order()
        # print(self._execute_main_method)
        # print(self._execute_order_before)
