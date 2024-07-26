import functools
from typing import Any, Literal
from collections import defaultdict
from abc import abstractmethod

import torch

from ..conversion import get_attribute_from_obj


class ExecuteOrderMixin:

    _execute_main_method = None  # type: ignore
    _execute_order_before = None
    _execute_order_after = None

    @property
    @abstractmethod
    def is_end() -> bool:
        pass

    @staticmethod
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

    @staticmethod
    def register_execute_main(tag: str):
        def wrapper(func):
            func._main_tag = tag
            return func

        return wrapper

    def _get_interval(self, func):
        interval = func._order_info['interval']
        if isinstance(interval, str):
            interval = get_attribute_from_obj(self, interval)
        return interval

    def _build_execute_order(self):
        self._execute_main_method = {}
        self._execute_order_before = defaultdict(list)
        self._execute_order_after = defaultdict(list)
        for name, func in self.__class__.__dict__.items():
            # print(f'has name: {name}', callable(func))
            if callable(func):
                if hasattr(func, '_main_tag'):
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

        for name, func in self._execute_main_method.items():

            func._execute_cnt = 0

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                is_end = self.is_end

                for _f in self._execute_order_before[name]:
                    intervel = self._get_interval(_f)
                    if is_end or (func._execute_cnt % intervel == 0):
                        _f(self)
                ret = func(self, *args, **kwargs)
                for _f in self._execute_order_after[name]:
                    intervel = self._get_interval(_f)
                    if is_end or (func._execute_cnt % intervel == 0):
                        _f(self)

                func._execute_cnt += 1
                return ret

            # print(f'set name {name}')
            self.__setattr__(func.__name__, wrapper)

    def __init__(self) -> None:
        # print('Execute Mixin')
        self._build_execute_order()
        # print(self._execute_main_method)
        # print(self._execute_order_before)
