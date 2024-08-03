import functools
from typing import Mapping
from collections import OrderedDict
from diffusers.utils.outputs import BaseOutput as HF_BaseOutput
from dataclasses import dataclass, is_dataclass


def parallel_enable_dataclass(cls):
    """
    A decorator that extends dataclass to accept either keyword arguments or a dictionary.
    """
    # cls = dataclass(cls)

    if not is_dataclass(cls):
        cls = dataclass(cls)

    # Modify the __init__ method to accept a dictionary as an argument
    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Mapping):
            init_dict = args[0]
            original_init(self, **init_dict)
        else:
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


# @parallel_enable_dataclass
# class BaseOutput(HF_BaseOutput):
#     aa: int
#     bb: str


# BaseOutput()
