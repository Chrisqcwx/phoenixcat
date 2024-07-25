import abc
import json
from typing import Callable


def _register_fn(_method_name, fn: str | Callable, cls=None):

    def _inner_wrapper(cls):
        nonlocal fn
        if isinstance(fn, str):
            fn = getattr(cls, fn, None)
            if fn is None:
                raise ValueError(f"Cannot find method {fn} in {cls}")
        setattr(cls, _method_name, fn)

    if cls is None:
        return _inner_wrapper
    return _inner_wrapper(cls)


def register_from_pretrained(load_method: str | Callable, cls=None):
    _register_fn('from_pretrained', load_method, cls)


def register_save_pretrained(save_method: str | Callable, cls=None):
    _register_fn('save_pretrained', save_method, cls)


# class ConvertToJsonMixin(abc.ABC):

#     @abc.abstractmethod
#     def to_json(self):
#         """Convert the model to a JSON object."""
#         raise NotImplementedError("This method should be implemented in the subclass.")

#     @abc.abstractmethod
#     @classmethod
#     def from_json(cls, json_obj: dict):
#         raise NotImplementedError("This method should be implemented in the subclass.")


def is_json_serializable(obj):
    # if hasattr(obj, 'to_json') and hasattr(obj, 'from_json'):
    #     return True
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False
