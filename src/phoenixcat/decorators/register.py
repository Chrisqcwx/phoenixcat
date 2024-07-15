import logging
from typing import Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)

_BUILDIN_REGISTERS = {}


class Register(dict):
    def __init__(self, name: str, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

        if name in _BUILDIN_REGISTERS:
            raise RuntimeError(f'The register {name} is already exist')
        logger.info(f'Create register {name}')
        _BUILDIN_REGISTERS[name] = self

    def register(self, name: Optional[str] = None):

        def _inner_register(fn):

            key = name if name is not None else fn.__name__

            if key in self._dict:
                raise RuntimeError(f'name {key} has already be register.')

            self._dict[key] = fn
            return fn

        return _inner_register

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


def list_registers():
    return list(_BUILDIN_REGISTERS.keys())


def get_register(name: str):
    if name in _BUILDIN_REGISTERS:
        return _BUILDIN_REGISTERS[name]
    return Register(name)
