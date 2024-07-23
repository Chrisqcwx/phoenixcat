import logging
from typing import Dict, Union, Any, Optional, AnyStr

import torch
from diffusers.optimization import get_scheduler as diffusers_get_scheduler

from ..conversion import get_obj_from_str
from ..decorators import Register

logger = logging.getLogger(__file__)

optimizer_register = Register('optimizer')
lr_shceduler_register = Register('lr_scheduler')


def _search(
    register: Register,
    name: Optional[AnyStr] = None,
    first_input: Any = None,
    build_params: Optional[Dict] = None,
    source: Optional[str] = None,
):
    if name is None:
        return None

    if source is not None and source not in register.keys():
        raise RuntimeError(f'Source {source} is not valid')

    if build_params is None:
        build_params = {}

    search_keys = [source] if source is not None else register.keys()
    for src in search_keys:
        cls_builder = search_keys[src]
        instance = cls_builder(name, first_input, **build_params)
        if instance is not None:
            logger.info(f'get {register.name} from {src}')
            return instance

    raise RuntimeError(f'Name {name} can not be found for {register.name}.')


@optimizer_register.register('torch')
def _get_torch_optimizer(name: str, params, build_params):
    if not name.startswith('torch.optim.'):
        name = f'torch.optim.{name}'
    optimizer_cls = get_obj_from_str(name)
    return optimizer_cls(params, **build_params)


@lr_shceduler_register.register('torch')
def _get_torch_lr_scheduler(name: str, optimizer, kwargs):
    lr_scheduler_cls = get_obj_from_str(f'torch.optim.lr_scheduler.{name}')
    if lr_scheduler_cls is None:
        return None
    return lr_scheduler_cls(optimizer, **kwargs)


@lr_shceduler_register.register('diffusers')
def _get_diffusers_lr_scheduler(name: str, optimizer, kwargs):
    try:
        lr_scheduler = diffusers_get_scheduler(name, optimizer, **kwargs)
    except:
        return None

    return lr_scheduler


def get_optimizer(name, params, optimizer_params):

    return _search(optimizer_register, name, params, optimizer_params)


def get_lr_scheduler(name, optimizer, lr_scheduler_params):

    return _search(lr_shceduler_register, name, optimizer, lr_scheduler_params)
