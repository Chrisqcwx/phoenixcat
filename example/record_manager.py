import os
import sys

sys.path.append("../src")
import logging

import torch
import torchvision
from torch import nn, Tensor
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.configuration_utils import register_to_config

from phoenixcat.logger.logging import init_logger
from phoenixcat.files import (
    RecordManager,
    set_record_manager_path,
    record_manager,
)

save_dir = './test_record_manager'

init_logger(os.path.join(save_dir, "record.log"))

logger = logging.getLogger(__name__)


record_manager_path = os.path.join(save_dir, "record.json")
set_record_manager_path(record_manager_path)


@record_manager("global.func")
def global_func(a, b=1, c=2, *, d=3, e=4):
    global logger
    logger.info("execute global_func")
    return f"a={a}, b={b}, c={c}, d={d}, e={e}"


class DummyNonSerializableClass:

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return "<DummyNonSerializableClass>"


class DummyClass:

    @record_manager("class.aaa.bbb.func")
    def class_method(self):
        global logger
        logger.info("execute class_method")
        return DummyNonSerializableClass()


logger.info("call global_func first")
global_func_result = global_func(1, 2, 3, d=4, e=5)
logger.info(f"global_func_result: {global_func_result}")
logger.info("call global_func second")
global_func_result = global_func(1, 2, 3, d=4, e=5)
logger.info(f"global_func_result: {global_func_result}")

logger.info("---------------")
a = DummyClass()
logger.info("call class_method first")
class_method_result = a.class_method()
logger.info(f"class_method_result: {str(class_method_result)}")
logger.info("call class_method second")
class_method_result = a.class_method()
logger.info(f"class_method_result: {str(class_method_result)}")
