import os
import logging

from .configuration.autosave_utils import auto_register_save_load
from .welcome import add_welcome_msg, welcome_print
from .constant import USER_NAME
from .environ import _environ_init

logger = logging.getLogger(__name__)


def _register_save_load():

    logger.debug("Registering save and load functions for phoenixcat.")

    # if is_accelerate_available():
    #     from accelerate import Accelerator, DataLoaderConfiguration

    #     auto_register_save_load(DataLoaderConfiguration)

    #     auto_register_save_load(
    #         Accelerator,
    #         ignore_list=[
    #             'dispatch_batches',
    #             'split_batches',
    #             'even_batches',
    #             'use_seedable_sampler',
    #         ],
    #     )


def _welcome():
    welcome_print(USER_NAME)


def _init_fn():
    _environ_init()
    _register_save_load()
    # _welcome()
