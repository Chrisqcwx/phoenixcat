import os
import logging
from diffusers.utils import is_accelerate_available

from .configuration.autosave_utils import auto_register_save_load
from .welcome import add_welcome_msg, welcome_print
from .constant import USER_NAME, CONFIG_HOME
from .files import load_json, safe_save_as_json

logger = logging.getLogger(__name__)

_default_environ_init = {
    "WANDB_MODE": "offline",
}

_ENVIRON_PATH = os.path.join(CONFIG_HOME, "environ.json")


def _environ_init():

    if not os.path.exists(_ENVIRON_PATH):
        config = {}
        safe_save_as_json(config, _ENVIRON_PATH)
    else:
        config = load_json(_ENVIRON_PATH)

    if not "environ" in config:
        config["environ"] = _default_environ_init

    for key, value in config["environ"].items():
        if not key in os.environ:
            os.environ[key] = value


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
    _welcome()
