import os
import logging

from .constant import USER_NAME, CONFIG_HOME
from .files import load_json, safe_save_as_json

logger = logging.getLogger(__name__)

# _default_environ_init = {
#     # "WANDB_MODE": "offline",
# }

_ENVIRON_PATH = os.path.join(CONFIG_HOME, "environ.json")

_environ_config = None


def get_default_environ_init():
    global _environ_config
    if _environ_config is not None:
        return _environ_config

    if not os.path.exists(_ENVIRON_PATH):
        config = {}
        safe_save_as_json(config, _ENVIRON_PATH)
    else:
        config = load_json(_ENVIRON_PATH)

    return config


def _environ_init():

    config = get_default_environ_init()

    for key, value in config.items():
        if not key in os.environ:
            os.environ[key] = value


def set_default_environ_init(set_dict: dict):

    config = get_default_environ_init()
    for key, value in set_dict.items():
        if value is not None:
            config[key] = value
        elif key in config:
            del config[key]
    safe_save_as_json(config, _ENVIRON_PATH)
    _environ_init()
