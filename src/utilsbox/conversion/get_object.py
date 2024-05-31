import importlib
from typing import Sequence, Union

import logging
logger = logging.getLogger(__name__)

def get_obj_from_str(strings: Union[str, Sequence[str]], reload=False):
    if not isinstance(strings, str):
        logger.debug(f"Recieve `strings` to be {strings} (type = {type(strings)}), not str.")
        strings = list(strings)
        strings = ".".join(strings)
    module, cls = strings.rsplit(".", 1)
    if reload:
        _module = importlib.import_module(module)
        importlib.reload(_module)
        logger.debug(f"Reload module `{module}`")
    return getattr(importlib.import_module(module), cls)
    