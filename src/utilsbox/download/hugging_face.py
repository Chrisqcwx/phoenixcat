import logging
import os
import sys

logger = logging.getLogger(__name__)

def is_huggingface_hub_imported():
    return "huggingface_hub" in sys.modules

def set_huggingface_mirror(url=None):
    if is_huggingface_hub_imported():
        logger.error("`huggingface_hub` is imported before setting the mirror.")

    url = "https://hf-mirror.com" if url is None else url
    os.environ["HF_ENDPOINT"] = url
    import huggingface_hub