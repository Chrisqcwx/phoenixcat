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
    logger.info(f"The `huggingface_hub` mirror is set to '{url}'")
    
def hf_hub_download(
    mirror_url: str = None,
    **kwargs
):
    set_huggingface_mirror(mirror_url)
    import huggingface_hub
    return huggingface_hub.hf_hub_download(**kwargs)
    
def load_dataset(
    mirror_url: str = None,
    **kwargs
):
    set_huggingface_mirror(mirror_url)
    import datasets
    return datasets.load_dataset(**kwargs)