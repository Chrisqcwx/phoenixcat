import os
import time
import logging
from huggingface_hub import snapshot_download


def retry(func, *args, max_retries=3, delay=0.5, silent=False, **kwargs):
    """
    Retry function if it raises an exception.

    Args:
        func (function): The function to be retried.
        *args (tuple): Positional arguments for the function.
        max_retries (int): Maximum number of retries. Default is 3.
        delay (float): Delay between retries in seconds. Default is 0.5.
        **kwargs (dict): Keyword arguments for the function.

    Returns:
        The result of the function.
    """
    for i in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if not silent:
                logging.info(
                    f"Function {func.__name__} executed successfully on attempt {i+1}"
                )
            return result
        except Exception as e:
            if not silent:
                logging.error(f"Function {func.__name__} failed on attempt {i+1}: {e}")
            if i == max_retries - 1:
                raise e
            time.sleep(delay)


def hf_snapshot_download(
    repo_id, repo_type="model", max_retries=3, mirror=None, **kwargs
):
    origin_hf_endpoint = (
        None if 'HF_ENDPOINT' not in os.environ else os.environ['HF_ENDPOINT']
    )
    if mirror is not None:
        os.environ['HF_ENDPOINT'] = mirror
    try:
        func = lambda: snapshot_download(repo_id=repo_id, repo_type=repo_type, **kwargs)
        return retry(func, max_retries=max_retries)
    finally:
        if origin_hf_endpoint is not None:
            os.environ['HF_ENDPOINT'] = origin_hf_endpoint
