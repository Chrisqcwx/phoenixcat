import os
import json
import logging
from typing import Optional

from .path import get_safe_save_path
from ..format import format_as_yaml

logger = logging.getLogger(__name__)


def safe_save_as_yaml(obj, save_path: str):
    """Save the data in yaml format.

    Args:
        obj (Any): The objective to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the data to save. Defaults to None.
    """

    s = format_as_yaml(obj)
    save_path = get_safe_save_path(save_path)

    with open(save_path, 'w') as f:
        f.write(s)

    logger.info(f'Save yaml to {save_path}.')


def safe_save_torchobj(obj, save_path: str):
    """Save the obj by using torch.save function.

    Args:
        obj (Any): The objective to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the objective to save. Defaults to None.
    """

    save_path = get_safe_save_path(save_path)

    import torch

    torch.save(obj, save_path)

    logger.info(f'Save torch object to {save_path}.')


def safe_save_csv(obj, save_path: str):
    """Save the data in csv format.

    Args:
        obj (Any): The objective to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the data to save. Defaults to None.
    """
    import pandas as pd

    if not isinstance(obj, pd.DataFrame):
        df = pd.DataFrame(obj)

    save_path = get_safe_save_path(save_path)
    df.to_csv(save_path, index=None)

    logger.info(f'Save csv to {save_path}.')


def safe_save_as_json(obj, save_path: str):
    """Save the data in json format.

    Args:
        obj (Any): The objective to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the data to save. Defaults to None.
    """

    save_path = get_safe_save_path(save_path)
    with open(save_path, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)

    logger.info(f'Save json to {save_path}.')
