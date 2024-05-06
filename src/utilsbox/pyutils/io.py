import os
import yaml
import json
from typing import Optional, Tuple
from collections import OrderedDict


import pandas as pd
from .format import format_as_yaml


def get_safe_save_path(save_dir: str, save_name: Optional[str] = None):
    if save_name is None:
        save_dir, save_name = os.path.split(save_dir)
    if save_dir.strip() != '':
        os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, save_name)


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def walk_extension_files(path: str, extension: str | Tuple[str]):
    """Traverse all images in the specified path.

    Args:
        path (_type_): The specified path.

    Returns:
        List: The list that collects the paths for all the images.
    """

    img_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                img_paths.append(os.path.join(root, file))
    return img_paths


def walk_images(path: str):
    return walk_extension_files(path, suffix=IMG_EXTENSIONS)


def safe_save_as_yaml(obj, save_dir: str, save_name: Optional[str] = None):
    """Save the data in yaml format.

    Args:
        obj (Any): The objective to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the data to save. Defaults to None.
    """

    s = format_as_yaml(obj)
    save_path = get_safe_save_path(save_dir, save_name)
    with open(save_path, 'w') as f:
        f.write(s)


def safe_save_torchobj(obj, save_dir: str, save_name: Optional[str] = None):
    """Save the obj by using torch.save function.

    Args:
        obj (Any): The objective to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the objective to save. Defaults to None.
    """

    save_path = get_safe_save_path(save_dir, save_name)

    import torch

    torch.save(obj, save_path)


def safe_save_csv(df: pd.DataFrame, save_dir: str, save_name: Optional[str] = None):
    """Save the data in csv format.

    Args:
        df (pd.DataFrame): The data to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the data to save. Defaults to None.
    """

    save_path = get_safe_save_path(save_dir, save_name)
    df.to_csv(save_path, index=None)


def safe_save_as_json(obj, save_dir: str, save_name: Optional[str] = None):
    """Save the data in json format.

    Args:
        obj (Any): The objective to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the data to save. Defaults to None.
    """

    save_path = get_safe_save_path(save_dir, save_name)
    with open(save_path, 'w') as f:
        json.dump(obj, save_path)
