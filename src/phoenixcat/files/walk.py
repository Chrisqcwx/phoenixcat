import os
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

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

    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    logger.info(
        f'Walk path: {path}. Find {len(file_paths)} files with extension {extension}'
    )
    return file_paths


def walk_images(path: str):
    return walk_extension_files(path, suffix=IMG_EXTENSIONS)


def walk_dict(data: Dict, key: str, create_if_not_exist: bool = False):
    keys = key.split(".")
    for k in keys:
        if k not in data:
            if create_if_not_exist:
                data[k] = {}
            else:
                raise KeyError(f"Key {key} not found in data")
        data = data[k]
    return data
