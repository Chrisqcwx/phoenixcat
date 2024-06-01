import os
import logging
from typing import Tuple

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
