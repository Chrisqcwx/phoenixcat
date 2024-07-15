import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_safe_save_path(save_dir: str, save_name: Optional[str] = None):
    if save_name is None:
        save_dir, save_name = os.path.split(save_dir)
    if save_dir.strip() != '':
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f'Create folder {save_dir}')
    return os.path.join(save_dir, save_name)
