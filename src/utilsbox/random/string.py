import logging
import time
import random

import numpy as np

logger = logging.getLogger(__name__)

_ALL_LOGITS = '0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'


def random_string(length: int = 6):
    """Generate a random string with the specified length.

    Args:
        length (int, optional): The string length. Defaults to 6.

    Returns:
        str: The randomly generated string.
    """
    results = random.choices(_ALL_LOGITS, k=length)
    results = ''.join(results)
    logger.info(f'Generate random string `{results}`.')
    return results
