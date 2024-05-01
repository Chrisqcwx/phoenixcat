import os
import random
import time
from typing import Optional

import torch
import numpy as np


def set_random_seed(self, seed: Optional[int] = None, benchmark: bool = True):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = benchmark
    if self.seed is None:
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


_ALL_LOGITS = '0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
_ALL_LOGITS_INDICES = np.arange(len(_ALL_LOGITS), dtype=np.int32)


def get_random_string(length: int = 6):
    """Generate a random string with the specified length.

    Args:
        length (int, optional): The string length. Defaults to 6.

    Returns:
        str: The randomly generated string.
    """

    seed = int(time.time() * 1000) % (2**30) ^ random.randint(0, 2**30)
    # print(seed)

    resindices = np.random.RandomState(seed).choice(_ALL_LOGITS_INDICES, length)
    return ''.join(map(lambda x: _ALL_LOGITS[x], resindices))
