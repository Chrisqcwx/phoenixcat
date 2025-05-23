# Copyright 2024 Hongyao Yu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
