import time
import yaml
import json
import logging
from typing import Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)


def format_as_split_line(content: Optional[str] = None, length: int = 60) -> str:
    if content is None:
        return '-' * length
    if len(content) > length - 4:
        logger.info('The length of content is larger than the specified length.')
        length = len(content) + 4

    total_num = length - len(content) - 2
    left_num = total_num // 2
    right_num = total_num - left_num
    return '-' * left_num + f' {content} ' + '_' * right_num


def format_number(num: int, base: int = 1000):
    prefix = ''
    if num < 0:
        prefix = '-'
        num = -num

    for suffix in ['', 'k', 'M', 'G']:
        if num < base:
            return f'{num:.4f} {suffix}'
        num /= base

    return f'{prefix}{num:.4f} T'


def format_time(time_time: Optional[float] = None, pattern: str = '%Y%m%d-%H%M%S'):
    if time_time is None:
        time_time = time.time()
    return time.strftime(pattern, time.localtime(time_time))


yaml.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        'tag:yaml.org,2002:map', data.items()
    ),
)
yaml.add_representer(
    tuple, lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data)
)


def format_as_yaml(obj) -> str:
    return yaml.dump(obj)


def format_as_json(obj) -> str:
    return json.dumps(obj)
