import time
import yaml
import json
from typing import Optional
from collections import OrderedDict


def print_split_line(content: Optional[str] = None, length: int = 60):
    """Print the content and surround it with '-' character for alignment.

    Args:
        content (_type_, optional): The content to print. Defaults to None.
        length (int, optional): The total length of content and '-' characters. Defaults to 60.
    """

    if content is None:
        print('-' * length)
        return
    if len(content) > length - 4:
        length = len(content) + 4

    total_num = length - len(content) - 2
    left_num = total_num // 2
    right_num = total_num - left_num
    print('-' * left_num, end=' ')
    print(content, end=' ')
    print('-' * right_num)


def format_number(num: int, base: int = 1000):
    assert num >= 0

    for suffix in ['', 'k', 'M', 'G']:
        if num < base:
            return f'{num:.4f} {suffix}'
        num /= base

    return f'{num:.4f} T'


def format_now_time(as_filename: bool = False):
    if as_filename:
        return time.strftime('%Y%m%d %H%M%S', time.localtime(time.time()))
    else:
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


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
