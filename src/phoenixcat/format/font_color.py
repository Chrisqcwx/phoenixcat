from enum import Enum
from typing import Literal


class Color(Enum):
    black = 30
    red = 31
    green = 32
    yellow = 33
    blue = 34
    purple = 35
    cyan = 36
    white = 37

    def __str__(self):
        return str(self.value)


class BackgroundColor(Enum):
    black = 40
    red = 41
    green = 42
    yellow = 43
    blue = 44
    purple = 45
    cyan = 46
    white = 47

    def __str__(self):
        return str(self.value)


class FontStyle(Enum):
    normal = 0
    bold = 1
    underline = 4
    blink = 5
    reverse = 7
    hidden = 8

    def __str__(self):
        return str(self.value)


def get_enum_value(value: int | str | None, enum_cls):

    if value is None:
        return None

    if isinstance(value, (int, enum_cls)):
        return str(value)

    return str(enum_cls[value].value)


def get_colored_str(
    content: str,
    color: (
        int
        | Literal['black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white']
        | None
    ) = None,
    background: (
        int
        | Literal['black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white']
        | None
    ) = None,
    style: (
        int
        | Literal['normal', 'bold', 'underline', 'blink', 'reverse', 'hidden']
        | None
    ) = None,
) -> str:
    # "\033[1;35;42m高亮紫色文字绿色背景\033[0m"

    font_styles = []
    for value, enum_cls in [
        (color, Color),
        (background, BackgroundColor),
        (style, FontStyle),
    ]:
        if value is not None:
            font_styles.append(get_enum_value(value, enum_cls))
    if not font_styles:
        return content
    return f"\033[{';'.join(font_styles)}m{content}\033[0m"
