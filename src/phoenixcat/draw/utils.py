# Copyright 2025 Hongyao Yu.
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

import os
from typing import Union, Tuple, Literal
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib import font_manager

_default_colors = [
    "#1F77B4",
    "#2CA02C",
    "#7626B4",
    "#4CCED6",
    "#FF7F0E",
    "#D62728",
    "#E317E5",
]

_default_markers = ["o", "D", "*", "p", "s", "v", "^", "X"]


from typing import Union, Tuple


def adjust_brightness(
    color: Union[str, Tuple[int, int, int]],
    factor: float,
    output_type: Literal["auto", "hex", "rgb"] = "auto",
) -> Union[str, Tuple[int, int, int]]:
    """
    Adjust the brightness of a color.
    - factor < 1: darken
    - factor > 1: lighten
    - factor = 1: no change

    Args:
        color (str | tuple): Input color, either a HEX string (e.g., "#336699")
                             or an RGB tuple (e.g., (51, 102, 153)).
        factor (float): Brightness adjustment factor.
        output_type (str): Output format, "auto", "hex", or "rgb".
                      - "auto": return in the same format as the input
                      - "hex": return HEX string
                      - "rgb": return RGB tuple
                      Defaults to "auto".

    Returns:
        str | tuple: Adjusted color in the chosen output_type.
    """
    # Convert input to RGB
    if isinstance(color, str):
        hex_color = color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        input_format = "hex"
    elif isinstance(color, tuple) and len(color) == 3:
        r, g, b = color
        input_format = "rgb"
    else:
        raise ValueError("color must be a HEX string or an RGB tuple")

    # Apply brightness factor
    r = min(max(0, int(r * factor)), 255)
    g = min(max(0, int(g * factor)), 255)
    b = min(max(0, int(b * factor)), 255)

    # Determine output format
    if output_type == "auto":
        output_type = input_format

    # Return in requested format
    if output_type == "hex":
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    elif output_type == "rgb":
        return (r, g, b)
    else:
        raise ValueError("output must be 'hex', 'rgb', or 'auto'")


@dataclass
class FontManager:

    font_path: str
    math_fonfamily: str = "cm"
    default_font_size: int = 14
    title_font_size: int | None = None
    axis_font_size: int | None = None
    legend_font_size: int | None = None
    tick_font_size: int | None = None

    def _get_font(self, size):
        return font_manager.FontProperties(
            fname=self.font_path,
            size=size or self.default_font_size,
            math_fontfamily=self.math_fonfamily,
        )

    def __post_init__(self):
        self.default = self._get_font(self.default_font_size)
        self.title = self._get_font(self.title_font_size)
        self.axis = self._get_font(self.axis_font_size)
        self.legend = self._get_font(self.legend_font_size)
        self.tick = self._get_font(self.tick_font_size)
        plt.rcParams.update({"font.size": self.default_font_size})


class AxesManager:

    def __init__(self, ax: Axes, font_manager: FontManager):
        self.ax = ax
        self.font_manager = font_manager
        self._init()

    def _init(self):
        self.set_xlabel(self.ax.get_xlabel())
        self.set_ylabel(self.ax.get_ylabel())
        self.set_title(self.ax.get_title())
        self.set_xticks(self.ax.get_xticks())
        self.set_yticks(self.ax.get_yticks())

    def set_xlabel(self, label):
        self.ax.set_xlabel(label, fontproperties=self.font_manager.axis)

    def set_ylabel(self, label):
        self.ax.set_ylabel(label, fontproperties=self.font_manager.axis)

    def set_title(self, title):
        self.ax.set_title(title, fontproperties=self.font_manager.title)

    def set_xticks(self, ticks):
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(ticks, fontproperties=self.font_manager.tick)

    def set_yticks(self, ticks):
        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(ticks, fontproperties=self.font_manager.tick)

    def set_legend(self, loc, framealpha=1.0, markerscale=1, handlelength=1.7):
        self.ax.legend(
            loc=loc,
            framealpha=framealpha,
            markerscale=markerscale,
            handlelength=handlelength,
            prop=self.font_manager.legend,
        )

    def grid(self, visible=True, axis: Literal["x", "y", "both"] = "y"):
        self.ax.grid(visible=visible, axis=axis)


class Painter:

    def __init__(self, fig_size: Tuple[int, int], font_manager: FontManager):
        self.fig_size = fig_size
        self.font_manager = font_manager
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(*fig_size)

        self.ax_manager = AxesManager(self.ax, self.font_manager)

    def save(self, path: str, dpi: int = 600, show: bool = False):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.fig.savefig(path, bbox_inches="tight", dpi=dpi)
        if show:
            plt.show()
