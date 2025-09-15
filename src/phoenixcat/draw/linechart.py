from typing import Union, Tuple, Literal
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .utils import FontManager, AxesManager, Painter
from ..auto.dataclass_utils import config_dataclass_wrapper


@config_dataclass_wrapper()
@dataclass
class LineInfo:
    x: list | np.ndarray | pd.Series
    y: list | np.ndarray | pd.Series
    label: str | None = None
    color: str | None = None
    marker: str = "o"
    alpha: float = 0.9
    linewidth: float = 1.0
    linestyle: str = "-"
    markersize: float = 7.0


class LineChartPainter(Painter):
    def __init__(self, fig_size: Tuple[int, int], font_manager: FontManager):
        super().__init__(fig_size=fig_size, font_manager=font_manager)

    def draw(self, line_info: LineInfo):
        self.ax.plot(
            line_info.x,
            line_info.y,
            label=line_info.label,
            color=line_info.color,
            marker=line_info.marker,
            alpha=line_info.alpha,
            linewidth=line_info.linewidth,
            linestyle=line_info.linestyle,
            markersize=line_info.markersize,
        )
