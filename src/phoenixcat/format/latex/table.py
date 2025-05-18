import re
import heapq
import logging
from typing import AnyStr, Dict, Callable, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_highlight_format_mapping = {
    "none": "{}",
    "bold": "\\mathbf{{{}}}",
    "italic": "\\mathit{{{}}}",
    "underline": "\\underline{{{}}}",
}

_default_highlight = "bold"


def register_highlight_format(name: str, format: str):
    if name in _highlight_format_mapping:
        logger.warning(f"Overwriting existing highlight format: {name}")
    _highlight_format_mapping[name] = format
    logger.info(f"Registered highlight format: {name}: {format}")


def list_all_highlight_format():
    return list(_highlight_format_mapping.keys())


def _get_last_number(string: str):
    match = re.search(r'\d+$', string)
    if match:
        return int(match.group())
    return None


def get_highlight_type_function(highlight_type: str):
    if "max" in highlight_type:
        if highlight_type == "max":
            return max
        max_num = _get_last_number(highlight_type)
        if max_num is None:
            return max
        return lambda x: heapq.nlargest(max_num, x)[-1]
    if "min" in highlight_type:
        if highlight_type == "min":
            return min
        min_num = _get_last_number(highlight_type)
        if min_num is None:
            return min
        return lambda x: heapq.nsmallest(min_num, x)[-1]
    raise ValueError(f"Invalid highlight type string: {highlight_type}")


@dataclass
class DataPoint:
    mean: float
    std: float | None = None
    highlight_format: str = _default_highlight
    decimal: int = None
    mean_rounded: float | None = None
    std_rounded: float | None = None

    def __post_init__(self):
        # if self.decimal is not None:
        self.set_decimal(self.decimal)
        self.set_highlight_format(self.highlight_format)

    def _preprocess_highlight_format(self, highlight_format):
        if highlight_format is None:
            return "{{{}}}"
        highlight_format_lower = highlight_format.lower()
        if highlight_format_lower in _highlight_format_mapping:
            highlight_format = _highlight_format_mapping[highlight_format_lower]
            return highlight_format
        return highlight_format

    def set_highlight_format(self, highlight_format=None):
        self.highlight_format = self._preprocess_highlight_format(highlight_format)

    def set_decimal(self, decimal):
        if decimal is None:
            self.decimal = None
            self.mean_rounded = self.mean
            self.std_rounded = self.std
            return
        self.decimal = decimal
        if self.std is not None:
            self.std_rounded = round(self.std, decimal)
        self.mean_rounded = round(self.mean, decimal)

    # compare two distributions
    def __lt__(self, other: "DataPoint"):
        return self.mean_rounded < other.mean_rounded

    def __eq__(self, other: "DataPoint"):
        return self.mean_rounded == other.mean_rounded

    def __gt__(self, other: "DataPoint"):
        return self.mean_rounded > other.mean_rounded

    def __le__(self, other: "DataPoint"):
        return self.mean_rounded <= other.mean_rounded

    def __ge__(self, other: "DataPoint"):
        return self.mean_rounded >= other.mean_rounded

    def __str__(self):

        decimal = self.decimal

        mean_s = f"{self.mean:.{decimal}f}" if decimal is not None else str(self.mean)
        # if self.highlight:
        mean_s = self.highlight_format.format(mean_s)
        std_s = ""
        if self.std is not None:
            std_s = (
                f"\\pm {self.std:.{decimal}f}"
                if decimal is not None
                else f"\\pm {self.std}"
            )
            std_s = f"_{{{std_s}}}"
        return f"${mean_s}{std_s}$"


class ListDataPoint:

    def __init__(self, mean, std=None, decimal=None, highlight_type=None):
        self.points = (
            [DataPoint(m, s) for m, s in zip(mean, std)]
            if std is not None
            else [DataPoint(m) for m in mean]
        )
        for d in self.points:
            # if decimal is not None:
            d.set_decimal(decimal)

        self.set_highlight(highlight_type)

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, i):
        return self.points[i]

    def _preprocess_highlight_type(
        self,
        highlight_type: (
            None | str | Callable | Dict[AnyStr, AnyStr] | Dict[Callable, AnyStr]
        ),
    ):
        if highlight_type is None:
            return {}

        if not isinstance(highlight_type, dict):
            highlight_type = {highlight_type: "bold"}

        processed_highlight_type = {}
        for key, value in highlight_type.items():
            if isinstance(key, str):
                key = get_highlight_type_function(key)
            processed_highlight_type[key] = value
        return processed_highlight_type

    def set_highlight(self, highlight_type=None, overwrite: bool = True):
        highlight_type = self._preprocess_highlight_type(highlight_type)

        if overwrite:
            for d in self.points:
                d.set_highlight_format(_highlight_format_mapping["none"])

        for anchor_fn, highlight_format in highlight_type.items():
            anchor = anchor_fn(self.points)
            for d in self.points:
                if d == anchor:
                    d.set_highlight_format(highlight_format)

    def get_list_str(self):
        return [str(d) for d in self.points]


class TableDataPoint:

    def __init__(self, list_points: list[list[str], ListDataPoint]):
        self.list_points = list_points

        if not all(
            len(list_point) == len(self.list_points[0])
            for list_point in self.list_points
        ):
            raise ValueError("All lists in list_points must have the same length")

    @property
    def transpose_str_matrix(self):
        return [list(map(str, col)) for col in zip(*self.list_points)]
        # return list(map(list, zip(*self.list_points)))

    def __str__(self):
        matrix = self.transpose_str_matrix
        return " \n".join([" & ".join(row) for row in matrix])

    def __getitem__(self, indice):
        return self.list_points[indice]

    def get_col_at(self, indice):
        return self.__getitem__(indice)

    def get_row_at(self, indice):
        return [list_point[indice] for list_point in self.list_points]

    def __len__(self):
        return len(self.list_points)

    @staticmethod
    def concat(table_data_points: list["TableDataPoint"]):
        return TableDataPoint(
            [
                list_point
                for table_data_point in table_data_points
                for list_point in table_data_point.list_points
            ]
        )
