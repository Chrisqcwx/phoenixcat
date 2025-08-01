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

import re
import heapq
import logging
from typing import AnyStr, Dict, Callable, Literal, TypeVar, Union, Generic
from dataclasses import dataclass

import pandas as pd

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

    def _preprocess_highlight_format(self, highlight_format: str):
        if highlight_format is None:
            return _highlight_format_mapping["none"]
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
        self.decimal = decimal
        self.highlight_type = highlight_type
        for d in self.points:
            d.set_decimal(decimal)

        self.set_highlight(highlight_type)

    @classmethod
    def concat(cls, list_data_points: list["ListDataPoint"]):
        all_means = [
            p.mean
            for list_data_point in list_data_points
            for p in list_data_point.points
        ]
        all_stds = [
            p.std
            for list_data_point in list_data_points
            for p in list_data_point.points
        ]
        # check decimal and highlight_type is the same
        if not all(
            list_data_point.decimal == list_data_points[0].decimal
            for list_data_point in list_data_points
        ):
            raise ValueError("Decimal must be the same for all data points")

        if not all(
            list_data_point.highlight_type == list_data_points[0].highlight_type
            for list_data_point in list_data_points
        ):
            raise ValueError("Highlight type must be the same for all data points")
        return cls(
            all_means,
            all_stds,
            decimal=list_data_points[0].decimal,
            highlight_type=list_data_points[0].highlight_type,
        )

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

    @property
    def fixed(self):
        return self.get_list_str()


class TableDataPoint:

    def __init__(self, list_points: list[list[str], ListDataPoint]):
        self.list_points = list_points

        if not all(
            len(list_point) == len(self.list_points[0])
            for list_point in self.list_points
        ):
            raise ValueError(
                f"All lists in list_points must have the same length, but found {[len(list_point) for list_point in self.list_points]}"
            )

    @property
    def str_matrix(self):
        return [list(map(str, col)) for col in self.list_points]

    @property
    def transpose_str_matrix(self):
        return [list(map(str, col)) for col in zip(*self.list_points)]
        # return list(map(list, zip(*self.list_points)))

    def __str__(self):
        matrix = self.transpose_str_matrix
        return "".join([" & ".join(row) + " \\\\\n" for row in matrix])

    def __getitem__(self, indice):
        return self.list_points[indice]

    def get_col_at(self, indice):
        return self.__getitem__(indice)

    def get_row_at(self, indice):
        return [list_point[indice] for list_point in self.list_points]

    def __len__(self):
        return len(self.list_points)

    @property
    def num_cols(self):
        return len(self)

    @property
    def num_rows(self):
        return len(self.list_points[0])

    @staticmethod
    def concat(
        table_data_points: list["TableDataPoint"], axis: Literal['col', 'row'] = 'col'
    ):
        if axis == "col":
            return TableDataPoint(
                [
                    list_point
                    for table_data_point in table_data_points
                    for list_point in table_data_point.list_points
                ]
            )
        elif axis == "row":
            if not all(
                len(table_data_point) == len(table_data_points[0])
                for table_data_point in table_data_points
            ):
                raise ValueError("All TableDataPoint must have the same length")

            all_cols = []
            for i in range(len(table_data_points[0])):
                cols = [table_data_point[i] for table_data_point in table_data_points]
                if isinstance(cols[0], ListDataPoint):
                    all_cols.append(ListDataPoint.concat(cols))
                else:
                    all_cols.append([item for col in cols for item in col])
            return TableDataPoint(all_cols)
        else:
            raise ValueError("Axis must be 'col' or 'row'")

    @property
    def fixed(self):
        lists = []
        for list_point in self.list_points:
            if isinstance(list_point, ListDataPoint):
                lists.append(list_point.fixed)
            else:
                lists.append(list_point)
        return TableDataPoint(lists)


@dataclass
class MultiCellInfo:
    context: str
    length: int | None = None


# @dataclass
# class MultiRowInfo(MultiCellInfo):
#     pass


# @dataclass
# class MultiColInfo(MultiCellInfo):
#     pass


# T = TypeVar("T", bound=Union[MultiRowInfo, MultiColInfo])


class ListMultiCellInfo:
    def __init__(self, infos: list[MultiCellInfo]):
        all_infos = []
        for info in infos:
            if isinstance(info, str):
                info = MultiCellInfo(info, 1)
            elif isinstance(info, MultiCellInfo):
                pass
            else:
                info = MultiCellInfo(info[0], info[1])
            all_infos.append(info)

        self.infos = all_infos

        # print(len(self))

    # def __len__(self):
    #     return len(self.infos)

    def __getitem__(self, item):
        return self.infos[item]

    def __iter__(self):
        return iter(self.infos)

    @property
    def length(self):
        # return None
        current_length = 0
        for info in self.infos:
            if info.length is None:
                return None
            current_length += info.length
        return current_length

    def fill_length(self, length):
        none_indices = []
        current_length = 0
        for idx, info in enumerate(self.infos):
            if info.length is None:
                none_indices.append(idx)
            else:
                current_length += info.length
        if len(none_indices) == 0:
            if not current_length == length:
                raise ValueError(f"Invalid length: {length}, expected {current_length}")
            return self
        if len(none_indices) == 1:
            if current_length > length:
                raise ValueError(f"Invalid length: {length}, expected {current_length}")
            self.infos[none_indices[0]].length = length - current_length
            return
        # len(none_indices) > 1
        raise ValueError("None length is larger than 1. Fail to fill length")


class ListMultiRowInfo(ListMultiCellInfo):

    def create_table(self):
        contents = []
        for info in self.infos:
            if info.length is None:
                raise RuntimeError("None length is not allowed")
            if info.length == 1:
                contents.append(info.context)
            else:
                contents.append(f"\\multirow{{{info.length}}}{{*}}{{{info.context}}}")
                contents.extend([""] * (info.length - 1))
        return TableDataPoint([contents])


class TableRow:

    def __init__(self, contents: list[ListMultiRowInfo, TableDataPoint]):
        self.contents = contents
        self.length = self._check_length()

    def create_table(self):
        tables = [
            content.create_table() if isinstance(content, ListMultiRowInfo) else content
            for content in self.contents
        ]
        return TableDataPoint.concat(tables, axis="col")

    def _check_length(self):
        standard_length = None
        for content in self.contents:
            if isinstance(content, ListMultiCellInfo):
                current_length = content.length
            elif isinstance(content, TableDataPoint):
                current_length = content.num_rows
            else:
                raise ValueError("Invalid content")

            if (
                standard_length is not None
                and current_length is not None
                and standard_length != current_length
            ):
                raise ValueError(
                    f"Invalid length, current length is {current_length}, reference length is {standard_length}"
                )
            standard_length = (
                standard_length if standard_length is not None else current_length
            )

        if standard_length is None:
            raise ValueError("Invalid length")

        for content in self.contents:
            if isinstance(content, ListMultiCellInfo):
                content.fill_length(standard_length)

        return standard_length
