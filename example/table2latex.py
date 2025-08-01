import copy
import numpy as np

from phoenixcat.format.latex.table import (
    ListDataPoint,
    TableDataPoint,
    ListMultiRowInfo,
    TableRow,
)

prefix_table = [
    'A',
    'B',
    'C',
    'D',
    'E',
]

prefix_table = [f"\\textbf{{{t}}}" for t in prefix_table]


def create_mainexp_table(file_path, add=0):
    data = np.loadtxt(file_path, delimiter=",") + add
    all_dists = [
        prefix_table,
        ListDataPoint(mean=data[:, 0], decimal=1, highlight_type=None),
        ListDataPoint(
            mean=data[:, 1] * 100, std=data[:, 3], decimal=1, highlight_type='max'
        ),
        ListDataPoint(
            mean=data[:, 2] * 100, std=data[:, 4], decimal=0, highlight_type='min'
        ),
        ListDataPoint(
            mean=data[:, 5],
            std=data[:, 6],
            decimal=2,
            highlight_type={'max': "bold", "min": "italic", "max2": "underline"},
        ),
    ]
    return TableDataPoint(all_dists)


data = create_mainexp_table("table.csv")
data2 = create_mainexp_table("table.csv", add=1)

print(str(data))

print('----------')

print(TableDataPoint.concat([data, data2]))

print('----------')

print(TableDataPoint.concat([data, data2], axis='row'))

print('----------')

print(TableDataPoint.concat([data.fixed, data2.fixed], axis='row'))

print('----------')

row_info_1 = ListMultiRowInfo([['ALL', None]])

row_info_2 = ListMultiRowInfo([['2', 2], ['3', 3]])

row_info_3 = ListMultiRowInfo(['aa', 'bb', 'cc', 'dd', 'ee'])

# print(TableRow.concat([TableRow(data, row_info_1), TableRow(data2, row_info_2)]))

new_table = TableRow([row_info_1, row_info_2, row_info_3, data]).create_table()

print(new_table)
