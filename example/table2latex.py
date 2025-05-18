import numpy as np

# from phoenixcat.format import ListDataPoint, create_table_from_list, merge_tables, table_to_string
from phoenixcat.format.latex.table import ListDataPoint, TableDataPoint

prefix_table = [
    'A',
    'B',
    'C',
    'D',
    'E',
]

prefix_table = [ f"\\textbf{{{t}}}" for t in prefix_table]

def create_mainexp_table(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    all_dists = [
        prefix_table,
        ListDataPoint(mean=data[:, 0], decimal=1, highlight_type=None),
        ListDataPoint(mean=data[:, 1]*100, std=data[:, 3], decimal=1, highlight_type='min'),
        ListDataPoint(mean=data[:, 2]*100, std=data[:, 4], decimal=0, highlight_type='min'),
        ListDataPoint(mean=data[:, 5], std=data[:, 6], decimal=2, highlight_type={
            'max': "bold",
            "min": "italic",
            "max2": "underline"
        }),
    ]
    return TableDataPoint(all_dists)

data = create_mainexp_table("table.csv")

print(str(data))