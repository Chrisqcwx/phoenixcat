from dataclasses import dataclass

@dataclass
class DataPoint:
    mean: float
    std: float | None = None
    highlight: bool = False
    decimal: int = None

    def set_decimal(self, decimal):
        self.decimal = decimal
        if self.std is not None:
            self.std = round(self.std, decimal)
        self.mean = round(self.mean, decimal)

    # compare two distributions
    def __lt__(self, other):
        return self.mean < other.mean
    
    def __eq__(self, other):
        return self.mean == other.mean

    def __str__(self):

        decimal = self.decimal
        
        mean_s = f"{self.mean:.{decimal}f}" if decimal is not None else str(self.mean)
        if self.highlight:
            mean_s = f"\\mathbf{{{mean_s}}}"
        std_s = ""
        if self.std is not None:
            std_s = f"\\pm {self.std:.{decimal}f}" if decimal is not None else f"\\pm {self.std}"
            std_s = f"_{{{std_s}}}"
        return f"${mean_s}{std_s}$"
    
class ListDataPoint:

    def __init__(self, mean, std=None, decimal=None, highlight_type=None):
        self.dists = [DataPoint(m, s) for m, s in zip(mean, std)] if std is not None else [DataPoint(m) for m in mean]
        for d in self.dists:
            # if decimal is not None:
            d.set_decimal(decimal)

        self.set_highlight(highlight_type)

    def __len__(self):
        return len(self.dists)
    
    def __getitem__(self, i):
        return self.dists[i]

    def set_highlight(self, highlight_type=None):
        if highlight_type is None:
            for d in self.dists:
                d.highlight = False
        elif highlight_type == "max":
            max_dist = max(self.dists)
            for d in self.dists:
                d.highlight = d == max_dist
        elif highlight_type == "min":
            min_dist = min(self.dists)
            for d in self.dists:
                d.highlight = d == min_dist
        else:
            raise ValueError("Invalid highlight_type")
        
class TableDataPoint:

    def __init__(self, list_dists: list[ListDataPoint]):
        self.list_dists = list_dists
        self.table = self.to_table()

    def __getitem__(self, indice):
        i, j = indice
        return self.list_dists[j][i]
    
    def __len__(self):
        return len(self.list_dists)

    def to_table(self):
        table = []
        for i in range(len(self.list_dists[0])):
            line_strs = []
            for j in range(len(self.list_dists)):
                single_s = self[i, j].__str__()
                line_strs.append(single_s)
            table.append(line_strs)
        return table
    
def create_table_from_list(list_dists: list[ListDataPoint]):
    return TableDataPoint(list_dists).table
    
def merge_tables(*tables):
    result_table = []
    for i in range(len(tables[0])):
        line = []
        for table in tables:
            line.extend(table[i])
        result_table.append(line)
    return result_table

def table_to_string(table):
    table_str = ""
    for line in table:
        line_str = " & ".join(line) + " \\\\\n"
        table_str += line_str
    return table_str


