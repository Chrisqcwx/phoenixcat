import numpy as np
from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.dataset, name):
                return getattr(self.dataset, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __hasattr__(self, name):
        return hasattr(self.dataset, name) or super().__hasattr__(name)
    
class DownSampleDatasetWrapper(DatasetWrapper):
    def __init__(self, dataset: Dataset, ratio: float = 0.1):
        super().__init__(dataset)
        self.ratio = ratio
        self.length = round(len(self.dataset) * self.ratio)

    def __getitem__(self, index: int):
        original_index = round(index / self.ratio)
        original_index = min(original_index, len(self.dataset) - 1)
        data = self.dataset[original_index]
        return original_index, data

    def __len__(self):
        return self.length
    
class RandomDownSampleDatasetWrapper(DatasetWrapper):
    def __init__(self, dataset: Dataset, ratio: float = 0.1):
        super().__init__(dataset)
        self.ratio = ratio
        self.length = round(len(self.dataset) * self.ratio)
        self.indices = np.random.choice(len(self.dataset), self.length, replace=False)

    def __getitem__(self, index: int):
        original_index = self.indices[index]
        data = self.dataset[original_index]
        return original_index, data

    def __len__(self):
        return self.length
    
class DatasetIndexWrapper(DatasetWrapper):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def __getitem__(self, index: int):
        data = self.dataset[index]
        return index, data

    def __len__(self):
        return len(self.dataset)