import os
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from ...configuration import pipeline_loadable


@pipeline_loadable
@dataclass
class DataManager:
    dataset: Dataset
    dataloader: DataLoader

    def save_pretrained(self, save_directory: str) -> None:
        """Save the dataset and dataloader to the specified directory."""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.dataset, f"{save_directory}/dataset.pt")
        torch.save(self.dataloader, f"{save_directory}/dataloader.pt")

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "DataManager":
        """Load the dataset and dataloader from the specified directory."""
        dataset = torch.load(f"{save_directory}/dataset.pt")
        dataloader = torch.load(f"{save_directory}/dataloader.pt")
        return cls(dataset, dataloader)


@pipeline_loadable
@dataclass
class DataManagerGroup:
    """Data Manager class."""

    _datasets: Dict[DataManager]

    def save_pretrained(self, save_directory: str) -> None:
        for name, data_manager in self._datasets.items():
            data_manager.save_pretrained(f"{save_directory}/{name}")

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "DataManagerGroup":
        datasets = {}
        for name in os.listdir(save_directory):
            subfolder = os.path.join(save_directory, name)
            if os.path.isdir(subfolder):
                datasets[name] = DataManager.from_pretrained(subfolder)
        return cls(datasets)
