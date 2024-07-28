# Copyright 2024 Sijin Yu.
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

from dataclasses import dataclass
from typing import Dict

import torch.distributed
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DataLoaderConfig:
    num_workers: int = 1
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    presistent_workers: bool = True


def getDataLoader(
    dataset: Dataset,
    batch_size: int,
    config: Dict,
    shuffle: bool = True,
    seed: int = 0,
    use_ddp: bool = False,
    device: str = "",
):
    if use_ddp:
        sampler = DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=config.get("drop_last", False),
        )
        shuffle = False
        rank = torch.distributed.get_rank()
        device = f"cuda:{rank}"
    else:
        sampler = None

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
        drop_last=config.get("drop_last", False),
        prefetch_factor=config.get("prefetch_factor", None),
        persistent_workers=config.get("presistent_workers", False),
        pin_memory_device=device,
    )

    return dataloader
