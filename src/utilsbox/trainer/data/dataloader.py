import torch.distributed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class DataLoaderConfig:
    def __init__(
        self,
        num_workers:        int  = 1,
        pin_memory:         bool = True,
        drop_last:          bool = False,
        prefetch_factor:    int  = 2,
        presistent_workers: bool = True,
    ) -> None:
        self.num_workers         = num_workers
        self.pin_memory          = pin_memory
        self.drop_last           = drop_last
        self.prefetch_factor     = prefetch_factor
        self.presistent_workers  = presistent_workers
        
def getDataLoader(
    dataset: Dataset,
    batch_size: int,
    config: DataLoaderConfig,
    shuffle: bool = True,
    seed: int = 0,
    use_ddp: bool = False,
    device: str = "cuda"
):
    if use_ddp:
        sampler = DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=config.drop_last
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
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        prefetch_factor=config.prefetch_factor,
        presistent_workers=config.presistent_workers,
        pin_memory_device=device
    )
    
    return dataloader