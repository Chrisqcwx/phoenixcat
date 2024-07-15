import functools
import logging

import torch.distributed

logger = logging.getLogger(__name__)

def setup_one_process(rank: int, world_size: int):
    if rank >= world_size:
        logger.error(f"`rank` must < `world_size`, but got `rank`={rank} and `world_size`={world_size}.")
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger.info(f"Successfully setup rank={rank} (world_size={world_size}).")
    
def is_main_process():
    return torch.distributed.get_rank() == 0
    
def register_to_ddp_train_func(train_func: function):
    @functools.wraps(train_func)
    def ddp_train_func(*args, **kwargs):
        rank = kwargs.pop("rank", -1)
        world_size = kwargs.pop("world_size", -1)
        setup_one_process(rank=rank, world_size=world_size)
        result = train_func(*args, **kwargs)
        torch.distributed.destroy_process_group()
        return result
    return ddp_train_func