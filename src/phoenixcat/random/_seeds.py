import logging
logger = logging.getLogger(__name__)

def seed_every_thing(seed: int = 0):
    """
    Set the random seed for various libraries to ensure reproducibility.

    Args:
    seed (int, Optional): The seed value to use for all random number generators. Default is 0.
    """
    _seed_python(seed)
    _seed_accelerate(seed)
    _seed_numpy(seed)
    _seed_torch(seed)

def _seed_python(seed):
    import random
    random.seed(seed)
    logger.info(f"The seed of `random` is set to {seed}.")

def _seed_torch(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.info(f"The seed of `torch` is set to {seed}.")
    except ImportError:
        logger.warning("`torch` not found in the environment.")
        
def _seed_numpy(seed):
    try:
        import numpy
        numpy.random.seed(seed)
        logger.info(f"The seed of `numpy` is set to {seed}.")
    except ImportError:
        logger.warning("`numpy` not found in the enviroment.")

def _seed_accelerate(seed):
    try:
        import accelerate
        accelerate.utils.set_seed(seed)
        logger.info(f"The seed of `accelerate` is set to {seed}.")
    except ImportError:
        logger.warning("`accelerate` not found in the enviroment.")
            
