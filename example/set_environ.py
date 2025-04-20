import os
import sys

sys.path.append("../src")

from phoenixcat.environ import set_default_environ_init


set_default_environ_init(
    {"HF_ENDPOINT": "https://hf-mirror.com", "WANDB_MODE": "offline"}
)
