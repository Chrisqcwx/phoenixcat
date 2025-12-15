import os
import sys

sys.path.append("../src")

from phoenixcat.cuda import wait_gpu_run
from phoenixcat.logger import init_logger

init_logger("wait_gpu_run.log", console_level="INFO")

CMD = ["bash", "wait_gpu_run.sh"]
gpu_use_num = 3
gpu_ids = "0-2,4,5-7"
threshold_gb = 40
check_interval = 1
confirm_times = 2

wait_gpu_run(
    CMD,
    gpu_use_num,
    gpu_ids,
    threshold_gb,
    check_interval,
    confirm_times,
)
