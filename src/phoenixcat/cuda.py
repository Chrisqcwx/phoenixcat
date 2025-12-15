import os
import time
import argparse
import logging
import subprocess
from pynvml import *  # 或 from nvml import *
from contextlib import contextmanager

from .data import RecentKCommon

logger = logging.getLogger(__name__)


def used_mem_gb(gpu_id):
    h = nvmlDeviceGetHandleByIndex(gpu_id)
    m = nvmlDeviceGetMemoryInfo(h)
    return m.used / 1024**3


def wait_gpu_run(
    CMD: list[str] | str,
    gpu_use_num: int = 1,
    gpu_ids: str = "0-7",
    threshold_gb: float = 5,
    check_interval: float = 5,
    confirm_times: int = 3,
):
    monitor_gpus = []
    gpu_ids = gpu_ids.split(",")
    for gpu_id in gpu_ids:
        if "-" in gpu_id:
            start, end = map(int, gpu_id.split("-"))
            monitor_gpus.extend(range(start, end + 1))
        else:
            monitor_gpus.append(int(gpu_id))

    logger.info(f"Monitor GPU ID: {monitor_gpus}")
    if isinstance(CMD, str):
        CMD = [CMD]

    logger.info(f"CMD wait to run: {' '.join(CMD)}")

    try:
        # 初始化
        nvmlInit()
        _nvml_initialized = True
    except NVMLError as e:
        _nvml_initialized = False

    # with nvml_context():
    if True:

        recent_k_common = RecentKCommon(confirm_times)

        while True:
            ok_gpus = [i for i in monitor_gpus if used_mem_gb(i) < threshold_gb]

            available_gpus = recent_k_common.push(ok_gpus)

            logger.info(f"Available GPU ID: {','.join(map(str, available_gpus))}")

            if len(available_gpus) >= gpu_use_num:
                use_gpus = list(available_gpus)[:gpu_use_num]
                use_gpus_str = ",".join(map(str, use_gpus))
                logger.info(
                    f"Available GPU ID: {','.join(map(str, available_gpus))}. Run on {use_gpus_str}"
                )
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = use_gpus_str

                if _nvml_initialized:
                    try:
                        nvmlShutdown()
                        _nvml_initialized = False  # 重置标志
                    except NVMLError as e:
                        pass

                subprocess.Popen(CMD, env=env)
                break
            else:
                print("")

            time.sleep(check_interval)


def _wait_gpu_run_handler(args):
    wait_gpu_run(
        args.cmd,
        args.gpu_use_num,
        args.gpu_ids,
        args.threshold_gb,
        args.check_interval,
        args.confirm_times,
    )


def _wait_gpu_run_register(parser: argparse._SubParsersAction):
    parser_wgr = parser.add_parser("launch", help="wait gpu run")
    parser_wgr.add_argument(
        "cmd",
        type=str,
        nargs="+",
        help="command to run",
    )
    parser_wgr.add_argument(
        "--gpu-use-num",
        '-k',
        type=int,
        default=1,
        help="number of gpu to use",
    )
    parser_wgr.add_argument(
        "--gpu-ids",
        "-g",
        type=str,
        default="0-7",
        help="gpu ids to use",
    )
    parser_wgr.add_argument(
        "--threshold-gb",
        "-s",
        type=float,
        default=5,
        help="threshold of gpu memory usage",
    )
    parser_wgr.add_argument(
        "--check-interval",
        "-i",
        type=float,
        default=5,
        help="interval of checking gpu memory usage",
    )
    parser_wgr.add_argument(
        "--confirm-times",
        "-t",
        type=int,
        default=3,
        help="number of confirm times",
    )
    parser_wgr.set_defaults(func=_wait_gpu_run_handler)
