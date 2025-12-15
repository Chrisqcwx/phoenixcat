import time
import logging
import subprocess
from pynvml import *  # 或 from nvml import *
from contextlib import contextmanager

from .data import RecentKCommon

logger = logging.getLogger(__name__)

# 定义全局标志
# _nvml_initialized = False


# @contextmanager
# def nvml_context():
#     """
#     NVML 上下文管理器，确保 Init 和 Shutdown 成对执行。
#     """
#     global _nvml_initialized

#     if _nvml_initialized:
#         # 如果已经初始化，直接 yield，不重复执行 nvmlInit
#         print("NVML 库已被初始化，进入上下文。")
#         yield
#         return

#     print("正在初始化 NVML 库...")
#     try:
#         # 初始化
#         nvmlInit()
#         _nvml_initialized = True
#         print("NVML 库初始化成功。")

#         # 允许程序块运行
#         yield

#     except NVMLError as e:
#         print(f"NVML 初始化失败: {e}")
#         raise  # 初始化失败，退出上下文

#     finally:
#         # 清理操作（无论 try 块是否成功都会执行）
#         if _nvml_initialized:
#             try:
#                 nvmlShutdown()
#                 _nvml_initialized = False  # 重置标志
#                 print("NVML 库已关闭。")
#             except NVMLError as e:
#                 print(f"NVML 关闭失败: {e}")


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
