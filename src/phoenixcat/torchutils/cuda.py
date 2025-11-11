import os
import logging
import subprocess

import torch

logger = logging.getLogger(__name__)


def _pick_gpu_by_torch() -> int | None:
    """
    Pick the GPU with the most free memory using torch.cuda.mem_get_info().
    Returns the GPU index; returns None on failure.
    """
    if not torch.cuda.is_available():
        return None
    try:
        best_i, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()  # bytes
            if free > best_free:
                best_free, best_i = free, i
        return best_i
    except Exception:
        return None


def _pick_gpu_by_nvidia_smi() -> int | None:
    """
    Fallback: call nvidia-smi to read each GPU's free memory (MiB) and pick the largest.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        frees = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        if not frees:
            return None
        # all_indices = list(range(len(frees)))
        if os.environ["CUDA_VISIBLE_DEVICES"]:
            frees = [
                frees[int(x)] for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            ]
        return max(list(range(len(frees))), key=lambda i: frees[i])
    except Exception:
        return None


def pick_best_gpu_index() -> int | None:
    """
    Combined strategy: prefer torch-based picking; fall back to nvidia-smi.
    """
    idx = _pick_gpu_by_torch()
    if idx is not None:
        return idx
    return _pick_gpu_by_nvidia_smi()


def auto_select_one_device():
    if torch.cuda.is_available():
        best_idx = pick_best_gpu_index()
        if best_idx is not None:
            try:
                torch.cuda.set_device(best_idx)
            except Exception:
                pass  # if set_device fails, fall back to the default CUDA device
            device = torch.device(f"cuda:{best_idx}")
            try:
                with torch.cuda.device(device):
                    free, total = torch.cuda.mem_get_info()
                logger.info(
                    f"[GPU] Auto-picked cuda:{best_idx} "
                    f"(free {free/1024/1024/1024:.1f} GB / total {total/1024/1024/1024:.1f} GB)"
                )
            except Exception:
                logger.info(f"[GPU] Auto-picked cuda:{best_idx}")
        else:
            device = torch.device("cuda")
            logger.info("[GPU] Auto-pick failed, fallback to default CUDA device.")
    else:
        device = torch.device("cpu")
        logger.info("[GPU] CUDA not available, using CPU.")
