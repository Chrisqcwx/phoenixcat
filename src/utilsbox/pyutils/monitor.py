import psutil
from collections import OrderedDict

from .format import format_as_yaml, format_number, format_now_time


def cpu_memory_infos():

    time_str = format_now_time()
    cpu_infos = OrderedDict(precent=psutil.cpu_percent(percpu=False))

    memory_infos_dict = psutil.virtual_memory()._asdict()

    memory_infos = OrderedDict(percent=memory_infos_dict.pop('percent'))

    for k, v in memory_infos_dict.items():
        # print(f'{k}: ')
        memory_infos[k] = f'{format_number(v)}B'

    final_infos = OrderedDict(time=time_str, cpu=cpu_infos, memory=memory_infos)
    return final_infos
