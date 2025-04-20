import os
import sys

sys.path.append("../src")
import phoenixcat
from pathlib import Path

from phoenixcat.files import DualFolderManager

src_dir = "../src"
dst_dir = "../test/copy_res"

dual_manager = DualFolderManager(src_dir, dst_dir)


def cocurrent_copy(manager: DualFolderManager):
    for item in manager.ls():
        if manager.is_file(item):
            if item.endswith(".py"):
                manager.copy(item)
        elif manager.is_dir(item):
            manager.cd(item)
            cocurrent_copy(manager)
            manager.parent()


cocurrent_copy(dual_manager)
