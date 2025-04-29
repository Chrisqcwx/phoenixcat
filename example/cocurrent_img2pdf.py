import os
import sys
import img2pdf

sys.path.append("../src")
import phoenixcat

from phoenixcat.files.walk import IMG_EXTENSIONS
from phoenixcat.files import DualFolderManager

src_dir = "./test_celeba"
dst_dir = "./test_celeba_pdf"

dual_manager = DualFolderManager(src_dir, dst_dir)


def cocurrent_img2pdf(manager: DualFolderManager):
    for item in manager.ls():
        if manager.is_file(item):
            if item.endswith(IMG_EXTENSIONS):
                with manager.open(item, 'rb', 'wb', write_extension='pdf') as (fr, fw):
                    fw.write(img2pdf.convert(fr))
        elif manager.is_dir(item):
            manager.cd(item)
            cocurrent_img2pdf(manager)
            manager.parent()


cocurrent_img2pdf(dual_manager)