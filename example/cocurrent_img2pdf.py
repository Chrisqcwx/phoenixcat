import os
import sys
import img2pdf

sys.path.append("../src")
import phoenixcat

from phoenixcat.files.walk import IMG_EXTENSIONS
from phoenixcat.files import DualFolderManager

src_dir = "./celeba"
dst_dir = "./celeba_pdf"

dual_manager = DualFolderManager(src_dir, dst_dir)


def cocurrent_img2pdf(manager: DualFolderManager):
    for item in manager.ls():
        if manager.is_file(item):
            if item.endswith(IMG_EXTENSIONS):
                src_path, dst_path = manager.get_target_path(item)
                src_path = str(src_path)
                dst_path = str(dst_path).rsplit(".", 1)[0] + ".pdf"
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                with open(dst_path, "wb") as f:
                    f.write(img2pdf.convert(src_path))
        elif manager.is_dir(item):
            manager.cd(item)
            cocurrent_img2pdf(manager)
            manager.parent()


cocurrent_img2pdf(dual_manager)