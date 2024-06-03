from utilsbox.download.hugging_face.api_download import download_from_huggingface
from utilsbox.logger.logging import init_logger

init_logger("dowanload_from_huggingface.log")

# replace it with your url
url="https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main"

# replace it with your dir
local_dir = ".download"

# set to `True` to make it faster if you are 🇨🇳 Chinese Mainland user
use_mirror = True

# to unenable overwrite downloaded files
continue_download = True

download_from_huggingface(url=url, use_mirror=use_mirror, local_path=local_dir, continue_download=continue_download)