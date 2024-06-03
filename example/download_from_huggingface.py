from utilsbox.download.hugging_face.api_download import download_from_huggingface

# replace it with your url
url="https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main"

# replace it with your dir
local_dir = ".download"

# set to `True` to make it faster if you are 🇨🇳 Chinese Mainland user
use_mirror = False

# to unenable overwrite downloaded files
continue_download = True

download_from_huggingface(url=url, use_mirror=use_mirror, local_path=local_dir, continue_download=continue_download)