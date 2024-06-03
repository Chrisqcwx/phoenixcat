import logging
import os

from tqdm.auto import tqdm

from ..web.get_file import download_file
from ..web.get_json import get_json
from ..web.url_split import url_spilt
from ...check.check_args import only_one_given

logger = logging.getLogger(__name__)

def download_one_dir_from_huggingface(
    repo_id: str,
    repo_type: str,
    path: str,
    local_path: os.PathLike,
    mkdirs: bool = True,
    root_url: str = "https://huggingface.co",
    retry: int = 10,
    wait: float = 1.0,
    use_progress_bar: bool = True,
    overwrite: bool = False,
    chunk_size: int = 8192
):
    dirname = os.path.dirname(local_path)
    if mkdirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    
    _all = get_json(f"{root_url}/api/{repo_type}/{repo_id}/tree/main/{path}", retry=retry, wait=wait)
    
    if use_progress_bar:
        progress_bar = tqdm(range(len(_all)), desc=f"Downloading: {repo_id}/tree/main/{path}")
    
    for item in _all:
        if item.get("type") == "directory":
            download_one_dir_from_huggingface(
                repo_id,
                repo_type,
                item.get("path"),
                os.path.join(local_path, item.get('path')),
                mkdirs,
                root_url,
                retry,
                wait,
                use_progress_bar,
                overwrite
            )
        elif item.get("type", None) == "file":
            file_name = os.path.join(local_path, item.get('path'))
            if (not overwrite) and (os.path.exists(file_name)):
                logger.info(f"File {file_name} already exists, skipping download.")
                continue
            download_file(
                f"{root_url}/{repo_type}/{repo_id}/tree/main/{item.get('path')}",
                file_name,
                mkdirs,
                chunk_size=chunk_size,
                retry=retry,
                wait=wait
            )
        progress_bar.update(1)


def download_from_huggingface(
    url: str = None,
    repo_id: str = None,
    repo_type: str = None,
    use_mirror: bool = True,
    local_path: os.PathLike = ".download",
    retry: int = 20,
    wait: float = 5.0,
    chunk_size: int = 8192,
    continue_download: bool = True
):
    if not only_one_given(url, repo_id):
        logger.error(f"Both `url` and `repo_id` are passed in.")
        raise ValueError(f"Both `url` and `repo_id` are passed in.")
    
    if url is not None:
        root_url, repo_type, author, repo_name, *_ = url_spilt(url)
        repo_id = f"{author}/{repo_name}"
    
    if use_mirror:
        root_url = "https://hf-mirror.com"
    elif url is None:
        root_url = "https://huggingface.co"
    
    download_one_dir_from_huggingface(
        repo_id,
        repo_type,
        "",
        local_path,
        True,
        root_url,
        retry=retry,
        wait=wait,
        chunk_size=chunk_size,
        overwrite=not continue_download
    )