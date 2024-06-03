import logging
import os
import requests
import time

logger = logging.getLogger(__name__)


def download_file(
    url: str,
    local_path: os.PathLike,
    stream: bool = True,
    mkdirs: bool = True,
    chunk_size: int = 8192,
    retry: int = 10,
    wait: float = 2.0
):
    dirname = os.path.dirname(local_path)
    if mkdirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    for i in range(retry):
        try:
            with requests.get(url, stream=stream) as response:
                with open(local_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        file.write(chunk)
            logger.info(f"Download {local_path} from {url}.")
            return True
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {str(e)}, retry {i + 1} / {retry} after {wait} sec.")
            time.sleep(wait)
    logger.error(f"Failed to download from {url}.")
    if os.path.exists(local_path):
        os.remove(local_path)