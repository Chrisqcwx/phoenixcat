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
    """Downloads a file from a specified URL and saves it to a local path.

    Args:
        url (str): The URL from which to download the file.
        local_path (os.PathLike): The local file system path where the file should be saved.
        stream (bool, optional): Whether to stream the download. Default is True, recommended for large files.
            Defaults to True.
        mkdirs (bool, optional): If True, create the directory for `local_path` if it doesn't exist. 
            Defaults to True.
        chunk_size (int, optional): Size of chunks to download at a time in bytes. Defaults to 8192.
        retry (int, optional): Number of times to retry the download in case of failure. Defaults to 10.
        wait (float, optional): Time to wait between retries in seconds. Defaults to 2.0.

    Returns:
        bool: True if the file was successfully downloaded and saved, False otherwise.
    """
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