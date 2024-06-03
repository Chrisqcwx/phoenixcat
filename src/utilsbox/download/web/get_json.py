import json
import logging
import os
import requests
import time

logger = logging.getLogger(__name__)


def get_json(
    url: str,
    local_file: os.PathLike = None,
    mkdirs: bool = True,
    retry: int = 10,
    wait: float = 2.0
):
    """Fetch JSON data from a given URL.

    Args:
        url (str): The URL from which to download the JSON data.
        local_file (os.PathLike, optional): Path to the file where the downloaded JSON data should be saved.
            If None, the data is not saved to a file. Defaults to None.
        mkdirs (bool, optional): If True, creates the directory path for the local_file if it does not already exist.
            Defaults to True.
        retry (int, optional): Number of retry attempts to download the data in case of failure. Defaults to 10.
        wait (float, optional): Time in seconds to wait between retry attempts. Defaults to 2.0 seconds.

    Returns:
        dict: A dictionary containing the JSON data if the download was successful. None if not successful.
    """
    for i in range(retry):
        try:
            response = requests.get(url)
        except Exception as e:
            response = None
            logger.warning(f"Failed to download from {url}: {str(e)}, retry {i + 1} / {retry} after {wait} sec.")
            time.sleep(wait)
    
    if response is not None:
        data = json.loads(response.text)
        logger.info(f"Get json data from {url}.")
        if local_file is not None:
            dirname = os.path.dirname(local_file)
            if mkdirs and dirname != "":
                os.makedirs(dirname, exist_ok=True)
            with open(local_file, "w") as file:
                json.dump(data, file, indent=4)
            logger.debug(f"Save {local_file}.")
        return data
    else:
        logger.error(f"Failed to get data from {url}.")
    