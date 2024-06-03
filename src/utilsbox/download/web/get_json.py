import json
import logging
import os
import requests
import time

logger = logging.getLogger(__name__)


def get_json(
    url,
    local_file: os.PathLike = None,
    mkdirs: bool = True,
    retry: int = 10,
    wait: float = 2.0
):
    
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
    