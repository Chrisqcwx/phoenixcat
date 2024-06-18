import logging
import os
import time

import boto3
from botocore.exceptions import ClientError
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

USER_CONFIG = dict(
    aws_access_key_id = "AKIAXO65CT57HVJ72JHL",
    aws_secret_access_key = "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",
    download_path = ".temp",
    selected_files = [
        "MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas.dtseries.nii"
    ],
    retries = 10,
    wait = 20
)

CONSTANT = dict(
    bucket_name = "hcp-openaccess",
    prefix = "HCP_1200",
)


def is_s3_file_exist(s3, source_path):
    try:
        s3.Object(CONSTANT["bucket_name"], source_path).load()
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.info(f"No such file: {source_path}")
            return False
        raise ClientError(e)
    
def download_one_file(s3, bucket, source_path, retries, wait):
    local_path = os.path.join(USER_CONFIG["download_path"], source_path)
    if os.path.isfile(local_path):
        return True
    for i in range(retries):
        try:
            if is_s3_file_exist(s3, source_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                bucket.download_file(source_path, local_path)
            return True
        except:
            logger.warning(f"Failed at {source_path}, retry: {i} / {retries}")
            time.sleep(wait)
    return False
                
def download_one_subj(s3, bucket, subj):
    for file in USER_CONFIG["selected_files"]:
        source_path = f"{CONSTANT['prefix']}/{subj}/{file.removeprefix('/')}"
        download_one_file(s3, bucket, source_path, retries=USER_CONFIG["retries"], wait=USER_CONFIG["wait"])

def download_all_subjects(s3, bucket):
    subjects_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subjects.txt")
    with open(subjects_file, "r") as f:
        num_subjs = 0
        for subj in f:
            num_subjs += 1
    num_files = num_subjs * len(USER_CONFIG["selected_files"])
    process_bar = tqdm(range(num_files), desc="Downloading from HCP 1200")
    with open(subjects_file, "r") as f:
        for subj in f:
            subj = subj.strip()
            download_one_subj(s3, bucket, subj)
            process_bar.update(1)

def main():
    boto3.setup_default_session(
        aws_access_key_id = USER_CONFIG["aws_access_key_id"],
        aws_secret_access_key = USER_CONFIG["aws_secret_access_key"]
    )

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(CONSTANT["bucket_name"])
    
    download_all_subjects(s3, bucket)