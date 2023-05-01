import logging
import os
from typing import Dict

import pandas as pd
import yaml
from google.cloud import storage


def load_config(config: str = "config.yaml") -> Dict:
    try:
        with open(config, "r") as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data
    except FileNotFoundError:
        logging.exception(f"Config file {config} not found. Check path provided.")
    except Exception as e:
        logging.exception(e)


def download_gcs_file(gcs_path: str) -> None:
    try:
        bucket_name = gcs_path.split("/")[2]
        file_path = "/".join(gcs_path.split("/")[3:])
        dest_file = os.path.basename(file_path)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        logging.info(f"Downloading file from {gcs_path} to {dest_file}")
        blob.download_to_filename(dest_file)
        logging.info(f"Successfuly downloaded file to {dest_file}")
    except Exception as e:
        logging.exception(e)


def gcs_to_df(gcs_path: str) -> pd.DataFrame:
    local_path = os.path.basename(gcs_path)
    if not os.path.exists(local_path):
        logging.info(f"File not found locally. Downloading {gcs_path}")
        download_gcs_file(gcs_path)
    try:
        logging.info(f"Loading {local_path}")
        df = pd.read_csv(local_path)
        return df
    except Exception as e:
        logging.exception(e)
