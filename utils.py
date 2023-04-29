import logging
import os
import pandas as pd
import yaml
from google.cloud import storage

def load_config(config="config.yaml"):
    try:
        with open(config, "r") as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data
    except FileNotFoundError:
        logging.exception(f"Config file {config} not found. Check path provided.")
    except Exception as e:
        logging.exception(e)

# add try catch
def download_gcs_file(gcs_path):
    bucket_name = gcs_path.split("/")[2]
    file_path = "/".join(gcs_path.split("/")[3:])
    dest_file = os.path.basename(file_path)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    logging.info(f"Downloading file from {gcs_path} to {dest_file}")
    blob.download_to_filename(dest_file)

# add try catch
def gcs_to_df(gcs_path):
    local_path = os.path.basename(gcs_path)
    if not os.path.exists(local_path):
        download_gcs_file(gcs_path)
    return pd.read_csv(local_path)
