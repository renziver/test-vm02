import logging
from utils import gcs_to_df, load_config, download_gcs_file

def set_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
if __name__ == "__main__":
    set_logger()
    config = load_config()
    gcs_to_df(config["task_1"]["gcp_filepath"])

