import logging
import sys

from model import batch_predict, prepare_data, train
from utils import gcs_to_df, load_config


def set_logger() -> None:
    file_handler = logging.FileHandler(filename="test-vm02.log")
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler, stream_handler],
    )


if __name__ == "__main__":
    set_logger()
    logging.info("Starting training and inferencing")
    config = load_config()
    data_cfg = config["data"]["dataset_config"]
    dataset = gcs_to_df(config["data"]["gcp_filepath"])
    train_ds, val_ds, test_ds = prepare_data(
        df=dataset,
        target_col=data_cfg["target"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
    )
    model_cfg = config["model"]
    model_data_cfg = config["data"]["model_data"]
    train(train_ds, val_ds, test_ds, model_data_cfg, model_cfg)
    test_data_cfg = config["data"]["test_data"]
    batch_predict(dataset, data_cfg["target"], model_data_cfg, test_data_cfg)
