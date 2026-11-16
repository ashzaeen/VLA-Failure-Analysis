import logging
import logging.config
import os
import sys
from datetime import date
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))


def setup_logging(yaml_path=None):
    if yaml_path is None:
        yaml_path = os.path.join(project_root, "configs","logging.yaml")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    today_str = date.today().isoformat()
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{today_str}.log")

    if "handlers" in config and "file" in config["handlers"]:
        config["handlers"]["file"]["filename"] = log_file

    logging.config.dictConfig(config)

def get_logger(name="my_logger"):
    return logging.getLogger(name)
