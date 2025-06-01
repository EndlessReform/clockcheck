import argparse
import os
import toml
from datasets import load_dataset, load_from_disk

from clockcheck.utils.config import Config
import clockcheck.models as models


def main():
    parser = argparse.ArgumentParser(description="Clockcheck harness")
    parser.add_argument(
        "--config",
        default="utils/config.toml",
        help="Path to configuration file (default: utils/config.toml)",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    try:
        with open(config_path, "r") as f:
            config_dict = toml.load(f)
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML configuration file: {e}")

    config = Config(**config_dict)

    dataset = (
        load_dataset(config.dataset_id)
        if config.dataset_id
        else load_from_disk(config.dataset_path)
    )
    tts_model = models.from_config(config.model)


if __name__ == "__main__":
    main()
