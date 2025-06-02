import argparse
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
import os

from clockcheck.utils.config import Config
import clockcheck.models as models
import clockcheck.transcribers as transcribers


async def main():
    parser = argparse.ArgumentParser(description="Clockcheck harness")
    parser.add_argument(
        "--config",
        default="utils/config.toml",
        help="Path to configuration file (default: utils/config.toml)",
    )
    args = parser.parse_args()

    load_dotenv()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    try:
        print(f"Loading config from {config_path}")
        config = Config.from_toml(config_path)
    except ValueError as e:
        raise ValueError(f"Invalid TOML configuration file: {e}")

    print(f"Config: {config}")

    dataset = (
        load_dataset(config.dataset_id)
        if config.dataset_id
        else load_from_disk(config.dataset_path)
    )
    tts_model = models.from_config(config.model)
    transcriber = transcribers.from_config(config.transcriber)

    ds_pred = await models.run_ds(dataset, tts_model, config.model)
    # ds_pred = load_from_disk("./datasets/dataset_oai_coral_0601")

    ds_pred = await transcribers.run_ds(ds_pred, transcriber, config.transcriber)
    # TODO proper error handling, file location
    # TODO add metadata
    ds_pred.save_to_disk("./datasets/dataset_oai_coral")
    print("Saved to ./datasets/dataset_oai_coral")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
