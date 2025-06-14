from pydantic import BaseModel
from typing import Literal, Optional
import tomllib


class ModelConfig(BaseModel):
    model_type: Literal["huggingface", "openai"]
    base_url: Optional[str] = None
    """
    Defaults to first-party OpenAI endpoint
    """
    model_id: Optional[str]
    voice: Optional[str]
    requests_per_minute: int = -1
    """
    If not set, generation will be in serial
    """


class TranscriptionConfig(BaseModel):
    model_type: Literal["openai", "whisper-cpp"]
    base_url: Optional[str] = None
    """
    Defaults to first-party OpenAI endpoint
    """
    model_id: Optional[str]
    requests_per_second: int = -1
    """
    If not set, generation will be in serial
    """


class Config(BaseModel):
    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None
    """
    Can be HuggingFace dataset ID or local path
    """
    output_dataset_path: Optional[str] = "./datasets/dataset_pred"

    model: ModelConfig
    transcriber: TranscriptionConfig

    @classmethod
    def from_toml(cls, file_path: str) -> "Config":
        """
        Initialize Config from a TOML file.

        Args:
            file_path: Path to the TOML configuration file

        Returns:
            Config object initialized with values from the TOML file
        """
        with open(file_path, "rb") as f:
            config_dict = tomllib.load(f)

        if config_dict.get("dataset_id") and config_dict.get("dataset_path"):
            raise ValueError(
                "Only one of 'dataset_id' or 'dataset_path' may be specified, not both."
            )
        if not config_dict.get("dataset_id") and not config_dict.get("dataset_path"):
            raise ValueError("One of 'dataset_id' or 'dataset_path' must be specified.")

        return cls(**config_dict)
