from pydantic import BaseModel
from typing import Literal, Optional
import tomllib


class ModelConfig(BaseModel):
    model_type: Literal["huggingface", "openai"]
    model_endpoint: Optional[str]
    model_id: Optional[str]
    voice: Optional[str]


class Config(BaseModel):
    dataset_id: str
    """
    Can be HuggingFace dataset ID or local path
    """

    model: ModelConfig
    # TODO add transcription model config

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
        return cls(**config_dict)
