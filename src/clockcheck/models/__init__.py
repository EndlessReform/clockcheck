from typing import Dict, Type

from clockcheck.models.contract import TTSModel
from clockcheck.utils.config import ModelConfig
from .openai_tts import OpenAITTSModel

# Map model types to their implementation classes
_MODEL_CLASSES: Dict[str, Type[TTSModel]] = {
    "openai_tts": OpenAITTSModel,
}


def from_config(config: ModelConfig) -> TTSModel:
    """Create a TTS model from configuration.

    Args:
        config: Model configuration containing the model type and parameters.

    Returns:
        An initialized TTS model instance.

    Raises:
        ValueError: If the model type is not found.
    """
    model_class = _MODEL_CLASSES.get(config.model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {config.model_type}")
    return model_class.from_config(config)


__all__ = ["TTSModel", "from_config"]
