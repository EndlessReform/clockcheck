from abc import ABC, abstractmethod
import numpy as np
from clockcheck.utils.config import ModelConfig


class TTSModel(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> "TTSModel":
        """
        Initialize a TTSModel instance from a ModelConfig BaseModel.

        Args:
            config: A ModelConfig instance containing model configuration.

        Returns:
            TTSModel: An initialized TTSModel instance.
        """
        pass

    @abstractmethod
    async def generate(self, text: str) -> np.ndarray:
        """
        Generate a PCM waveform at 24kHz from input text.

        Args:
            text (str): Input text to synthesize.

        Returns:
            np.ndarray: PCM waveform at 24kHz (dtype=np.float32 or np.int16).
        """
        pass
