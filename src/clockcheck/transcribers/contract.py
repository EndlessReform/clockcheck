from abc import ABC, abstractmethod
import numpy as np
from clockcheck.utils.config import ModelConfig


class Transcriber(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> "Transcriber":
        """
        Initialize a Transcriber instance from a ModelConfig BaseModel.

        Args:
            config: A ModelConfig instance containing model configuration.

        Returns:
            Transcriber: An initialized Transcriber instance.
        """
        pass

    @abstractmethod
    async def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a 24kHz waveform ndarray to text using OpenAI's API.

        Args:
            audio (np.ndarray): PCM waveform at 24kHz (dtype=np.float32 or np.int16).

        Returns:
            str: The transcribed text.
        """
        pass
