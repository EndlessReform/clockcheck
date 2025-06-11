from io import BytesIO
from openai import AsyncOpenAI
import numpy as np
import soundfile as sf
from typing import Optional

from clockcheck.transcribers.contract import Transcriber
from clockcheck.utils.config import TranscriptionConfig


class OpenAITranscriber(Transcriber):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini-transcribe",
        endpoint: Optional[str] = None,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
        self.model = model

    @classmethod
    def from_config(cls, config: TranscriptionConfig) -> "OpenAITranscriber":
        api_key = config.api_key if hasattr(config, "api_key") else None
        model = (
            config.model_id
            if getattr(config, "model_id", None)
            else "gpt-4o-mini-transcribe"
        )
        endpoint = getattr(config, "base_url", None)
        return cls(api_key=api_key, model=model, endpoint=endpoint)

    async def transcribe(self, audio: np.ndarray) -> str:
        pseudo_file = BytesIO()
        sf.write(pseudo_file, audio, 24000, format="WAV")
        pseudo_file.seek(0)
        pseudo_file.name = "speech.wav"
        response = await self.client.audio.transcriptions.create(
            model=self.model, file=pseudo_file
        )
        return response.text
