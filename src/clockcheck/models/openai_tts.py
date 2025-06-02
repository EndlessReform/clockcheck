import io
import librosa
import numpy as np
from openai import AsyncOpenAI
import soundfile as sf
from typing import Optional

from clockcheck.models.contract import TTSModel
from clockcheck.utils.config import ModelConfig


class OpenAITTSModel(TTSModel):
    def __init__(
        self,
        api_key: str,
        model: str = "tts-1",
        endpoint: Optional[str] = None,
        voice: str = "nova",
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
        self.model = model
        self.voice = voice

    @classmethod
    def from_config(cls, config: ModelConfig) -> "OpenAITTSModel":
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not config.base_url:
            raise ValueError(
                "OpenAI API key not found in environment, but you're hitting OpenAI's first-party API."
            )
        model = config.model_id or "tts-1"
        voice = config.voice or "nova"
        return cls(api_key=api_key, model=model, endpoint=config.base_url, voice=voice)

    async def generate(self, text: str) -> np.ndarray:
        response = await self.client.audio.speech.create(
            model=self.model, voice=self.voice, input=text, response_format="wav"
        )
        # The response is an MP3 file. We need to decode to PCM 24kHz
        audio_bytes = response.content
        with io.BytesIO(audio_bytes) as buf:
            buf.seek(0)
            data, samplerate = sf.read(buf, dtype="float32")
        # Resample if necessary
        if samplerate != 24000:
            data = librosa.resample(data.T, orig_sr=samplerate, target_sr=24000)
        return data.astype(np.float32)
