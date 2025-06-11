import asyncio
from io import BytesIO
import httpx
import numpy as np
import soundfile as sf
import uuid
from typing import Optional

from clockcheck.transcribers.contract import Transcriber
from clockcheck.utils.config import TranscriptionConfig


class WhisperCppTranscriber(Transcriber):
    def __init__(
        self,
        model: str = "whisper-1",
        endpoint: Optional[str] = None,
    ):
        self.client = httpx.AsyncClient()
        self.model = model

    @classmethod
    def from_config(cls, config: TranscriptionConfig) -> "WhisperCppTranscriber":
        model = config.model_id if getattr(config, "model_id", None) else "whisper-1"
        endpoint = getattr(config, "base_url", None)
        return cls(model=model, endpoint=endpoint)

    async def transcribe(self, audio: np.ndarray) -> str:
        pseudo_file = BytesIO()
        import librosa

        if hasattr(audio, "shape") and audio.shape[-1] > 0:
            audio = librosa.resample(audio, orig_sr=24000, target_sr=16000)

        sf.write(pseudo_file, audio, 16000, format="WAV")
        pseudo_file.seek(0)
        pseudo_file.name = f"{uuid.uuid4().hex}.wav"

        endpoint = getattr(self, "endpoint", None) or "http://127.0.0.1:5000"
        url = f"{endpoint.rstrip('/')}/inference"

        files = {
            "file": (pseudo_file.name, pseudo_file, "audio/wav"),
        }
        data = {
            "temperature": "0.5",
            "response_format": "json",
        }

        max_attempts = 4
        delay = 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.client.post(url, files=files, data=data)
                response.raise_for_status()
                result = response.json()
                return result.get("text", "")
            except (httpx.HTTPError, Exception) as e:
                if attempt == max_attempts:
                    print(f"Failed to transcribe audio: {e}")
                    return None
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
