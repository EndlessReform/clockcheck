import asyncio
from aiolimiter import AsyncLimiter
from datasets import Dataset
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, Type

from clockcheck.transcribers.contract import Transcriber
from clockcheck.utils.config import ModelConfig
from .openai_asr import OpenAITranscriber

# Map transcriber types to their implementation classes
_TRANSCRIBER_CLASSES: Dict[str, Type[Transcriber]] = {
    "openai": OpenAITranscriber,
}


def from_config(config: ModelConfig) -> Transcriber:
    """Create a Transcriber from configuration.

    Args:
        config: Model configuration containing the model type and parameters.

    Returns:
        An initialized Transcriber instance.

    Raises:
        ValueError: If the transcriber type is not found.
    """
    transcriber_class = _TRANSCRIBER_CLASSES.get(config.model_type)
    if not transcriber_class:
        raise ValueError(f"Unknown transcriber type: {config.model_type}")
    return transcriber_class.from_config(config)


async def run_ds(ds: Dataset, transcriber: Transcriber, config: ModelConfig) -> Dataset:
    """Run transcription on a dataset using the given transcriber and config.

    Args:
        ds: Hugging Face Dataset containing audio data (expects 'audio' field).
        transcriber: Transcriber instance.
        config: ModelConfig with rate limiting info.

    Returns:
        Dataset with an added 'text' field containing transcriptions.
    """
    actual_limiter: asyncio.Semaphore | AsyncLimiter

    if getattr(config, "requests_per_minute", -1) == -1:
        actual_limiter = asyncio.Semaphore(1)
    else:
        if getattr(config, "requests_per_minute", 0) <= 0:
            raise ValueError(
                "config.requests_per_minute must be positive for rate limiting, or -1 for sequential."
            )
        actual_limiter = AsyncLimiter(config.requests_per_minute, 60.0)

    async def process_item(item_data: dict) -> dict | None:
        async with actual_limiter:
            try:
                audio_data = item_data.get("audio")
                if audio_data is None:
                    print(f"Skipping item due to missing 'audio' field: {item_data}")
                    return None
                text = await transcriber.transcribe(audio_data)
                return {**item_data, "transcribed_text": text}
            except Exception as e:
                print(f"Failed to transcribe audio: {e}")
                return None

    jobs = [process_item(item) for item in ds]
    if not jobs:
        print("Warning: Input dataset is empty.")
        return Dataset.from_list([])

    results = await tqdm_asyncio.gather(*jobs, desc="Transcribing audio")
    successful_results = [r for r in results if r is not None]

    if not successful_results and len(jobs) > 0:
        print("Warning: All transcription jobs failed or returned None.")

    return Dataset.from_list(successful_results)


__all__ = ["Transcriber", "from_config", "run_ds"]
