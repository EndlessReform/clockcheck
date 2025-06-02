import asyncio
from aiolimiter import AsyncLimiter
from datasets import Dataset
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, Type

from clockcheck.models.contract import TTSModel
from clockcheck.utils.config import ModelConfig
from .openai_tts import OpenAITTSModel

# Map model types to their implementation classes
_MODEL_CLASSES: Dict[str, Type[TTSModel]] = {
    "openai": OpenAITTSModel,
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


async def run_ds(ds: Dataset, model: TTSModel, config: ModelConfig) -> Dataset:
    actual_limiter: asyncio.Semaphore | AsyncLimiter  # Type hint for clarity

    if config.requests_per_minute == -1:
        # Path 1: Sequential processing (one at a time, as fast as possible)
        # Use asyncio.Semaphore(1) for this.
        actual_limiter = asyncio.Semaphore(1)
        # print("Using asyncio.Semaphore(1) for sequential processing.") # Optional: for debugging
    else:
        # Path 2: Rate-limited concurrent processing
        if config.requests_per_minute <= 0:
            raise ValueError(
                "config.requests_per_minute must be positive for rate limiting, or -1 for sequential."
            )
        # aiolimiter.AsyncLimiter(max_rate, period_seconds)
        # If config.requests_per_minute is, e.g., 60 (meaning 60 per minute),
        # then max_rate is 60, and period is 60.0 seconds.
        actual_limiter = AsyncLimiter(config.requests_per_minute, 60.0)
        # print(f"Using aiolimiter.AsyncLimiter({config.requests_per_minute}, 60.0) for rate-limited processing.") # Optional

    # This inner function will perform the generation for a single item,
    # respecting the 'actual_limiter' defined above.
    async def process_item(item_data: dict) -> dict | None:
        # 'actual_limiter' is captured from the outer scope.
        # Both asyncio.Semaphore and aiolimiter.AsyncLimiter support 'async with'.
        async with actual_limiter:
            try:
                text_to_process = item_data.get("text")
                if text_to_process is None:
                    print(f"Skipping item due to missing 'text' field: {item_data}")
                    return None  # Ensure this item is filtered out

                # Call the model's async generate method
                audio_output = await model.generate(text_to_process)

                # Return a new dictionary with the original item data plus the audio
                return {**item_data, "audio": audio_output}
            except Exception as e:
                # Log the error and the text that failed, if possible
                failed_text = item_data.get("text", "UNKNOWN_TEXT_IN_ITEM")
                print(f"Failed to generate audio for text '{failed_text}': {e}")
                return None  # Mark as None to filter out later

    # Create a list of coroutine jobs by applying process_item to each item in the dataset
    # ds is assumed to be an iterable (e.g., Hugging Face Dataset)
    jobs = [process_item(item) for item in ds]

    if not jobs:
        print("Warning: Input dataset is empty.")
        return Dataset.from_list([])  # Return an empty dataset

    # Execute all jobs, with progress bar via tqdm_asyncio
    # The concurrency/rate is managed by the 'actual_limiter' inside each 'process_item' call.
    results = await tqdm_asyncio.gather(*jobs, desc="Generating audio")

    # Filter out failed jobs (which returned None)
    successful_results = [r for r in results if r is not None]

    if not successful_results and len(jobs) > 0:
        print("Warning: All audio generation jobs failed or returned None.")

    # Reconstruct the dataset from the successful results
    return Dataset.from_list(successful_results)


__all__ = ["TTSModel", "from_config"]
