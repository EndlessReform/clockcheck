# ClockCheck

ClockCheck is a simple benchmark for testing TTS WER (Word Error Rate) when reading an un-normalized time of day (ex. "1:23 AM"). Surprisingly, this is a hard problem for open source TTS models.

## Usage

### With Whisper C++

Start server:

```bash
./build/bin/whisper-server -m ./models/ggml-large-v3-turbo-q5_0.bin --host 0.0.0.0 --port 5000
```

Transcribe:

```bash
uv run src/clockcheck/harness.py --config ./config/config_whisper.toml
```


