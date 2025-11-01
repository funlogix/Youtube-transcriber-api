# YouTube Transcription API

This Space provides a FastAPI-based endpoint to transcribe YouTube videos using OpenAI Whisper. It supports Whisper transcription and falls back to YouTube captions if needed.

## Endpoints

- `POST /transcribe` — Transcribes a YouTube video
- `GET /health` — Health check

## Environment Variables

Set these in the Hugging Face Space settings:

- `API_TOKEN` — Your secret token for authentication
- `WHISPER_MODEL` — Whisper model name (e.g., `tiny`, `base`)
