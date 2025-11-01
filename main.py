from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import yt_dlp
import whisper
import os
import uuid
import logging
import datetime
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re

# --- Config ---
API_TOKEN = os.getenv("API_TOKEN", "your-secret-token")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- App ---
app = FastAPI()

# --- Models ---
class VideoRequest(BaseModel):
    video_url: str

# --- Auth ---
def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")

# --- Time Formatter ---
def format_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))

# --- Health Check ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- Transcription Endpoint ---
def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

@app.post("/transcribe")
def transcribe_video(req: VideoRequest, _: str = Depends(verify_token)):
    temp_id = str(uuid.uuid4())
    video_path = f"{temp_id}.mp4"

    try:
        logging.info(f"Transcription request for: {req.video_url}")
        model = whisper.load_model(WHISPER_MODEL)

        yt_opts = {
            'format': 'mp4',
            'outtmpl': video_path,
            'socket_timeout': 30,
        }

        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download([req.video_url])
        logging.info("Video downloaded")

        result = model.transcribe(video_path)
        os.remove(video_path)
        logging.info("Whisper transcription complete")

        segments = [
            {
                "start": format_time(seg["start"]),
                "end": format_time(seg["end"]),
                "text": seg["text"].strip()
            }
            for seg in result.get("segments", [])
        ]

        return {
            "source": "whisper",
            "transcript": result["text"],
            "segments": segments
        }

    except Exception as whisper_error:
        logging.warning(f"Whisper failed: {str(whisper_error)}")
        try:
            video_id = extract_video_id(req.video_url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            segments = [
                {
                    "start": format_time(seg["start"]),
                    "end": format_time(seg["start"] + seg["duration"]),
                    "text": seg["text"].strip()
                }
                for seg in transcript
            ]
            full_text = " ".join(seg["text"] for seg in transcript)
            logging.info("Fallback to YouTube captions succeeded")

            return {
                "source": "youtube_captions",
                "transcript": full_text,
                "segments": segments
            }

        except (TranscriptsDisabled, NoTranscriptFound, Exception) as caption_error:
            logging.error(f"Fallback failed: {str(caption_error)}")
            raise HTTPException(status_code=500, detail="Transcription failed: Whisper and YouTube captions unavailable")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860)) # 7860 for Hugging Face, 10000 for Render
    uvicorn.run(app, host="0.0.0.0", port=port)