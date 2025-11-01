from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import yt_dlp
import os
import uuid
import logging
import datetime
import re
import subprocess
import requests
import time
from threading import Lock
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# --- Config ---
API_TOKEN = os.getenv("API_TOKEN", "your-secret-token")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key")
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MAX_FILE_MB = 25  # Per-file limit from Groq

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- FastAPI App ---
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

# --- Helpers ---
def format_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def convert_to_mp3(video_path: str) -> str:
    mp3_path = video_path.replace(".mp4", ".mp3")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-q:a", "4", mp3_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
        raise RuntimeError("MP3 conversion failed")
    return mp3_path

def get_audio_duration(mp3_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", mp3_path],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0

# --- Groq Rate Limit Tracker ---
GROQ_LIMITS = {
    "rpm": 20,
    "rpd": 2000,
    "ash": 7200,   # audio seconds/hour
    "asd": 28800,  # audio seconds/day
}
usage_lock = Lock()
usage_stats = {
    "minute": {"count": 0, "reset": time.time() + 60},
    "day": {"count": 0, "reset": time.time() + 86400},
    "hour_audio": {"seconds": 0, "reset": time.time() + 3600},
    "day_audio": {"seconds": 0, "reset": time.time() + 86400},
}

def reset_if_needed(key):
    now = time.time()
    if now > usage_stats[key]["reset"]:
        if "count" in usage_stats[key]:
            usage_stats[key]["count"] = 0
        if "seconds" in usage_stats[key]:
            usage_stats[key]["seconds"] = 0
        usage_stats[key]["reset"] = now + (
            60 if key == "minute" else 3600 if key == "hour_audio" else 86400
        )

def check_and_update_limits(audio_seconds: float):
    """Raise HTTPException if limits would be exceeded."""
    with usage_lock:
        for key in usage_stats.keys():
            reset_if_needed(key)
        if usage_stats["minute"]["count"] >= GROQ_LIMITS["rpm"]:
            raise HTTPException(status_code=429, detail="Groq RPM limit reached")
        if usage_stats["day"]["count"] >= GROQ_LIMITS["rpd"]:
            raise HTTPException(status_code=429, detail="Groq RPD limit reached")
        if usage_stats["hour_audio"]["seconds"] + audio_seconds > GROQ_LIMITS["ash"]:
            raise HTTPException(status_code=429, detail="Groq ASH limit reached")
        if usage_stats["day_audio"]["seconds"] + audio_seconds > GROQ_LIMITS["asd"]:
            raise HTTPException(status_code=429, detail="Groq ASD limit reached")

        usage_stats["minute"]["count"] += 1
        usage_stats["day"]["count"] += 1
        usage_stats["hour_audio"]["seconds"] += audio_seconds
        usage_stats["day_audio"]["seconds"] += audio_seconds

# --- Groq Transcription ---
def call_groq_transcription(mp3_path: str):
    file_size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
    if file_size_mb > GROQ_MAX_FILE_MB:
        raise HTTPException(status_code=413, detail="Audio file exceeds Groq 25MB limit")

    audio_seconds = get_audio_duration(mp3_path)
    check_and_update_limits(audio_seconds)

    with open(mp3_path, "rb") as audio_file:
        files = {"file": audio_file}
        data = {"model": "whisper-large-v3-turbo", "response_format": "verbose_json"}
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        logging.info(f"Sending {round(file_size_mb,2)}MB ({round(audio_seconds)}s) to Groq STT...")
        resp = requests.post(GROQ_API_URL, headers=headers, data=data, files=files, timeout=300)

    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error: {resp.text}")

    return resp.json()

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/transcribe")
def transcribe_video(req: VideoRequest, _: str = Depends(verify_token)):
    temp_id = str(uuid.uuid4())
    video_path = f"{temp_id}.mp4"

    try:
        logging.info(f"Request received for: {req.video_url}")
        yt_opts = {
            'format': 'bestaudio/best',
            'outtmpl': video_path,
            'quiet': True,
            'socket_timeout': 30
        }
        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download([req.video_url])

        if not os.path.exists(video_path):
            raise RuntimeError("Download failed")

        mp3_path = convert_to_mp3(video_path)
        logging.info("Video converted to MP3")

        result = call_groq_transcription(mp3_path)
        segments = [
            {
                "start": format_time(seg["start"]),
                "end": format_time(seg["end"]),
                "text": seg["text"].strip()
            }
            for seg in result.get("segments", [])
        ]

        return {
            "source": "groq_whisper",
            "transcript": result.get("text", ""),
            "segments": segments,
        }

    except Exception as groq_error:
        logging.warning(f"Groq transcription failed: {str(groq_error)}")
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
            logging.error(f"YouTube captions fallback failed: {str(caption_error)}")
            raise HTTPException(status_code=500, detail="Transcription failed: Groq and YouTube captions unavailable")

    finally:
        for f in [video_path, video_path.replace(".mp4", ".mp3")]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
