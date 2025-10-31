from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import yt_dlp
import whisper
import os
import uuid
import logging
import datetime

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
@app.post("/transcribe")
def transcribe_video(req: VideoRequest, _: str = Depends(verify_token)):
    model = whisper.load_model(WHISPER_MODEL) # Load only when needed
    temp_id = str(uuid.uuid4())
    audio_path = f"{temp_id}.mp3"

    try:
        logging.info(f"Transcription request for: {req.video_url}")

        yt_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'socket_timeout': 30,
        }

        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download([req.video_url])
        logging.info("Audio downloaded")

        result = model.transcribe(audio_path)
        os.remove(audio_path)
        logging.info("Transcription complete")

        segments = [
            {
                "start": format_time(seg["start"]),
                "end": format_time(seg["end"]),
                "text": seg["text"].strip()
            }
            for seg in result.get("segments", [])
        ]

        return {
            "transcript": result["text"],
            "segments": segments
        }

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)