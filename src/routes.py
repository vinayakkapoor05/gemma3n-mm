# src/routes.py
import os, pathlib, shutil, tempfile
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from src.core import generate_response
from av import open as av_open  

TARGET_FPS    = int(os.getenv("TARGET_FPS", "3"))
MAX_FRAMES    = int(os.getenv("MAX_FRAMES", "30"))
GRADIO_TEMP   = tempfile.gettempdir()
IMAGE_FILE_TYPES = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_FILE_TYPES = (".mp4", ".mov", ".webm")
AUDIO_FILE_TYPES = (".mp3", ".wav")

def extract_frames_to_tempdir(
    video_path: str,
    target_fps: float,
    max_frames: int | None = None,
    parent_dir: str | None = None,
    prefix: str = "frames_",
) -> str:
    temp_dir = tempfile.mkdtemp(prefix=prefix, dir=parent_dir)
    container = av_open(video_path)
    stream    = container.streams.video[0]
    tb, dur   = stream.time_base, float(stream.duration * stream.time_base)
    interval  = 1.0 / target_fps
    total     = min(int(dur * target_fps), max_frames or 10_000)
    times     = [i*interval for i in range(total)]
    idx       = 0

    for frame in container.decode(video=0):
        if frame.pts is None: continue
        ts = float(frame.pts * tb)
        if idx < len(times) and abs(ts - times[idx]) < (interval/2):
            path = pathlib.Path(temp_dir)/f"frame_{idx:04d}.jpg"
            frame.to_image().save(path)
            idx += 1
            if idx >= total:
                break

    container.close()
    return temp_dir


def save_to_temp(upload: UploadFile) -> str:
    suffix = pathlib.Path(upload.filename).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tmp.name, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp.name


def register_routes(app: FastAPI):
    @app.post("/chat/multimodal")
    async def chat_multimodal(
        system_prompt: str      = Form(...),
        user_text: str          = Form(""),
        files: List[UploadFile] = File([]),
        max_new_tokens: int     = Form(50),
    ):
        paths = [save_to_temp(f) for f in files]

        video_paths = [p for p in paths if pathlib.Path(p).suffix.lower() in VIDEO_FILE_TYPES]
        content = []
        if user_text:
            content.append({"type":"text", "text": user_text})

        if len(video_paths) == 1 and len(paths) == 1:
            frame_dir = extract_frames_to_tempdir(
                video_paths[0],
                target_fps=TARGET_FPS,
                max_frames=MAX_FRAMES,
                parent_dir=GRADIO_TEMP,
            )
            for frame in sorted(pathlib.Path(frame_dir).glob("*.jpg")):
                content.append({"type":"image", "image": frame.as_posix()})
        else:
            for p in paths:
                ext = pathlib.Path(p).suffix.lower()
                if ext in IMAGE_FILE_TYPES:
                    content.append({"type":"image", "image": p})
                elif ext in AUDIO_FILE_TYPES:
                    content.append({"type":"audio", "audio": p})
                elif ext in VIDEO_FILE_TYPES:
                    content.append({"type":"video", "video": p})
                else:
                    raise HTTPException(400, f"Unsupported file type {ext}")

        raw_msgs = [
            {"role":"system", "content":[{"type":"text","text":system_prompt}]},
            {"role":"user",   "content": content},
        ]

        try:
            reply = generate_response(raw_msgs, max_new_tokens)
            return {"reply": reply}
        except Exception as e:
            raise HTTPException(500, detail=str(e))
