# src/utils.py
import os, pathlib, shutil, tempfile
from typing import List
from fastapi import UploadFile
from av import open as av_open  

TARGET_FPS    = int(os.getenv("TARGET_FPS", "3"))
MAX_FRAMES    = int(os.getenv("MAX_FRAMES", "30"))
TEMP_DIR   = tempfile.gettempdir()
IMAGE_FILE_TYPES = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_FILE_TYPES = (".mp4", ".mov", ".webm")
AUDIO_FILE_TYPES = (".mp3", ".wav", ".ogg")


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


