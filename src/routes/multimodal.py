# src/routes/multimodal.py
import pathlib
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from src.core import generate_response
from src.utils import (
    save_to_temp, 
    extract_frames_to_tempdir,
    AUDIO_FILE_TYPES,
    IMAGE_FILE_TYPES,
    VIDEO_FILE_TYPES,
    TARGET_FPS,
    MAX_FRAMES,
    TEMP_DIR
)

router = APIRouter(prefix="/multimodal", tags=["multimodal"])


@router.post("/")
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
            parent_dir=TEMP_DIR,
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


@router.post("/audio_vision")
async def audio_vision_understanding(
    audio_file: UploadFile = File(...),
    image_file: UploadFile = File(...),
    user_text: str = Form(""),
    max_new_tokens: int = Form(150),
):
    if not audio_file.filename.lower().endswith(AUDIO_FILE_TYPES):
        raise HTTPException(400, "Audio file must be in supported format")
    if not image_file.filename.lower().endswith(IMAGE_FILE_TYPES):
        raise HTTPException(400, "Image file must be in supported format")
    
    audio_path = save_to_temp(audio_file)
    image_path = save_to_temp(image_file)
    
    system_prompt = "You are an expert multimodal analyst. Analyze both the audio and visual information provided to create a comprehensive understanding of the environment and situation. Correlate information from both modalities to provide insights that wouldn't be possible from either alone. Describe the scene, events, context, and any relationships between what you hear and see."
    
    content = []
    if user_text:
        content.append({"type":"text", "text": user_text})
    content.extend([
        {"type":"audio", "audio": audio_path},
        {"type":"image", "image": image_path}
    ])
    
    raw_msgs = [
        {"role":"system", "content":[{"type":"text","text":system_prompt}]},
        {"role":"user", "content": content},
    ]
    
    try:
        reply = generate_response(raw_msgs, max_new_tokens)
        return {"reply": reply, "task": "multimodal_audio_vision"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/audio_video")
async def audio_video_understanding(
    audio_file: UploadFile = File(...),
    video_file: UploadFile = File(...),
    user_text: str = Form(""),
    max_new_tokens: int = Form(200),
):
    if not audio_file.filename.lower().endswith(AUDIO_FILE_TYPES):
        raise HTTPException(400, "Audio file must be in supported format")
    if not video_file.filename.lower().endswith(VIDEO_FILE_TYPES):
        raise HTTPException(400, "Video file must be in supported format")
    
    audio_path = save_to_temp(audio_file)
    video_path = save_to_temp(video_file)
    
    system_prompt = "You are an expert multimodal analyst. Analyze both the audio and video information provided to create a comprehensive understanding of the environment and situation. Correlate information from both modalities, including temporal alignment between audio and visual events. Describe the scene, events, context, and any relationships between what you hear and see over time."
    
    frame_dir = extract_frames_to_tempdir(
        video_path,
        target_fps=TARGET_FPS,
        max_frames=MAX_FRAMES,
        parent_dir=TEMP_DIR,
    )
    
    content = []
    if user_text:
        content.append({"type":"text", "text": user_text})
    content.append({"type":"audio", "audio": audio_path})
    
    for frame in sorted(pathlib.Path(frame_dir).glob("*.jpg")):
        content.append({"type":"image", "image": frame.as_posix()})
    
    raw_msgs = [
        {"role":"system", "content":[{"type":"text","text":system_prompt}]},
        {"role":"user", "content": content},
    ]
    
    try:
        reply = generate_response(raw_msgs, max_new_tokens)
        return {"reply": reply, "task": "multimodal_audio_video"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))