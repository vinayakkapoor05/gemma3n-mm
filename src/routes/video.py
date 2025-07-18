# src/routes/video.py
import pathlib
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from src.core import generate_response
from src.utils import save_to_temp, extract_frames_to_tempdir, VIDEO_FILE_TYPES, TARGET_FPS, MAX_FRAMES, TEMP_DIR

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/captioning")
async def video_captioning(
    file: UploadFile = File(...),
    user_text: str = Form(""),
    max_new_tokens: int = Form(150),
):
    if not file.filename.lower().endswith(VIDEO_FILE_TYPES):
        raise HTTPException(400, "Only video files are supported")
    
    video_path = save_to_temp(file)
    system_prompt = "You are an expert video analyst. Provide detailed, accurate captions describing the video content including actions, scenes, objects, people, and any notable events or patterns. Describe the temporal progression of events."
    
    frame_dir = extract_frames_to_tempdir(
        video_path,
        target_fps=TARGET_FPS,
        max_frames=MAX_FRAMES,
        parent_dir=TEMP_DIR,
    )
    
    content = []
    if user_text:
        content.append({"type":"text", "text": user_text})
    
    for frame in sorted(pathlib.Path(frame_dir).glob("*.jpg")):
        content.append({"type":"image", "image": frame.as_posix()})
    
    raw_msgs = [
        {"role":"system", "content":[{"type":"text","text":system_prompt}]},
        {"role":"user", "content": content},
    ]
    
    try:
        reply = generate_response(raw_msgs, max_new_tokens)
        return {"reply": reply, "task": "video_captioning"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/event_detection")
async def video_event_detection(
    file: UploadFile = File(...),
    event_description: str = Form(...),
    max_new_tokens: int = Form(100),
):
    if not file.filename.lower().endswith(VIDEO_FILE_TYPES):
        raise HTTPException(400, "Only video files are supported")
    
    video_path = save_to_temp(file)
    system_prompt = f"You are an expert video event detector. Analyze the video frames and determine if the following event is occurring: '{event_description}'. Respond with 'YES' if the event is detected, 'NO' if it's not detected, followed by a detailed explanation of what you see in the video and when/where the event occurs if detected."
    
    frame_dir = extract_frames_to_tempdir(
        video_path,
        target_fps=TARGET_FPS,
        max_frames=MAX_FRAMES,
        parent_dir=TEMP_DIR,
    )
    
    content = []
    for frame in sorted(pathlib.Path(frame_dir).glob("*.jpg")):
        content.append({"type":"image", "image": frame.as_posix()})
    
    raw_msgs = [
        {"role":"system", "content":[{"type":"text","text":system_prompt}]},
        {"role":"user", "content": content},
    ]
    
    try:
        reply = generate_response(raw_msgs, max_new_tokens)
        return {"reply": reply, "task": "video_event_detection", "event": event_description}
    except Exception as e:
        raise HTTPException(500, detail=str(e))