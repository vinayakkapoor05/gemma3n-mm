# src/routes/audio.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from src.core import generate_response
from src.utils import save_to_temp, AUDIO_FILE_TYPES

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/captioning")
async def audio_captioning(
    file: UploadFile = File(...),
    user_text: str = Form(""),
    max_new_tokens: int = Form(100),
):
    if not file.filename.lower().endswith(AUDIO_FILE_TYPES):
        raise HTTPException(400, "Only audio files are supported")
    
    audio_path = save_to_temp(file)
    system_prompt = "You are an expert audio analyst. Provide detailed, accurate captions describing the audio content including sounds, speech, music, environment, and any notable events or patterns you detect."
    
    content = []
    if user_text:
        content.append({"type":"text", "text": user_text})
    content.append({"type":"audio", "audio": audio_path})
    
    raw_msgs = [
        {"role":"system", "content":[{"type":"text","text":system_prompt}]},
        {"role":"user", "content": content},
    ]
    
    try:
        reply = generate_response(raw_msgs, max_new_tokens)
        return {"reply": reply, "task": "audio_captioning"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/event_detection")
async def audio_event_detection(
    file: UploadFile = File(...),
    event_description: str = Form(...),
    max_new_tokens: int = Form(50),
):
    if not file.filename.lower().endswith(AUDIO_FILE_TYPES):
        raise HTTPException(400, "Only audio files are supported")
    
    audio_path = save_to_temp(file)
    system_prompt = f"You are an expert audio event detector. Analyze the audio and determine if the following event is occurring: '{event_description}'. Respond with 'YES' if the event is detected, 'NO' if it's not detected, followed by a brief explanation of what you hear."
    
    content = [{"type":"audio", "audio": audio_path}]
    
    raw_msgs = [
        {"role":"system", "content":[{"type":"text","text":system_prompt}]},
        {"role":"user", "content": content},
    ]
    
    try:
        reply = generate_response(raw_msgs, max_new_tokens)
        return {"reply": reply, "task": "audio_event_detection", "event": event_description}
    except Exception as e:
        raise HTTPException(500, detail=str(e))