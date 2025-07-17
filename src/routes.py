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

    # Audio Routes
    @app.post("/audio/captioning")
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

    @app.post("/audio/event_detection")
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

    # Vision Routes
    @app.post("/vision/image_classification")
    async def image_classification(
        file: UploadFile = File(...),
        categories: str = Form(""),
        max_new_tokens: int = Form(50),
    ):
        if not file.filename.lower().endswith(IMAGE_FILE_TYPES):
            raise HTTPException(400, "Only image files are supported")
        
        image_path = save_to_temp(file)
        
        if categories:
            system_prompt = f"You are an expert image classifier. Classify this image into one of the following categories: {categories}. Respond with the most appropriate category and a brief explanation."
        else:
            system_prompt = "You are an expert image classifier. Analyze this image and provide a detailed classification including the main subject, scene type, and any notable features."
        
        content = [{"type":"image", "image": image_path}]
        
        raw_msgs = [
            {"role":"system", "content":[{"type":"text","text":system_prompt}]},
            {"role":"user", "content": content},
        ]
        
        try:
            reply = generate_response(raw_msgs, max_new_tokens)
            return {"reply": reply, "task": "image_classification"}
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @app.post("/vision/image_event_detection")
    async def image_event_detection(
        file: UploadFile = File(...),
        event_description: str = Form(...),
        max_new_tokens: int = Form(50),
    ):
        if not file.filename.lower().endswith(IMAGE_FILE_TYPES):
            raise HTTPException(400, "Only image files are supported")
        
        image_path = save_to_temp(file)
        system_prompt = f"You are an expert image event detector. Analyze the image and determine if the following event is occurring: '{event_description}'. Respond with 'YES' if the event is detected, 'NO' if it's not detected, followed by a brief explanation of what you see."
        
        content = [{"type":"image", "image": image_path}]
        
        raw_msgs = [
            {"role":"system", "content":[{"type":"text","text":system_prompt}]},
            {"role":"user", "content": content},
        ]
        
        try:
            reply = generate_response(raw_msgs, max_new_tokens)
            return {"reply": reply, "task": "image_event_detection", "event": event_description}
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @app.post("/vision/image_change_detection")
    async def image_change_detection(
        file1: UploadFile = File(...),
        file2: UploadFile = File(...),
        max_new_tokens: int = Form(100),
    ):
        if not (file1.filename.lower().endswith(IMAGE_FILE_TYPES) and 
                file2.filename.lower().endswith(IMAGE_FILE_TYPES)):
            raise HTTPException(400, "Only image files are supported")
        
        image1_path = save_to_temp(file1)
        image2_path = save_to_temp(file2)
        
        system_prompt = "You are an expert in image comparison and change detection. Compare these two images and identify what has changed between them. Describe any differences in objects, positions, appearances, or scenes. Be specific about what was added, removed, or modified."
        
        content = [
            {"type":"image", "image": image1_path},
            {"type":"image", "image": image2_path}
        ]
        
        raw_msgs = [
            {"role":"system", "content":[{"type":"text","text":system_prompt}]},
            {"role":"user", "content": content},
        ]
        
        try:
            reply = generate_response(raw_msgs, max_new_tokens)
            return {"reply": reply, "task": "image_change_detection"}
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @app.post("/vision/bounding_box_detection")
    async def bounding_box_detection(
        file: UploadFile = File(...),
        features: str = Form(...),
        max_new_tokens: int = Form(150),
        draw_boxes: bool = Form(False),
    ):
        raise NotImplementedError("Bounding box detection is not implemented yet.")


    # Video Routes
    @app.post("/video/captioning")
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
            parent_dir=GRADIO_TEMP,
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

    @app.post("/video/event_detection")
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
            parent_dir=GRADIO_TEMP,
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

    # Multimodal Routes
    @app.post("/multimodal/audio_vision")
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

    @app.post("/multimodal/audio_video")
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
            parent_dir=GRADIO_TEMP,
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

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "model": "gemma-3n-multimodal"}

    # Available endpoints
    @app.get("/endpoints")
    async def list_endpoints():
        return {
            "available_endpoints": [
                "/chat/multimodal - General multimodal chat with custom system prompts",
                "/audio/captioning - Generate captions for audio content",
                "/audio/event_detection - Detect specific events in audio",
                "/vision/image_classification - Classify images into categories",
                "/vision/image_event_detection - Detect specific events in images",
                "/vision/image_change_detection - Compare two images for changes",
                "/vision/bounding_box_detection - NOT IMPLEMENTED YET",
                "/video/captioning - Generate captions for video content",
                "/video/event_detection - Detect specific events in video",
                "/multimodal/audio_vision - Combined audio and image analysis",
                "/multimodal/audio_video - Combined audio and video analysis",
                "/health - Health check",
                "/endpoints - List all available endpoints"
            ]
        }
