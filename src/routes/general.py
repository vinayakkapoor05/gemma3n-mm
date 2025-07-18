# src/routes/general.py
from fastapi import APIRouter

router = APIRouter(tags=["general"])


@router.get("/health")
async def health_check():
    return {"status": "ok", "model": "gemma-3n-multimodal"}


@router.get("/endpoints")
async def list_endpoints():
    return {
        "available_endpoints": [
            "/multimodal/ - General multimodal chat with custom system prompts",
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