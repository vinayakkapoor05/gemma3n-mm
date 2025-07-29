# src/routes/__init__.py
from fastapi import FastAPI
from .audio import router as audio_router
from .vision import router as vision_router
from .video import router as video_router
from .multimodal import router as multimodal_router
from .general import router as general_router
from .object_detection import router as object_detection_router


def register_routes(app: FastAPI):
    app.include_router(audio_router)
    app.include_router(vision_router)
    app.include_router(video_router)
    app.include_router(multimodal_router)
    app.include_router(general_router)
    app.include_router(object_detection_router)

__all__ = [
    "register_routes",
    "audio_router",
    "vision_router", 
    "video_router",
    "multimodal_router",
    "general_router",
    "object_detection_router"
]

