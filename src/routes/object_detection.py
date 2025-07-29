# src/routes/bounding_box_detection.py

import json
import base64
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

from src.core import generate_response
from src.utils import save_to_temp, IMAGE_FILE_TYPES

router = APIRouter(prefix="/vision", tags=["vision"])

class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class BoundingBoxDetectionResponse(BaseModel):
    task: str = Field("bounding_box_detection", Literal=True)
    object: str
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[BBox] = None
    image_with_boxes: Optional[str] = None   

@router.post(
    "/bounding_box_detection",
    response_model=BoundingBoxDetectionResponse,
)
async def bounding_box_detection(
    file: UploadFile = File(...),
    object_name: str = Form(...),
    max_new_tokens: int = Form(150),
    draw_boxes: bool = Form(False),
):
    raise NotImplementedError("Bounding box detection is not implemented yet.")