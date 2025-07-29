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
    task: str = Field("bounding_box_detection", const=True)
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
    object_name: str = Form(..., description="The object to detect"),
    max_new_tokens: int = Form(150),
    draw_boxes: bool = Form(False, description="If true, return image with drawn boxes as base64‑PNG"),
):
    if not file.filename.lower().endswith(IMAGE_FILE_TYPES):
        raise HTTPException(400, "Only image files are supported")
    image_path = save_to_temp(file)

    system_prompt = (
        f"You are an expert object detector. Analyze the image and say whether '{object_name}' "
        f"is present.\n\nOutput valid JSON **only** with keys:\n"
        f"- detected (boolean)\n"
        f"- confidence (0.0–1.0)\n"
        f"- bbox (object with x_min, y_min, x_max, y_max) or null\n\n"
        f"Example:\n"
        f"{{\n"
        f"  \"detected\": true,\n"
        f"  \"confidence\": 0.87,\n"
        f"  \"bbox\": {{\"x_min\": 34.5, \"y_min\": 22.1, \"x_max\": 130.0, \"y_max\": 150.2}}\n"
        f"}}"
    )

    raw_msgs = [
        {"role": "system", "content": [{"type":"text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type":"image", "image": image_path}]},
    ]

    try:
        reply = generate_response(raw_msgs, max_new_tokens)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    try:
        payload = json.loads(reply)
    except json.JSONDecodeError:
        raise HTTPException(502, detail=f"Model returned invalid JSON:\n{reply}")

    image_b64 = None
    if draw_boxes and payload.get("detected") and payload.get("bbox"):
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            bb = payload["bbox"]
            draw.rectangle([bb["x_min"], bb["y_min"], bb["x_max"], bb["y_max"]], outline="red", width=3)

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            image_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        except Exception as e:
            image_b64 = None

    return JSONResponse({
        "task": "bounding_box_detection",
        "object": object_name,
        "detected": payload["detected"],
        "confidence": payload["confidence"],
        "bbox": payload.get("bbox"),
        "image_with_boxes": image_b64
    })
