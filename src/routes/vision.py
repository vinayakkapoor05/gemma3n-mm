# src/routes/vision.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from src.core import generate_response
from src.utils import save_to_temp, IMAGE_FILE_TYPES

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/image_classification")
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


@router.post("/image_event_detection")
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


@router.post("/image_change_detection")
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


@router.post("/bounding_box_detection")
async def bounding_box_detection(
    file: UploadFile = File(...),
    features: str = Form(...),
    max_new_tokens: int = Form(150),
    draw_boxes: bool = Form(False),
):
    raise NotImplementedError("Bounding box detection is not implemented yet.")