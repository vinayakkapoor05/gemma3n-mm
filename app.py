# app.py
import os
from pathlib import Path
import tempfile

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from transformers import pipeline

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.bfloat16 if DEVICE >= 0 else torch.float32

# Text-generation pipeline
TEXT_MODEL = os.getenv("TEXT_MODEL", "google/gemma-3n-e4b-it")
text_pipe = pipeline(
    "text-generation",
    model=TEXT_MODEL,
    device=DEVICE,
    torch_dtype=DTYPE,
)

# Image-to-text pipeline
IMG_MODEL = os.getenv("IMG_MODEL", "google/gemma-3n-e4b-it")
img_pipe = pipeline(
    "image-text-to-text",
    model=IMG_MODEL,
    device=DEVICE,
    torch_dtype=DTYPE,
)

app = FastAPI(title="Gemma-3n mm")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(payload: dict):
    if "inputs" in payload:
        payload["text_inputs"] = payload.pop("inputs")

    if "text_inputs" not in payload:
        raise HTTPException(
            status_code=400,
            detail="Request JSON must include 'inputs'"
        )
    try:
        outputs = text_pipe(**payload)
        return {"generated": outputs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/caption")
async def caption(image: UploadFile = File(...), text: str = None):
    suffix = Path(image.filename).suffix or ".jpg"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name

        if text:
            outputs = img_pipe(tmp_path, text=text)
        else:
            outputs = img_pipe(tmp_path)

        return {"caption": outputs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass
