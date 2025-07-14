import os
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3n")

app = FastAPI(
    title="Gemma3n API",
    description="API for interacting with the Gemma3N model hosted on Ollama.",
    version="0.0.1",
)

class GenerateResponse(BaseModel):
    text: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    prompt: str = Form(""),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
):
    full_prompt = prompt
    if audio:
        audio_bytes = await audio.read()
        transcript = base64.b64encode(audio_bytes)
        full_prompt += f"\n\n[Audio transcript]\n{transcript}"

    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
    }

    if image:
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode()
        payload["images"] = image_base64
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")

    data = resp.json()
    return GenerateResponse(text=data.get("text", "").strip())