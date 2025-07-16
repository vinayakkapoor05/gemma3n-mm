# app.py
import os
import torch
from fastapi import FastAPI
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from src.routes import register_routes
from src.periodic import setup_periodic_tasks
import src.core as core

MODEL_ID = os.getenv("IMG_MODEL", "google/gemma-3n-e4b-it")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

model = Gemma3nForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=(torch.bfloat16 if DEVICE == "cuda" else torch.float32),
    device_map=0
)
processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")

app = FastAPI(title="Gemma-3n Multimodal")

core.model = model
core.processor = processor
core.DEVICE = DEVICE

register_routes(app)

GIST_URL = os.getenv("GIST_URL", "")
setup_periodic_tasks(app, GIST_URL)
