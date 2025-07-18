# app.py
from fastapi import FastAPI
from src.routes import register_routes
import src.core as core

print("Starting Gemma-3n")

app = FastAPI(title="Gemma-3n Multimodal")

# initialize model and processor
core.initialize_model()

register_routes(app)
