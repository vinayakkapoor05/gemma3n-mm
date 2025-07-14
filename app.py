# app.py
import torch
from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline

app = FastAPI()

MODEL_ID = "google/gemma-3n-e4b-it"

processor = AutoProcessor.from_pretrained(MODEL_ID, device_map="auto")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)
model.eval()

text_pipe = pipeline(
    "text-generation",
    model="google/gemma-3-4b-it",
    device=0,
    torch_dtype=torch.bfloat16,
)

#endpoints
@app.post("/generate/text")
async def generate_text(prompt: str = Form(...)):
    return text_pipe(prompt)

@app.post("/generate/multimodal")
async def generate_multimodal(
    prompt: str = Form(...),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
):
    content = []
    content.append({"type": "text", "text": prompt})

    if image:
        img_bytes = await image.read()
        content.insert(0, {"type": "image", "image": img_bytes})

    if audio:
        audio_bytes = await audio.read()
        content.insert(0, {"type": "audio", "audio": audio_bytes})

    messages = [{"role": "user", "content": content}]

    # preprocess + generate
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device, dtype=model.dtype)
    outputs = model.generate(**inputs, max_new_tokens=128)
    text = processor.batch_decode(outputs, skip_special_tokens=True)
    return {"response": text[0]}
