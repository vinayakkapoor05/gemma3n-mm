import os
import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# initialize model and processor
MODEL_ID = os.getenv("IMG_MODEL", "google/gemma-3n-e4b-it")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Gemma3nForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map=0,
    attn_implementation="eager"
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    padding_side="left"
)

app = FastAPI(title="Gemma-3n mm (raw model)")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/caption")
async def caption():
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/"
                           "huggingface/documentation-images/"
                           "resolve/main/pipeline-cat-chonk.jpeg"
                },
                {"type": "text", "text": "What is shown in this image?"},
            ]
        },
    ]

    try:
        # tokenize + prepare inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(DEVICE)

        # generate
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            cache_implementation="static"
        )

        # decode output
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return {"caption": caption}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
