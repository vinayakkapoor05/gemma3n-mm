# src/core.py
from typing import List
import os
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

model: Gemma3nForConditionalGeneration = None
processor: AutoProcessor = None
DEVICE: str = None

def initialize_model():
    global model, processor, DEVICE
    
    if model is None or processor is None:
        print("Loading Gemma-3n model...")
        MODEL_ID = os.getenv("IMG_MODEL", "google/gemma-3n-e2b-it")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if we have a local cache directory
        local_model_path = f"/hf_cache/google/gemma-3n-e2b-it"
        use_local = os.path.exists(local_model_path) and os.listdir(local_model_path)
        
        if use_local:
            print(f"Using local cached model from {local_model_path}")
            model_path = local_model_path
            local_files_only = True
        else:
            print(f"Using model from HuggingFace Hub: {MODEL_ID}")
            model_path = MODEL_ID
            local_files_only = False
        
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=(torch.bfloat16 if DEVICE == "cuda" else torch.float32),
            device_map=0 if DEVICE == "cuda" else None,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(
            model_path, 
            padding_side="left",
            local_files_only=local_files_only
        )
        print(f"Model loaded on {DEVICE}")
    
    return model, processor


def build_raw_messages(msg_dicts: List[dict]) -> List[dict]:
    raw = []
    for m in msg_dicts:
        t = m['type']
        if t == 'text':
            raw.append({
                'role': m['role'],
                'content': [{'type': 'text', 'text': m['content']}]
            })
        else:
            entry = {'role': m['role'], 'content': [{'type': t}]}
            if isinstance(m['content'], (bytes, bytearray)):
                entry['content'][0]['bytes'] = m['content']
            else:
                entry['content'][0]['url'] = m['content']
            raw.append(entry)

    return raw


def generate_response(raw_messages: List[dict], max_new_tokens: int) -> str:
    initialize_model()
    
    inputs = processor.apply_chat_template(
        raw_messages,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
        add_generation_prompt=True
    )
    if DEVICE == "cuda":
        inputs = inputs.to(device="cuda", dtype=torch.bfloat16)
    else:
        inputs = inputs.to(device="cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        cache_implementation='static'
    )
    return processor.decode(outputs[0], skip_special_tokens=True)

