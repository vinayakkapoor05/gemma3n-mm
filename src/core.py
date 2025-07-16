# src/core.py
from typing import List
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

model: Gemma3nForConditionalGeneration = None
processor: AutoProcessor = None
DEVICE: str = None


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
            raw.append({
                'role': m['role'],
                'content': [{'type': t, 'url': m['content']}]
            })
    return raw


def generate_response(raw_messages: List[dict], max_new_tokens: int) -> str:
    inputs = processor.apply_chat_template(
        raw_messages,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
        add_generation_prompt=True
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        cache_implementation='static'
    )
    return processor.decode(outputs[0], skip_special_tokens=True)

