# src/schemas.py
from pydantic import BaseModel
from typing import List, Literal, Optional

class Message(BaseModel):
    role: Literal["system", "user"]
    type: Literal["text", "image", "audio", "video"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = 50
