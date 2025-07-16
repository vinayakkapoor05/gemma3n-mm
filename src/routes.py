# src/routes.py
from fastapi import FastAPI, HTTPException
from src.schemas import ChatRequest
from src.core import build_raw_messages, generate_response


def register_routes(app: FastAPI):
    @app.get('/health')
    async def health():
        return {'status': 'ok'}

    @app.post('/text')
    async def text_only(req: ChatRequest):
        filtered = [m for m in req.messages if m.type == 'text']
        if not filtered:
            raise HTTPException(400, 'No text messages provided.')
        raw = build_raw_messages([m.dict() for m in filtered])
        try:
            reply = generate_response(raw, req.max_new_tokens)
            return {'reply': reply}
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @app.post('/image')
    async def image_only(req: ChatRequest):
        filtered = [m for m in req.messages if m.type == 'image']
        if not filtered:
            raise HTTPException(400, 'No image messages provided.')
        raw = build_raw_messages([m.dict() for m in req.messages])
        try:
            caption = generate_response(raw, req.max_new_tokens)
            return {'caption': caption}
        except Exception as e:
            raise HTTPException(500, detail=str(e))
