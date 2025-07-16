# src/periodic.py
import asyncio
import json
import requests
from src.schemas import Message
from src.core import build_raw_messages, generate_response


def setup_periodic_tasks(app, gist_url: str):
    @app.on_event('startup')
    async def _startup_loop():
        if not gist_url:
            print('No GIST_URL provided, skipping periodic tasks.')
            return

        async def _loop():
            while True:
                try:
                    resp = requests.get(gist_url)
                    resp.raise_for_status()
                    text = resp.text.strip()

                    try:
                        job = json.loads(text)
                        msgs = [Message(**m) for m in job['messages']]
                        max_toks = job.get('max_new_tokens', 50)
                    except json.JSONDecodeError:
                        lines = text.splitlines()
                        system = lines[0]
                        mode = lines[1].strip()
                        user_msg = lines[2]
                        max_toks = 50
                        msgs = [
                            Message(role='system', type='text', content=system),
                            Message(role='user',   type=mode,   content=user_msg)
                        ]

                    raw = build_raw_messages([m.dict() for m in msgs])
                    out = generate_response(raw, max_toks)
                    print(f"[Periodic Task] → {out}", flush=True)

                except Exception as e:
                    print(f"[Periodic Task] ERROR: {e}", flush=True)

                await asyncio.sleep(300)

        asyncio.create_task(_loop())
