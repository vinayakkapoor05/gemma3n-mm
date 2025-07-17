import argparse
import os
import pathlib

def parse_args():
    parser = argparse.ArgumentParser(description="Gemma-3n")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mode", type=str, default="image", choices=["audio", "video", "image", "multimodal"])
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    import uvicorn
    uvicorn.run(
        "app:app",           
        host=args.host,
        port=args.port,
        log_level="info",
        workers=1,
        reload=True,
    )
