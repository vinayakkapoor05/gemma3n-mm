# gemma3n.py

import argparse
import sys
import os
import subprocess
from pathlib import Path

def serve_command(args):
    import uvicorn
    
    print(f"Starting Gemma3n server on {args.host}:{args.port}")
    print(f"Mode: {args.mode}")
    
    if args.model:
        os.environ["IMG_MODEL"] = args.model
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        log_level="info",
        workers=1,
        reload=args.reload,
    )

def cli_command(args):
    cmd = [sys.executable, "cli.py"]
    
    cmd.append(args.task)
    
    if hasattr(args, 'modes') and args.modes:
        cmd.extend(["--modes"] + args.modes)
    
    if hasattr(args, 'event_description') and args.event_description:
        cmd.extend(["--event-description", args.event_description])
    
    if hasattr(args, 'user_text') and args.user_text:
        cmd.extend(["--user-text", args.user_text])
    
    if hasattr(args, 'max_tokens') and args.max_tokens:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    
    if hasattr(args, 'period') and args.period:
        cmd.extend(["--period", str(args.period)])
    
    if hasattr(args, 'yaml_url') and args.yaml_url:
        cmd.extend(["--yaml-url", args.yaml_url])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CLI command failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Gemma3n - Multimodal AI Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # serve command
    serve_parser = subparsers.add_parser('serve', help='Start the FastAPI server')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0',
                             help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8080,
                             help='Port to bind to')
    serve_parser.add_argument('--mode', type=str, default='multimodal',
                             choices=['audio', 'video', 'image', 'multimodal'],
                             help='Server mode')
    serve_parser.add_argument('--model', type=str,
                             help='Model ID to use (overrides IMG_MODEL env var)')
    serve_parser.add_argument('--reload', action='store_true',
                             help='Enable auto-reload during development')
    serve_parser.set_defaults(func=serve_command)
    
    # cli command
    cli_parser = subparsers.add_parser('cli', help='Run CLI operations')
    cli_parser.add_argument('task', choices=['caption', 'detect', 'dynamic-prompting'],
                           help='Task to perform')
    cli_parser.add_argument('--modes', nargs='+', choices=['image', 'audio', 'video'],
                           default=['image'], help='Media modes to process')
    cli_parser.add_argument('--event-description', type=str,
                           help='Event description for detection task')
    cli_parser.add_argument('--user-text', type=str, default='',
                           help='Additional user text/prompt')
    cli_parser.add_argument('--max-tokens', type=int, default=100,
                           help='Maximum tokens to generate')
    cli_parser.add_argument('--period', type=int, default=0,
                           help='Run periodically every N minutes (0 = run once)')
    cli_parser.add_argument('--yaml-url', type=str,
                           help='URL to YAML configuration file (required for dynamic-prompting)')
    cli_parser.set_defaults(func=cli_command)
    
    # parse args
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    args.func(args)

if __name__ == '__main__':
    main() 