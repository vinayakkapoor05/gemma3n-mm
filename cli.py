# cli.py
import argparse
import asyncio
import os
import pathlib
import sys
import tempfile
import time
from typing import List, Optional, Dict, Any
import requests
import yaml
import schedule
import logging
from datetime import datetime

from src.core import generate_response, build_raw_messages
from src.utils import extract_frames_to_tempdir, TARGET_FPS, MAX_FRAMES, TEMP_DIR
from src.utils import IMAGE_FILE_TYPES, AUDIO_FILE_TYPES, VIDEO_FILE_TYPES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_FILES = {
    'image': 'image.jpg',
    'audio': 'audio.mp3', 
    'video': 'video.mp4'
}

def get_file_path(filename):
    if os.path.exists(filename):
        return filename
    
    data_path = os.path.join('data', filename)
    if os.path.exists(data_path):
        return data_path
    
    return filename

class GemmaCliProcessor:
    
    def __init__(self):
        self.temp_files = []
    
    def __del__(self):
        # clean up temp files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

    def process_image_captioning(self, image_path: str, user_text: str = "", max_tokens: int = 100) -> str:
        system_prompt = "You are an expert image analyst. Provide detailed, accurate captions describing the image content including objects, scenes, people, actions, and any notable features."
        
        content = []
        if user_text:
            content.append({"type": "text", "text": user_text})
        content.append({"type": "image", "image": image_path})
        
        raw_msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        
        return generate_response(raw_msgs, max_tokens)
    
    def process_image_detection(self, image_path: str, event_description: str, max_tokens: int = 50) -> str:
        system_prompt = f"You are an expert image event detector. Analyze the image and determine if the following event is occurring: '{event_description}'. Respond with 'YES' if the event is detected, 'NO' if it's not detected, followed by a brief explanation of what you see."
        
        content = [{"type": "image", "image": image_path}]
        
        raw_msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        
        return generate_response(raw_msgs, max_tokens)
    
    def process_audio_captioning(self, audio_path: str, user_text: str = "", max_tokens: int = 100) -> str:
        system_prompt = "You are an expert audio analyst. Provide detailed, accurate captions describing the audio content including sounds, speech, music, environment, and any notable events or patterns you detect."
        
        content = []
        if user_text:
            content.append({"type": "text", "text": user_text})
        content.append({"type": "audio", "audio": audio_path})
        
        raw_msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        
        return generate_response(raw_msgs, max_tokens)
    
    def process_audio_detection(self, audio_path: str, event_description: str, max_tokens: int = 50) -> str:
        system_prompt = f"You are an expert audio event detector. Analyze the audio and determine if the following event is occurring: '{event_description}'. Respond with 'YES' if the event is detected, 'NO' if it's not detected, followed by a brief explanation of what you hear."
        
        content = [{"type": "audio", "audio": audio_path}]
        
        raw_msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        
        return generate_response(raw_msgs, max_tokens)
    
    def process_video_captioning(self, video_path: str, user_text: str = "", max_tokens: int = 150) -> str:
        system_prompt = "You are an expert video analyst. Provide detailed, accurate captions describing the video content including actions, scenes, objects, people, and any notable events or patterns. Describe the temporal progression of events."
        
        frame_dir = extract_frames_to_tempdir(
            video_path,
            target_fps=TARGET_FPS,
            max_frames=MAX_FRAMES,
            parent_dir=TEMP_DIR,
        )
        
        content = []
        if user_text:
            content.append({"type": "text", "text": user_text})
        
        for frame in sorted(pathlib.Path(frame_dir).glob("*.jpg")):
            content.append({"type": "image", "image": frame.as_posix()})
        
        raw_msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        
        return generate_response(raw_msgs, max_tokens)
    
    def process_video_detection(self, video_path: str, event_description: str, max_tokens: int = 50) -> str:
        system_prompt = f"You are an expert video event detector. Analyze the video and determine if the following event is occurring: '{event_description}'. Respond with 'YES' if the event is detected, 'NO' if it's not detected, followed by a brief explanation of what you see."
        
        frame_dir = extract_frames_to_tempdir(
            video_path,
            target_fps=TARGET_FPS,
            max_frames=MAX_FRAMES,
            parent_dir=TEMP_DIR,
        )
        
        content = []
        for frame in sorted(pathlib.Path(frame_dir).glob("*.jpg")):
            content.append({"type": "image", "image": frame.as_posix()})
        
        raw_msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        
        return generate_response(raw_msgs, max_tokens)
    
    def process_multimodal(self, files: List[str], system_prompt: str, user_text: str = "", max_tokens: int = 200) -> str:
        content = []
        if user_text:
            content.append({"type": "text", "text": user_text})
        
        video_files = [f for f in files if pathlib.Path(f).suffix.lower() in VIDEO_FILE_TYPES]
        
        if len(video_files) == 1 and len(files) == 1:
            # single video file, we can extract frames
            frame_dir = extract_frames_to_tempdir(
                video_files[0],
                target_fps=TARGET_FPS,
                max_frames=MAX_FRAMES,
                parent_dir=TEMP_DIR,
            )
            for frame in sorted(pathlib.Path(frame_dir).glob("*.jpg")):
                content.append({"type": "image", "image": frame.as_posix()})
        else:
            for file_path in files:
                ext = pathlib.Path(file_path).suffix.lower()
                if ext in IMAGE_FILE_TYPES:
                    content.append({"type": "image", "image": file_path})
                elif ext in AUDIO_FILE_TYPES:
                    content.append({"type": "audio", "audio": file_path})
                elif ext in VIDEO_FILE_TYPES:
                    content.append({"type": "video", "video": file_path})
        
        raw_msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        
        return generate_response(raw_msgs, max_tokens)


class DynamicPromptProcessor:
    
    def __init__(self, yaml_url: str):
        self.yaml_url = yaml_url
        self.last_config = None
        self.processor = GemmaCliProcessor()
    
    def fetch_config(self) -> Optional[Dict[str, Any]]:
        # fetch config from remote yaml
        try:
            response = requests.get(self.yaml_url, timeout=30)
            response.raise_for_status()
            config = yaml.safe_load(response.text)
            return config
        except Exception as e:
            logger.error(f"Failed to fetch config from {self.yaml_url}: {e}")
            return None
    
    def process_dynamic_prompt(self):
        config = self.fetch_config()
        if not config:
            logger.warning("No configuration available, skipping...")
            return
        
        config_changed = config != self.last_config
        if config_changed:
            logger.info("Configuration changed, processing new configuration...")
            self.last_config = config
        else:
            logger.info("Configuration unchanged, but processing anyway (scheduled run)...")
        
        try:
            system_prompt = config.get('system_prompt', '')
            user_prompt = config.get('user_prompt', '')
            modes = config.get('modes', [])
            max_tokens = config.get('max_tokens', 200)
            
            files = []
            for mode in modes:
                if mode in DEFAULT_FILES:
                    file_path = get_file_path(DEFAULT_FILES[mode])
                    if os.path.exists(file_path):
                        files.append(file_path)
                        logger.info(f"Using {mode} file: {file_path}")
                    else:
                        logger.warning(f"File not found: {file_path}")
            
            if not files:
                logger.warning("No valid files found for processing")
                return
            
            result = self.processor.process_multimodal(files, system_prompt, user_prompt, max_tokens)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            config_status = "NEW CONFIG" if config_changed else "SCHEDULED RUN"
            print(f"\n[{timestamp}] Dynamic Prompt Result ({config_status}):")
            print("="*60)
            print(f"System: {system_prompt}")
            print(f"User: {user_prompt}")
            print(f"Modes: {modes}")
            print(f"Files: {files}")
            print("-"*60)
            print(result)
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error processing dynamic prompt: {e}")


def run_task(processor: GemmaCliProcessor, args):
    try:
        files = []
        for mode in args.modes:
            if mode in DEFAULT_FILES:
                file_path = get_file_path(DEFAULT_FILES[mode])
                if os.path.exists(file_path):
                    files.append(file_path)
                    logger.info(f"Using {mode} file: {file_path}")
                else:
                    logger.error(f"File not found: {file_path}")
                    return
        
        if not files:
            logger.error("No valid files found for processing")
            return
        
        if args.task == 'caption':
            if len(files) == 1:
                file_path = files[0]
                ext = pathlib.Path(file_path).suffix.lower()
                
                if ext in IMAGE_FILE_TYPES:
                    result = processor.process_image_captioning(file_path, args.user_text, args.max_tokens)
                elif ext in AUDIO_FILE_TYPES:
                    result = processor.process_audio_captioning(file_path, args.user_text, args.max_tokens)
                elif ext in VIDEO_FILE_TYPES:
                    result = processor.process_video_captioning(file_path, args.user_text, args.max_tokens)
                else:
                    logger.error(f"Unsupported file type: {ext}")
                    return
            else:
                system_prompt = "You are an expert multimodal analyst. Provide detailed, accurate captions describing the content across all provided media types."
                result = processor.process_multimodal(files, system_prompt, args.user_text, args.max_tokens)
        
        elif args.task == 'detect':
            if not args.event_description:
                logger.error("Event description is required for detection task")
                return
            
            if len(files) == 1:
                file_path = files[0]
                ext = pathlib.Path(file_path).suffix.lower()
                
                if ext in IMAGE_FILE_TYPES:
                    result = processor.process_image_detection(file_path, args.event_description, args.max_tokens)
                elif ext in AUDIO_FILE_TYPES:
                    result = processor.process_audio_detection(file_path, args.event_description, args.max_tokens)
                elif ext in VIDEO_FILE_TYPES:
                    result = processor.process_video_detection(file_path, args.event_description, args.max_tokens)
                else:
                    logger.error(f"Unsupported file type: {ext}")
                    return
            else:
                system_prompt = f"You are an expert multimodal event detector. Analyze all provided media and determine if the following event is occurring: '{args.event_description}'. Respond with 'YES' if detected, 'NO' if not, followed by explanation."
                result = processor.process_multimodal(files, system_prompt, args.user_text, args.max_tokens)
        
        else:
            logger.error(f"Unknown task: {args.task}")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Task: {args.task.upper()}")
        print("="*60)
        print(f"Modes: {args.modes}")
        print(f"Files: {files}")
        if args.event_description:
            print(f"Event: {args.event_description}")
        print("-"*60)
        print(result)
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running task: {e}")


def main():
    parser = argparse.ArgumentParser(description='Gemma3n CLI Tool')
    parser.add_argument('task', choices=['caption', 'detect', 'dynamic-prompting'], 
                       help='Task to perform')
    parser.add_argument('--modes', nargs='+', choices=['image', 'audio', 'video'], 
                       default=['image'], help='Media modes to process')
    parser.add_argument('--event-description', type=str, 
                       help='Event description for detection task')
    parser.add_argument('--user-text', type=str, default='', 
                       help='Additional user text/prompt')
    parser.add_argument('--max-tokens', type=int, default=100, 
                       help='Maximum tokens to generate')
    parser.add_argument('--period', type=int, default=0, 
                       help='Run periodically every N minutes (0 = run once)')
    parser.add_argument('--yaml-url', type=str, 
                       help='URL to YAML configuration file (required for dynamic-prompting)')
    
    args = parser.parse_args()
    
    processor = GemmaCliProcessor()
    
    if args.task == 'dynamic-prompting':
        if not args.yaml_url:
            logger.error("--yaml-url is required for dynamic-prompting mode")
            sys.exit(1)
        
        dynamic_processor = DynamicPromptProcessor(args.yaml_url)
        
        dynamic_processor.process_dynamic_prompt()
        
        schedule.every(5).minutes.do(dynamic_processor.process_dynamic_prompt)
        
        logger.info("Dynamic prompting mode started (5 minute intervals)")
        while True:
            schedule.run_pending()
            time.sleep(60)  
    
    else:
        if args.period > 0:
            logger.info(f"Starting periodic execution every {args.period} minutes")
            
            run_task(processor, args)
            
            schedule.every(args.period).minutes.do(run_task, processor, args)
            
            while True:
                schedule.run_pending()
                time.sleep(60)  
        else:
            run_task(processor, args)


if __name__ == '__main__':
    main() 