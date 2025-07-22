# waggle_cli.py
import argparse
import os
import pathlib
import sys
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from waggle.plugin import Plugin
from waggle.data.vision import Camera, ImageFolder
from waggle.data.audio import Microphone, AudioFolder

from cli import GemmaCliProcessor
from src.utils import IMAGE_FILE_TYPES, AUDIO_FILE_TYPES, VIDEO_FILE_TYPES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_FILES = {
    'image': 'assets/image.jpg',
    'audio': 'assets/audio.mp3', 
    'video': 'assets/video.mp4'
}

def get_file_path(filename):
    if os.path.exists(filename):
        return filename
    
    assets_path = os.path.join('assets', filename)
    if os.path.exists(assets_path):
        return assets_path
    
    data_path = os.path.join('data', filename)
    if os.path.exists(data_path):
        return data_path
    
    return filename

class WaggleMediaProcessor:
    
    def __init__(self):
        self.processor = GemmaCliProcessor()
        
    def capture_live_image(self, camera_device=None):
        try:
            if camera_device:
                camera = Camera(camera_device)
            else:
                camera = Camera()
            
            sample = camera.snapshot()
            
            temp_path = f"/tmp/live_capture_{int(time.time())}.jpg"
            sample.save(temp_path)
            
            return temp_path, sample.timestamp
        except Exception as e:
            logger.error(f"Failed to capture live image: {e}")
            return None, None
    
    def capture_live_audio(self, duration=10):
        try:
            with Microphone() as microphone:
                sample = microphone.record(duration)
                
                temp_path = f"/tmp/live_audio_{int(time.time())}.ogg"
                sample.save(temp_path)
                
                return temp_path, sample.timestamp
        except Exception as e:
            logger.error(f"Failed to capture live audio: {e}")
            return None, None
    
    def process_and_publish(self, plugin, task, modes, files, event_description=None, 
                          user_text="", max_tokens=100, use_live_capture=False, 
                          audio_duration=10, camera_device=None):
        
        processed_files = []
        capture_timestamps = {}
        
        if use_live_capture:
            if 'image' in modes:
                live_image_path, timestamp = self.capture_live_image(camera_device)
                if live_image_path:
                    processed_files.append(live_image_path)
                    capture_timestamps[live_image_path] = timestamp
                    logger.info(f"Captured live image: {live_image_path}")
            
            if 'audio' in modes:
                live_audio_path, timestamp = self.capture_live_audio(audio_duration)
                if live_audio_path:
                    processed_files.append(live_audio_path)
                    capture_timestamps[live_audio_path] = timestamp
                    logger.info(f"Captured live audio: {live_audio_path}")
        else:
            for mode in modes:
                if mode in DEFAULT_FILES:
                    file_path = get_file_path(DEFAULT_FILES[mode])
                    if os.path.exists(file_path):
                        processed_files.append(file_path)
                        logger.info(f"Using {mode} file: {file_path}")
                    else:
                        logger.warning(f"File not found: {file_path}")
        
        if not processed_files:
            logger.error("No valid files found for processing")
            return
        
        try:
            with plugin.timeit("plugin.duration.processing"):
                if task == 'caption':
                    result = self._process_captioning(processed_files, user_text, max_tokens)
                elif task == 'detect':
                    if not event_description:
                        logger.error("Event description is required for detection task")
                        return
                    result = self._process_detection(processed_files, event_description, max_tokens)
                else:
                    logger.error(f"Unknown task: {task}")
                    return
            
            timestamp = int(time.time() * 1e9)  
            
            plugin.publish(f"gemma3n.{task}.result", result, timestamp=timestamp, 
                         meta={
                             "modes": modes,
                             "files": [os.path.basename(f) for f in processed_files],
                             "model": "gemma-3n",
                             "event_description": event_description,
                             "live_capture": use_live_capture
                         })
            
            if len(processed_files) > 1:
                for file_path in processed_files:
                    file_mode = self._get_file_mode(file_path)
                    if file_mode:
                        plugin.publish(f"gemma3n.{task}.{file_mode}", result, 
                                     timestamp=capture_timestamps.get(file_path, timestamp),
                                     meta={
                                         "file": os.path.basename(file_path),
                                         "mode": file_mode,
                                         "live_capture": file_path in capture_timestamps
                                     })
            
            for file_path in processed_files:
                if file_path in capture_timestamps:
                    plugin.upload_file(file_path, 
                                     meta={
                                         "mode": self._get_file_mode(file_path),
                                         "capture_timestamp": capture_timestamps[file_path]
                                     })
            
            logger.info(f"Successfully processed and published results for modes: {modes}")
            
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            capture_status = "LIVE CAPTURE" if use_live_capture else "FILE PROCESSING"
            print(f"\n[{timestamp_str}] {task.upper()} Results ({capture_status}):")
            print("="*60)
            print(f"Modes: {modes}")
            print(f"Files: {[os.path.basename(f) for f in processed_files]}")
            if event_description:
                print(f"Event: {event_description}")
            print("-"*60)
            print(result)
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            plugin.publish("gemma3n.error", str(e), 
                         meta={"modes": modes, "task": task})
    
    def _get_file_mode(self, file_path):
        ext = pathlib.Path(file_path).suffix.lower()
        if ext in IMAGE_FILE_TYPES:
            return "image"
        elif ext in AUDIO_FILE_TYPES:
            return "audio"
        elif ext in VIDEO_FILE_TYPES:
            return "video"
        return None
    
    def _process_captioning(self, files, user_text, max_tokens):
        if len(files) == 1:
            file_path = files[0]
            ext = pathlib.Path(file_path).suffix.lower()
            
            if ext in IMAGE_FILE_TYPES:
                return self.processor.process_image_captioning(file_path, user_text, max_tokens)
            elif ext in AUDIO_FILE_TYPES:
                return self.processor.process_audio_captioning(file_path, user_text, max_tokens)
            elif ext in VIDEO_FILE_TYPES:
                return self.processor.process_video_captioning(file_path, user_text, max_tokens)
        else:
            system_prompt = "You are an expert multimodal analyst. Provide detailed, accurate captions describing the content across all provided media types."
            return self.processor.process_multimodal(files, system_prompt, user_text, max_tokens)
    
    def _process_detection(self, files, event_description, max_tokens):
        if len(files) == 1:
            file_path = files[0]
            ext = pathlib.Path(file_path).suffix.lower()
            
            if ext in IMAGE_FILE_TYPES:
                return self.processor.process_image_detection(file_path, event_description, max_tokens)
            elif ext in AUDIO_FILE_TYPES:
                return self.processor.process_audio_detection(file_path, event_description, max_tokens)
            elif ext in VIDEO_FILE_TYPES:
                return self.processor.process_video_detection(file_path, event_description, max_tokens)
        else:
            system_prompt = f"You are an expert multimodal event detector. Analyze all provided media and determine if the following event is occurring: '{event_description}'. Respond with 'YES' if detected, 'NO' if not, followed by explanation."
            return self.processor.process_multimodal(files, system_prompt, "", max_tokens)


def main():
    parser = argparse.ArgumentParser(description='Gemma3n Waggle CLI Tool - Process media and publish to Waggle ecosystem')
    parser.add_argument('task', choices=['caption', 'detect'], 
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
                       help='Run periodically every N seconds (0 = run once)')
    parser.add_argument('--live-capture', action='store_true',
                       help='Use live camera/microphone capture instead of files')
    parser.add_argument('--audio-duration', type=int, default=10,
                       help='Duration in seconds for live audio capture')
    parser.add_argument('--camera-device', type=str,
                       help='Camera device ID or URL for live capture')
    parser.add_argument('--log-dir', type=str,
                       help='Directory for pywaggle run logs')
    
    args = parser.parse_args()
    
    if args.log_dir:
        os.environ['PYWAGGLE_LOG_DIR'] = args.log_dir
        logger.info(f"Pywaggle logs will be saved to: {args.log_dir}")
    
    processor = WaggleMediaProcessor()
    
    def run_processing():
        try:
            with Plugin() as plugin:
                processor.process_and_publish(
                    plugin=plugin,
                    task=args.task,
                    modes=args.modes,
                    files=[],  
                    event_description=args.event_description,
                    user_text=args.user_text,
                    max_tokens=args.max_tokens,
                    use_live_capture=args.live_capture,
                    audio_duration=args.audio_duration,
                    camera_device=args.camera_device
                )
        except Exception as e:
            logger.error(f"Error in processing cycle: {e}")
    
    if args.period > 0:
        logger.info(f"Starting periodic execution every {args.period} seconds")
        logger.info(f"Task: {args.task}, Modes: {args.modes}, Live capture: {args.live_capture}")
        
        run_processing()
        
        while True:
            time.sleep(args.period)
            run_processing()
    else:
        logger.info(f"Running single execution - Task: {args.task}, Modes: {args.modes}")
        run_processing()


if __name__ == '__main__':
    main() 