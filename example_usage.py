
import requests
import json
import os
from pathlib import Path


BASE_URL = "http://localhost:8080"  


def test_audio_captioning(audio_file_path: str):
    url = f"{BASE_URL}/audio/captioning"
    
    with open(audio_file_path, 'rb') as f:
        files = {'file': f}
        data = {'user_text': 'Describe this audio in detail', 'max_new_tokens': 100}
        
        response = requests.post(url, files=files, data=data)
        
    print("Audio Captioning Result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_audio_event_detection(audio_file_path: str, event: str):
    url = f"{BASE_URL}/audio/event_detection"
    
    with open(audio_file_path, 'rb') as f:
        files = {'file': f}
        data = {'event_description': event, 'max_new_tokens': 50}
        
        response = requests.post(url, files=files, data=data)
        
    print(f"Audio Event Detection Result (Event: {event}):")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_image_classification(image_file_path: str, categories: str = ""):
    url = f"{BASE_URL}/vision/image_classification"
    
    with open(image_file_path, 'rb') as f:
        files = {'file': f}
        data = {'categories': categories, 'max_new_tokens': 50}
        
        response = requests.post(url, files=files, data=data)
        
    print("Image Classification Result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_image_event_detection(image_file_path: str, event: str):
    url = f"{BASE_URL}/vision/image_event_detection"
    
    with open(image_file_path, 'rb') as f:
        files = {'file': f}
        data = {'event_description': event, 'max_new_tokens': 50}
        
        response = requests.post(url, files=files, data=data)
        
    print(f"Image Event Detection Result (Event: {event}):")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_image_change_detection(image1_path: str, image2_path: str):
    url = f"{BASE_URL}/vision/image_change_detection"
    
    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        files = {'file1': f1, 'file2': f2}
        data = {'max_new_tokens': 100}
        
        response = requests.post(url, files=files, data=data)
        
    print("Image Change Detection Result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def test_bounding_box_detection(image_file_path: str, categories: str, draw_boxes: bool = False, save_to_beehive: bool = False):
    pass

def test_video_captioning(video_file_path: str):
    url = f"{BASE_URL}/video/captioning"
    
    with open(video_file_path, 'rb') as f:
        files = {'file': f}
        data = {'user_text': 'Describe what happens in this video', 'max_new_tokens': 150}
        
        response = requests.post(url, files=files, data=data)
        
    print("Video Captioning Result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_video_event_detection(video_file_path: str, event: str):
    url = f"{BASE_URL}/video/event_detection"
    
    with open(video_file_path, 'rb') as f:
        files = {'file': f}
        data = {'event_description': event, 'max_new_tokens': 100}
        
        response = requests.post(url, files=files, data=data)
        
    print(f"Video Event Detection Result (Event: {event}):")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_multimodal_audio_vision(audio_file_path: str, image_file_path: str):
    url = f"{BASE_URL}/multimodal/audio_vision"
    
    with open(audio_file_path, 'rb') as af, open(image_file_path, 'rb') as if_:
        files = {'audio_file': af, 'image_file': if_}
        data = {'user_text': 'Analyze the relationship between the audio and image', 'max_new_tokens': 150}
        
        response = requests.post(url, files=files, data=data)
        
    print("Multimodal Audio-Vision Result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_multimodal_audio_video(audio_file_path: str, video_file_path: str):
    url = f"{BASE_URL}/multimodal/audio_video"
    
    with open(audio_file_path, 'rb') as af, open(video_file_path, 'rb') as vf:
        files = {'audio_file': af, 'video_file': vf}
        data = {'user_text': 'Analyze the relationship between the audio and video', 'max_new_tokens': 200}
        
        response = requests.post(url, files=files, data=data)
        
    print("Multimodal Audio-Video Result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def list_available_endpoints():
    url = f"{BASE_URL}/endpoints"
    response = requests.get(url)
    
    print("Available Endpoints:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def main():
    print("Gemma-3n mm API Examples")
    print("=" * 50)
    
    list_available_endpoints()
    
    audio_file = "assets/audio.mp3"
    image_file = "assets/image.jpg"
    image_file_2 = "assets/image_2.jpg"
    video_file = "assets/video.mp4"
    person_file = "assets/person.jpg"

    if os.path.exists(audio_file):
        test_audio_captioning(audio_file)
        test_audio_event_detection(audio_file, "music playing")
    
    if os.path.exists(image_file):
        test_image_classification(image_file, "nature, urban, indoor, outdoor")
        test_image_event_detection(image_file, "people walking")
    
    if os.path.exists(image_file_2) and os.path.exists(image_file):
        test_image_change_detection(image_file, image_file_2)
    
    if os.path.exists(video_file):
        test_video_captioning(video_file)
        test_video_event_detection(video_file, "person speaking")
    
    if os.path.exists(audio_file) and os.path.exists(image_file):
        test_multimodal_audio_vision(audio_file, image_file)
    
    if os.path.exists(audio_file) and os.path.exists(video_file):
        test_multimodal_audio_video(audio_file, video_file)
    

if __name__ == "__main__":
    main() 