# Gemma3n-mm


```bash
docker build -t gemma3n:dev --target build-base .
docker run --env-file .env -p 8080:8080 gemma3n:dev
```

```bash
docker build --build-arg HF_TOKEN=your_token -t gemma3n:dev .
```
locally

```bash
pip install -r requirements.txt
```

### Server Mode

```bash
python gemma3n.py serve --host 0.0.0.0 --port 8080
```


### CLI Mode

#### Basic Usage

```bash
python gemma3n.py cli <task> [options]
```


#### Captioning Tasks

```bash
# Caption an image
python cli.py caption --modes image

# Caption audio
python cli.py caption --modes audio

# Caption video
python cli.py caption --modes video

# Multimodal captioning (image + audio)
python cli.py caption --modes image audio

# Custom user prompt
python cli.py caption --modes image --user-text "Focus on the animals"
```

#### Detection Tasks

```bash
# Detect events in an image
python cli.py detect --modes image --event-description "people walking"

# Detect events in audio
python cli.py detect --modes audio --event-description "music playing"

# Detect events in video
python cli.py detect --modes video --event-description "person speaking"

# Multimodal event detection
python cli.py detect --modes image audio --event-description "doorbell ringing"
```

#### Periodic Processing

```bash
# Run every 10 minutes
python cli.py caption --modes image --period 10

# Run detection every 5 minutes
python cli.py detect --modes audio --event-description "alarm sound" --period 5
```

#### Dynamic Prompting Mode

Remote config via YAML files

```bash
# Start dynamic prompting with remote config
python cli.py dynamic-prompting --yaml-url "https://gist.githubusercontent.com/user/id/raw/config.yaml"
```

- Polls YAML file every 5 minutes
- Only processes when configuration changes
- Supports all media modes and custom prompts


YAML file structure:

```yaml
system_prompt: |
  You are an expert analyst. Analyze the provided content for...

user_prompt: |
  Please provide a detailed analysis focusing on...

modes:
  - image
  - audio

max_tokens: 200
```

## API Endpoints

### Audio Processing
- `POST /audio/captioning` - Generate audio descriptions
- `POST /audio/event_detection` - Detect events in audio

### Vision Processing
- `POST /vision/image_classification` - Classify images
- `POST /vision/image_event_detection` - Detect events in images

### Video Processing
- `POST /video/captioning` - Generate video descriptions
- `POST /video/event_detection` - Detect events in videos

### Multimodal Processing
- `POST /multimodal/` - Custom multimodal analysis
- `POST /multimodal/audio_vision` - Combined audio and image analysis
- `POST /multimodal/audio_video` - Combined audio and video analysis

### Utility Endpoints
- `GET /health` - Health check
- `GET /endpoints` - List all available endpoints

## Examples

### CLI Examples

```bash
# Simple image captioning
python cli.py caption --modes image

# Wildfire monitoring with periodic checks
python cli.py detect --modes image audio --event-description "wildfire" --period 15

# Dynamic monitoring
python cli.py dynamic-prompting --yaml-url "https://example.com/config.yaml"
```

### Server Examples

```bash
curl -X POST "http://localhost:8080/audio/captioning" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "user_text=Describe the audio content"

curl -X POST "http://localhost:8080/multimodal/" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image.jpg" \
  -F "files=@audio.mp3" \
  -F "system_prompt=You are a helpful analyst" \
  -F "user_text=What's happening here?"
```
