docker build -t gemma3n-mm .
docker run --gpus all -e HF_TOKEN=<your_hf_token> -p 8000:8000 --shm-size=1g -v ~/.cache/huggingface:/root/.cache/huggingface gemma3n-mm:latest