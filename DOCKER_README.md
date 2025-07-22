```bash
export DOCKER_BUILDKIT=1
docker build   --no-cache   --secret id=hf_token,src=hf_token.txt   -t gemma3n:latest   .
```

```bash
docker run -d --gpus all -p 8080:8000 \
  -e HF_TOKEN="$(cat hf_token.txt)" \
  -v "$(pwd)/hf_cache:/hf_cache" \
  --name gemma3n_server \
  gemma3n:latest serve
```