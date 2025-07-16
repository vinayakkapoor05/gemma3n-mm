docker build -t gemma3n:dev --target build-base .

docker run --gpus all -it --rm   -v "$(pwd)":/workspace   -v ~/.cache/huggingface:/root/.cache/huggingface   -e HF_TOKEN="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"   -w /workspace   -p 8080:8000   gemma3n:dev