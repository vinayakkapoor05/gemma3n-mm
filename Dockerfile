ARG CUDA_VERSION=12.8.1
ARG IMAGE_DISTRO=ubuntu22.04
ARG PYTHON_VERSION=3.12

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS base

ARG MAX_JOBS=32
ARG NVCC_THREADS=2
ARG TORCH_CUDA_ARCH_LIST="9.0a"

ENV MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    DEBIAN_FRONTEND=noninteractive \
    CC=/usr/bin/gcc-12 \
    CXX=/usr/bin/g++-12 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends \ 
    python3 python3-venv python3-distutils \       
      curl gcc-12 g++-12 git \
      libibverbs-dev libjpeg-turbo8-dev libpng-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin sh

ARG PYTHON_VERSION
RUN uv venv -p python3 --seed --python-preference only-managed /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

FROM base AS torch-base
RUN uv pip install -U \
      torch torchvision torchaudio triton \
      --index-url https://download.pytorch.org/whl/cu128

FROM torch-base AS build-base

RUN mkdir /wheels
RUN uv pip install pynvml
COPY requirements.txt .
RUN uv pip install -r requirements.txt

RUN uv clean && \
    apt autoremove --purge -y && \
    apt clean && \
    rm -rf /var/cache/apt/*

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_TOKEN=""

# create a local cache directory for the model
RUN mkdir -p /hf_cache/google/gemma-3n-e4b-it

ENV HF_HOME=/hf_cache
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Download the model only if HF_TOKEN is provided
RUN if [ -n "$HF_TOKEN" ]; then \
    echo "Downloading Gemma-3n model with authentication..."; \
    huggingface-cli download \
        google/gemma-3n-e4b-it \
        --repo-type model \
        --cache-dir /hf_cache \
        --local-dir /hf_cache/google/gemma-3n-e4b-it \
        --resume \
        --force; \
    echo "Model downloaded successfully"; \
else \
    echo "No HF_TOKEN provided, model will be downloaded at runtime"; \
fi

# Copy project files
WORKDIR /app
COPY main.py app.py ./
COPY src/ ./src/
COPY *.py ./

EXPOSE 8000

ENTRYPOINT ["python3", "gemma3n.py"]
