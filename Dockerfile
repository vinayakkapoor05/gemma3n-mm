ARG CUDA_VERSION=11.8.0
ARG IMAGE_DISTRO=ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO}

ARG TARGETARCH
ARG HF_TOKEN

ENV DEBIAN_FRONTEND=noninteractive \
    CC=/usr/bin/gcc-12 \
    CXX=/usr/bin/g++-12 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf_cache

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-distutils \
      curl gcc-12 g++-12 git \
      libibverbs-dev libjpeg-turbo8-dev libpng-dev zlib1g-dev \
      libexpat1 zlib1g libbz2-1.0 liblzma5 libffi7 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --copies /opt/venv

RUN pip install --no-cache-dir -U \
    torch torchvision torchaudio triton \
    uvicorn[standard] fastapi \
    pynvml \
    --extra-index-url https://download.pytorch.org/whl/cu118

WORKDIR /wheels

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir opencv-python-headless

RUN mkdir -p /hf_cache/google/gemma-3n-e2b-it

RUN bash -euxc ' \
      huggingface-cli download google/gemma-3n-e2b-it \
        --repo-type model \
        --cache-dir /hf_cache \
        --local-dir /hf_cache/google/gemma-3n-e2b-it \
        --token "$HF_TOKEN" \
        --resume --force \
    '
RUN apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

WORKDIR /app
COPY gemma3n.py main.py app.py waggle_cli.py cli.py /app/
COPY src/ /app/src/

RUN if [ -d /app/src ] && [ ! -f /app/src/__init__.py ]; then \
      touch /app/src/__init__.py; \
    fi

EXPOSE 8000
ENTRYPOINT ["python3", "gemma3n.py"]