ARG CUDA_VERSION=11.8.0
ARG IMAGE_DISTRO=ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS deps
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive \
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      python3 python3-venv python3-distutils \
      curl gcc-12 g++-12 git \
      libibverbs-dev libjpeg-turbo8-dev libpng-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin sh

FROM deps AS venv

RUN python3 -m venv --copies /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN uv pip install -U \
      torch torchvision torchaudio triton \
      uvicorn[standard] fastapi \
      --extra-index-url https://download.pytorch.org/whl/cu118

FROM venv AS build
WORKDIR /wheels

RUN uv pip install pynvml

COPY requirements.txt .
RUN uv pip install -r requirements.txt

RUN uv clean && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/cache/apt/*

ENV HF_HOME=/hf_cache
RUN mkdir -p /hf_cache/google/gemma-3n-e2b-it

ARG HF_TOKEN
RUN bash -euxc ' \
      huggingface-cli download google/gemma-3n-e2b-it \
        --repo-type model \
        --cache-dir /hf_cache \
        --local-dir /hf_cache/google/gemma-3n-e2b-it \
        --token "$HF_TOKEN" \
        --resume --force \
    '

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${IMAGE_DISTRO} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-distutils libexpat1 zlib1g libbz2-1.0 liblzma5 libffi7 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf_cache

COPY --from=build /opt/venv /opt/venv
COPY --from=build /hf_cache /hf_cache

RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless

WORKDIR /app
COPY gemma3n.py main.py app.py waggle_cli.py cli.py /app/
COPY src/ /app/src/

RUN if [ -d /app/src ] && [ ! -f /app/src/__init__.py ]; then \
      touch /app/src/__init__.py; \
    fi

EXPOSE 8000
ENTRYPOINT ["python3", "gemma3n.py"]