# syntax=docker/dockerfile:1.4
ARG CUDA_VERSION=12.8.1
ARG IMAGE_DISTRO=ubuntu22.04
ARG TARGETARCH

FROM nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS deps
ENV DEBIAN_FRONTEND=noninteractive \
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12

RUN --mount=type=cache,id=apt_lists_${TARGETARCH},target=/var/lib/apt/lists \
    --mount=type=cache,id=apt_archives_${TARGETARCH},target=/var/cache/apt/archives \
    apt-get update && apt-get upgrade -y && \
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

RUN --mount=type=cache,id=pip_cache_${TARGETARCH},target=/root/.cache/pip \
    uv pip install -U \
      torch torchvision torchaudio triton \
      uvicorn[standard] fastapi \
      --extra-index-url https://download.pytorch.org/whl/cu128

FROM venv AS build
WORKDIR /wheels

RUN uv pip install pynvml

COPY requirements.txt .
RUN --mount=type=cache,id=pip_cache_${TARGETARCH},target=/root/.cache/pip \
    uv pip install -r requirements.txt

RUN uv clean && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/cache/apt/*

ENV HF_HOME=/hf_cache
RUN mkdir -p /hf_cache/google/gemma-3n-e4b-it

RUN --mount=type=secret,id=hf_token \
    bash -euxc ' \
      export HF_TOKEN="$(cat /run/secrets/hf_token)" && \
      huggingface-cli download google/gemma-3n-e4b-it \
        --repo-type model \
        --cache-dir /hf_cache \
        --local-dir /hf_cache/google/gemma-3n-e4b-it \
        --resume --force \
    '

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${IMAGE_DISTRO} AS runtime

RUN --mount=type=cache,id=apt_lists_rt_${TARGETARCH},target=/var/lib/apt/lists \
    --mount=type=cache,id=apt_archives_rt_${TARGETARCH},target=/var/cache/apt/archives \
    apt-get update && \
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