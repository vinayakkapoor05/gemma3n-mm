# syntax=docker/dockerfile:1.4
ARG CUDA_VERSION=12.8.1
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
      --extra-index-url https://download.pytorch.org/whl/cu128

FROM venv AS build
WORKDIR /wheels

RUN uv pip install pynvml

COPY requirements.txt .
RUN uv pip install -r requirements.txt

RUN uv clean && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/cache/apt/*

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${IMAGE_DISTRO} AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-distutils libexpat1 zlib1g libbz2-1.0 liblzma5 libffi7 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf_cache
COPY --from=build /opt/venv /opt/venv

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