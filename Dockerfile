
ARG CUDA_VERSION=12.8.1
ARG IMAGE_DISTRO=ubuntu22.04
ARG PYTHON_VERSION=3.12

# Base
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS base

# Scaling
ARG MAX_JOBS=32
ENV MAX_JOBS=${MAX_JOBS}
ARG NVCC_THREADS=2
ENV NVCC_THREADS=${NVCC_THREADS}

ARG TORCH_CUDA_ARCH_LIST="9.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Update apt packages and install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt upgrade -y
RUN apt install -y --no-install-recommends \
    curl \
    gcc-12 g++-12 \
    git \
    libibverbs-dev \
    libjpeg-turbo8-dev \
    libpng-dev \
    zlib1g-dev

# Set compiler paths
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Setup build workspace
WORKDIR /workspace

# Prep build venv
ARG PYTHON_VERSION
RUN uv venv -p ${PYTHON_VERSION} --seed --python-preference only-managed
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

FROM base AS torch-base
RUN uv pip install -U torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu128

FROM torch-base AS build-base
RUN mkdir /wheels

# Install pynvml
RUN uv pip install pynvml

# Install requirements.txt
COPY requirements.txt .

RUN uv pip install -r requirements.txt

# Clean uv cache
RUN uv clean

# Clean apt cache
RUN apt autoremove --purge -y
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /var/cache/apt/archives

# Enable hf-transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY app.py .

EXPOSE 8000

ENV HF_TOKEN=""

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
