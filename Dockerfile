FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3-pip python3-dev git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip \
 && pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

ENV HF_TOKEN=""

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
