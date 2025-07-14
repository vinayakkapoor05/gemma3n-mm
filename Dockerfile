FROM ollama/ollama

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

ENTRYPOINT ["python3"]
CMD ["main.py"]
