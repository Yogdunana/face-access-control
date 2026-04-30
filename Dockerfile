FROM python:3.11-slim

LABEL maintainer="Face Access Control Contributors"
LABEL description="Face Access Control Platform - CPU version"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config.yaml .
COPY README.md .

RUN mkdir -p data/face_data data/models

EXPOSE 8000

CMD ["python", "-m", "src.core.main"]
