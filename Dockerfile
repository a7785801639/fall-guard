FROM python:3.11-slim

ARG COMPUTE_PROFILE=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Для профиля cpu ставим лёгкий PyTorch CPU (~180 МБ вместо 2.8 ГБ CUDA),
# а для будущего ПК с NVIDIA (COMPUTE_PROFILE=gpu) — полноценный CUDA + onnxruntime-gpu.
RUN if [ "$COMPUTE_PROFILE" = "gpu" ]; then \
      pip install --upgrade pip && \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
      pip install -r /app/requirements.txt onnxruntime-gpu ; \
    else \
      pip install --upgrade pip && \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
      pip install -r /app/requirements.txt ; \
    fi

COPY app /app/app
COPY tests /app/tests

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8090/api/health || exit 1

CMD ["python", "-m", "app.main"]
