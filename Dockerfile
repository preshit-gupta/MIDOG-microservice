FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for OpenCV, OpenSlide, and imaging
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

# Install matching torchvision for PyTorch 2.2.0 (CUDA 12.1)
RUN pip install --no-cache-dir torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm KongNet_Det_MIDOG_1 weights (MIDOG Challenge Winner) into Docker cache
# and verify that NucleusDetector and torchvision work without error during build
RUN python -c "import torchvision; from tiatoolbox.models.engine.nucleus_detector import NucleusDetector; NucleusDetector(model='KongNet_Det_MIDOG_1')"
RUN python -c "from ultralytics import YOLO" || true

# Copy application files (best.pt* is optional via wildcard)
COPY main.py .
COPY best.pt* ./

EXPOSE 8080
ENV PORT=8080
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV MODEL_ENGINE=kongnet

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
