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
# and record weights SHA-256 to /app/WEIGHTS_SHA256 (SPEC-06 §4 / WP-7.1)
RUN python -c "import torchvision, hashlib, os; \
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector; \
d = NucleusDetector(model='KongNet_Det_MIDOG_1', verbose=False); \
from tiatoolbox import constants; \
cache = getattr(constants, 'TIATOOLBOX_CACHE_DIR', os.path.expanduser('~/.tiatoolbox')); \
sha = None; \
for root, _, files in os.walk(cache): \
    for f in files: \
        if 'KongNet_Det_MIDOG_1' in f and f.endswith(('.pth', '.pt', '.tar', '.bin')): \
            sha = hashlib.sha256(open(os.path.join(root, f), 'rb').read()).hexdigest(); \
            break; \
if sha: \
    with open('/app/WEIGHTS_SHA256', 'w') as out: out.write(sha); \
    print(f'Recorded KongNet weights SHA256: {sha}')"

# Copy application files
COPY main.py engine.py ./

EXPOSE 8080
ENV PORT=8080
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV MODEL_ENGINE=kongnet

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
