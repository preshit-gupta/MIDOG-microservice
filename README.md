# MIDOG Mitosis Detector (KongNet & YOLO) on Google Cloud Vertex AI

This repository contains the complete production-grade deployment package for serving the official **MIDOG Challenge 1st-Place Winner model (`KongNet_Det_MIDOG_1`)** alongside **YOLOv8** on **Google Cloud Vertex AI** as an autoscaling, low-latency microservice integrated with **OncoGemma** and **Gemini 1.5 Flash**.

---

## Architecture: Two-Stage MDFS (Mitosis Detection, Fast & Slow)

This service implements the Stage 1 High-Recall Candidate Sweeper of the two-stage MDFS architecture (Jahanifar et al., MedIA 2024), feeding mitosis candidates to Gemini 1.5 Flash:

```
                      +---------------------------------------+
                      |         OncoGemma Slide Viewer        |
                      |          (Digital Pathology)          |
                      +-------------------+-------------------+
                                          |
                                          | Tile: 512x512 Base64 JPEG/PNG
                                          v
                      +---------------------------------------+
                      |         Vertex AI Endpoint            |
                      |   (us-central1-aiplatform.googleapis) |
                      +-------------------+-------------------+
                                          |
                                          | HTTP POST /predict (Port 8080)
                                          v
                      +---------------------------------------+
                      |      Stage 1: Candidate Detector      |
                      |  - Model: KongNet_Det_MIDOG_1         |
                      |    (1st-Place MIDOG Challenge Winner) |
                      |  - Alternate: YOLOv8 (best.pt)        |
                      |  - GPU: NVIDIA T4 / L4 (Sub-100ms)    |
                      +-------------------+-------------------+
                                          |
                                          | Mitosis Candidates:
                                          | [cx, cy, width, height, conf]
                                          v
                      +---------------------------------------+
                      |    Stage 2: Gemini 1.5 Flash Referee   |
                      |   (Multi-modal van Diest Criteria)    |
                      +-------------------+-------------------+
                                          |
                                          | Final Certified Mitoses
                                          v
                      +---------------------------------------+
                      |     Pathologist UI & Decision Log     |
                      +---------------------------------------+
```

---

## Project Structure

```
d:\Projects\MIDOG\
├── main.py                  # FastAPI microservice conforming to Vertex AI contracts
├── requirements.txt         # Python runtime dependencies
├── Dockerfile               # CUDA-optimized container definition with pre-baked weights
├── deploy.ps1               # Automated end-to-end PowerShell deployment script
├── test_predict.py          # Client script for local and Vertex AI endpoint smoke tests
├── scripts/
│   └── setup_weights.py     # Helper to inspect, download, or bootstrap weights
└── README.md                # This comprehensive deployment guide
```

---

## Step 1: Model Selection & Pre-Warming

Vertex AI custom containers perform fastest when model weights are baked directly into the Docker image, eliminating cold-start downloads:

1. **Option A (Recommended & Default): KongNet_Det_MIDOG_1 (Challenge 1st Place Winner)**
   No manual weight download is needed! TIAToolbox automatically downloads and pre-warms the verified MIDOG Challenge winner weights during container build:
   ```powershell
   # Inspect model details
   python scripts\setup_weights.py --check-winner
   ```

2. **Option B: Custom Fine-Tuned YOLOv8x Checkpoint**
   If you have a custom trained `best.pt` file:
   ```powershell
   Copy-Item "C:\path\to\your\model.pt" "d:\Projects\MIDOG\best.pt"
   ```

---

## Step 2: Automated Deployment (PowerShell)

A fully automated PowerShell script orchestrates the entire GCP pipeline:

```powershell
# 1. Deploy KongNet_Det_MIDOG_1 on NVIDIA T4 GPU (Recommended for OncoGemma)
.\deploy.ps1 -ProjectId "oncogemma" -Region "us-central1" -DeployGPU

# 2. Or deploy on CPU only (Cost-saving testing option)
.\deploy.ps1 -ProjectId "oncogemma" -Region "us-central1" -DeployCPU

# 3. Or deploy custom YOLO if best.pt is present
.\deploy.ps1 -ProjectId "oncogemma" -Region "us-central1" -Engine yolo -DeployGPU
```

The script automatically executes:
1. Validates the selected engine (`kongnet` or `yolo`).
2. Enables `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`, and `aiplatform.googleapis.com`.
3. Creates the Artifact Registry repository `oncogemma-models`.
4. Submits the container build using **Google Cloud Build** (pre-warming KongNet weights inside the image).
5. Registers the container in the **Vertex AI Model Registry**.
6. Creates the **Vertex AI Endpoint** (`mitosis-detector-endpoint`).
7. Deploys the model to the endpoint with the specified compute resources (1x NVIDIA Tesla T4 GPU or CPU).
8. Outputs the final numeric `ENDPOINT_ID` and OncoGemma configuration line.

---

## Step 3: Manual Deployment Steps (Alternative via gcloud CLI)

If you prefer executing the `gcloud` commands manually step-by-step:

### 1. Set Environment Variables
```powershell
$PROJECT_ID = "oncogemma"
$REGION = "us-central1"
$REPO_NAME = "oncogemma-models"
$IMAGE_NAME = "yolov8-midog:v1"
```

### 2. Configure GCP
```powershell
gcloud config set project $PROJECT_ID
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com
```

### 3. Create Artifact Registry Repository
```powershell
gcloud artifacts repositories create $REPO_NAME `
    --repository-format=docker `
    --location=$REGION `
    --description="OncoGemma Model Containers"
```

### 4. Build and Push via Cloud Build
```powershell
gcloud builds submit --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"
```

### 5. Upload Model to Vertex AI Model Registry
```powershell
$IMAGE_URI = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

gcloud ai models upload `
    --region=$REGION `
    --display-name="yolov8-midog-v1" `
    --container-image-uri=$IMAGE_URI `
    --container-predict-route="/predict" `
    --container-health-route="/health" `
    --container-ports=8080
```

### 6. Create Vertex AI Endpoint
```powershell
gcloud ai endpoints create `
    --region=$REGION `
    --display-name="mitosis-detector-endpoint"
```

### 7. Deploy Model to Endpoint

**Option A: NVIDIA Tesla T4 GPU (Recommended for Sub-100ms Inference)**
```powershell
gcloud ai endpoints deploy-model YOUR_ENDPOINT_ID `
    --region=$REGION `
    --model="projects/${PROJECT_ID}/locations/${REGION}/models/YOUR_MODEL_ID" `
    --display-name="yolov8-midog-gpu-deploy" `
    --machine-type="n1-standard-4" `
    --accelerator="type=nvidia-tesla-t4,count=1" `
    --min-replica-count=1 `
    --max-replica-count=1
```

**Option B: CPU Only (Cost-Effective Baseline)**
```powershell
gcloud ai endpoints deploy-model YOUR_ENDPOINT_ID `
    --region=$REGION `
    --model="projects/${PROJECT_ID}/locations/${REGION}/models/YOUR_MODEL_ID" `
    --display-name="yolov8-midog-cpu-deploy" `
    --machine-type="e2-standard-4" `
    --min-replica-count=1 `
    --max-replica-count=1
```

---

## Step 4: Verify the Deployment

Use `test_predict.py` to test the live endpoint:

```powershell
# Test your deployed Vertex AI Endpoint
python test_predict.py `
    --endpoint-id YOUR_NUMERIC_ENDPOINT_ID `
    --project-id "oncogemma" `
    --region "us-central1"
```

Or test a specific tile image:
```powershell
python test_predict.py `
    --endpoint-id YOUR_NUMERIC_ENDPOINT_ID `
    --image "path\to\histology_crop.jpg"
```

---

## Step 5: Connect to OncoGemma

Once deployment completes, open your OncoGemma application directory and update your `.env` file:

```env
# =====================================================================
# MIDOG Mitosis Detection Microservice (Vertex AI)
# =====================================================================
MIDOG_VERTEX_ENDPOINT_ID=1234567890123456789
VERTEX_PROJECT_ID=oncogemma
VERTEX_REGION=us-central1
```

Your OncoGemma viewer will now route slide tiles directly to this Vertex AI endpoint and visualize mitotic figure bounding boxes in real-time.

---

## API Contract Specification

### 1. Health Probe (`GET /health`)
* **Response (HTTP 200):**
```json
{
  "status": "healthy"
}
```

### 2. Prediction (`POST /predict`)
* **Request:**
```json
{
  "instances": [
    {
      "image_bytes": "<base64_encoded_jpeg_or_png>"
    }
  ],
  "parameters": {
    "conf": 0.25,
    "imgsz": 512
  }
}
```

* **Response (HTTP 200):**
```json
{
  "predictions": [
    {
      "boxes": [
        {
          "cx": 256.45,
          "cy": 182.12,
          "width": 34.20,
          "height": 38.15,
          "confidence": 0.9124
        }
      ]
    }
  ]
}
```
* **Coordinate Schema:**
  * `cx`: Center X coordinate in pixels relative to the tile.
  * `cy`: Center Y coordinate in pixels relative to the tile.
  * `width`: Bounding box width in pixels.
  * `height`: Bounding box height in pixels.
  * `confidence`: Detection probability score (`0.0` - `1.0`).
