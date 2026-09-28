# MIDOG Mitosis Detector (`KongNet_Det_MIDOG_1`) on Google Cloud Vertex AI

This repository contains the production-grade deployment package for serving the official **MIDOG Challenge 1st-Place Winner model (`KongNet_Det_MIDOG_1`)** on **Google Cloud Vertex AI** as an autoscaling, deterministic microservice conforming to the **OncoGemma v6 contract** (SPEC-06 §4).

---

## v6 Service Contract (v2)

### Endpoints

1. **`GET /health`** (and `$AIP_HEALTH_ROUTE`)
   - Returns `200` with model status and weights SHA-256:
     ```json
     {
       "status": "healthy",
       "model": "KongNet_Det_MIDOG_1",
       "weights_sha256": "<sha256_hash>"
     }
     ```
   - Returns `503` if the model engine is not loaded.

2. **`GET /metadata`**
   - Returns model metadata and resolution constraints:
     ```json
     {
       "model": "KongNet_Det_MIDOG_1",
       "tiatoolbox": "2.1.3",
       "weights_sha256": "<sha256_hash>",
       "input_mpp": 0.25,
       "patch_px": 512,
       "output": "points",
       "deterministic": true,
       "contract": "v2"
     }
     ```

3. **`POST /predict`** (and `$AIP_PREDICT_ROUTE`)
   - Vertex AI prediction route accepting instance batches.

### Request & Response Formats

#### v2 Contract Format (Primary)
- **Input:** Lossless PNG image encoded in Base64 and the scan resolution in `mpp` (microns per pixel).
- **Validation:**
  - Image must be a valid PNG (non-PNG rejected per-instance with `{"points": [], "error": "invalid_format: input image must be PNG"}`).
  - Image resolution `mpp` must match `input_mpp` within 1% (otherwise `{"points": [], "error": "mpp_mismatch: expected 0.25, got <mpp>"}`).
  - Dimensions must equal `patch_px` (512×512) (otherwise `{"points": [], "error": "invalid_size: expected 512x512, got <w>x<h>"}`).
  - Detections are filtered by `min_prob` (default: `0.01`).

```json
{
  "instances": [
    {
      "image_png_b64": "<base64_encoded_png>",
      "mpp": 0.25,
      "min_prob": 0.01
    }
  ],
  "parameters": {
    "min_prob": 0.01
  }
}
```

**Response:**
```json
{
  "predictions": [
    {
      "points": [
        {"x": 256.0, "y": 256.0, "prob": 0.9421},
        {"x": 112.5, "y": 320.0, "prob": 0.5812}
      ],
      "error": null
    }
  ],
  "model_sha256": "<sha256_hash>"
}
```

#### Legacy Format (v5 Backward Compatibility)
Maintains backward compatibility with legacy OncoGemma v5 callers using `image_bytes`:
```json
{
  "instances": [
    {
      "image_bytes": "<base64_encoded_image>",
      "confidence_threshold": 0.25
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "boxes": [
        {
          "cx": 256.0,
          "cy": 256.0,
          "width": 48.0,
          "height": 48.0,
          "confidence": 0.9421
        }
      ]
    }
  ],
  "model_sha256": "<sha256_hash>"
}
```

---

## Testing & Acceptance

### Running Offline Unit & Contract Tests
Offline contract tests run with `FakeEngine` without needing a GPU or TIAToolbox:

```powershell
pip install -r requirements-dev.txt
python -m pytest tests -q
```

### Smoke Testing Local Container
```powershell
# Start local server or container
python main.py

# Run v2 smoke test
python test_predict.py --local

# Run legacy smoke test
python test_predict.py --local --legacy
```

---

## Deployment (PowerShell)

Deploy to Google Cloud Vertex AI with a single command:

```powershell
# Deploy KongNet_Det_MIDOG_1 on NVIDIA T4 GPU (Recommended for OncoGemma)
.\deploy.ps1 -ProjectId "oncogemma" -Region "us-central1" -DeployGPU

# Deploy on CPU (Testing / Development)
.\deploy.ps1 -ProjectId "oncogemma" -Region "us-central1" -DeployCPU
```
