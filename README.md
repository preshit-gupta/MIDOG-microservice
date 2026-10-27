# MIDOG Mitosis Detector (`KongNet_Det_MIDOG_1`) on Google Cloud Vertex AI

This repository contains the production-grade deployment package for serving the official **MIDOG Challenge 1st-Place Winner model (`KongNet_Det_MIDOG_1`)** on **Google Cloud Vertex AI** as an autoscaling, deterministic microservice conforming to the **OncoGemma v6 contract** (SPEC-06 §4).

---

## Contract v3: logits (OncoGemma v7.2)

v3 keeps the v2 request and response and adds one field: every point also carries its **`logit`**.
`/metadata` reports `"contract": "v3"`, `"point_fields": ["x", "y", "prob", "logit"]`,
`"peak_rule": "logit_peaks"`, `"peak_min_distance_um": 5.0`, `"nms_radius_um": 7.5` and
`"preproc": "imagenet_mean_std"`. A v2 client keeps working: it ignores the new field.

```json
{"predictions": [{"points": [{"x": 256.0, "y": 255.5, "prob": 1.0, "logit": 118.42}], "error": null}],
 "model_sha256": "978afb33..."}
```

**Why.** The model's probability map is a float32 sigmoid, which reads exactly 1.0 wherever the logit
passes about 17. On MIDOG++ breast, 80% of in-tissue candidates score 1.0, so peaks found on the
probability map sit anywhere inside the plateau and their probabilities cannot rank them. v3 finds
peaks on the **logit** map instead:
- local maxima at least 5 µm apart, at or above `logit(min_prob)` (inclusive);
- greedy suppression within 7.5 µm, strongest logit first (equal logits by row, then column);
- `prob` is the float32 sigmoid of the logit, as v2 returned it.

On 6 MIDOG++ breast images, logit peaks return the same 456 candidates as probability peaks, and
the logit ranks them better: AUROC 0.846 against 0.772.

**Normalisation.** TIAToolbox's `infer_batch` does not normalise; its engines call the model's
`preproc` (ImageNet mean and std) per patch. `KongNetEngine.logit_map` calls it explicitly.
Without it, a test tile's logits move by 144 on average and the detection moves off the figure.

**Checks (gate G1):**
- `pytest tests`: contract and engine tests, offline (no TIAToolbox, no GPU).
- `KONGNET_WEIGHTS=/path/KongNet_Det_MIDOG_1.pth pytest tests/test_golden_logits.py` reproduces the
  recorded logits of two deterministic synthetic tiles (`tests/golden_tiles.py`,
  `tests/fixtures/golden_logits.json`, from `scripts/golden_logits.py`). Run it in the built image
  with `KONGNET_WEIGHTS=/root/.tiatoolbox/models/KongNet_Det_MIDOG_1.pth`.
- `scripts/positive_control.py --contract v3 ...` on a labelled MIDOG++ image reports F1 by
  threshold and the AUROC of logit and prob among candidates at p >= 0.3.

The legacy v5 path (`image_bytes`) is unchanged: KongNet's own 0.99 peak threshold on probabilities.

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
       "tiatoolbox": "2.0.1",
       "weights_sha256": "<sha256_hash>",
       "input_mpp": 0.25,
       "ioconfig_input_mpp": 0.5,
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
  - Image resolution `mpp` must match `input_mpp` within 1% (otherwise `{"points": [], "error": "mpp_mismatch: expected 0.5, got <mpp>"}`). The service takes **0.25 µm/px** (native 40×), measured on MIDOG++ image 094 (82 labelled mitotic figures, 0.2298 µm/px): F1 0.87 at native resolution versus at most 0.27 when downsampled to the 0.5 µm/px in the model's TIAToolbox IO config, which `/metadata` reports as `ioconfig_input_mpp`. `patch_px` (512) comes from the IO config.
  - Coordinates are image pixels (`x` = column, `y` = row). TIAToolbox 2.0.1's patch-mode output has the two axes transposed; the service swaps them back (same measurement: F1 0.12 as returned, 0.87 swapped).
  - Dimensions must equal `patch_px` (512×512) (otherwise `{"points": [], "error": "invalid_size: expected 512x512, got <w>x<h>"}`).
  - Detections are filtered by `min_prob` (default: `0.01`). `min_prob` is also passed to KongNet's post-processing as its peak threshold (`threshold_abs`, 0.99 by default), so every candidate down to `min_prob` is returned and clients can sweep thresholds offline.
  - Weights: the image build records the SHA-256 of `KongNet_Det_MIDOG_1.pth` in `/app/WEIGHTS_SHA256`; the service refuses to start (503) if the loaded weights differ or the input resolution cannot be read. The legacy path keeps the model's 0.99 threshold.

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
