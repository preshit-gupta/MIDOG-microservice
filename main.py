import base64
import io
import os
import logging
from typing import Any, Dict, List
from fastapi import FastAPI, Request, HTTPException
from PIL import Image
import numpy as np

from engine import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("midog-detector")

app = FastAPI(title="MIDOG Mitosis Detector for Vertex AI (v2 Contract)")

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")


def loaded_engine():
    """The engine, or 503 with the reason it could not load (logged; the next call retries)."""
    try:
        return get_engine()
    except Exception as err:  # any load failure means "not ready": report it, never serve without it
        logger.error(f"Model engine failed to load: {type(err).__name__}: {err}")
        raise HTTPException(status_code=503, detail=f"Model engine not loaded: {type(err).__name__}")


# Load the engine at startup; a failure is logged and /health answers 503 until it loads.
try:
    get_engine()
except Exception as err:  # the probes report it; see loaded_engine
    logger.error(f"Engine failed to load at startup: {type(err).__name__}: {err}")


@app.get(HEALTH_ROUTE)
@app.get("/health")
def health() -> Dict[str, Any]:
    """Vertex AI liveness and readiness probe."""
    engine = loaded_engine()
    return {
        "status": "healthy",
        "model": "KongNet_Det_MIDOG_1",
        "weights_sha256": engine.weights_sha256,
    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    """v2 contract metadata endpoint (SPEC-06 §4)."""
    engine = loaded_engine()
    return {
        "model": "KongNet_Det_MIDOG_1",
        "tiatoolbox": engine.tiatoolbox_version,
        "weights_sha256": engine.weights_sha256,
        "input_mpp": engine.input_mpp,
        "patch_px": engine.patch_px,
        "output": "points",
        "deterministic": True,
        "contract": "v2",
    }


@app.post(PREDICT_ROUTE)
@app.post("/predict")
async def predict(request: Request) -> Dict[str, Any]:
    """
    Accepts Vertex AI prediction payloads.
    Differentiates between v2 (lossless PNG, resolution-checked) and legacy formats per instance.
    """
    engine = loaded_engine()

    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    instances = body.get("instances", [])
    parameters = body.get("parameters", {})
    default_min_prob = float(parameters.get("min_prob", 0.01))
    default_conf = float(parameters.get("conf", 0.25))

    weights_sha = engine.weights_sha256

    if not instances:
        return {"predictions": [], "model_sha256": weights_sha}

    predictions: List[Dict[str, Any]] = []

    for idx, item in enumerate(instances):
        # 1. v2 Contract Format: image_png_b64 + mpp
        if "image_png_b64" in item:
            b64_str = item.get("image_png_b64")
            if not b64_str:
                predictions.append({"points": [], "error": "empty_image_data"})
                continue

            try:
                raw_bytes = base64.b64decode(b64_str)
            except Exception as e:
                predictions.append({"points": [], "error": f"invalid_base64: {e}"})
                continue

            # Check lossless PNG format (magic bytes: \x89PNG\r\n\x1a\n)
            if len(raw_bytes) < 8 or raw_bytes[:8] != b"\x89PNG\r\n\x1a\n":
                predictions.append({"points": [], "error": "invalid_format: input image must be PNG"})
                continue

            # Check MPP resolution
            mpp = item.get("mpp")
            if mpp is None:
                predictions.append({"points": [], "error": "missing_mpp: 'mpp' field is required in v2 contract"})
                continue

            try:
                mpp_val = float(mpp)
            except (ValueError, TypeError):
                predictions.append({"points": [], "error": f"invalid_mpp: expected float, got {mpp}"})
                continue

            expected_mpp = float(engine.input_mpp)
            if abs(mpp_val - expected_mpp) / expected_mpp > 0.01:
                predictions.append({
                    "points": [],
                    "error": f"mpp_mismatch: expected {expected_mpp}, got {mpp_val}",
                })
                continue

            # Check image size
            expected_px = int(engine.patch_px)
            try:
                img = Image.open(io.BytesIO(raw_bytes))
                if img.width != expected_px or img.height != expected_px:
                    predictions.append({
                        "points": [],
                        "error": f"invalid_size: expected {expected_px}x{expected_px}, got {img.width}x{img.height}",
                    })
                    continue
                img_rgb = np.array(img.convert("RGB"))
            except Exception as e:
                predictions.append({"points": [], "error": f"image_decode_error: {e}"})
                continue

            # Predict and filter by min_prob. min_prob is also KongNet's post-processing peak
            # threshold (its own default is 0.99), so candidates down to min_prob are returned.
            item_min_prob = float(item.get("min_prob", default_min_prob))
            try:
                raw_points = engine.predict(img_rgb, threshold_abs=item_min_prob)
                filtered_points = [
                    {
                        "x": round(float(x), 2),
                        "y": round(float(y), 2),
                        "prob": round(float(p), 4),
                    }
                    for x, y, p in raw_points
                    if float(p) >= item_min_prob
                ]
                predictions.append({"points": filtered_points, "error": None})
            except Exception as e:
                logger.error(f"Inference error on v2 instance #{idx}: {e}")
                predictions.append({"points": [], "error": f"inference_error: {e}"})

        # 2. Legacy v5 Format: image_bytes
        elif "image_bytes" in item:
            logger.info("legacy_contract_used")
            b64_str = item.get("image_bytes")
            if not b64_str:
                predictions.append({"boxes": []})
                continue

            try:
                img_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_rgb = np.array(img)

                item_conf = float(item.get("confidence_threshold", default_conf))
                # Unchanged v5 behaviour: KongNet's own post-processing threshold (0.99).
                raw_points = engine.predict(img_rgb)

                boxes = [
                    {
                        "cx": round(float(x), 2),
                        "cy": round(float(y), 2),
                        "width": 48.0,
                        "height": 48.0,
                        "confidence": round(float(p), 4),
                    }
                    for x, y, p in raw_points
                    if float(p) >= item_conf
                ]
                predictions.append({"boxes": boxes})
            except Exception as err:
                logger.error(f"Error processing legacy instance #{idx}: {err}")
                predictions.append({"boxes": [], "error": str(err)})

        else:
            predictions.append({
                "points": [],
                "error": "invalid_instance: expected 'image_png_b64' (v2) or 'image_bytes' (legacy)",
            })

    return {
        "predictions": predictions,
        "model_sha256": weights_sha,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AIP_HTTP_PORT", os.environ.get("PORT", "8080")))
    uvicorn.run(app, host="0.0.0.0", port=port)
