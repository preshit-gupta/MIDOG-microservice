import base64
import io
import math
import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Request, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image
import numpy as np

from engine import NMS_RADIUS_UM, PEAK_MIN_DISTANCE_UM, float32_sigmoid, get_engine, logit_of
from tile_pack import TilePackError, read_tile_pack

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("midog-detector")

app = FastAPI(title="MIDOG Mitosis Detector for Vertex AI (v3 Contract)")

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
    """Contract metadata (SPEC-06 §4; v3: OncoGemma v7.2 Stage 4 spec §4.2)."""
    engine = loaded_engine()
    return {
        "model": "KongNet_Det_MIDOG_1",
        "tiatoolbox": engine.tiatoolbox_version,
        "weights_sha256": engine.weights_sha256,
        "input_mpp": engine.input_mpp,
        "ioconfig_input_mpp": engine.ioconfig_input_mpp,
        "patch_px": engine.patch_px,
        "output": "points",
        "point_fields": ["x", "y", "prob", "logit"],
        "deterministic": True,
        "contract": "v3",
        "peak_rule": "logit_peaks",
        "peak_min_distance_um": PEAK_MIN_DISTANCE_UM,
        "nms_radius_um": NMS_RADIUS_UM,
        "preproc": "imagenet_mean_std",
        "max_batch": getattr(engine, "max_batch", 1),
        "precision": getattr(engine, "precision", "fp32"),
        "instances": ["image_png_b64", "tile_pack_uri", "image_bytes"],
    }


def _points(raw_points) -> List[Dict[str, Any]]:
    points = []
    for x, y, logit in raw_points:
        if not math.isfinite(float(logit)):
            raise ValueError(f"non-finite logit {logit} at ({x}, {y})")
        points.append({
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "prob": round(float32_sigmoid(logit), 4),
            "logit": round(float(logit), 4),
        })
    return points


def _run_batch(engine, imgs: List[np.ndarray], floors: List[float]) -> List[Dict[str, Any]]:
    """Contract v3 for several patches through one batched engine call; one answer per patch, in order.

    A failed batch is retried patch by patch, so one bad patch costs only its own answer."""
    try:
        return [{"points": _points(r), "error": None} for r in engine.predict_logits_batch(imgs, floors)]
    except Exception as e:  # fall back to single patches so the error lands on the patch that caused it
        logger.error(f"Batched inference failed ({e}); retrying {len(imgs)} patches one by one")
    out = []
    for img, floor in zip(imgs, floors):
        try:
            out.append({"points": _points(engine.predict_logits(img, min_logit=floor)), "error": None})
        except Exception as e:
            out.append({"points": [], "error": f"inference_error: {e}"})
    return out


def _check_mpp(engine, mpp: Any) -> Optional[str]:
    if mpp is None:
        return "missing_mpp: 'mpp' field is required in v2 contract"
    try:
        mpp_val = float(mpp)
    except (ValueError, TypeError):
        return f"invalid_mpp: expected float, got {mpp}"
    expected_mpp = float(engine.input_mpp)
    if abs(mpp_val - expected_mpp) / expected_mpp > 0.01:
        return f"mpp_mismatch: expected {expected_mpp}, got {mpp_val}"
    return None


@app.post(PREDICT_ROUTE)
@app.post("/predict")
async def predict(request: Request) -> Dict[str, Any]:
    """
    Accepts Vertex AI prediction payloads.
    Differentiates between v2/v3 (lossless PNG, resolution-checked; v3 points also carry their
    logit) and legacy formats per instance.
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
    # v3 patches waiting for the batched engine call: (prediction index, patch, logit floor).
    pending: List[Tuple[int, np.ndarray, float]] = []
    # Tile packs (spec §12.1, "send references, not pixels"): (prediction index, patches, logit floor).
    packs: List[Tuple[int, List[np.ndarray], float]] = []

    for idx, item in enumerate(instances):
        # 1. v2/v3 Contract Format: image_png_b64 + mpp (v3 answers are v2 answers plus logits)
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
            mpp_error = _check_mpp(engine, item.get("mpp"))
            if mpp_error:
                predictions.append({"points": [], "error": mpp_error})
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

            # Contract v3: peaks of the logit map at or above logit(min_prob), each with its logit
            # and the float32 sigmoid of it as prob (what v2 returned). See engine.py.
            try:
                item_min_prob = float(item.get("min_prob", default_min_prob))
                min_logit = logit_of(item_min_prob)
            except (TypeError, ValueError) as e:
                predictions.append({"points": [], "error": f"invalid_min_prob: {e}"})
                continue
            predictions.append({"points": [], "error": None})  # filled after the batched call
            pending.append((idx, img_rgb, min_logit))

        # 1b. Tile pack: a GCS npz of N patches; one prediction with one answer per patch, in pack order.
        elif "tile_pack_uri" in item:
            mpp_error = _check_mpp(engine, item.get("mpp"))
            if mpp_error:
                predictions.append({"tiles": [], "error": mpp_error})
                continue
            try:
                min_logit = logit_of(float(item.get("min_prob", default_min_prob)))
            except (TypeError, ValueError) as e:
                predictions.append({"tiles": [], "error": f"invalid_min_prob: {e}"})
                continue
            try:
                tiles = await run_in_threadpool(read_tile_pack, str(item["tile_pack_uri"]), int(engine.patch_px))
            except TilePackError as e:
                predictions.append({"tiles": [], "error": str(e)})
                continue
            predictions.append({"tiles": [], "error": None})
            packs.append((idx, tiles, min_logit))

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
                "error": "invalid_instance: expected 'image_png_b64' (v2), 'tile_pack_uri' or 'image_bytes' (legacy)",
            })

    # One batched engine call for every v3 patch of the request, pack patches included, off the event loop.
    patches = [img for _, img, _ in pending] + [t for _, tiles, _ in packs for t in tiles]
    floors = [f for _, _, f in pending] + [f for _, tiles, f in packs for _ in tiles]
    if patches:
        answers = await run_in_threadpool(_run_batch, engine, patches, floors)
        for k, (idx, _, _) in enumerate(pending):
            predictions[idx] = answers[k]
        cursor = len(pending)
        for idx, tiles, _ in packs:
            predictions[idx] = {"tiles": answers[cursor:cursor + len(tiles)], "error": None}
            cursor += len(tiles)

    return {
        "predictions": predictions,
        "model_sha256": weights_sha,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AIP_HTTP_PORT", os.environ.get("PORT", "8080")))
    uvicorn.run(app, host="0.0.0.0", port=port)
