import base64
import io
import os
import sys
import tempfile
import logging
from typing import Any, Dict, List
from fastapi import FastAPI, Request, HTTPException
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("midog-detector")

app = FastAPI(title="MIDOG Mitosis Detector for Vertex AI")

MODEL_ENGINE = os.environ.get("MODEL_ENGINE", "auto").lower()
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

active_engine = None
model_instance = None


def init_engine():
    global active_engine, model_instance

    # Determine desired engine
    target_engine = MODEL_ENGINE
    if target_engine == "auto":
        # If a non-placeholder best.pt exists (>10MB), default to YOLO; otherwise use KongNet winner
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10 * 1024 * 1024:
            target_engine = "yolo"
        else:
            target_engine = "kongnet"

    logger.info(f"Initializing MIDOG Detector engine: '{target_engine}'")

    if target_engine == "kongnet":
        try:
            logger.info("Loading KongNet_Det_MIDOG_1 (1st Place Winner MIDOG Challenge)...")
            from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_instance = NucleusDetector(
                model="KongNet_Det_MIDOG_1",
                batch_size=1,
                device=device,
                verbose=False
            )
            active_engine = "kongnet"
            logger.info(f"KongNet_Det_MIDOG_1 successfully loaded on device: {device}")
            return
        except Exception as e:
            logger.error(f"Failed to load KongNet_Det_MIDOG_1 via tiatoolbox: {e}")
            if os.path.exists(MODEL_PATH):
                logger.info(f"Attempting fallback to YOLO with {MODEL_PATH}...")
                target_engine = "yolo"
            else:
                raise e

    if target_engine == "yolo":
        try:
            logger.info(f"Loading YOLO model from: {MODEL_PATH}")
            from ultralytics import YOLO
            model_instance = YOLO(MODEL_PATH)
            active_engine = "yolo"
            logger.info("YOLO model successfully loaded.")
            return
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise e


try:
    init_engine()
except Exception as err:
    logger.error(f"Engine initialization error: {err}")
    active_engine = None
    model_instance = None


@app.get(HEALTH_ROUTE)
@app.get("/health")
def health() -> Dict[str, Any]:
    """Vertex AI liveness and readiness probe."""
    if model_instance is None or active_engine is None:
        raise HTTPException(status_code=503, detail="Model engine not loaded")
    return {
        "status": "healthy",
        "engine": active_engine,
        "model": "KongNet_Det_MIDOG_1" if active_engine == "kongnet" else MODEL_PATH
    }


def predict_kongnet(img: Image.Image, conf_threshold: float, img_size: int) -> List[Dict[str, Any]]:
    """Runs inference using TIAToolbox KongNet_Det_MIDOG_1."""
    boxes_out = []
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        img.save(tmp_path, format="PNG")

    try:
        raw_output = model_instance.run(
            images=[tmp_path],
            output_type="dict",
            patch_mode=True,
            auto_get_mask=False
        )

        inst_dict = {}
        if isinstance(raw_output, dict):
            if tmp_path in raw_output:
                inst_dict = raw_output[tmp_path]
            elif 0 in raw_output:
                inst_dict = raw_output[0]
            else:
                for v in raw_output.values():
                    if isinstance(v, dict):
                        inst_dict = v
                        break
        elif isinstance(raw_output, (list, tuple)) and len(raw_output) > 0:
            inst_dict = raw_output[0]

        if isinstance(inst_dict, dict):
            for _, inst in inst_dict.items():
                if not isinstance(inst, dict):
                    continue
                prob = float(inst.get("prob", inst.get("confidence", 1.0)))
                if prob < conf_threshold:
                    continue

                centroid = inst.get("centroid")
                box = inst.get("box")

                if centroid is not None and len(centroid) >= 2:
                    cx = float(centroid[0])
                    cy = float(centroid[1])
                elif box is not None and len(box) >= 4:
                    cx = float(box[0] + box[2]) / 2.0
                    cy = float(box[1] + box[3]) / 2.0
                else:
                    continue

                if box is not None and len(box) >= 4:
                    w = float(box[2] - box[0])
                    h = float(box[3] - box[1])
                else:
                    w, h = 48.0, 48.0

                boxes_out.append({
                    "cx": round(cx, 2),
                    "cy": round(cy, 2),
                    "width": round(w, 2),
                    "height": round(h, 2),
                    "confidence": round(prob, 4)
                })

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return boxes_out


def predict_yolo(img: Image.Image, conf_threshold: float, img_size: int) -> List[Dict[str, Any]]:
    """Runs inference using Ultralytics YOLO."""
    results = model_instance.predict(
        source=img,
        imgsz=img_size,
        conf=conf_threshold,
        verbose=False
    )
    boxes_out = []
    for r in results:
        for box in r.boxes:
            xywh = box.xywh[0].tolist()
            conf = float(box.conf[0])
            boxes_out.append({
                "cx": round(xywh[0], 2),
                "cy": round(xywh[1], 2),
                "width": round(xywh[2], 2),
                "height": round(xywh[3], 2),
                "confidence": round(conf, 4)
            })
    return boxes_out


@app.post(PREDICT_ROUTE)
@app.post("/predict")
async def predict(request: Request) -> Dict[str, List[Dict[str, Any]]]:
    """
    Accepts Vertex AI prediction payloads:
    {
      "instances": [
        {"image_bytes": "<base64_encoded_jpeg_or_png>"}
      ],
      "parameters": {
        "conf": 0.25,
        "imgsz": 512
      }
    }
    """
    if model_instance is None or active_engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    instances = body.get("instances", [])
    parameters = body.get("parameters", {})
    conf_threshold = float(parameters.get("conf", 0.25))
    img_size = int(parameters.get("imgsz", 512))

    if not instances:
        return {"predictions": []}

    predictions = []

    for idx, item in enumerate(instances):
        b64_str = item.get("image_bytes")
        if not b64_str:
            predictions.append({"boxes": []})
            continue

        try:
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            item_conf = float(item.get("confidence_threshold", conf_threshold))

            if active_engine == "kongnet":
                boxes = predict_kongnet(img, item_conf, img_size)
            else:
                boxes = predict_yolo(img, item_conf, img_size)

            predictions.append({"boxes": boxes})

        except Exception as err:
            logger.error(f"Error processing instance #{idx}: {err}")
            predictions.append({"boxes": [], "error": str(err)})

    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AIP_HTTP_PORT", os.environ.get("PORT", "8080")))
    uvicorn.run(app, host="0.0.0.0", port=port)
