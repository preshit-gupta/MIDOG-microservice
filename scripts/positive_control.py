"""Positive control for a deployed detector: labelled MIDOG++ image in, precision/recall/F1 out.

Tiles the whole image into 512 px patches at its native resolution (stride 448), sends each
through the deployed endpoint, and matches detections to the labelled mitotic figures within
MIDOG's 7.5 um radius (greedy by probability). Coordinates are used exactly as the service
returns them: the service, not this script, is responsible for their frame.

Needs `pip install google-cloud-aiplatform numpy pillow` and Application Default Credentials.

    python scripts/positive_control.py --image 094.tiff --labels MIDOG++.json \\
        --endpoint 6276949705008087040 --project oncogemma --region us-central1 --min-f1 0.85
"""
import argparse
import base64
import io
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MATCH_RADIUS_UM = 7.5
PATCH_PX, STRIDE_PX = 512, 448
THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def tiff_mpp(path: Path) -> float:
    """um/px from the TIFF resolution tags (ResolutionUnit inch or cm); an error if absent."""
    with Image.open(path) as image:
        tags = image.tag_v2
        if 282 not in tags or 296 not in tags:
            raise SystemExit(f"{path} has no resolution tags; pass --mpp")
        per_unit = float(tags[282])
        unit_um = {2: 25400.0, 3: 10000.0}.get(int(tags[296]))
        if unit_um is None or per_unit <= 0:
            raise SystemExit(f"{path}: unsupported resolution unit {tags[296]}; pass --mpp")
        return unit_um / per_unit


def labels(path: Path, file_name: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    image = [i for i in data["images"] if i["file_name"] == file_name]
    if not image:
        raise SystemExit(f"{file_name} is not in {path}")
    points = {"mitotic figure": [], "not mitotic figure": []}
    for a in data["annotations"]:
        if a["image_id"] == image[0]["id"]:
            x0, y0, x1, y1 = a["bbox"]  # MIDOG++ boxes are corner pairs
            points[names[a["category_id"]]].append(((x0 + x1) / 2, (y0 + y1) / 2))
    return np.array(points["mitotic figure"]), np.array(points["not mitotic figure"])


def starts(length: int) -> list[int]:
    s = list(range(0, max(length - PATCH_PX, 0) + 1, STRIDE_PX))
    if s[-1] + PATCH_PX < length:
        s.append(length - PATCH_PX)
    return s


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def dedupe(points, radius):
    kept = []
    for x, y, p in sorted(points, key=lambda t: -t[2]):
        if all(math.hypot(x - kx, y - ky) > radius for kx, ky, _ in kept):
            kept.append((x, y, p))
    return kept


def true_positives(points, truth, radius) -> int:
    used = np.zeros(len(truth), dtype=bool)
    tp = 0
    for x, y, _ in sorted(points, key=lambda t: -t[2]):
        if not len(truth):
            break
        d = np.hypot(truth[:, 0] - x, truth[:, 1] - y)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= radius:
            used[j], tp = True, tp + 1
    return tp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--mpp", type=float, help="default: from the TIFF resolution tags")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--contract", choices=("v2", "legacy"), default="v2")
    parser.add_argument("--min-f1", type=float, help="exit 1 if the best F1 is lower")
    args = parser.parse_args()

    from google.cloud import aiplatform

    mpp = args.mpp or tiff_mpp(args.image)
    mf, imposters = labels(args.labels, args.image.name)
    image = Image.open(args.image).convert("RGB")
    endpoint = aiplatform.Endpoint(args.endpoint, project=args.project, location=args.region)
    service_mpp = 0.25  # the v2 contract's input_mpp (1% tolerance)
    if args.contract == "v2" and abs(mpp - service_mpp) / service_mpp > 0.01:
        # Resample to the contract's resolution, as the pipeline does (SPEC-06 §5.1), and
        # move the labels with it. The legacy contract is sent at the native resolution.
        factor = mpp / service_mpp
        image = image.resize((round(image.width * factor), round(image.height * factor)), Image.LANCZOS)
        mf, imposters, mpp = mf * factor, imposters * factor, service_mpp
    radius_px = MATCH_RADIUS_UM / mpp
    width, height = image.size

    def detect(origin):
        x, y = origin
        patch = image.crop((x, y, x + PATCH_PX, y + PATCH_PX))
        if args.contract == "v2":
            instance, parameters = {"image_png_b64": png_b64(patch), "mpp": service_mpp}, {"min_prob": 0.01}
        else:
            instance, parameters = {"image_bytes": png_b64(patch), "confidence_threshold": 0.0}, None
        answer = endpoint.predict(instances=[instance], parameters=parameters)
        prediction = answer.predictions[0]
        if prediction.get("error"):
            raise SystemExit(f"patch {origin}: {prediction['error']}")
        if args.contract == "v2":
            found = [(x + q["x"], y + q["y"], q["prob"]) for q in prediction["points"]]
        else:
            found = [(x + b["cx"], y + b["cy"], b["confidence"]) for b in prediction["boxes"]]
        return found, answer.deployed_model_id

    origins = [(x, y) for y in starts(height) for x in starts(width)]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(detect, origins))
    points = dedupe([p for found, _ in results for p in found], radius_px)
    print(f"{args.image.name}: {mpp:.4f} um/px, {len(mf)} mitotic figures, {len(imposters)} imposters; "
          f"{len(origins)} {args.contract} requests to deployed model(s) {sorted({m for _, m in results})}")

    best = 0.0
    for t in THRESHOLDS:
        sel = [p for p in points if p[2] >= t]
        tp = true_positives(sel, mf, radius_px)
        fp = len(sel) - tp
        misses = [p for p in sel if true_positives([p], mf, radius_px) == 0]
        on_imposters = true_positives(misses, imposters, radius_px)
        precision = tp / len(sel) if sel else 0.0
        recall = tp / len(mf) if len(mf) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = max(best, f1)
        print(f"  t>={t:<5} det {len(sel):4d}  TP {tp:3d}  FP {fp:4d} (on imposters {on_imposters:3d})  "
              f"P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}")
    if args.min_f1 is not None and best < args.min_f1:
        print(f"FAIL: best F1 {best:.3f} < {args.min_f1}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
