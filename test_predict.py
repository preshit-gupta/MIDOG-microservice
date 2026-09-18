"""
Test script to send sample prediction requests to local container or deployed Vertex AI Endpoint.

Usage:
    # 1. Test local container (http://localhost:8080)
    python test_predict.py --local

    # 2. Test local container with a custom image
    python test_predict.py --local --image path/to/sample_tile.jpg

    # 3. Test deployed Vertex AI Endpoint
    python test_predict.py --endpoint-id 1234567890123456789 --project-id oncogemma --region us-central1
"""

import os
import sys
import io
import time
import json
import base64
import argparse
import urllib.request
import subprocess
from PIL import Image, ImageDraw


def create_synthetic_tile() -> bytes:
    """Generate a synthetic 512x512 tile for quick smoke testing."""
    img = Image.new("RGB", (512, 512), color=(235, 215, 230)) # H&E background tone
    draw = ImageDraw.Draw(img)

    # Draw simulated nuclei and dense chromatin clumps (mitosis candidates)
    draw.ellipse([240, 240, 272, 272], fill=(60, 20, 90))
    draw.ellipse([100, 150, 125, 175], fill=(70, 30, 100))
    draw.ellipse([380, 400, 410, 430], fill=(50, 15, 80))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def load_image_bytes(image_path: str = None) -> bytes:
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return f.read()
    print("[INFO] No input image provided. Generating synthetic 512x512 H&E tile...")
    return create_synthetic_tile()


def test_local(image_bytes: bytes, host: str = "http://localhost:8080"):
    print(f"\n[INFO] Testing local server at: {host}")
    b64_str = base64.b64encode(image_bytes).decode("utf-8")

    # 1. Health check
    try:
        health_req = urllib.request.Request(f"{host}/health")
        with urllib.request.urlopen(health_req, timeout=5) as resp:
            health_data = json.loads(resp.read().decode("utf-8"))
            print(f"[HEALTH CHECK] Status: {resp.status} Response: {health_data}")
    except Exception as e:
        print(f"[HEALTH CHECK ERROR] Could not reach {host}/health: {e}")
        return

    # 2. Prediction check
    payload = {
        "instances": [
            {"image_bytes": b64_str}
        ],
        "parameters": {
            "conf": 0.25,
            "imgsz": 512
        }
    }
    payload_json = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/predict",
        data=payload_json,
        headers={"Content-Type": "application/json"}
    )

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed_ms = (time.time() - start_time) * 1000
            res_body = json.loads(resp.read().decode("utf-8"))
            print(f"\n[SUCCESS] Response received in {elapsed_ms:.1f} ms")
            print(json.dumps(res_body, indent=2))
            
            predictions = res_body.get("predictions", [])
            for i, p in enumerate(predictions):
                boxes = p.get("boxes", [])
                print(f"--> Instance #{i+1}: {len(boxes)} mitotic figures detected.")
                for b_idx, b in enumerate(boxes[:5]):
                    print(f"    Candidate #{b_idx+1}: center=({b.get('cx')}, {b.get('cy')}), conf={b.get('confidence')}")
    except Exception as e:
        print(f"[PREDICTION ERROR] Failed to run prediction: {e}")


def test_vertex_endpoint(image_bytes: bytes, endpoint_id: str, project_id: str, region: str):
    print(f"\n[INFO] Testing Vertex AI Endpoint: {endpoint_id} in {region} ({project_id})")
    b64_str = base64.b64encode(image_bytes).decode("utf-8")

    # Obtain gcloud OAuth access token
    try:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            shell=True,
            text=True
        ).strip()
    except Exception as e:
        print(f"[ERROR] Could not obtain gcloud access token: {e}")
        return

    endpoint_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/{endpoint_id}:predict"

    payload = {
        "instances": [
            {"image_bytes": b64_str}
        ],
        "parameters": {
            "conf": 0.25,
            "imgsz": 512
        }
    }
    payload_json = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        endpoint_url,
        data=payload_json,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    )

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            elapsed_ms = (time.time() - start_time) * 1000
            res_body = json.loads(resp.read().decode("utf-8"))
            print(f"\n[SUCCESS] Vertex AI response received in {elapsed_ms:.1f} ms")
            print(json.dumps(res_body, indent=2))
    except Exception as e:
        print(f"[VERTEX PREDICTION ERROR] Failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv8-MIDOG Inference")
    parser.add_argument("--local", action="store_true", help="Test local container at http://localhost:8080")
    parser.add_argument("--host", type=str, default="http://localhost:8080", help="Local server URL")
    parser.add_argument("--image", type=str, default=None, help="Path to input image/tile")
    parser.add_argument("--endpoint-id", type=str, default=None, help="Vertex AI numeric endpoint ID")
    parser.add_argument("--project-id", type=str, default="oncogemma", help="GCP Project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP Region")

    args = parser.parse_args()
    img_data = load_image_bytes(args.image)

    if args.local:
        test_local(img_data, host=args.host)
    elif args.endpoint_id:
        test_vertex_endpoint(img_data, args.endpoint_id, args.project_id, args.region)
    else:
        print("Please specify either --local or --endpoint-id <ID>. See --help for details.")
