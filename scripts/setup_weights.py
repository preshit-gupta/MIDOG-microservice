"""
Helper script to verify or download model weights for MIDOG Mitosis Detector (KongNet / YOLO).
Usage:
    python scripts/setup_weights.py --check-winner
    python scripts/setup_weights.py --url https://example.com/path/to/midog_best.pt
    python scripts/setup_weights.py --bootstrap-starter
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

TARGET_FILE = Path("best.pt")


def check_kongnet_winner():
    print("=" * 60)
    print("  MIDOG Challenge Winner: KongNet_Det_MIDOG_1")
    print("=" * 60)
    print("Developed by: Tissue Image Analytics (TIA) Centre, University of Warwick")
    print("Performance: 1st Place on MIDOG Challenge (F1 = 0.7400)")
    print("License    : Creative Commons Attribution (CC BY 4.0)")
    print("Status     : Built-in to container via TIAToolbox")
    print("No manual best.pt file is required when deploying KongNet_Det_MIDOG_1.")
    print("Weights are fetched and cached automatically during container build / startup.")
    try:
        from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
        print("\n[INFO] Initializing KongNet_Det_MIDOG_1 via TIAToolbox...")
        detector = NucleusDetector(model="KongNet_Det_MIDOG_1", verbose=False)
        print("[SUCCESS] KongNet_Det_MIDOG_1 is verified and ready for deployment!")
        return True
    except ImportError:
        print("[INFO] tiatoolbox is not installed in the local environment.")
        print("[INFO] It will be installed and cached inside the Docker container automatically.")
        return True
    except Exception as e:
        print(f"[WARNING] Could not pre-verify KongNet locally: {e}")
        return False


def setup_weights(download_url: str = None, bootstrap_starter: bool = False, check_winner: bool = False):
    if check_winner:
        return check_kongnet_winner()

    print("=" * 60)
    print("  MIDOG Model Weight Setup")
    print("=" * 60)

    if TARGET_FILE.exists():
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        print(f"[INFO] Found local checkpoint: '{TARGET_FILE.resolve()}' ({size_mb:.2f} MB)")
        if size_mb < 8.0:
            print("[NOTE] This checkpoint is under 8MB (likely the generic COCO starter).")
            print("       To deploy the 1st-place MIDOG Challenge Winner, use -Engine kongnet in deploy.ps1.")
        return True

    print(f"[INFO] '{TARGET_FILE}' not found in current directory.")

    if download_url:
        print(f"[INFO] Downloading weights from: {download_url} ...")
        try:
            urllib.request.urlretrieve(download_url, TARGET_FILE)
            size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] Downloaded '{TARGET_FILE}' ({size_mb:.2f} MB)")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to download weights: {e}")
            return False

    print("\nModel Selection Guidance:")
    print(" 1. [RECOMMENDED] Deploy KongNet_Det_MIDOG_1 (MIDOG Challenge 1st Place Winner):")
    print("    No manual file download needed! Run:")
    print("    .\\deploy.ps1 -Engine kongnet -DeployGPU")
    print(" 2. If you have custom fine-tuned YOLOv8x weights on MIDOG++, copy your checkpoint here:")
    print(f"    --> {TARGET_FILE.resolve()}")
    print(" 3. To test the pipeline with a starter YOLOv8 checkpoint:")
    print("    python scripts/setup_weights.py --bootstrap-starter\n")

    if bootstrap_starter:
        print("[INFO] Bootstrapping starter YOLOv8 checkpoint for pipeline verification...")
        try:
            from ultralytics import YOLO
            starter = YOLO("yolov8n.pt")
            import shutil
            shutil.copy("yolov8n.pt", TARGET_FILE)
            size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] Starter checkpoint created as '{TARGET_FILE}' ({size_mb:.2f} MB)")
            return True
        except ImportError:
            print("[INFO] 'ultralytics' not installed locally. Downloading starter weights directly...")
            url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
            urllib.request.urlretrieve(url, TARGET_FILE)
            size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] Downloaded starter weights to '{TARGET_FILE}' ({size_mb:.2f} MB)")
            return True
        except Exception as ex:
            print(f"[ERROR] Could not bootstrap starter weights: {ex}")
            return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup model weights for MIDOG Vertex AI deployment")
    parser.add_argument("--check-winner", action="store_true", help="Inspect and verify KongNet_Det_MIDOG_1 challenge winner")
    parser.add_argument("--url", type=str, default=None, help="Direct download URL for custom MIDOG weights")
    parser.add_argument("--bootstrap-starter", action="store_true", help="Bootstrap starter YOLOv8 weights for testing")
    args = parser.parse_args()

    success = setup_weights(download_url=args.url, bootstrap_starter=args.bootstrap_starter, check_winner=args.check_winner)
    if not success and not TARGET_FILE.exists() and not args.check_winner:
        sys.exit(1)
