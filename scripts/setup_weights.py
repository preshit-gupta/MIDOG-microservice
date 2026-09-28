"""
Helper script to verify or download model weights for MIDOG Mitosis Detector (KongNet_Det_MIDOG_1).
Usage:
    python scripts/setup_weights.py
    python scripts/setup_weights.py --check-winner
"""

import os
import sys
import hashlib
import argparse


def check_kongnet_winner():
    print("=" * 60)
    print("  MIDOG Challenge Winner: KongNet_Det_MIDOG_1")
    print("=" * 60)
    print("Developed by: Tissue Image Analytics (TIA) Centre, University of Warwick")
    print("Performance: 1st Place on MIDOG Challenge (F1 = 0.7400)")
    print("License    : Creative Commons Attribution (CC BY 4.0)")
    print("Status     : Built-in to container via TIAToolbox")
    print("No manual checkpoint file is required.")
    print("Weights are fetched and cached automatically during container build / startup.")
    try:
        from tiatoolbox.models.engine.nucleus_detector import NucleusDetector
        from tiatoolbox import constants

        print("\n[INFO] Initializing KongNet_Det_MIDOG_1 via TIAToolbox...")
        _ = NucleusDetector(model="KongNet_Det_MIDOG_1", verbose=False)
        print("[SUCCESS] KongNet_Det_MIDOG_1 is verified and ready for deployment!")

        cache_dir = getattr(constants, "TIATOOLBOX_CACHE_DIR", os.path.expanduser("~/.tiatoolbox"))
        for root, _, files in os.walk(cache_dir):
            for f in files:
                if "KongNet_Det_MIDOG_1" in f and f.endswith((".pth", ".pt", ".tar", ".bin")):
                    weights_path = os.path.join(root, f)
                    h = hashlib.sha256(open(weights_path, "rb").read()).hexdigest()
                    print(f"[INFO] Weights file: {weights_path}")
                    print(f"[INFO] Weights SHA-256: {h}")
                    with open("WEIGHTS_SHA256", "w") as out:
                        out.write(h)
                    print("[INFO] Wrote hash to WEIGHTS_SHA256")
                    return True
        return True
    except ImportError:
        print("[INFO] tiatoolbox is not installed in the local environment.")
        print("[INFO] It will be installed and cached inside the Docker container automatically.")
        return True
    except Exception as e:
        print(f"[WARNING] Could not pre-verify KongNet locally: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup model weights for MIDOG Vertex AI deployment")
    parser.add_argument("--check-winner", action="store_true", help="Inspect and verify KongNet_Det_MIDOG_1 challenge winner")
    args = parser.parse_args()

    check_kongnet_winner()
