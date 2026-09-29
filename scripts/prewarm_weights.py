"""Image build step: download KongNet_Det_MIDOG_1, record its weights' SHA-256, check its IO config.

Run from /app by the Dockerfile. Any failure exits non-zero and fails the build, so an
image never ships without known weights (SPEC-06 §4).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiatoolbox import rcParam  # noqa: E402
from tiatoolbox.models.engine.nucleus_detector import NucleusDetector  # noqa: E402

from engine import (  # noqa: E402
    MODEL_NAME,
    WEIGHTS_SHA256_FILE,
    compute_file_sha256,
    input_mpp_from_ioconfig,
    patch_px_from_ioconfig,
    weights_path,
)


def main() -> None:
    detector = NucleusDetector(model=MODEL_NAME, batch_size=1, device="cpu", verbose=False)
    weights = weights_path(rcParam["TIATOOLBOX_HOME"])
    if not weights.is_file():
        raise SystemExit(f"{MODEL_NAME}: weights not found at {weights} after loading the model")
    sha = compute_file_sha256(weights)
    input_mpp = input_mpp_from_ioconfig(detector.ioconfig)
    patch_px = patch_px_from_ioconfig(detector.ioconfig)
    WEIGHTS_SHA256_FILE.write_text(sha + "\n", encoding="utf-8")
    print(f"{MODEL_NAME}: weights {weights} sha256 {sha}; input {input_mpp} um/px, {patch_px} px patches")


if __name__ == "__main__":
    main()
