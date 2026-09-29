"""KongNet_Det_MIDOG_1 through TIAToolbox, with the v6 guarantees (SPEC-06 §4).

The engine starts only if it knows the model's input resolution and patch size (from
the model's own TIAToolbox IO config) and the SHA-256 of the weights it loaded; the
service answers 503 otherwise. Detector output without a probability is an error,
never a default.
"""
import hashlib
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("midog-detector")

MODEL_NAME = "KongNet_Det_MIDOG_1"
# Written by scripts/prewarm_weights.py during the image build.
WEIGHTS_SHA256_FILE = Path(os.environ.get("WEIGHTS_SHA256_FILE", "/app/WEIGHTS_SHA256"))

Point = Tuple[float, float, float]


class EngineError(RuntimeError):
    """The detector cannot run, or produced output that cannot be trusted."""


def compute_file_sha256(path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def weights_path(tiatoolbox_home) -> Path:
    """Where TIAToolbox caches the pretrained weights: models/<name>.pth under TIATOOLBOX_HOME."""
    return Path(tiatoolbox_home) / "models" / f"{MODEL_NAME}.pth"


def loaded_weights_sha256(weights: Path, recorded: Path = WEIGHTS_SHA256_FILE) -> str:
    """SHA-256 of the weights file. It must match the hash the image build recorded, if there is one."""
    if not weights.is_file():
        raise EngineError(f"weights file {weights} does not exist")
    sha = compute_file_sha256(weights)
    if recorded.is_file():
        expected = recorded.read_text(encoding="utf-8").strip()
        if expected != sha:
            raise EngineError(f"weights {weights} have sha256 {sha}, but the image build recorded {expected}")
    return sha


def input_mpp_from_ioconfig(ioconfig: Any) -> float:
    """The model's input resolution in µm/px, from its TIAToolbox IO config."""
    resolutions = getattr(ioconfig, "input_resolutions", None)
    if not resolutions:
        raise EngineError(f"{MODEL_NAME} IO config has no input resolution")
    resolution = resolutions[0]
    if isinstance(resolution, dict):
        units, value = resolution.get("units"), resolution.get("resolution")
    else:
        units, value = getattr(resolution, "units", None), getattr(resolution, "resolution", None)
    if units != "mpp" or value is None:
        raise EngineError(f"{MODEL_NAME} input resolution is {resolution!r}, not a resolution in mpp")
    return float(value)


def patch_px_from_ioconfig(ioconfig: Any) -> int:
    """The model's square input patch size in pixels, from its TIAToolbox IO config."""
    shape = getattr(ioconfig, "patch_input_shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != int(shape[1]):
        raise EngineError(f"{MODEL_NAME} IO config patch_input_shape is {shape!r}, not a square patch")
    return int(shape[0])


def points_from_output(raw_output: Any) -> List[Point]:
    """(x, y, prob) for the one patch in a NucleusDetector patch-mode ``dict`` output.

    TIAToolbox 2.1.3 returns ``{"x": [arr], "y": [arr], "classes": [arr], "probabilities": [arr]}``,
    one array per input image.
    """
    required = ("x", "y", "probabilities")
    if not isinstance(raw_output, dict) or any(key not in raw_output for key in required):
        found = sorted(raw_output) if isinstance(raw_output, dict) else type(raw_output).__name__
        raise EngineError(f"unexpected detector output, need {required}: {found}")
    columns = []
    for key in required:
        per_image = raw_output[key]
        if len(per_image) != 1:
            raise EngineError(f"detector output '{key}' has {len(per_image)} entries for one patch")
        column = per_image[0]
        if hasattr(column, "compute"):
            column = column.compute()
        columns.append(np.asarray(column, dtype=float).ravel())
    xs, ys, probs = columns
    if not len(xs) == len(ys) == len(probs):
        raise EngineError(f"detector output lengths differ: x {len(xs)}, y {len(ys)}, probabilities {len(probs)}")
    return [(float(x), float(y), float(p)) for x, y, p in zip(xs, ys, probs)]


class KongNetEngine:
    """
    Inference engine wrapping TIAToolbox KongNet_Det_MIDOG_1.
    Adheres to v6 determinism, resolution verification, and weights reporting.
    """

    def __init__(self, device: Optional[str] = None):
        import tiatoolbox
        import torch
        from tiatoolbox import rcParam
        from tiatoolbox.models.engine.nucleus_detector import NucleusDetector

        # Deterministic PyTorch settings (SPEC-06 §4)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading TIAToolbox NucleusDetector(model='{MODEL_NAME}') on {self.device}...")
        self.detector = NucleusDetector(
            model=MODEL_NAME,
            batch_size=1,
            device=self.device,
            verbose=False,
        )
        self.tiatoolbox_version = tiatoolbox.__version__
        self.input_mpp = input_mpp_from_ioconfig(self.detector.ioconfig)
        self.patch_px = patch_px_from_ioconfig(self.detector.ioconfig)
        self.weights_sha256 = loaded_weights_sha256(weights_path(rcParam["TIATOOLBOX_HOME"]))
        # One patch at a time through the detector: fixed batch size, deterministic order.
        self._lock = threading.Lock()
        logger.info(
            f"KongNetEngine ready: tiatoolbox {self.tiatoolbox_version}, input {self.input_mpp} um/px, "
            f"{self.patch_px} px patches, weights sha256 {self.weights_sha256}"
        )

    def predict(self, img_rgb: np.ndarray, threshold_abs: Optional[float] = None) -> List[Point]:
        """
        Runs inference on an RGB numpy array (patch_px x patch_px).
        Returns list of (x, y, prob).

        ``threshold_abs`` is the peak threshold of KongNet's post-processing. None keeps
        the model's own value (0.99 for KongNet_Det_MIDOG_1), as the legacy contract did.
        """
        run_params = {} if threshold_abs is None else {"threshold_abs": float(threshold_abs)}
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            Image.fromarray(img_rgb).save(tmp_path, format="PNG")
            with self._lock:
                raw_output = self.detector.run(
                    images=[tmp_path],
                    output_type="dict",
                    patch_mode=True,
                    auto_get_mask=False,
                    **run_params,
                )
            return points_from_output(raw_output)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class FakeEngine:
    """Fake engine for offline contract tests without GPU or TIAToolbox."""

    def __init__(
        self,
        input_mpp: float = 0.25,
        patch_px: int = 512,
        weights_sha256: str = "fake_sha256_kongnet_weights",
        tiatoolbox_version: str = "2.1.3",
        mock_predictions: Optional[List[Point]] = None,
    ):
        self.input_mpp = input_mpp
        self.patch_px = patch_px
        self.weights_sha256 = weights_sha256
        self.tiatoolbox_version = tiatoolbox_version
        self.mock_predictions = (
            mock_predictions
            if mock_predictions is not None
            else [
                (256.0, 256.0, 0.95),
                (100.0, 150.0, 0.50),
                (300.0, 300.0, 0.005),  # Below default min_prob 0.01
            ]
        )
        # The post-processing threshold of every predict call, for the contract tests.
        self.thresholds: List[Optional[float]] = []

    def predict(self, img_rgb: np.ndarray, threshold_abs: Optional[float] = None) -> List[Point]:
        self.thresholds.append(threshold_abs)
        return self.mock_predictions


_engine_instance: Optional[Any] = None


def get_engine():
    """Module-level factory to retrieve or create the active engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = KongNetEngine()
    return _engine_instance


def set_engine(engine: Optional[Any]) -> None:
    """Module-level factory override for testing."""
    global _engine_instance
    _engine_instance = engine
