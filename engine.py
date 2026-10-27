"""KongNet_Det_MIDOG_1 through TIAToolbox, with the v6 guarantees (SPEC-06 §4).

The engine starts only if it knows the model's patch size (from the model's own
TIAToolbox IO config) and the SHA-256 of the weights it loaded; the service answers 503
otherwise. Detector output without a probability is an error, never a default.

Two facts were measured on the live endpoint against MIDOG++ image 094 (82 labelled
mitotic figures, 0.2298 um/px) on 2026-09-29, with tiatoolbox 2.0.1:
- Resolution: native 40x input (~0.25 um/px) reaches F1 0.87; the same image
  downsampled to the 0.5 um/px in the model's IO config reaches at most F1 0.27. The
  service therefore takes 0.25 um/px (SPEC-06 AC5) and reports the IO config's value only
  for transparency.
- Coordinates: patch-mode output has x and y transposed. Taken as returned, 11 of 104
  detections match a labelled figure (F1 0.12); swapped back, 75 of 90 do (F1 0.87).

Contract v3 (OncoGemma v7.2 Stage 4 spec §4.2) reads the model's logits. The probability
map is a float32 sigmoid, which is exactly 1.0 wherever the logit passes about 17: 80% of
in-tissue candidates on MIDOG++ breast score 1.0, so peaks found on it sit anywhere inside
the plateau and their probabilities cannot be ranked. Peaks are therefore found on the logit
map (5 um apart, at or above logit(min_prob)), suppressed greedily within 7.5 um by logit,
and returned with their logit; ``prob`` is the float32 sigmoid, for v2 readers. On 6 MIDOG++
breast images the logit-peak sweep returns the same 456 candidates as the probability peaks,
and its logit ranks them better (AUROC 0.846 against 0.772).

TIAToolbox's ``infer_batch`` does not normalise its input; its engines call the model's
``preproc`` (ImageNet mean and std) per patch first. ``logit_map`` does the same, explicitly.
"""
import hashlib
import logging
import math
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
# The resolution the model is served at (see the module docstring for the measurement).
MODEL_INPUT_MPP = 0.25
# Contract v3 peak rule (see the module docstring).
PEAK_MIN_DISTANCE_UM = 5.0
NMS_RADIUS_UM = 7.5

Point = Tuple[float, float, float]
LogitPoint = Tuple[float, float, float]  # (x, y, logit) in input-image pixels


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
    """(x, y, prob) in input-image pixels for the one patch in a NucleusDetector patch-mode output.

    TIAToolbox 2.0.1 returns ``{"x": [arr], "y": [arr], "classes": [arr], "probabilities": [arr]}``,
    one array per input image, with the two coordinate axes transposed: its "y" array holds
    the image column (x) and its "x" array the image row (y). See the module docstring.
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
    return [(float(col), float(row), float(p)) for row, col, p in zip(xs, ys, probs)]


def logit_of(prob: float) -> float:
    """The logit of a probability in (0, 1): the v3 peak floor for a client's ``min_prob``."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"min_prob must lie strictly between 0 and 1, got {prob}")
    return math.log(prob / (1.0 - prob))


def float32_sigmoid(logit: float) -> float:
    """The probability the model's own float32 sigmoid gives: 1.0 for every logit above about 17."""
    value = np.float32(logit)
    return float(np.float32(1.0) / (np.float32(1.0) + np.exp(-value, dtype=np.float32)))


def peaks_from_logit_map(logits: np.ndarray, mpp: float, min_logit: float,
                         min_distance_um: float = PEAK_MIN_DISTANCE_UM,
                         nms_radius_um: float = NMS_RADIUS_UM) -> List[LogitPoint]:
    """(x, y, logit) detections of one patch's mitosis logit map (H x W), highest logit first.

    Local maxima at least ``min_distance_um`` apart and at or above ``min_logit`` (skimage
    ``peak_local_max``, borders included), then greedy suppression of any peak within
    ``nms_radius_um`` of a stronger one. Equal logits are ordered by row, then column, so
    the output does not depend on the peak finder's internal order. x is the column, y the row.
    """
    from skimage.feature import peak_local_max

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2:
        raise EngineError(f"a logit map is 2-D, got shape {logits.shape}")
    if not np.isfinite(logits).all():
        raise EngineError("the logit map has non-finite values")
    distance_px = max(1, int(round(min_distance_um / mpp)))
    # peak_local_max keeps values strictly above threshold_abs; the floor is inclusive, so ask for a
    # hair below it and apply ">= min_logit" exactly afterwards.
    below_floor = min_logit - 1e-6 * max(1.0, abs(min_logit))
    found = peak_local_max(logits, min_distance=distance_px, threshold_abs=below_floor, exclude_border=False)
    rows, cols = found[:, 0], found[:, 1]
    values = logits[rows, cols].astype(float)
    keep_floor = values >= min_logit
    rows, cols, values = rows[keep_floor], cols[keep_floor], values[keep_floor]
    order = np.lexsort((cols, rows, -values))
    radius_px = nms_radius_um / mpp
    kept: List[LogitPoint] = []
    for i in order:
        x, y = float(cols[i]), float(rows[i])
        if all(math.hypot(x - kx, y - ky) >= radius_px for kx, ky, _ in kept):
            kept.append((x, y, float(values[i])))
    return kept


class KongNetEngine:
    """
    Inference engine wrapping TIAToolbox KongNet_Det_MIDOG_1.
    Adheres to v6 determinism, resolution verification, and weights reporting.
    Contract v3 runs the same weights through ``logit_map`` (see the module docstring).
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
        self.input_mpp = MODEL_INPUT_MPP
        self.ioconfig_input_mpp = input_mpp_from_ioconfig(self.detector.ioconfig)
        self.patch_px = patch_px_from_ioconfig(self.detector.ioconfig)
        weights = weights_path(rcParam["TIATOOLBOX_HOME"])
        self.weights_sha256 = loaded_weights_sha256(weights)
        # The same architecture and weights for contract v3's logits, built as TIAToolbox builds
        # the named pretrained model, from the verified weights file.
        from tiatoolbox.models.architecture import get_pretrained_model

        self.model, _ = get_pretrained_model(MODEL_NAME, pretrained_weights=str(weights))
        self.model.eval()
        self.model.to(self.device)
        self.target_channels = list(getattr(self.model, "target_channels", None) or [0])
        # One patch at a time through the detector: fixed batch size, deterministic order.
        self._lock = threading.Lock()
        logger.info(
            f"KongNetEngine ready: tiatoolbox {self.tiatoolbox_version}, input {self.input_mpp} um/px "
            f"(IO config says {self.ioconfig_input_mpp}), "
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
                    # tiatoolbox 2.0.1 leaves "probabilities" out of the dict output unless asked
                    # (v1 then reported every detection as confidence 1.0).
                    return_probabilities=True,
                    **run_params,
                )
            return points_from_output(raw_output)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def logit_map(self, img_rgb: np.ndarray) -> np.ndarray:
        """The mitosis channel's logits for one RGB patch (H x W x 3 uint8) -> H x W float32.

        ``preproc`` (ImageNet normalisation) first, as TIAToolbox's engines do per patch, then
        the forward pass without the sigmoid ``infer_batch`` applies.
        """
        import torch

        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise EngineError(f"expected an H x W x 3 RGB patch, got shape {img_rgb.shape}")
        batch = np.ascontiguousarray(self.model.preproc(np.asarray(img_rgb)[None]), dtype=np.float32)
        with self._lock, torch.inference_mode():
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
            out = self.model(x)
            out = out[0] if isinstance(out, (list, tuple)) else out
            logits = out[:, self.target_channels].permute(0, 2, 3, 1).float().cpu().numpy()
        return logits[0, :, :, 0]

    def predict_logits(self, img_rgb: np.ndarray, min_logit: float) -> List[LogitPoint]:
        """Contract v3: (x, y, logit) detections of one patch at or above ``min_logit``."""
        return peaks_from_logit_map(self.logit_map(img_rgb), self.input_mpp, min_logit)


class FakeEngine:
    """Fake engine for offline contract tests without GPU or TIAToolbox."""

    def __init__(
        self,
        input_mpp: float = 0.25,
        patch_px: int = 512,
        weights_sha256: str = "fake_sha256_kongnet_weights",
        tiatoolbox_version: str = "2.0.1",
        mock_predictions: Optional[List[Point]] = None,
    ):
        self.input_mpp = input_mpp
        self.ioconfig_input_mpp = 0.5
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
        # The logit floor of every predict_logits call (contract v3).
        self.logit_floors: List[float] = []

    def predict(self, img_rgb: np.ndarray, threshold_abs: Optional[float] = None) -> List[Point]:
        self.thresholds.append(threshold_abs)
        return self.mock_predictions

    def predict_logits(self, img_rgb: np.ndarray, min_logit: float) -> List[LogitPoint]:
        """The mock predictions as logits, floored like the real engine (no peak finding)."""
        self.logit_floors.append(min_logit)
        points = []
        for x, y, p in self.mock_predictions:
            value = math.inf if p >= 1.0 else (-math.inf if p <= 0.0 else math.log(p / (1.0 - p)))
            if value >= min_logit:
                points.append((x, y, value))
        return points


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
