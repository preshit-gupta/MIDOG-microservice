"""The engine's checks, without TIAToolbox or a GPU: nothing about the model is assumed."""
from types import SimpleNamespace

import numpy as np
import pytest

from engine import (
    EngineError,
    compute_file_sha256,
    input_mpp_from_ioconfig,
    loaded_weights_sha256,
    patch_px_from_ioconfig,
    points_from_output,
    weights_path,
)


def ioconfig(resolutions=({"units": "mpp", "resolution": 0.5},), shape=(512, 512)):
    return SimpleNamespace(input_resolutions=list(resolutions), patch_input_shape=list(shape))


def test_input_resolution_and_patch_size_come_from_the_model_config():
    # TIAToolbox 2.1.3 pretrained_model.yaml, KongNet_Det_MIDOG_1: 0.5 mpp, 512 x 512.
    assert input_mpp_from_ioconfig(ioconfig()) == 0.5
    assert patch_px_from_ioconfig(ioconfig()) == 512
    assert input_mpp_from_ioconfig(SimpleNamespace(input_resolutions=[SimpleNamespace(units="mpp", resolution=0.25)])) == 0.25


@pytest.mark.parametrize("resolutions", [(), ({"units": "baseline", "resolution": 1.0},), ({"units": "mpp"},)])
def test_an_unknown_input_resolution_is_an_error_not_0_25(resolutions):
    with pytest.raises(EngineError):
        input_mpp_from_ioconfig(ioconfig(resolutions=resolutions))
    with pytest.raises(EngineError):
        input_mpp_from_ioconfig(SimpleNamespace())


@pytest.mark.parametrize("shape", [(512, 256), (512,)])
def test_a_non_square_patch_is_an_error(shape):
    with pytest.raises(EngineError):
        patch_px_from_ioconfig(ioconfig(shape=shape))


def test_weights_hash_is_the_file_hash_and_must_match_the_build(tmp_path):
    weights = weights_path(tmp_path)
    assert weights == tmp_path / "models" / "KongNet_Det_MIDOG_1.pth"
    recorded = tmp_path / "WEIGHTS_SHA256"
    with pytest.raises(EngineError, match="does not exist"):
        loaded_weights_sha256(weights, recorded)

    weights.parent.mkdir()
    weights.write_bytes(b"kongnet weights")
    sha = compute_file_sha256(weights)
    assert loaded_weights_sha256(weights, recorded) == sha  # no build record (local run)
    recorded.write_text(sha + "\n", encoding="utf-8")
    assert loaded_weights_sha256(weights, recorded) == sha
    recorded.write_text("0" * 64, encoding="utf-8")
    with pytest.raises(EngineError, match="image build recorded"):
        loaded_weights_sha256(weights, recorded)


def test_points_are_read_with_the_axes_swapped_back():
    """tiatoolbox 2.0.1 patch mode returns the image row as "x" and the column as "y" (measured on MIDOG++ 094)."""
    output = {
        "x": [np.array([10.0, 20.5])],   # image rows
        "y": [np.array([30.0, 40.0])],   # image columns
        "classes": [np.array([0, 0])],
        "probabilities": [np.array([0.995, 0.42])],
    }
    assert points_from_output(output) == [(30.0, 10.0, 0.995), (40.0, 20.5, 0.42)]
    assert points_from_output({"x": [[]], "y": [[]], "probabilities": [[]]}) == []


@pytest.mark.parametrize(
    "output",
    [
        {"x": [np.array([1.0])], "y": [np.array([2.0])]},  # no probabilities: v2 invented 1.0
        {"x": [np.array([1.0])], "y": [np.array([2.0])], "probabilities": [np.array([])]},
        {"x": [np.array([1.0]), np.array([2.0])], "y": [np.array([2.0])], "probabilities": [np.array([0.9])]},
        {"x": [], "y": [], "probabilities": []},
        [{"centroid": [1.0, 2.0]}],
    ],
)
def test_unexpected_detector_output_is_an_error(output):
    with pytest.raises(EngineError):
        points_from_output(output)


def test_the_engine_asks_the_detector_for_probabilities_and_passes_the_threshold():
    """tiatoolbox 2.0.1 returns only x, y and classes unless return_probabilities=True."""
    import threading

    from engine import KongNetEngine

    calls = []

    class Detector:
        def run(self, **kwargs):
            calls.append(kwargs)
            return {"x": [np.array([5.0])], "y": [np.array([6.0])], "classes": [np.array([0])],
                    "probabilities": [np.array([0.97])]}

    engine = object.__new__(KongNetEngine)  # skip loading TIAToolbox
    engine.detector, engine._lock = Detector(), threading.Lock()
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    assert engine.predict(image, threshold_abs=0.05) == [(6.0, 5.0, 0.97)]
    assert engine.predict(image) == [(6.0, 5.0, 0.97)]
    assert all(call["return_probabilities"] is True and call["patch_mode"] is True for call in calls)
    assert calls[0]["threshold_abs"] == 0.05 and "threshold_abs" not in calls[1]
