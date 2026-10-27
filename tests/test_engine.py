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


# --- contract v3: logits (OncoGemma v7.2 Stage 4 spec §4.2) ------------------------------------------------------

from engine import float32_sigmoid, logit_of, peaks_from_logit_map  # noqa: E402

MPP = 0.25  # 5 um = 20 px, 7.5 um = 30 px


def blob_map(peaks, size=128, sigma=4.0, base=-12.0):
    """A logit map: a flat background with Gaussian bumps of the given heights at (x, y)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    m = np.full((size, size), base, dtype=np.float32)
    for x, y, height in peaks:
        m = np.maximum(m, base + (height - base) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)))
    return m


def test_peaks_are_columns_and_rows_highest_logit_first():
    m = blob_map([(20, 90, 40.0), (100, 30, 118.0)])
    assert peaks_from_logit_map(m, MPP, logit_of(0.01)) == [(100.0, 30.0, 118.0), (20.0, 90.0, 40.0)]


def test_peaks_within_7_5_um_keep_only_the_stronger():
    # 25 px apart (6.25 um): both are local maxima 5 um apart, NMS keeps the higher logit.
    m = blob_map([(40, 40, 30.0), (65, 40, 90.0)], sigma=3.0)
    assert peaks_from_logit_map(m, MPP, logit_of(0.01)) == [(65.0, 40.0, 90.0)]
    # 40 px apart (10 um): both stay.
    m = blob_map([(40, 40, 30.0), (80, 40, 90.0)], sigma=3.0)
    assert [p[:2] for p in peaks_from_logit_map(m, MPP, logit_of(0.01))] == [(80.0, 40.0), (40.0, 40.0)]


def test_the_floor_is_logit_of_min_prob_inclusive():
    m = blob_map([(30, 30, -4.0), (90, 90, -5.0)])
    assert [p[2] for p in peaks_from_logit_map(m, MPP, logit_of(0.01))] == [-4.0]  # logit(0.01) = -4.595
    assert peaks_from_logit_map(m, MPP, -4.0) == [(30.0, 30.0, -4.0)]


def test_saturated_peaks_keep_distinct_logits_and_a_plateau_gives_one_peak():
    m = blob_map([(30, 30, 150.0), (90, 90, 40.0)])
    probs = [float32_sigmoid(p[2]) for p in peaks_from_logit_map(m, MPP, logit_of(0.01))]
    assert probs == [1.0, 1.0]  # both saturate as probabilities; their logits (150, 40) still rank them
    flat = np.full((64, 64), -12.0, dtype=np.float32)
    flat[20:24, 20:24] = 50.0  # a 4 x 4 plateau: one detection, at its first row-major pixel
    assert peaks_from_logit_map(flat, MPP, 0.0) == [(20.0, 20.0, 50.0)]


def test_peaks_are_deterministic_and_reject_bad_maps():
    m = blob_map([(30, 30, 20.0), (90, 30, 20.0), (60, 100, 20.0)])
    first = peaks_from_logit_map(m, MPP, 0.0)
    assert first == peaks_from_logit_map(m.copy(), MPP, 0.0)
    assert [p[:2] for p in first] == [(30.0, 30.0), (90.0, 30.0), (60.0, 100.0)]  # equal logits: row, then column
    with pytest.raises(EngineError):
        peaks_from_logit_map(np.zeros((4, 4, 1)), MPP, 0.0)
    bad = m.copy()
    bad[0, 0] = np.nan
    with pytest.raises(EngineError):
        peaks_from_logit_map(bad, MPP, 0.0)


def test_logit_of_and_the_float32_sigmoid():
    assert logit_of(0.5) == 0.0 and logit_of(0.01) == pytest.approx(-4.59512, abs=1e-5)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            logit_of(bad)
    assert float32_sigmoid(17.5) == 1.0 and float32_sigmoid(16.0) < 1.0  # the plateau the v3 contract avoids
    assert float32_sigmoid(0.0) == 0.5 and float32_sigmoid(-4.6) == pytest.approx(0.01, abs=1e-4)


def test_logit_map_normalises_with_preproc_and_skips_the_sigmoid():
    """TIAToolbox's infer_batch does not normalise; the engine must call preproc before the forward pass."""
    torch = pytest.importorskip("torch")
    import threading

    from engine import KongNetEngine

    seen = {}

    class Model(torch.nn.Module):
        def preproc(self, batch):
            seen["preproc_in"] = batch.copy()
            return (batch / 255.0 - 0.5) / 0.25  # a stand-in normalisation the test can recognise

        def forward(self, x):
            seen["forward_in"] = x.detach().clone()
            return torch.cat([x[:, :1] * 100.0, x[:, 1:2]], dim=1)  # channel 0: large values, never squashed

    engine = object.__new__(KongNetEngine)
    engine.model, engine.target_channels, engine.device, engine._lock = Model(), [0], "cpu", threading.Lock()
    tile = np.full((8, 8, 3), 255, dtype=np.uint8)
    out = engine.logit_map(tile)
    assert seen["preproc_in"].shape == (1, 8, 8, 3)
    assert seen["forward_in"].shape == (1, 3, 8, 8) and float(seen["forward_in"].max()) == pytest.approx(2.0)
    assert out.shape == (8, 8) and float(out.max()) == pytest.approx(200.0)  # a logit, not sigmoid(200) = 1
    with pytest.raises(EngineError):
        engine.logit_map(np.zeros((8, 8), dtype=np.uint8))
