import io
import base64
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from engine import FakeEngine, set_engine


def make_png_b64(width: int = 512, height: int = 512) -> str:
    """Creates a base64-encoded lossless PNG image."""
    img = Image.new("RGB", (width, height), color=(240, 220, 230))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_jpeg_b64(width: int = 512, height: int = 512) -> str:
    """Creates a base64-encoded JPEG image (invalid for v2)."""
    img = Image.new("RGB", (width, height), color=(240, 220, 230))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture(autouse=True)
def setup_fake_engine():
    """Injects FakeEngine into module-level factory before every test."""
    fake = FakeEngine(
        input_mpp=0.25,
        patch_px=512,
        weights_sha256="abc123sha256fake",
        tiatoolbox_version="2.1.3",
        mock_predictions=[
            (256.0, 256.0, 0.95),
            (100.0, 150.0, 0.50),
            (300.0, 300.0, 0.005),
        ],
    )
    set_engine(fake)
    yield fake
    set_engine(None)


def test_metadata_fields():
    client = TestClient(app)
    resp = client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "KongNet_Det_MIDOG_1"
    assert data["tiatoolbox"] == "2.1.3"
    assert data["weights_sha256"] == "abc123sha256fake"
    assert data["input_mpp"] == 0.25
    assert data["patch_px"] == 512
    assert data["output"] == "points"
    assert data["deterministic"] is True
    assert data["contract"] == "v2"


def test_health_success_and_failure():
    client = TestClient(app)
    # Healthy when engine loaded
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model"] == "KongNet_Det_MIDOG_1"
    assert data["weights_sha256"] == "abc123sha256fake"

    # 503 when engine is None
    set_engine(None)
    resp_fail = client.get("/health")
    assert resp_fail.status_code == 503


def test_predict_v2_happy_path():
    client = TestClient(app)
    png_b64 = make_png_b64(512, 512)

    payload = {
        "instances": [
            {"image_png_b64": png_b64, "mpp": 0.25}
        ],
        "parameters": {
            "min_prob": 0.01
        }
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["model_sha256"] == "abc123sha256fake"
    assert len(data["predictions"]) == 1
    pred = data["predictions"][0]
    assert pred["error"] is None
    # Detections >= 0.01: (256, 256, 0.95) and (100, 150, 0.50). 0.005 is excluded.
    assert len(pred["points"]) == 2
    assert pred["points"][0] == {"x": 256.0, "y": 256.0, "prob": 0.95}
    assert pred["points"][1] == {"x": 100.0, "y": 150.0, "prob": 0.50}


def test_predict_v2_mpp_mismatch():
    client = TestClient(app)
    png_b64 = make_png_b64(512, 512)

    # 0.20 differs from 0.25 by 20%, exceeding 1% tolerance
    payload = {
        "instances": [
            {"image_png_b64": png_b64, "mpp": 0.20}
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    pred = data["predictions"][0]
    assert pred["points"] == []
    assert "mpp_mismatch" in pred["error"]
    assert "expected 0.25, got 0.2" in pred["error"]


def test_predict_v2_wrong_size():
    client = TestClient(app)
    wrong_size_png = make_png_b64(256, 256)

    payload = {
        "instances": [
            {"image_png_b64": wrong_size_png, "mpp": 0.25}
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    pred = data["predictions"][0]
    assert pred["points"] == []
    assert "invalid_size" in pred["error"]
    assert "expected 512x512, got 256x256" in pred["error"]


def test_predict_v2_non_png_input():
    client = TestClient(app)
    jpeg_b64 = make_jpeg_b64(512, 512)

    payload = {
        "instances": [
            {"image_png_b64": jpeg_b64, "mpp": 0.25}
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    pred = data["predictions"][0]
    assert pred["points"] == []
    assert "invalid_format: input image must be PNG" in pred["error"]


def test_predict_v2_min_prob_filtering():
    client = TestClient(app)
    png_b64 = make_png_b64(512, 512)

    # Filter with min_prob = 0.60
    payload = {
        "instances": [
            {"image_png_b64": png_b64, "mpp": 0.25, "min_prob": 0.60}
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    pred = data["predictions"][0]
    assert pred["error"] is None
    # Only 0.95 exceeds 0.60
    assert len(pred["points"]) == 1
    assert pred["points"][0] == {"x": 256.0, "y": 256.0, "prob": 0.95}


def test_predict_legacy_path():
    client = TestClient(app)
    # Legacy sends image_bytes (can be JPEG or PNG)
    jpeg_b64 = make_jpeg_b64(512, 512)

    payload = {
        "instances": [
            {"image_bytes": jpeg_b64, "confidence_threshold": 0.25}
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["model_sha256"] == "abc123sha256fake"
    assert len(data["predictions"]) == 1
    pred = data["predictions"][0]
    assert "boxes" in pred
    # Boxes with prob >= 0.25: (256, 256, 0.95) and (100, 150, 0.50)
    assert len(pred["boxes"]) == 2
    assert pred["boxes"][0]["cx"] == 256.0
    assert pred["boxes"][0]["cy"] == 256.0
    assert pred["boxes"][0]["width"] == 48.0
    assert pred["boxes"][0]["height"] == 48.0
    assert pred["boxes"][0]["confidence"] == 0.95


def test_v2_min_prob_is_the_detector_threshold_and_legacy_keeps_the_model_default(setup_fake_engine):
    """KongNet keeps only peaks >= 0.99 unless told otherwise; v2 asks for candidates down to min_prob."""
    client = TestClient(app)
    client.post("/predict", json={"instances": [{"image_png_b64": make_png_b64(), "mpp": 0.25}],
                                  "parameters": {"min_prob": 0.05}})
    client.post("/predict", json={"instances": [{"image_png_b64": make_png_b64(), "mpp": 0.25, "min_prob": 0.3}]})
    client.post("/predict", json={"instances": [{"image_png_b64": make_png_b64(), "mpp": 0.25}]})
    client.post("/predict", json={"instances": [{"image_bytes": make_jpeg_b64(), "confidence_threshold": 0.25}]})
    assert setup_fake_engine.thresholds == [0.05, 0.3, 0.01, None]


def test_an_engine_that_cannot_load_is_reported_by_every_route(monkeypatch):
    import engine as engine_module

    def broken():
        raise engine_module.EngineError("weights file /root/.tiatoolbox/models/KongNet_Det_MIDOG_1.pth does not exist")

    set_engine(None)
    monkeypatch.setattr(engine_module, "KongNetEngine", broken)
    client = TestClient(app)
    for response in (client.get("/health"), client.get("/metadata"),
                     client.post("/predict", json={"instances": [{"image_png_b64": make_png_b64(), "mpp": 0.25}]})):
        assert response.status_code == 503
        assert "EngineError" in response.json()["detail"]


def test_the_service_takes_0_25_um_per_px_and_refuses_0_5(setup_fake_engine):
    """KongNet reaches F1 0.87 at native 40x and at most 0.27 at 0.5 um/px (MIDOG++ 094, SPEC-06 AC5)."""
    client = TestClient(app)
    meta = client.get("/metadata").json()
    assert meta["input_mpp"] == 0.25 and meta["ioconfig_input_mpp"] == 0.5
    refused = client.post("/predict", json={"instances": [{"image_png_b64": make_png_b64(), "mpp": 0.5}]}).json()
    assert refused["predictions"][0]["error"] == "mpp_mismatch: expected 0.25, got 0.5"
