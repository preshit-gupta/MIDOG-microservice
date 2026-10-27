"""Throughput (OncoGemma v7.2 Stage 3 spec §12.1): one batched engine call per request, and tile packs.

The real-weights test (KONGNET_WEIGHTS set) checks that a patch's logits in a batch equal its single-patch
logits within the golden tolerance; the rest run on the fake engine.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine import FakeEngine, set_engine
from main import app
from tests.test_contract import make_png_b64
from tile_pack import TilePackError, read_tile_pack, tile_pack_bytes

ATOL = 0.05


@pytest.fixture
def fake():
    engine = FakeEngine(input_mpp=0.25, patch_px=512, weights_sha256="sha",
                        mock_predictions=[(256.0, 256.0, 0.95), (10.0, 20.0, 0.5)])
    set_engine(engine)
    yield engine
    set_engine(None)


@pytest.fixture
def pack_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TILE_PACK_ROOT", str(tmp_path))

    def write(name, tiles):
        path = tmp_path / "bucket" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tile_pack_bytes(tiles) if isinstance(tiles, list) else tiles)
        return f"gs://bucket/{name}"

    return write


def tile(value=200):
    return np.full((512, 512, 3), value, np.uint8)


def test_every_v3_instance_of_a_request_goes_through_one_batched_call(fake):
    body = {"instances": [{"image_png_b64": make_png_b64(), "mpp": 0.25} for _ in range(5)]}
    data = TestClient(app).post("/predict", json=body).json()
    assert fake.batches == [5] and len(data["predictions"]) == 5
    assert all(p["error"] is None and len(p["points"]) == 2 for p in data["predictions"])


def test_a_bad_instance_keeps_its_place_and_its_error(fake):
    body = {"instances": [{"image_png_b64": make_png_b64(), "mpp": 0.25},
                          {"image_png_b64": make_png_b64(), "mpp": 0.5},
                          {"image_png_b64": make_png_b64(), "mpp": 0.25}]}
    preds = TestClient(app).post("/predict", json=body).json()["predictions"]
    assert fake.batches == [2]
    assert [p["error"] is None for p in preds] == [True, False, True] and "mpp_mismatch" in preds[1]["error"]


def test_a_tile_pack_answers_one_prediction_with_one_entry_per_tile(fake, pack_root):
    uri = pack_root("cases/a/pack.npz", [tile(), tile(100), tile(50)])
    body = {"instances": [{"tile_pack_uri": uri, "mpp": 0.25, "min_prob": 0.6},
                          {"image_png_b64": make_png_b64(), "mpp": 0.25}]}
    preds = TestClient(app).post("/predict", json=body).json()["predictions"]
    assert fake.batches == [4]  # the PNG and the pack's three tiles in one call
    assert preds[0]["error"] is None and len(preds[0]["tiles"]) == 3
    assert [len(t["points"]) for t in preds[0]["tiles"]] == [1, 1, 1]  # min_prob 0.6 keeps only 0.95
    assert len(preds[1]["points"]) == 2


def test_tile_packs_of_the_wrong_shape_or_location_are_refused(fake, pack_root):
    small = pack_root("small.npz", [np.zeros((256, 256, 3), np.uint8)])
    junk = pack_root("junk.npz", b"not an npz")
    for uri, error in ((small, "invalid_pack"), (junk, "invalid_pack"), ("gs://bucket/missing.npz", "pack_read_error"),
                       ("https://example.com/x.npz", "invalid_pack_uri")):
        with pytest.raises(TilePackError, match=error):
            read_tile_pack(uri, 512)
    body = {"instances": [{"tile_pack_uri": small, "mpp": 0.25}]}
    pred = TestClient(app).post("/predict", json=body).json()["predictions"][0]
    assert pred["tiles"] == [] and pred["error"].startswith("invalid_pack") and fake.batches == []


def test_metadata_reports_the_batch_and_the_instances(fake):
    data = TestClient(app).get("/metadata").json()
    assert data["instances"] == ["image_png_b64", "tile_pack_uri", "image_bytes"]


def test_batched_logits_match_single_patch_logits_with_the_real_weights():
    weights = os.environ.get("KONGNET_WEIGHTS")
    if not weights or not Path(weights).is_file():
        pytest.skip("set KONGNET_WEIGHTS to the KongNet_Det_MIDOG_1.pth weights file")
    pytest.importorskip("tiatoolbox")
    from scripts.golden_logits import engine_for
    from tests.golden_tiles import TILES

    eng = engine_for(Path(weights))
    eng.max_batch = 4
    patches = [make() for make in TILES.values()] * 2 + [TILES["synthetic"]()[:, ::-1].copy()]
    batched = eng.logit_maps(patches)
    eng.max_batch = 1
    single = [eng.logit_maps([p])[0] for p in patches]
    worst = max(float(np.abs(b - s).max()) for b, s in zip(batched, single))
    assert worst <= ATOL, worst
    floors = [-4.6] * len(patches)
    eng.max_batch = 4
    got = eng.predict_logits_batch(patches, floors)
    want = [eng.predict_logits(p, -4.6) for p in patches]
    assert [[(x, y) for x, y, _ in g] for g in got] == [[(x, y) for x, y, _ in w] for w in want]
    assert all(abs(a[2] - b[2]) <= ATOL for g, w in zip(got, want) for a, b in zip(g, w))
    print(json.dumps({"max_abs_logit_difference": worst}))
