"""Gate G1 (contract v3): the engine's logit path reproduces the recorded logits with the real weights.

Needs TIAToolbox, torch and the weights; set KONGNET_WEIGHTS to KongNet_Det_MIDOG_1.pth (in the image:
/root/.tiatoolbox/models/KongNet_Det_MIDOG_1.pth). Skipped otherwise. The fixture comes from
scripts/golden_logits.py on CPU; a GPU build differs by float noise, well inside the tolerance.

What it guards: the ImageNet normalisation (``preproc``) before the forward pass. Without it the same
synthetic tile's logits move by 144 on average, reach +263, and the candidate moves off the figure.
"""
import json
import os
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "golden_logits.json"
ATOL = 0.05  # logits; CPU vs GPU and BLAS differences are far smaller


@pytest.fixture(scope="module")
def engine():
    weights = os.environ.get("KONGNET_WEIGHTS")
    if not weights or not Path(weights).is_file():
        pytest.skip("set KONGNET_WEIGHTS to the KongNet_Det_MIDOG_1.pth weights file")
    pytest.importorskip("tiatoolbox")
    from engine import compute_file_sha256
    from scripts.golden_logits import engine_for

    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert compute_file_sha256(weights) == golden["weights_sha256"], "these are not the weights the fixture used"
    return engine_for(Path(weights)), golden


def test_logit_maps_match_the_golden_record(engine):
    import numpy as np

    from tests.golden_tiles import PROBES, TILES

    eng, golden = engine
    for name, make in TILES.items():
        want = golden["tiles"][name]
        m = eng.logit_map(make())
        assert list(m.shape) == want["shape"]
        assert np.isclose([m.min(), m.max(), m.mean()], [want["min"], want["max"], want["mean"]], atol=ATOL).all(), name
        for (r, c), (_, _, value) in zip(PROBES, want["probes"]):
            assert abs(float(m[r, c]) - value) <= ATOL, (name, r, c)


def test_detections_match_the_golden_record(engine):
    from engine import logit_of
    from tests.golden_tiles import TILES

    eng, golden = engine
    for name, make in TILES.items():
        got = eng.predict_logits(make(), logit_of(golden["min_prob"]))
        want = golden["tiles"][name]["peaks"]
        assert [(x, y) for x, y, _ in got] == [(x, y) for x, y, _ in want], name
        assert all(abs(g[2] - w[2]) <= ATOL for g, w in zip(got, want)), name
