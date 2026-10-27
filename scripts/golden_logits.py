"""Record the golden logits of the test tiles with the real weights (contract v3, gate G1).

Builds the model exactly as the engine does (TIAToolbox's named pretrained model from the given
weights file) and runs ``KongNetEngine.logit_map`` and the v3 peak rule on the deterministic tiles of
tests/golden_tiles.py. tests/test_golden_logits.py checks a later build against the file.

    python scripts/golden_logits.py --weights /path/KongNet_Det_MIDOG_1.pth --out tests/fixtures/golden_logits.json
    python scripts/golden_logits.py --check tests/fixtures/golden_logits.json   # the image build runs this

Without --weights the weights come from TIAToolbox's cache (where the image build pre-warms them).
"""
import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import MODEL_INPUT_MPP, MODEL_NAME, KongNetEngine, compute_file_sha256, logit_of  # noqa: E402
from tests.golden_tiles import PROBES, TILES  # noqa: E402

MIN_PROB = 0.01
ATOL = 0.05  # logits: CPU vs GPU and BLAS differences are far smaller; a missing preproc moves them by ~144


def engine_for(weights: Path) -> KongNetEngine:
    """The engine's logit path on CPU, without TIAToolbox's NucleusDetector or its cache."""
    from tiatoolbox.models.architecture import get_pretrained_model

    engine = object.__new__(KongNetEngine)
    engine.model, _ = get_pretrained_model(MODEL_NAME, pretrained_weights=str(weights))
    engine.model.eval()
    engine.device = "cpu"
    engine.model.to("cpu")
    engine.target_channels = list(getattr(engine.model, "target_channels", None) or [0])
    engine.input_mpp = MODEL_INPUT_MPP
    engine._lock = threading.Lock()
    return engine


def record(engine: KongNetEngine) -> dict:
    out = {}
    for name, make in TILES.items():
        m = engine.logit_map(make())
        peaks = engine.predict_logits(make(), logit_of(MIN_PROB))
        out[name] = {
            "shape": list(m.shape),
            "min": float(m.min()), "max": float(m.max()), "mean": float(m.mean()),
            "probes": [[r, c, float(m[r, c])] for r, c in PROBES],
            "peaks": [[x, y, round(v, 4)] for x, y, v in peaks],
        }
    return out


def differences(got: dict, want: dict) -> list:
    """Every way the recorded tiles differ beyond ATOL; empty when they match."""
    problems = []
    for name, w in want.items():
        g = got[name]
        if g["shape"] != w["shape"]:
            problems.append(f"{name}: shape {g['shape']} != {w['shape']}")
        for key in ("min", "max", "mean"):
            if abs(g[key] - w[key]) > ATOL:
                problems.append(f"{name}: {key} {g[key]:.3f} != {w[key]:.3f}")
        for (r, c, gv), (_, _, wv) in zip(g["probes"], w["probes"]):
            if abs(gv - wv) > ATOL:
                problems.append(f"{name}: logit at ({r}, {c}) {gv:.3f} != {wv:.3f}")
        if [p[:2] for p in g["peaks"]] != [p[:2] for p in w["peaks"]]:
            problems.append(f"{name}: peaks {g['peaks']} != {w['peaks']}")
        elif any(abs(gp[2] - wp[2]) > ATOL for gp, wp in zip(g["peaks"], w["peaks"])):
            problems.append(f"{name}: peak logits {g['peaks']} != {w['peaks']}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, help="default: TIAToolbox's cached KongNet_Det_MIDOG_1.pth")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="record a new fixture")
    group.add_argument("--check", type=Path, help="compare with a recorded fixture; exit 1 on a difference")
    args = ap.parse_args()
    weights = args.weights
    if weights is None:
        from tiatoolbox import rcParam

        from engine import weights_path

        weights = weights_path(rcParam["TIATOOLBOX_HOME"])
    sha = compute_file_sha256(weights)
    if args.check is not None:
        want = json.loads(args.check.read_text(encoding="utf-8"))
        if sha != want["weights_sha256"]:
            raise SystemExit(f"weights {weights} have sha256 {sha}; the fixture was recorded with {want['weights_sha256']}")
        problems = differences(record(engine_for(weights)), want["tiles"])
        if problems:
            raise SystemExit("golden logits differ:\n  " + "\n  ".join(problems))
        print(f"golden logits match ({len(want['tiles'])} tiles, tolerance {ATOL})")
        return
    doc = {"model": MODEL_NAME, "weights_sha256": sha, "mpp": MODEL_INPUT_MPP,
           "min_prob": MIN_PROB, "device": "cpu", "tiles": record(engine_for(weights))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    for name, t in doc["tiles"].items():
        print(f"{name}: logit range {t['min']:.2f} .. {t['max']:.2f}, {len(t['peaks'])} peaks >= p {MIN_PROB}")


if __name__ == "__main__":
    main()
