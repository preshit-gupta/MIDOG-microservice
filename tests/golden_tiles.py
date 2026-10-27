"""Deterministic 512 px test tiles for the golden-logit check (contract v3, gate G1).

Synthetic, so no third-party image data ships with the repository: a constant tile, and an
H&E-like tile (eosin background, haematoxylin nuclei, one compact dark figure) from a fixed seed.
"""
import numpy as np

SIZE = 512


def constant_tile() -> np.ndarray:
    return np.full((SIZE, SIZE, 3), (240, 220, 230), dtype=np.uint8)


def synthetic_tile(seed: int = 20261027) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    img = np.empty((SIZE, SIZE, 3), dtype=np.float32)
    img[...] = (236.0, 190.0, 214.0)  # eosin pink
    img += rng.normal(0.0, 6.0, img.shape).astype(np.float32)
    for _ in range(60):  # interphase nuclei: pale purple ellipses
        cx, cy = rng.uniform(20, SIZE - 20, 2)
        a, b = rng.uniform(14, 24), rng.uniform(10, 18)
        theta = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
        v = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
        inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        img[inside] = img[inside] * 0.35 + np.array((120.0, 80.0, 160.0)) * 0.65
    # One compact, very dark, irregular figure: the shape a mitotic figure has.
    cx, cy = 300.0, 220.0
    r = 9.0 + 3.0 * np.sin(5 * np.arctan2(yy - cy, xx - cx))
    figure = np.hypot(xx - cx, yy - cy) <= r
    img[figure] = (40.0, 20.0, 70.0)
    return np.clip(img, 0, 255).astype(np.uint8)


TILES = {"constant": constant_tile, "synthetic": synthetic_tile}
# Fixed pixels whose logits are recorded, (row, col).
PROBES = ((0, 0), (256, 256), (220, 300), (100, 400), (511, 511))
