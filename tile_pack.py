"""Tile packs: many patches in one GCS object instead of base64 PNGs in the request (OncoGemma v7.2 spec §12.1).

A pack is an ``.npz`` with one array ``tiles`` of shape (N, patch_px, patch_px, 3), uint8 RGB, at the service's
0.25 um/px, written by the caller to a bucket in the endpoint's region. The service reads it with its own service
account (it needs ``storage.objects.get`` on the bucket) through the Cloud Storage JSON API, with no extra library:
the token comes from the metadata server. ``TILE_PACK_ROOT`` (a local directory) replaces GCS in tests.

The request is a few hundred bytes whatever N is, so the online-prediction payload limit no longer caps the batch.
"""
import io
import json
import os
import urllib.parse
import urllib.request
from typing import List

import numpy as np

MAX_TILES = int(os.environ.get("MIDOG_MAX_PACK_TILES", "256"))
METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
GCS_MEDIA_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o/{name}?alt=media"
TIMEOUT_S = 30


class TilePackError(ValueError):
    """The pack could not be read, or is not N patches of the service's size."""


def _token() -> str:
    req = urllib.request.Request(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read())["access_token"]


def _read_bytes(uri: str) -> bytes:
    if not uri.startswith("gs://"):
        raise TilePackError(f"invalid_pack_uri: expected gs://bucket/object, got {uri!r}")
    bucket, _, name = uri[len("gs://"):].partition("/")
    if not bucket or not name:
        raise TilePackError(f"invalid_pack_uri: {uri!r}")
    local_root = os.environ.get("TILE_PACK_ROOT")
    try:
        if local_root:
            with open(os.path.join(local_root, bucket, *name.split("/")), "rb") as f:
                return f.read()
        url = GCS_MEDIA_URL.format(bucket=bucket, name=urllib.parse.quote(name, safe=""))
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return resp.read()
    except (OSError, ValueError, KeyError) as e:
        raise TilePackError(f"pack_read_error: {type(e).__name__}: {e}") from e


def read_tile_pack(uri: str, patch_px: int) -> List[np.ndarray]:
    """The pack's patches, in order; refuses anything but N x patch_px x patch_px x 3 uint8 with N <= MAX_TILES."""
    data = _read_bytes(uri)
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            tiles = z["tiles"]
    except Exception as e:
        raise TilePackError(f"invalid_pack: {type(e).__name__}: {e}") from e
    if tiles.dtype != np.uint8 or tiles.ndim != 4 or tiles.shape[1:] != (patch_px, patch_px, 3):
        raise TilePackError(f"invalid_pack: expected N x {patch_px} x {patch_px} x 3 uint8, "
                            f"got {tiles.shape} {tiles.dtype}")
    if not 0 < len(tiles) <= MAX_TILES:
        raise TilePackError(f"invalid_pack: {len(tiles)} tiles; between 1 and {MAX_TILES} per pack")
    return [tiles[k] for k in range(len(tiles))]


def tile_pack_bytes(tiles: List[np.ndarray]) -> bytes:
    """The npz a caller uploads (the OncoGemma worker writes the same layout)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, tiles=np.stack(tiles).astype(np.uint8))
    return buf.getvalue()
