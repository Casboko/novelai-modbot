from __future__ import annotations

from io import BytesIO
from typing import Iterable

import imagehash
from PIL import Image


def chunked(iterable: Iterable, size: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)


def compute_phash(data: bytes, hash_size: int = 8) -> str:
    with Image.open(BytesIO(data)) as image:
        phash = imagehash.phash(image.convert("RGB"), hash_size=hash_size)
    return str(phash)
