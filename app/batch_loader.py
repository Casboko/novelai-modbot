from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, Sequence

import aiohttp
from PIL import Image, UnidentifiedImageError

from .http_client import FetchResult, ImageFetcher, RateLimiter


@dataclass(slots=True)
class ImageRequest:
    identifier: str
    urls: tuple[str, ...]


@dataclass(slots=True)
class ImageLoadResult:
    request: ImageRequest
    image: Image.Image | None
    used_url: str | None
    note: str | None
    fetch: FetchResult | None


async def load_images(
    requests: Sequence[ImageRequest],
    *,
    qps: float = 5.0,
    concurrency: int = 4,
) -> list[ImageLoadResult]:
    if not requests:
        return []
    timeout = aiohttp.ClientTimeout(total=None)
    limiter = RateLimiter(qps)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        fetcher = ImageFetcher(session, limiter)
        semaphore = asyncio.Semaphore(concurrency)

        async def worker(request: ImageRequest) -> ImageLoadResult:
            async with semaphore:
                return await _fetch_single(fetcher, request)

        tasks = [asyncio.create_task(worker(req)) for req in requests]
        results = await asyncio.gather(*tasks)
    return list(results)


async def _fetch_single(fetcher: ImageFetcher, request: ImageRequest) -> ImageLoadResult:
    last_note: str | None = None
    for url in request.urls:
        if not url:
            continue
        fetch = await fetcher.fetch(url)
        if fetch.data is None:
            last_note = fetch.note
            continue
        try:
            with Image.open(BytesIO(fetch.data)) as image:
                loaded = image.convert("RGB")
                loaded.load()
        except UnidentifiedImageError:
            last_note = "decode_error"
            continue
        return ImageLoadResult(request, loaded, url, None, fetch)
    return ImageLoadResult(request, None, None, last_note, None)

