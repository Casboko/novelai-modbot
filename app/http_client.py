from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp


class RateLimiter:
    def __init__(self, qps: float) -> None:
        self._interval = 1.0 / max(qps, 1.0)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            sleep_for = self._interval - (now - self._last)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
                now = time.monotonic()
            self._last = now


@dataclass(slots=True)
class FetchResult:
    data: Optional[bytes]
    content_type: Optional[str]
    size: Optional[int]
    note: Optional[str]


class ImageFetcher:
    def __init__(self, session: aiohttp.ClientSession, limiter: RateLimiter) -> None:
        self._session = session
        self._limiter = limiter

    async def fetch(self, url: str) -> FetchResult:
        head_ct, head_size, head_note = await self._head(url)
        if head_note and head_note != "method_not_allowed":
            return FetchResult(None, head_ct, head_size, head_note)
        if head_ct and not head_ct.lower().startswith("image/"):
            return FetchResult(None, head_ct, head_size, "not_image")
        data, get_ct, get_size, get_note = await self._get(url)
        if data is None:
            return FetchResult(None, get_ct or head_ct, get_size or head_size, get_note or head_note)
        final_ct = get_ct or head_ct
        if final_ct and not final_ct.lower().startswith("image/"):
            return FetchResult(None, final_ct, len(data), "not_image")
        return FetchResult(data, final_ct, get_size or head_size or len(data), None)

    async def _head(self, url: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
        for _ in range(3):
            await self._limiter.wait()
            try:
                async with self._session.head(url, allow_redirects=True) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(self._retry_after(resp))
                        continue
                    if resp.status in {405, 501}:
                        return None, None, "method_not_allowed"
                    if resp.status >= 400:
                        return None, None, f"head_status_{resp.status}"
                    content_type = resp.headers.get("Content-Type")
                    length = resp.headers.get("Content-Length")
                    size = int(length) if length and length.isdigit() else None
                    return content_type, size, None
            except aiohttp.ClientError as exc:
                return None, None, f"head_error:{exc.__class__.__name__}"
        return None, None, "head_error:retry_exceeded"

    async def _get(self, url: str) -> tuple[Optional[bytes], Optional[str], Optional[int], Optional[str]]:
        for _ in range(3):
            await self._limiter.wait()
            try:
                async with self._session.get(url, allow_redirects=True) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(self._retry_after(resp))
                        continue
                    if resp.status >= 400:
                        return None, None, None, f"get_status_{resp.status}"
                    content = await resp.read()
                    content_type = resp.headers.get("Content-Type")
                    size = len(content)
                    return content, content_type, size, None
            except aiohttp.ClientError as exc:
                return None, None, None, f"get_error:{exc.__class__.__name__}"
        return None, None, None, "get_error:retry_exceeded"

    @staticmethod
    def _retry_after(resp: aiohttp.ClientResponse) -> float:
        retry = resp.headers.get("Retry-After")
        try:
            if retry is None:
                return 1.0
            return float(retry)
        except ValueError:
            return 1.0

