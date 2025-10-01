from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
import random
from typing import Optional
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

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
    def __init__(
        self,
        session: aiohttp.ClientSession,
        limiter: RateLimiter,
        *,
        max_retries: int = 4,
        max_backoff: float = 60.0,
    ) -> None:
        self._session = session
        self._limiter = limiter
        self._max_retries = max(0, int(max_retries))
        self._max_backoff = max_backoff

    async def fetch(self, url: str) -> FetchResult:
        parsed = urlparse(url)
        if parsed.scheme.lower() == "file":
            return self._fetch_local(parsed)
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

    def _fetch_local(self, parsed_url) -> FetchResult:
        path = self._path_from_file_url(parsed_url)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return FetchResult(None, None, None, "local_missing")
        except OSError as exc:
            return FetchResult(None, None, None, f"local_error:{exc.__class__.__name__}")
        content_type, _ = mimetypes.guess_type(str(path))
        content_type = content_type or "application/octet-stream"
        return FetchResult(data, content_type, len(data), "local")

    async def _head(self, url: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
        last_note: Optional[str] = None
        for attempt in range(self._max_retries + 1):
            await self._limiter.wait()
            try:
                async with self._session.head(url, allow_redirects=True) as resp:
                    status = resp.status
                    if status == 429 or 500 <= status < 600:
                        last_note = f"head_status_{status}"
                        await self._sleep_with_backoff(attempt, self._retry_after(resp))
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
                last_note = f"head_error:{exc.__class__.__name__}"
                await self._sleep_with_backoff(attempt, None)
                continue
        return None, None, last_note or "head_error:retry_exceeded"

    async def _get(self, url: str) -> tuple[Optional[bytes], Optional[str], Optional[int], Optional[str]]:
        last_note: Optional[str] = None
        for attempt in range(self._max_retries + 1):
            await self._limiter.wait()
            try:
                async with self._session.get(url, allow_redirects=True) as resp:
                    status = resp.status
                    if status == 429 or 500 <= status < 600:
                        last_note = f"get_status_{status}"
                        await self._sleep_with_backoff(attempt, self._retry_after(resp))
                        continue
                    if resp.status >= 400:
                        return None, None, None, f"get_status_{resp.status}"
                    content = await resp.read()
                    content_type = resp.headers.get("Content-Type")
                    size = len(content)
                    return content, content_type, size, None
            except aiohttp.ClientError as exc:
                last_note = f"get_error:{exc.__class__.__name__}"
                await self._sleep_with_backoff(attempt, None)
                continue
        return None, None, None, last_note or "get_error:retry_exceeded"

    @staticmethod
    def _retry_after(resp: aiohttp.ClientResponse) -> Optional[float]:
        retry = resp.headers.get("Retry-After")
        if retry is None:
            return None
        try:
            return float(retry)
        except ValueError:
            return None

    async def _sleep_with_backoff(self, attempt: int, retry_after: Optional[float]) -> None:
        if attempt >= self._max_retries:
            return
        delay = self._compute_backoff(attempt, retry_after)
        if delay > 0:
            await asyncio.sleep(delay)

    def _compute_backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            delay = retry_after
        else:
            base = 1.0 * (2 ** attempt)
            jitter = random.uniform(0, 0.5)
            delay = base + jitter
        return max(0.1, min(delay, self._max_backoff))

    @staticmethod
    def _path_from_file_url(parsed_url) -> Path:
        if parsed_url.scheme.lower() != "file":
            raise ValueError("URL scheme must be file")
        path_str = url2pathname(parsed_url.path)
        netloc = parsed_url.netloc
        if netloc:
            if os.name == "nt":
                path_str = f"//{netloc}{path_str}"
            else:
                path_str = f"/{netloc}{path_str}"
        if not path_str:
            path_str = "/"
        return Path(path_str)
