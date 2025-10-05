from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import logging
import mimetypes
from dataclasses import dataclass
from datetime import datetime, time as time_cls, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable, Optional

import aiohttp
import discord
import imagehash
from discord import abc

from .config import get_settings
from .profiles import PartitionPaths, ProfileContext
from .http_client import ImageFetcher, RateLimiter
from .util import compute_phash


logger = logging.getLogger(__name__)


FIELDNAMES = [
    "guild_id",
    "channel_id",
    "message_id",
    "message_link",
    "created_at",
    "author_id",
    "is_nsfw_channel",
    "source",
    "attachment_id",
    "filename",
    "content_type",
    "file_size",
    "url",
    "phash_hex",
    "phash_distance_ref",
    "note",
]


def _temporary_path(path: Path) -> Path:
    if path.suffix:
        return path.with_suffix(path.suffix + ".tmp")
    return path.with_name(path.name + ".tmp")


@dataclass
class ScanOptions:
    guild_id: int
    since: Optional[datetime]
    channel_filters: set[str]
    include_archived_threads: bool
    resume: bool
    qps: float
    out_path: Path
    state_path: Path
    cache_path: Path
    attachments_index_path: Path
    profile: ProfileContext


class CursorStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._state: dict[str, int] = {}
        self._dirty = False

    def load(self) -> None:
        if self._path.exists():
            self._state = json.loads(self._path.read_text("utf-8"))

    def get(self, channel_id: int) -> Optional[int]:
        value = self._state.get(str(channel_id))
        return int(value) if value is not None else None

    def update(self, channel_id: int, message_id: int) -> None:
        if self._state.get(str(channel_id)) == message_id:
            return
        self._state[str(channel_id)] = message_id
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state, ensure_ascii=False, indent=2)
        tmp_path = _temporary_path(self._path)
        tmp_path.write_text(payload, "utf-8")
        tmp_path.replace(self._path)
        self._dirty = False


class PhashCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str] = {}
        self._dirty = False

    def load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                url = row.get("url")
                phash = row.get("phash_hex")
                if url and phash:
                    self._data[url] = phash

    def get(self, url: str) -> Optional[str]:
        return self._data.get(url)

    def set(self, url: str, phash_hex: str) -> None:
        if self._data.get(url) == phash_hex:
            return
        self._data[url] = phash_hex
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _temporary_path(self._path)
        with tmp_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["url", "phash_hex"])
            writer.writeheader()
            for url, phash in self._data.items():
                writer.writerow({"url": url, "phash_hex": phash})
        tmp_path.replace(self._path)
        self._dirty = False


class CsvSink:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        exists = self._path.exists()
        self._file = self._path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        if not exists:
            self._writer.writeheader()

    def write(self, record: dict[str, object]) -> None:
        assert self._writer is not None
        self._writer.writerow(record)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None


class PhashIndex:
    def __init__(self) -> None:
        self._hashes: list[imagehash.ImageHash] = []

    def register(self, phash_hex: str) -> None:
        self._hashes.append(imagehash.hex_to_hash(phash_hex))

    def nearest_distance(self, phash_hex: str) -> Optional[int]:
        if not self._hashes:
            return None
        target = imagehash.hex_to_hash(phash_hex)
        return min(int(target - existing) for existing in self._hashes)


class ImageFetcher:
    def __init__(self, session: aiohttp.ClientSession, limiter: RateLimiter) -> None:
        self._session = session
        self._limiter = limiter

    async def fetch(self, url: str) -> tuple[Optional[bytes], Optional[str], Optional[int], Optional[str]]:
        head_ct, head_size, head_note = await self._head(url)
        if head_note and head_note != "method_not_allowed":
            return None, head_ct, head_size, head_note
        if head_ct and not head_ct.lower().startswith("image/"):
            return None, head_ct, head_size, "not_image"
        data, get_ct, get_size, get_note = await self._get(url)
        if data is None:
            note = get_note or head_note
            return None, get_ct or head_ct, get_size or head_size, note
        content_type = get_ct or head_ct
        if content_type and not content_type.lower().startswith("image/"):
            return None, content_type, len(data), "not_image"
        return data, content_type, get_size or head_size, None

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
                    length_header = resp.headers.get("Content-Length")
                    size = int(length_header) if length_header and length_header.isdigit() else None
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


class ScanClient(discord.Client):
    def __init__(self, runner: "ScanRunner", *, intents: discord.Intents) -> None:
        super().__init__(intents=intents)
        self._runner = runner
        self._started = False

    async def on_ready(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            await self._runner.execute(self)
        finally:
            await self.close()


class ScanRunner:
    def __init__(self, options: ScanOptions) -> None:
        self._options = options
        self._cursor = CursorStore(options.state_path)
        self._cache = PhashCache(options.cache_path)
        self._sink = CsvSink(options.out_path)
        self._phash_index = PhashIndex()
        self._attachments_index_path = options.attachments_index_path
        self._profile_context = options.profile
        self._session: Optional[aiohttp.ClientSession] = None
        self._fetcher: Optional[ImageFetcher] = None

    async def run(self, token: str) -> None:
        self._cursor.load()
        self._cache.load()
        self._sink.open()

        timeout = aiohttp.ClientTimeout(total=None)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._fetcher = ImageFetcher(self._session, RateLimiter(self._options.qps))

        intents = discord.Intents.default()
        intents.message_content = True
        client = ScanClient(self, intents=intents)
        try:
            await client.start(token)
        finally:
            with contextlib.suppress(Exception):
                await client.close()
            if self._session is not None:
                await self._session.close()
            self._session = None
            self._fetcher = None
            self._sink.close()
            self._cursor.save()
            self._cache.save()

    async def execute(self, client: discord.Client) -> None:
        guild = client.get_guild(self._options.guild_id)
        if guild is None:
            guild = await client.fetch_guild(self._options.guild_id)
        if self._fetcher is None:
            raise RuntimeError("Image fetcher is not initialised")
        await self._scan_guild(guild, self._fetcher)

    async def _scan_guild(self, guild: discord.Guild, fetcher: ImageFetcher) -> None:
        logger.info("scan started", extra={"guild_id": guild.id})
        async for destination in self._iter_targets(guild):
            await self._scan_destination(destination, fetcher)
        logger.info("scan completed", extra={"guild_id": guild.id})

    async def _iter_targets(self, guild: discord.Guild) -> AsyncIterator[abc.Messageable]:
        for channel in guild.text_channels:
            if not self._is_selected(channel):
                continue
            yield channel
            for thread in channel.threads:
                if self._is_selected(thread):
                    yield thread
            if not self._options.include_archived_threads:
                continue
            try:
                async for thread in channel.archived_threads(limit=None):
                    if self._is_selected(thread):
                        yield thread
            except discord.Forbidden:
                logger.warning("archived thread access denied", extra={"channel_id": channel.id})

    def _is_selected(self, channel: abc.Messageable) -> bool:
        if not self._options.channel_filters:
            return True
        channel_id = str(getattr(channel, "id", ""))
        channel_name = getattr(channel, "name", "")
        parent_name = getattr(getattr(channel, "parent", None), "name", "")
        return (
            channel_id in self._options.channel_filters
            or channel_name in self._options.channel_filters
            or parent_name in self._options.channel_filters
        )

    async def _scan_destination(self, destination: abc.Messageable, fetcher: ImageFetcher) -> None:
        channel_id = getattr(destination, "id", None)
        logger.info("scanning", extra={"channel_id": channel_id})
        after = None
        if self._options.resume:
            last_id = self._cursor.get(channel_id)
            if last_id is not None:
                after = discord.Object(id=last_id)
        if after is None and self._options.since is not None:
            after = self._options.since
        try:
            async for message in destination.history(limit=None, oldest_first=True, after=after):
                await self._process_message(destination, message, fetcher)
                if channel_id is not None:
                    self._cursor.update(channel_id, message.id)
                    self._cursor.save()
        except discord.Forbidden:
            logger.warning("permission denied", extra={"channel_id": channel_id})
        except discord.HTTPException as exc:
            logger.error("history fetch failed", extra={"channel_id": channel_id, "error": str(exc)})

    async def _process_message(
        self,
        destination: abc.Messageable,
        message: discord.Message,
        fetcher: ImageFetcher,
    ) -> None:
        channel_id = getattr(destination, "id", 0)
        for attachment in message.attachments:
            await self._process_attachment(destination, message, attachment, fetcher)
        for embed in message.embeds:
            await self._process_embed(destination, message, embed, fetcher)
        logger.debug(
            "message processed",
            extra={"channel_id": channel_id, "message_id": message.id},
        )

    async def _process_attachment(
        self,
        destination: abc.Messageable,
        message: discord.Message,
        attachment: discord.Attachment,
        fetcher: ImageFetcher,
    ) -> None:
        if not self._is_supported_attachment(attachment):
            return
        url_candidates = [attachment.url]
        if attachment.proxy_url:
            url_candidates.append(attachment.proxy_url)
        await self._record_image(
            destination,
            message,
            source="attachment",
            attachment_id=str(attachment.id),
            filename=attachment.filename,
            content_type=attachment.content_type,
            file_size=attachment.size,
            url_candidates=url_candidates,
            fetcher=fetcher,
        )

    async def _process_embed(
        self,
        destination: abc.Messageable,
        message: discord.Message,
        embed: discord.Embed,
        fetcher: ImageFetcher,
    ) -> None:
        for proxy in (embed.image, embed.thumbnail):
            if proxy is None or not proxy.url or proxy.url.startswith("attachment://"):
                continue
            filename = Path(proxy.url).name or None
            await self._record_image(
                destination,
                message,
                source="embed",
                attachment_id="",
                filename=filename,
                content_type=getattr(proxy, "content_type", None),
                file_size=None,
                url_candidates=[proxy.url, getattr(proxy, "proxy_url", None)],
                fetcher=fetcher,
            )

    async def _record_image(
        self,
        destination: abc.Messageable,
        message: discord.Message,
        *,
        source: str,
        attachment_id: str,
        filename: Optional[str],
        content_type: Optional[str],
        file_size: Optional[int],
        url_candidates: Iterable[Optional[str]],
        fetcher: ImageFetcher,
    ) -> None:
        guild = message.guild
        if guild is None:
            return
        urls = [url for url in url_candidates if url]
        if not urls:
            return
        phash_hex = None
        used_url = None
        final_content_type = content_type
        final_size = file_size
        note = None
        for url in urls:
            cached = self._cache.get(url)
            if cached:
                phash_hex = cached
                used_url = url
                break
            data, ctype, size, fetch_note = await fetcher.fetch(url)
            final_content_type = ctype or final_content_type
            final_size = size or final_size
            if data is None:
                note = fetch_note
                continue
            try:
                phash_hex = compute_phash(data)
            except Exception as exc:  # noqa: BLE001
                note = f"phash_error:{exc.__class__.__name__}"
                continue
            self._cache.set(url, phash_hex)
            used_url = url
            break
        if phash_hex is None and note is None:
            note = "unprocessed"
        distance = self._phash_index.nearest_distance(phash_hex) if phash_hex else None
        record = {
            "guild_id": str(guild.id),
            "channel_id": str(getattr(destination, "id", "")),
            "message_id": str(message.id),
            "message_link": self._build_message_link(guild.id, destination, message.id),
            "created_at": message.created_at.astimezone(timezone.utc).isoformat(),
            "author_id": str(message.author.id) if message.author else "",
            "is_nsfw_channel": str(self._is_nsfw(destination)).lower(),
            "source": source,
            "attachment_id": attachment_id,
            "filename": filename or "",
            "content_type": final_content_type or "",
            "file_size": final_size if final_size is not None else "",
            "url": used_url or urls[0],
            "phash_hex": phash_hex or "",
            "phash_distance_ref": distance if distance is not None else "",
            "note": note or "",
        }
        if phash_hex:
            self._phash_index.register(phash_hex)
        self._sink.write(record)

    @staticmethod
    def _build_message_link(guild_id: int, destination: abc.Messageable, message_id: int) -> str:
        channel_id = getattr(destination, "id", "")
        return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

    @staticmethod
    def _is_nsfw(destination: abc.Messageable) -> bool:
        if isinstance(destination, discord.Thread):
            parent = destination.parent
            return bool(parent.is_nsfw()) if parent else False
        if hasattr(destination, "is_nsfw"):
            return bool(destination.is_nsfw())
        return False

    @staticmethod
    def _is_supported_attachment(attachment: discord.Attachment) -> bool:
        if attachment.content_type and attachment.content_type.lower().startswith("image/"):
            return True
        mime_guess, _ = mimetypes.guess_type(attachment.filename or "")
        return bool(mime_guess and mime_guess.lower().startswith("image/"))


def parse_args() -> ScanOptions:
    parser = argparse.ArgumentParser(description="P0 image history scanner")
    parser.add_argument("--guild", type=int, help="Guild ID override")
    parser.add_argument(
        "--since",
        type=str,
        help="ISO8601 datetime, 'auto' (default), or 'none' for full history",
    )
    parser.add_argument("--channels", type=str, help="Comma separated channel IDs or names")
    parser.add_argument("--include-archived-threads", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qps", type=float, default=5.0)
    parser.add_argument("--profile", type=str, help="Profile name to operate on")
    parser.add_argument(
        "--date",
        type=str,
        help="Partition date (YYYY-MM-DD, today, yesterday). Default: today in profile timezone",
    )
    parser.add_argument("--out", type=Path, help="Override output CSV path")
    parser.add_argument("--state", type=Path, help="Override cursor state path")
    parser.add_argument("--cache", type=Path, help="Override pHash cache CSV path")
    parser.add_argument(
        "--attachments-index",
        type=Path,
        help="Override attachments index JSON path",
    )
    args = parser.parse_args()

    settings = get_settings()
    context = settings.build_profile_context(profile=args.profile, date=args.date)
    paths = PartitionPaths(context)
    guild_id = args.guild or settings.guild_id
    if guild_id is None:
        raise SystemExit("Guild ID must be provided via --guild or GUILD_ID")

    since_dt = None
    since_arg = args.since.strip() if args.since else None
    if since_arg:
        lowered = since_arg.lower()
        if lowered in {"none", "all"}:
            since_dt = None
        elif lowered == "auto":
            since_dt = datetime.combine(context.date, time_cls(0, 0), tzinfo=context.tzinfo)
        else:
            since_dt = datetime.fromisoformat(since_arg)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=context.tzinfo)
            else:
                since_dt = since_dt.astimezone(context.tzinfo)
    else:
        since_dt = datetime.combine(context.date, time_cls(0, 0), tzinfo=context.tzinfo)

    filters: set[str] = set()
    if args.channels:
        parts = [token.strip() for token in args.channels.split(",")]
        filters = {token for token in parts if token}

    return ScanOptions(
        guild_id=int(guild_id),
        since=since_dt,
        channel_filters=filters,
        include_archived_threads=args.include_archived_threads,
        resume=args.resume,
        qps=max(args.qps, 1.0),
        out_path=args.out or paths.stage_file("p0"),
        state_path=args.state or paths.p0_state_path(),
        cache_path=args.cache or paths.p0_cache_path(),
        attachments_index_path=args.attachments_index or paths.attachments_index(),
        profile=context,
    )


async def async_main() -> None:
    options = parse_args()
    settings = get_settings()
    runner = ScanRunner(options)
    await runner.run(settings.discord_bot_token)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
