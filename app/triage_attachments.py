from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


AttachmentMap = Dict[str, Dict[str, tuple[dict[str, object], ...]]]


@dataclass(slots=True)
class P0Index:
    """In-memory index of p0 scan attachments keyed by phash and message id."""

    _data: AttachmentMap
    total_rows: int

    @classmethod
    def from_csv(cls, path: Path) -> "P0Index":
        data: AttachmentMap = {}
        total_rows = 0
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total_rows += 1
                phash = (row.get("phash_hex") or "").strip().lower()
                message_id = (row.get("message_id") or "").strip()
                attachment_id = (row.get("attachment_id") or "").strip()
                if not phash or not attachment_id:
                    continue
                attachment = _build_attachment(row)
                bucket = data.setdefault(phash, {})
                key = message_id or ""
                existing = list(bucket.get(key, ()))
                if any(item["id"] == attachment["id"] for item in existing):
                    continue
                existing.append(attachment)
                bucket[key] = tuple(existing)
        return cls(_data=data, total_rows=total_rows)

    def get(self, phash: str) -> dict[str, tuple[dict[str, object], ...]]:
        return self._data.get((phash or "").strip().lower(), {})


def _build_attachment(row: dict[str, str]) -> dict[str, object]:
    file_size_raw = (row.get("file_size") or "").strip()
    try:
        file_size = int(file_size_raw) if file_size_raw else None
    except ValueError:
        file_size = None
    attachment: dict[str, object] = {
        "id": (row.get("attachment_id") or "").strip(),
        "filename": (row.get("filename") or "").strip() or None,
        "content_type": (row.get("content_type") or "").strip() or None,
        "file_size": file_size,
        "url": (row.get("url") or "").strip() or None,
        "source": "p0",
    }
    return attachment
