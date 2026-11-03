from __future__ import annotations

import hashlib
import re
from pathlib import Path


def serial_series_artifact_root(output_root: Path, series_id: str) -> Path:
    raw_series_id = str(series_id or "").strip()
    slug = re.sub(
        r"[^A-Za-z0-9._-]+", "-", raw_series_id
    ).strip(".-_") or "series"
    slug = slug[:48]
    identity = hashlib.sha256(raw_series_id.encode("utf-8")).hexdigest()[:10]
    return Path(output_root) / "serial_series" / f"{slug}-{identity}"
