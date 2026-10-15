"""Shared Krea style definitions and the user's persisted random selection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import cached_property
from pathlib import Path
from threading import RLock
from typing import Any

import yaml


STYLE_CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs" / "styles"


class SelectionConflict(ValueError):
    """The saved selection changed after the browser loaded it."""


class StyleLibrary:
    def __init__(
        self,
        catalog_path: Path = STYLE_CONFIG_DIR / "krea2_catalog.json",
        selection_path: Path = STYLE_CONFIG_DIR / "krea2_selection.yaml",
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.selection_path = Path(selection_path)
        self._lock = RLock()

    @cached_property
    def catalog(self) -> dict[str, Any]:
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        styles = data.get("styles")
        if not isinstance(styles, list) or not styles:
            raise ValueError("style catalog must contain styles")
        ids: set[str] = set()
        for style in styles:
            if not isinstance(style, dict):
                raise ValueError("style catalog entries must be mappings")
            for field in ("id", "name", "descriptor"):
                if not isinstance(style.get(field), str) or not style[field].strip():
                    raise ValueError(f"style catalog entry is missing {field}")
            if style["id"] in ids:
                raise ValueError(f"duplicate catalog style: {style['id']}")
            ids.add(style["id"])
        return data

    def _validate_ids(self, ids: object) -> list[str]:
        if not isinstance(ids, list) or not ids:
            raise ValueError("至少保留一種風格，才能進行 random 抽選。")
        if any(not isinstance(style_id, str) for style_id in ids):
            raise ValueError("風格清單必須由風格 ID 組成。")
        if len(set(ids)) != len(ids):
            raise ValueError("風格清單含有重複項目。")
        known = {style["id"] for style in self.catalog["styles"]}
        unknown = set(ids) - known
        if unknown:
            raise ValueError(f"未知的風格：{', '.join(sorted(unknown))}")
        return ids

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            raw = self.selection_path.read_bytes()
            selection = yaml.safe_load(raw)
            if not isinstance(selection, dict):
                raise ValueError("style selection must be a mapping")
            enabled = self._validate_ids(selection.get("enabled"))
            return {
                **self.catalog,
                "enabled_ids": enabled,
                "revision": hashlib.sha256(raw).hexdigest(),
            }

    def profiles(self) -> dict[str, dict[str, Any]]:
        state = self.snapshot()
        enabled = set(state["enabled_ids"])
        return {
            style["id"]: {
                "label": style["name"], "prompt": style["descriptor"],
                **{field: style[field] for field in ("composition", "palette", "motion", "avoid")
                   if field in style},
            }
            for style in state["styles"] if style["id"] in enabled
        }

    def save(self, enabled_ids: object, *, revision: str) -> dict[str, Any]:
        ids = set(self._validate_ids(enabled_ids))
        with self._lock:
            if self.snapshot()["revision"] != revision:
                raise SelectionConflict("設定已在其他頁面更新。請重新載入後再儲存。")
            ordered = [style["id"] for style in self.catalog["styles"] if style["id"] in ids]
            text = "# Selected styles for uniform random generation.\n"
            text += yaml.safe_dump({"enabled": ordered}, allow_unicode=True, sort_keys=False)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.selection_path.parent,
                    prefix=".krea2-selection-", suffix=".tmp", delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self.selection_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            state = self.snapshot()
            return {"enabled_ids": state["enabled_ids"], "revision": state["revision"]}
