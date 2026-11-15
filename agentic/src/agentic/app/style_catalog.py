"""Style definitions shared by every model, and each model's enabled styles.

``configs/styles/catalog.json`` defines every style once. ``configs/styles/<model>.yaml``
filters that catalog down to the styles one image model is good at, so a new model
gets its own file and no character config lists styles.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from functools import cached_property
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from agentic.runtime.yaml_config import safe_yaml_load


STYLE_CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs" / "styles"
_MODEL_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


class SelectionConflict(ValueError):
    """The saved selection changed after the browser loaded it."""


class StyleLibrary:
    def __init__(self, model: str = "krea2", config_dir: Path = STYLE_CONFIG_DIR) -> None:
        if not isinstance(model, str) or not _MODEL_NAME.fullmatch(model):
            raise ValueError(f"style model must be a lowercase name such as 'krea2', got {model!r}")
        self.model = model
        self.config_dir = Path(config_dir)
        self.catalog_path = self.config_dir / "catalog.json"
        self.model_path = self.config_dir / f"{model}.yaml"
        if not self.model_path.is_file():
            raise ValueError(
                f"model '{model}' has no style file; expected {self.model_path.name} in {self.config_dir}"
            )
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

    @cached_property
    def _catalog_ids(self) -> frozenset[str]:
        return frozenset(style["id"] for style in self.catalog["styles"])

    def _validate_ids(self, ids: object) -> list[str]:
        if not isinstance(ids, list) or not ids:
            raise ValueError("至少保留一種風格，才能進行 random 抽選。")
        if any(not isinstance(style_id, str) for style_id in ids):
            raise ValueError("風格清單必須由風格 ID 組成。")
        if len(set(ids)) != len(ids):
            raise ValueError("風格清單含有重複項目。")
        unknown = set(ids) - self._catalog_ids
        if unknown:
            raise ValueError(f"未知的風格：{', '.join(sorted(unknown))}")
        return ids

    def _read(self) -> tuple[bytes, list[str]]:
        raw = self.model_path.read_bytes()
        data = safe_yaml_load(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{self.model_path.name} must be a mapping with an 'enabled' list")
        try:
            return raw, self._validate_ids(data.get("enabled"))
        except ValueError as exc:
            raise ValueError(f"{self.model_path.name} enabled: {exc}") from exc

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            raw, enabled = self._read()
            return {**self.catalog, "enabled_ids": enabled, "revision": hashlib.sha256(raw).hexdigest()}

    def profiles(self) -> dict[str, dict[str, Any]]:
        """Return the enabled styles in catalog order, so a seeded draw is stable."""
        with self._lock:
            ids = set(self._read()[1])
        return {
            style["id"]: {
                "label": style["name"], "prompt": style["descriptor"],
                **{field: style[field] for field in ("composition", "palette", "motion", "avoid")
                   if field in style},
            }
            for style in self.catalog["styles"] if style["id"] in ids
        }

    def save(self, enabled_ids: object, *, revision: str) -> dict[str, Any]:
        ids = set(self._validate_ids(enabled_ids))
        with self._lock:
            raw, _ = self._read()
            if hashlib.sha256(raw).hexdigest() != revision:
                raise SelectionConflict("設定已在其他頁面更新。請重新載入後再儲存。")
            ordered = [style["id"] for style in self.catalog["styles"] if style["id"] in ids]
            comments = [line for line in raw.decode("utf-8").splitlines(keepends=True)
                        if line.startswith("#")]
            text = "".join(comments) + yaml.safe_dump({"enabled": ordered}, allow_unicode=True, sort_keys=False)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", dir=self.model_path.parent,
                    prefix=f".{self.model}-styles-", suffix=".tmp", delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self.model_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            state = self.snapshot()
            return {"enabled_ids": state["enabled_ids"], "revision": state["revision"]}
