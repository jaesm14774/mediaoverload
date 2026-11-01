"""Safe YAML parsing with bounded reuse for repeated config reads."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any

import yaml


@lru_cache(maxsize=8)
def _parse_yaml_content(content: str | bytes) -> Any:
    return yaml.safe_load(content)


def safe_yaml_load(content: str | bytes) -> Any:
    """Parse identical YAML content once and isolate each caller's mapping."""
    return deepcopy(_parse_yaml_content(content))
