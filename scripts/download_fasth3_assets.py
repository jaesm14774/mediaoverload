"""Download official non-GGUF MiniMax H3/FastH3 assets with resume and verification."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.assets.fasth3 import FASTH3_ASSETS, FastH3Asset, inspect_fast_h3_asset  # noqa: E402


DEFAULT_MODEL_ROOT = Path(r"E:\comfyui\_extra\models")


def _format_bytes(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def _curl_path() -> str:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl.exe is required for resumable Windows model downloads")
    return curl


def _download(asset: FastH3Asset, model_root: Path, *, timeout_seconds: int) -> dict[str, object]:
    target = asset.target_path(model_root)
    before = inspect_fast_h3_asset(asset, model_root)
    if before["status"] == "ready":
        return {**before, "action": "reuse"}
    target.parent.mkdir(parents=True, exist_ok=True)
    part = Path(f"{target}.part")
    command = [
        _curl_path(),
        "--location",
        "--fail",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--max-time",
        str(timeout_seconds),
        "--continue-at",
        "-",
        "--output",
        str(part),
        asset.source,
    ]
    print(
        f"DOWNLOAD {asset.name} ({_format_bytes(asset.expected_size)}) -> {target}",
        flush=True,
    )
    completed = subprocess.run(command, check=False, timeout=timeout_seconds + 60)
    if completed.returncode:
        raise RuntimeError(
            f"curl failed for {asset.name} with exit code {completed.returncode}; "
            f"resumable partial file: {part}"
        )
    # Inspect the .part file directly before exposing it as a ready model.
    if not part.exists():
        raise RuntimeError(f"Download produced no partial file for {asset.name}")
    if part.stat().st_size != asset.expected_size:
        raise RuntimeError(
            f"Incomplete download for {asset.name}: {part.stat().st_size} bytes, "
            f"expected {asset.expected_size}; kept {part}"
        )
    import hashlib

    digest = hashlib.sha256()
    with part.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != asset.sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {asset.name}: {digest.hexdigest()} != {asset.sha256}; "
            f"kept {part}"
        )
    part.replace(target)
    final = inspect_fast_h3_asset(asset, model_root)
    if final["status"] != "ready":
        raise RuntimeError(f"Asset verification failed for {asset.name}: {final}")
    return {**final, "action": "downloaded"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = args.model_root.expanduser().resolve()
    statuses = [inspect_fast_h3_asset(asset, model_root) for asset in FASTH3_ASSETS]
    if args.status:
        payload: dict[str, object] = {"model_root": str(model_root), "assets": statuses}
    elif args.dry_run:
        payload = {
            "model_root": str(model_root),
            "total_bytes": sum(asset.expected_size for asset in FASTH3_ASSETS),
            "assets": [{**status, "action": "download"} for status in statuses],
        }
    else:
        results: list[dict[str, object]] = []
        for asset in FASTH3_ASSETS:
            results.append(_download(asset, model_root, timeout_seconds=args.timeout_seconds))
        payload = {"model_root": str(model_root), "assets": results}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Model root: {model_root}")
        for item in payload["assets"]:
            print(f"[{item['status']}] {item['name']} -> {item['path']}")
        if "total_bytes" in payload:
            print(f"Total: {_format_bytes(int(payload['total_bytes']))}")
    return 0 if args.dry_run or all(item["status"] == "ready" for item in payload["assets"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
