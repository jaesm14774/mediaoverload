from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.assets.minimax_h3 import PROFILES, download_profile, inspect_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a MiniMax H3 ComfyUI profile")
    parser.add_argument(
        "--comfy-root",
        default="D:/ComfyUI_windows_portable",
        help="Portable ComfyUI root, not the nested ComfyUI directory",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced-lowvram")
    parser.add_argument("--status", action="store_true", help="Only inspect files")
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan without writing files")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    if args.status or args.dry_run:
        result = inspect_profile(args.profile, comfy_root)
        if args.dry_run and not args.status:
            result = download_profile(args.profile, comfy_root, dry_run=True)
    else:
        result = download_profile(args.profile, comfy_root)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"MiniMax H3 profile: {result['profile']}")
        print(f"ComfyUI root: {result['comfy_root']}")
        print(f"Total profile size: {result.get('total_size_human', 'n/a')}")
        for asset in result.get("assets", []):
            print(f"[{asset['status']}] {asset['name']} -> {asset['path']}")
        print(f"Ready: {result.get('ready', False)}")
    return 0 if result.get("ready", False) or args.dry_run or args.status else 1


if __name__ == "__main__":
    raise SystemExit(main())
