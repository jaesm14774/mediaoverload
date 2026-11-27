#!/usr/bin/env python3
"""Operator helpers for the Google Flow render backend (AGENTIC_RENDER_BACKEND=flow_handoff).

  python scripts/flow_handoff.py next   <handoff_dir> [--wait SECONDS]
  python scripts/flow_handoff.py finish <handoff_dir> <seq> [--downloads DIR] [--count N]
  python scripts/flow_handoff.py fail   <handoff_dir> <seq> "reason"

``finish`` moves the N newest media files from the browser Downloads folder into ``NNNN.out``
and creates ``NNNN.done`` so the blocked pipeline continues.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".mp4"}


def pending(directory: Path) -> list[int]:
    seqs = sorted(int(p.name[:4]) for p in directory.glob("????.request.json") if p.name[:4].isdigit())
    return [s for s in seqs if not (directory / f"{s:04d}.done").exists()]


def cmd_next(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    deadline = time.monotonic() + args.wait
    while True:
        todo = pending(directory)
        if todo:
            seq = todo[0]
            print(f"PENDING seq={seq:04d} more_pending={len(todo) - 1}")
            print(f"read: {(directory / f'{seq:04d}.request.md').resolve()}")
            return 0
        if time.monotonic() >= deadline:
            print("NOTHING_PENDING")
            return 0
        time.sleep(1.0)


def cmd_finish(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    request = directory / f"{args.seq:04d}.request.json"
    if not request.exists():
        print(f"no such request: {request}", file=sys.stderr)
        return 1
    downloads = Path(args.downloads).expanduser()
    if not downloads.is_dir():
        print(f"downloads folder not found: {downloads}", file=sys.stderr)
        return 1
    kind = json.loads(request.read_text(encoding="utf-8")).get("kind")
    suffixes = {".mp4"} if kind == "video" else MEDIA - {".mp4"}
    since = request.stat().st_mtime
    # Only files of the right kind downloaded after the request was written count.
    files = sorted((p for p in downloads.iterdir() if p.suffix.lower() in suffixes and p.stat().st_mtime >= since),
                   key=lambda p: p.stat().st_mtime, reverse=True)[: args.count]
    if len(files) < args.count:
        print(f"only {len(files)} new media file(s) in {downloads} since request {args.seq:04d}", file=sys.stderr)
        return 1
    out = directory / f"{args.seq:04d}.out"
    out.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(reversed(files), start=1):
        shutil.move(str(source), str(out / f"{args.seq:04d}_{index:02d}{source.suffix.lower()}"))
        print(f"moved {source.name}")
    (directory / f"{args.seq:04d}.done").touch()
    print(f"DONE seq={args.seq:04d}")
    return 0


def cmd_fail(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    (directory / f"{args.seq:04d}.failed").write_text(args.reason, encoding="utf-8")
    (directory / f"{args.seq:04d}.done").touch()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    nxt = sub.add_parser("next")
    nxt.add_argument("directory")
    nxt.add_argument("--wait", type=float, default=0.0)
    nxt.set_defaults(handler=cmd_next)
    fin = sub.add_parser("finish")
    fin.add_argument("directory")
    fin.add_argument("seq", type=int)
    fin.add_argument("--downloads", default="~/Downloads")
    fin.add_argument("--count", type=int, default=1)
    fin.set_defaults(handler=cmd_finish)
    fail = sub.add_parser("fail")
    fail.add_argument("directory")
    fail.add_argument("seq", type=int)
    fail.add_argument("reason")
    fail.set_defaults(handler=cmd_fail)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
