#!/usr/bin/env python3
"""Serve one Flow handoff request with gflow-cli instead of driving Chrome by hand.

  python scripts/flow_gflow.py next <handoff_dir> [--wait SECONDS]   # same as flow_handoff.py next
  python scripts/flow_gflow.py run  <handoff_dir> <seq> [--project ID] [--dry-run]

Hard rules carried over from the mediaoverload-flow skill (never relaxed here):
  * image model nano2 (Nano Banana 2.1), never nano-pro; no prompt tools
  * video model omni-1.2 at 360p, no upscale, no Veo, count 1
  * --max-credits is the request's own credit figure; a quote above it is refused before submit
  * Omni 1.2 + end frame is capped at 10 s by Flow (above that it drops the end frame), so the
    clip is clamped to 10 s when an end frame is present; the provider pads/fits the delivery itself
  * no automatic retry: a failure leaves the request pending so the operator decides (credits)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

END_FRAME_MAX_SECONDS = 10
GUARD_SUFFIX = (
    " Only the character(s) present in the start frame appear; no humans, no new characters; keep their "
    "appearance identical to the start frame; normal limb count (two arms, two legs); ambient sound only, "
    "no background music."
)


def credits_for(seconds: int) -> int:
    return seconds if seconds <= 10 else 10 + 2 * (seconds - 10)


def gflow_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"GFLOW_CLI_TELEMETRY": "0", "DO_NOT_TRACK": "1", "PYTHONIOENCODING": "utf-8"})
    return env


def _run(argv: list[str], dry: bool) -> int:
    print("$ " + " ".join(f'"{a}"' if " " in a else a for a in argv), flush=True)
    if dry:
        return 0
    return subprocess.run(argv, env=gflow_env()).returncode


def image_aspect(raw: str) -> str:
    head = str(raw).split(" ", 1)[0]
    return head if head in {"9:16", "16:9", "1:1", "4:3", "3:4"} else "16:9"


def video_aspect(raw: str) -> str:
    head = str(raw).split(" ", 1)[0]
    return head if head in {"9:16", "16:9"} else "16:9"  # 1:1 delivery: ask 16:9, pipeline letterboxes


def serve_image(req: dict, out: Path, seq: int, project: str, dry: bool) -> int:
    for index in range(1, max(1, int(req.get("count") or 1)) + 1):
        target = out / f"{seq:04d}_{index:02d}.png"
        argv = ["gflow", "image", "t2i", str(req["prompt"]).replace("\n", " "), "--model", "nano2",
                "--aspect", image_aspect(req.get("aspect", "16:9")), "--count", "1", "--out-file", str(target)]
        if project:
            argv += ["--project", project]
        if _run(argv, dry):
            return 1
        # gflow may rewrite the suffix (png -> jpg); the provider accepts png/jpg/jpeg/webp.
        if not dry and not any(out.glob(f"{seq:04d}_{index:02d}.*")):
            print(f"missing output {target}", file=sys.stderr)
            return 1
    return 0


def serve_video(req: dict, out: Path, seq: int, project: str, dry: bool) -> int:
    start, end = str(req.get("start_frame") or ""), str(req.get("end_frame") or "")
    if not start:
        print("video request has no start_frame", file=sys.stderr)
        return 1
    seconds = int(req["seconds"])
    if end and seconds > END_FRAME_MAX_SECONDS:
        print(f"clamping {seconds}s -> {END_FRAME_MAX_SECONDS}s (Omni 1.2 + end frame limit)", flush=True)
        seconds = END_FRAME_MAX_SECONDS
    target = out / f"{seq:04d}_01.mp4"
    argv = ["gflow", "video", "i2v", "--initial-frame", start]
    if end:
        argv += ["--end-frame", end]
    argv += [str(req["prompt"]).replace("\n", " ") + GUARD_SUFFIX, "--model", "omni-1.2", "--duration", str(seconds),
             "--resolution", "360p", "--aspect", video_aspect(req.get("aspect", "16:9")), "--count", "1",
             "--max-credits", str(credits_for(seconds)), "--out-file", str(target)]
    if project:
        argv += ["--project", project]
    if _run(argv, dry):
        return 1
    if not dry and not target.is_file():
        print(f"missing output {target}", file=sys.stderr)
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    request_path = directory / f"{args.seq:04d}.request.json"
    if not request_path.is_file():
        print(f"no such request: {request_path}", file=sys.stderr)
        return 1
    if (directory / f"{args.seq:04d}.done").exists():
        print("already done", file=sys.stderr)
        return 1
    req = json.loads(request_path.read_text(encoding="utf-8"))
    out = directory / f"{args.seq:04d}.out"
    out.mkdir(parents=True, exist_ok=True)
    handler = {"image": serve_image, "video": serve_video}.get(req.get("kind"))
    if handler is None:
        print(f"unsupported request kind: {req.get('kind')}", file=sys.stderr)
        return 1
    code = handler(req, out, args.seq, args.project or "", args.dry_run)
    if code == 0 and not args.dry_run:
        (directory / f"{args.seq:04d}.done").touch()
        print(f"DONE seq={args.seq:04d}")
    elif code:
        print(f"NOT DONE seq={args.seq:04d}: left pending for the operator (no auto-retry)", file=sys.stderr)
    return code


def cmd_next(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    deadline = time.monotonic() + args.wait
    while True:
        todo = sorted(int(p.name[:4]) for p in directory.glob("????.request.json")
                      if p.name[:4].isdigit() and not (directory / f"{p.name[:4]}.done").exists())
        if todo:
            print(f"PENDING seq={todo[0]:04d} more_pending={len(todo) - 1}")
            print(f"read: {(directory / f'{todo[0]:04d}.request.md').resolve()}")
            return 0
        if time.monotonic() >= deadline:
            print("NOTHING_PENDING")
            return 0
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    nxt = sub.add_parser("next")
    nxt.add_argument("directory")
    nxt.add_argument("--wait", type=float, default=0.0)
    nxt.set_defaults(handler=cmd_next)
    run = sub.add_parser("run")
    run.add_argument("directory")
    run.add_argument("seq", type=int)
    run.add_argument("--project", default="")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(handler=cmd_run)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
