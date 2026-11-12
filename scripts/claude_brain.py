#!/usr/bin/env python3
"""Operator helpers for A/B runs where Claude Code is the LLM.

  python scripts/claude_brain.py pick-news --out output/ab/<name>/news.json
  python scripts/claude_brain.py next  <brain_dir> [--wait SECONDS]
  python scripts/claude_brain.py status <brain_dir>

Answers are written by the operator straight to ``<brain_dir>/NNNN.response.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "agentic" / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from agentic.runtime.claude_brain import (  # noqa: E402
    FINISHED_FILE,
    pending_sequences,
    response_path,
)


def _pick_news(args: argparse.Namespace) -> int:
    import os

    os.chdir(REPO_ROOT)  # news credentials live in media_overload.env at the repo root
    from agentic.tools.context_services import NewsContextService

    selection = NewsContextService().select_news()
    if selection is None:
        print("No usable news was selected (MySQL not configured, or no eligible row).", file=sys.stderr)
        return 1
    payload = selection.to_dict()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")
    print(f"title: {payload['title']}")
    print(f"source: {payload['source']}  category: {payload['category']}  created_at: {payload['created_at']}")
    print(f"url: {payload['url']}")
    print(f"content_chars: {len(payload['content'])}")
    return 0


def _next(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    deadline = time.monotonic() + args.wait
    while True:
        pending = pending_sequences(directory)
        if pending:
            seq = pending[0]
            markdown = directory / f"{seq:04d}.request.md"
            request = json.loads((directory / f"{seq:04d}.request.json").read_text(encoding="utf-8"))
            print(f"PENDING seq={seq:04d} modality={request['modality']} "
                  f"attempt={request['attempt']}/{request['max_attempts']} more_pending={len(pending) - 1}")
            print(f"read:   {markdown.resolve()}")
            print(f"answer: write the raw answer text to {response_path(directory, seq).resolve()}")
            return 0
        if (directory / FINISHED_FILE).exists():
            print("RUN_FINISHED (no pending requests)")
            return 0
        if time.monotonic() >= deadline:
            print("NOTHING_PENDING (run still going; call next --wait again)")
            return 0
        time.sleep(1.0)


def _status(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    total = len(list(directory.glob("*.request.json")))
    pending = pending_sequences(directory)
    finished = (directory / FINISHED_FILE).exists()
    print(f"requests={total} pending={[f'{seq:04d}' for seq in pending]} finished={finished}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("pick-news", help="Select one news article the way a news-driven run does and save it")
    pick.add_argument("--out", required=True)
    pick.set_defaults(handler=_pick_news)
    nxt = sub.add_parser("next", help="Show the oldest unanswered LLM request")
    nxt.add_argument("directory")
    nxt.add_argument("--wait", type=float, default=0.0, help="Block up to this many seconds for a request")
    nxt.set_defaults(handler=_next)
    status = sub.add_parser("status")
    status.add_argument("directory")
    status.set_defaults(handler=_status)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
