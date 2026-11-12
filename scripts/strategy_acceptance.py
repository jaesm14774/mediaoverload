"""Print how often a human accepted each generation strategy.

Reads the run manifests under logs/runs. A run counts as accepted when it was
published, and as declined when the reviewer rejected it or asked for a
regeneration. Crashes, review timeouts, and runs that never asked to publish
are undecided and stay out of the rate. serial_video keeps its decisions in its
own database, so its runs show here as undecided.

    python scripts/strategy_acceptance.py [--since 2026-11-01] [--runs-root logs/runs]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TARGET_RATE = 0.8


def _outcome(manifest: dict) -> str:
    publish = ((manifest.get("stage_status") or {}).get("publish") or {}).get("status")
    if publish == "success":
        return "accepted"
    if manifest.get("status") == "regeneration_requested":
        return "declined"
    if "reviewer rejected" in str(manifest.get("failure_reason") or ""):
        return "declined"
    return "undecided"


def tally(runs_root: Path, since: str = "") -> list[dict]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"accepted": 0, "declined": 0, "undecided": 0}
    )
    for path in Path(runs_root).glob("*/run_manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        strategy = str(manifest.get("source_generation_type") or "").strip()
        if not strategy or str(manifest.get("updated_at") or "") < since:
            continue
        counts[strategy][_outcome(manifest)] += 1
    rows = []
    for strategy, count in counts.items():
        decided = count["accepted"] + count["declined"]
        rows.append(
            {
                "strategy": strategy,
                **count,
                "rate": count["accepted"] / decided if decided else None,
            }
        )
    return sorted(rows, key=lambda row: (row["rate"] is None, -(row["rate"] or 0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-root", default=str(Path(__file__).resolve().parents[1] / "logs" / "runs")
    )
    parser.add_argument("--since", default="", help="ISO date; older runs are skipped")
    args = parser.parse_args()
    print(f"{'strategy':32}{'accepted':>9}{'declined':>9}{'undecided':>10}{'rate':>7}")
    for row in tally(Path(args.runs_root), args.since):
        rate = "n/a" if row["rate"] is None else f"{row['rate']:.0%}"
        flag = "" if row["rate"] is None or row["rate"] >= TARGET_RATE else "  below 80%"
        print(
            f"{row['strategy']:32}{row['accepted']:>9}{row['declined']:>9}"
            f"{row['undecided']:>10}{rate:>7}{flag}"
        )


if __name__ == "__main__":
    main()
