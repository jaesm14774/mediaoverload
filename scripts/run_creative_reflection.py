from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect recent MediaOverload runs and write an evidence-first reflection report.")
    parser.add_argument("--count", type=int, default=20, help="Number of newest run manifests to inspect.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root; defaults to the parent of scripts/.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Report directory; defaults to logs/reflections.")
    args = parser.parse_args()

    repo_root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    sys.path.insert(0, str(repo_root / "agentic" / "src"))
    from agentic.runtime.creative_reflection import DEFAULT_OUTPUT_DIR, inspect_recent_runs, write_report

    report = inspect_recent_runs(repo_root, count=max(1, args.count))
    output_dir = (args.output_dir or repo_root / DEFAULT_OUTPUT_DIR).resolve()
    json_path, markdown_path, memory_path = write_report(report, output_dir)
    print(f"inspected={report['run_count_inspected']}")
    print(f"batch_counts={report['batch_counts']}")
    print(f"next_experiment={report['recommended_next_experiment']}")
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")
    print(f"memory={memory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

