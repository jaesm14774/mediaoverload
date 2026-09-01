from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
for candidate in (REPO_ROOT, AGENTIC_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from agentic.runtime.reference_style_benchmark import ReferenceStyleBenchmark, ReferenceStyleBenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Krea 2 Turbo against a local visual reference collection")
    parser.add_argument("--collection-root", required=True, help="Folder containing reference images and optional MP4 files")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "output" / "krea2_style_benchmark"))
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "krea2_reference_style.yaml"))
    parser.add_argument("--limit", type=int, default=10, help="Number of image items to run; 0 means every discovered image")
    parser.add_argument("--seed-base", type=int, default=20260830)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--no-img2img-rescue", action="store_true")
    parser.add_argument("--seed-probe", action="store_true", help="Repeat a winning prompt twice with the same seed and compare output hashes")
    parser.add_argument("--execute", action="store_true", help="Submit real Krea2 jobs to ComfyUI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_config = config_data.get("model") if isinstance(config_data, dict) else {}
    benchmark_config = config_data.get("benchmark") if isinstance(config_data, dict) else {}
    model_config = model_config if isinstance(model_config, dict) else {}
    benchmark_config = benchmark_config if isinstance(benchmark_config, dict) else {}
    config = ReferenceStyleBenchmarkConfig(
        repo_root=REPO_ROOT,
        collection_root=Path(args.collection_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        logs_root=REPO_ROOT / "logs",
        config_path=config_path,
        max_attempts=max(1, min(5, int(args.max_attempts or benchmark_config.get("max_attempts", 5)))),
        score_threshold=int(benchmark_config.get("score_threshold", 80)),
        seed_base=int(args.seed_base if args.seed_base != 20260830 else benchmark_config.get("seed_base", 20260830)),
        width=int(model_config.get("width", 1024)),
        height=int(model_config.get("height", 576)),
        steps=int(model_config.get("steps", 8)),
        limit=int(args.limit),
        use_img2img_rescue=not args.no_img2img_rescue,
        seed_probe=bool(args.seed_probe),
        execute=bool(args.execute),
    )
    report = ReferenceStyleBenchmark(config).run()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.execute and not report.get("acceptance_passed", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
