"""Render real H3 strategy acceptance cases through MediaOverload's public tool."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agentic/src"))
from agentic.assets.registry import AssetRegistry
from agentic.runtime.registry import ToolRegistry
from agentic.tools.wan2gp import Wan2GPToolset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=["t2va", "i2va", "fl2va", "l2va", "ref2va", "ref2va_video"])
    parser.add_argument("--output", type=Path, default=ROOT / "output/benchmarks/wan2gp_h3_routes_20261009")
    parser.add_argument("--source-video", type=Path, default=ROOT / "output/benchmarks/wan2gp_20261008/wan_matched.mp4")
    parser.add_argument("--length", type=int, default=124)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--height", type=int, default=352)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output = args.output.resolve()
    fixtures = output / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    first, last = fixtures / "opening.png", fixtures / "ending.png"
    for target, seconds in ((first, "0"), (last, "5")):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", seconds, "-i", str(args.source_video), "-frames:v", "1", str(target)], check=True)
    prompt = json.loads((ROOT / "output/benchmarks/wan2gp_20261008/manifest.json").read_text("utf-8"))["prompt"]
    assets = AssetRegistry(ROOT / "agentic", asset_root=Path("D:/ComfyUI_windows_portable"))
    tools = ToolRegistry()
    backend = Wan2GPToolset(assets, output)
    backend.register_tools(tools)
    results = []
    try:
        for case in args.modes:
            mode = "ref2va" if case == "ref2va_video" else case
            payload = {"workflow_name": f"wan2gp_h3_{mode}", "h3_mode": mode, "run_dir": str(output / case), "prompt": prompt, "width": args.width, "height": args.height, "length": args.length, "steps": 16, "seed": 20261008, "model_profile": "q4"}
            if mode in {"i2va", "fl2va"}:
                payload["image_path"] = str(first)
            if mode in {"l2va", "fl2va"}:
                payload["last_image_path"] = str(last)
            if case == "ref2va":
                payload.update(reference_image_paths=[str(first), str(last)], prompt="Use <Picture 1> and <Picture 2> as references for the same Kirby character and watercolor meadow. " + prompt)
            if case == "ref2va_video":
                payload.update(reference_video_paths=[str(args.source_video.resolve())], prompt="Use <Video 1> as a reference for the character and seed-catching motion. " + prompt)
            print(f"START {case}", flush=True)
            result = tools.call("wan2gp.render_h3", payload)
            results.append({"case": case, **result})
            (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    finally:
        backend.client.close()


if __name__ == "__main__":
    main()
