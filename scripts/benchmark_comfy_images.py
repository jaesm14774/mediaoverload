"""Real ComfyUI image acceptance check; never queues behind an occupied server.

User Given an idle image provider When successive candidate seeds are rendered
Then saved images, latency and provider cache evidence are recorded.
No fake GPU, model substitution, or advertised timing is used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentic" / "src"))
from agentic.tools.comfy_backend import AgenticComfyCommunicator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True, help="Comfy API graph JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--timeout", type=int, default=420, help="Maximum seconds per owned prompt")
    parser.add_argument("--target-seconds", type=float, default=210)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    args = parser.parse_args()
    if args.timeout <= 0 or args.target_seconds <= 0:
        parser.error("Latency budgets must be positive")
    base = f"http://{args.host}:{args.port}"

    def get(path: str):
        with urlopen(base + path, timeout=10) as response:
            return json.load(response)

    def require_idle() -> None:
        queue = get("/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise RuntimeError("ComfyUI is occupied; benchmark did not queue or interrupt work")

    require_idle()
    original = json.loads(args.workflow.read_text(encoding="utf-8-sig"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    label = "latency_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"workflow": str(args.workflow.resolve()), "graph": original,
              "target_seconds": args.target_seconds, "system": get("/system_stats"), "renders": []}
    communicator = AgenticComfyCommunicator(args.host, args.port, timeout=args.timeout)
    try:
        for index in range(args.count):
            require_idle()
            graph = json.loads(json.dumps(original))
            for node in graph.values():
                inputs = node.get("inputs", {})
                if node.get("class_type") in {"KSampler", "KSamplerAdvanced", "RandomNoise"}:
                    for key in ("seed", "noise_seed"):
                        if key in inputs:
                            inputs[key] += index
                if node.get("class_type") == "SaveImage":
                    inputs["filename_prefix"] = f"benchmarks/{label}_{index + 1:02d}"
            started = dt.datetime.now().isoformat()
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu",
                 "--format=csv,noheader"], capture_output=True, text=True, check=False,
            ).stdout.strip()
            before = time.perf_counter()
            success, files = communicator.process_workflow(graph, [], str(output), auto_close=False)
            render = {"started_at": started, "seconds": time.perf_counter() - before,
                      "success": success, "files": files, "gpu_before": gpu}
            history = get("/history?max_items=50")
            owned = {pid: item for pid, item in history.items()
                     if item.get("prompt", [None] * 4)[3].get("client_id") == communicator.client_id}
            render["history"] = owned
            # Frontend logs are diagnostic evidence, never a production dependency.
            try:
                render["logs"] = [item for item in get("/internal/logs/raw")["entries"]
                                  if item["t"] >= started and "clip missing" not in item["m"]]
            except Exception as exc:
                render["diagnostic_error"] = str(exc)
            if success:
                render["images"] = []
                for file in files:
                    with Image.open(file) as image:
                        image.load()
                        render["images"].append({"file": file, "size": list(image.size)})
            render["target_met"] = success and render["seconds"] <= args.target_seconds
            report["renders"].append(render)
            (output / "benchmark.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: render[key] for key in
                              ("seconds", "success", "files", "gpu_before", "target_met")}), flush=True)
            if not success:
                break
    finally:
        if communicator.ws:
            communicator.ws.close()
    return 0 if len(report["renders"]) == args.count and all(
        item["target_met"] for item in report["renders"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
