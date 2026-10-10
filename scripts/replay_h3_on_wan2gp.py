"""Replay a retained L2VA benchmark through the official WanGP API."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agentic/src"))
import requests
from agentic.tools.wan2gp import Wan2GPClient, build_h3_settings, deliver_video, probe_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="5fbf5fbe6369")
    parser.add_argument("--output", type=Path, default=ROOT / "output/benchmarks/wan2gp_h3_15s_replay_20261009")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    source = ROOT / "logs/runs" / args.run
    old_node = json.loads((source / "nodes/native-h3-render_attempt_1.json").read_text("utf-8"))
    old = json.loads(Path(old_node["outputs"]["summary_path"]).read_text("utf-8"))
    payload = old["payload"]
    if old_node["outputs"].get("h3_mode") != "l2va":
        raise ValueError("This replay entry point requires a retained L2VA run")
    seed = int(old["render_attempts"][0]["seed"])
    native = probe_video(Path(old["saved_files"][0]))
    for key in ("width", "height", "length"):
        if native[key] != int(payload[key]):
            raise ValueError(f"Retained Comfy output does not match {key}")
    times = {}
    for line in (source / "events.jsonl").read_text("utf-8").splitlines():
        event = json.loads(line)
        if event.get("payload", {}).get("node_id") == "native-h3-render" and event["event"] in {"node.started", "node.completed"}:
            times[event["event"]] = datetime.fromisoformat(event["timestamp"])
    old_seconds = (times["node.completed"] - times["node.started"]).total_seconds()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Historical prompts may contain instructions that current production rejects.
    # This benchmark preserves the original bytes without weakening that contract.
    settings, evidence = build_h3_settings("l2va", {**payload, "prompt": "A character moves through the scene.", "seed": seed})
    settings["prompt"] = payload["prompt"]
    settings_path = output / "wan2gp_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    response = requests.get("http://127.0.0.1:8188/queue", timeout=5)
    response.raise_for_status()
    current = response.json()
    if current.get("queue_running") or current.get("queue_pending"):
        raise RuntimeError("ComfyUI is occupied; historical replay will not interrupt a GPU job")
    requests.post("http://127.0.0.1:8188/free", json={"unload_models": True, "free_memory": True}, timeout=30).raise_for_status()
    client = Wan2GPClient(output)
    report = {"source_run": args.run, "source_summary_path": old_node["outputs"]["summary_path"], "source_generation_seconds": old_seconds, "source_actual": native, "source_seed": seed, "source_model_profile": payload["model_profile"], "boundary": "historical official-API benchmark; production rejects this old instruction-containing prompt", "prompt_sha256": hashlib.sha256(payload["prompt"].encode("utf-8")).hexdigest(), "settings_path": str(settings_path), **evidence, "success": False}
    started = time.perf_counter()
    try:
        result = client.request({"settings": settings}, output / "wan2gp_events.jsonl")
        if settings["seed"] != seed or settings["prompt"] != payload["prompt"]:
            raise RuntimeError("Replayed input differs from the retained run")
        files = [Path(path) for path in result["generated_files"] if Path(path).suffix.lower() == ".mp4"]
        if len(files) != 1:
            raise RuntimeError(f"Expected one MP4, got {files}")
        target = output / "wan2gp_h3_l2va_replay.mp4"
        actual = deliver_video(files[0], target, evidence["delivery"])
        report.update(success=True, outputs={"saved_files": [str(target)], "raw_video_path": str(files[0])}, wall_seconds=time.perf_counter() - started, generation_seconds=result["wall_seconds"], worker_startup_seconds=client.startup_seconds, actual=actual)
        report["historical_speed_ratio"] = old_seconds / report["wall_seconds"]
        print(json.dumps({key: report[key] for key in ("success", "wall_seconds", "historical_speed_ratio", "outputs")}, ensure_ascii=False), flush=True)
    finally:
        client.close()
        (output / "replay_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
