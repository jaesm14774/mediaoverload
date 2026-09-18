"""Run a reproducible ComfyUI H3/FastH3 speed and output comparison."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.assets.fasth3 import FASTH3_ASSETS, inspect_fast_h3_assets  # noqa: E402
from agentic.tools.comfy_backend import AgenticComfyCommunicator  # noqa: E402


PROMPT = (
    "A polished 2D anime short about Kirby discovering a tiny glowing seed in a windy meadow. "
    "[0s-2s] Kirby notices the seed rolling toward a cliff as grass bends in a strong gust. "
    "[2s-4s] Kirby runs, reaches out, and catches it just before it falls. "
    "[4s-5s] Kirby lifts the seed; warm light spreads across the meadow as the wind settles. "
    "Use a clear beginning, middle, and payoff, readable subject silhouette, deliberate camera movement, "
    "continuous motion, and native stereo audio with wind, a soft impact, and a warm uplifting musical resolve."
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    workflow_path: Path
    steps: int
    runtime: str
    expected_assets: tuple[str, ...]


class GpuSampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fasth3-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, object]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = _nvidia_smi()
            if sample:
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)


def _nvidia_smi() -> dict[str, object] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode or not completed.stdout.strip():
        return None
    row = completed.stdout.strip().splitlines()[0].split(",")
    if len(row) < 7:
        return None
    def number(value: str) -> float | int | None:
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return int(parsed) if parsed.is_integer() else parsed
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "timestamp": row[0].strip(),
        "name": row[1].strip(),
        "memory_used_mib": number(row[2]),
        "memory_free_mib": number(row[3]),
        "utilization_gpu_percent": number(row[4]),
        "power_draw_w": number(row[5]),
        "temperature_c": number(row[6]),
    }


def _system_stats(host: str, port: int) -> dict[str, object] | None:
    try:
        with request.urlopen(f"http://{host}:{port}/system_stats", timeout=15) as response:
            return json.loads(response.read())
    except Exception:
        return None


def _object_info(host: str, port: int) -> dict[str, Any] | None:
    try:
        with request.urlopen(f"http://{host}:{port}/object_info", timeout=30) as response:
            payload = json.loads(response.read())
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _load_workflow(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Workflow is empty: {path}")
    return payload


def _prepare_workflow(
    workflow: dict[str, Any], *, seed: int, run_prefix: str, steps: int, prompt: str = PROMPT
) -> dict[str, Any]:
    runtime = json.loads(json.dumps(workflow))
    prompt_nodes = 0
    for node_id, node in runtime.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.setdefault("inputs", {})
        if class_type == "MiniMaxH3ImageToVideo":
            inputs.update({"prompt": prompt, "width": 608, "height": 352, "length": 124})
            prompt_nodes += 1
        elif class_type == "BasicScheduler":
            inputs["steps"] = steps
        elif class_type == "RandomNoise":
            inputs["noise_seed"] = seed
        elif class_type == "CreateVideo":
            inputs["fps"] = 24.0
        elif class_type == "SaveVideo":
            inputs["filename_prefix"] = run_prefix
    if prompt_nodes != 1:
        raise ValueError(f"Expected exactly one MiniMaxH3ImageToVideo node, found {prompt_nodes}")
    return runtime


def _safe_case_assets(case: BenchmarkCase, model_root: Path, comfy_root: Path) -> list[dict[str, object]]:
    expected = set(case.expected_assets)
    if case.name == "baseline_gguf":
        candidates = {
            "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf": comfy_root / "ComfyUI" / "models" / "unet" / "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf",
            "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf": comfy_root / "ComfyUI" / "models" / "clip" / "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
            "minimax_h3_video_vae_fp16.safetensors": comfy_root / "ComfyUI" / "models" / "vae" / "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors": comfy_root / "ComfyUI" / "models" / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
        }
    else:
        candidates = {
            asset.name: asset.target_path(model_root)
            for asset in FASTH3_ASSETS
        }
    result: list[dict[str, object]] = []
    for name in sorted(expected):
        path = candidates.get(name)
        result.append({"name": name, "path": str(path) if path else None, "ready": bool(path and path.is_file())})
    return result


def _ffprobe(path: Path) -> dict[str, object] | None:
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def _contact_sheet(video_path: Path, destination: Path) -> str | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vf", "select='not(mod(n\\,15))',scale=320:-1,tile=4x2", "-frames:v", "1", str(destination),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return str(destination) if completed.returncode == 0 and destination.is_file() else None


def _render_one(
    communicator: AgenticComfyCommunicator,
    workflow: dict[str, Any],
    output_dir: Path,
    run_prefix: str,
    *,
    free_before: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if free_before:
        communicator.free_memory()
    sampler = GpuSampler()
    before_stats = _system_stats(communicator.host, communicator.port)
    sampler.start()
    started = time.perf_counter()
    prompt_id: str | None = None
    try:
        prompt_id = str(communicator.queue_prompt(workflow)["prompt_id"])
        communicator.wait_for_completion(prompt_id)
        success, saved_files = communicator.save_results(prompt_id, str(output_dir), run_prefix)
        if not success:
            raise RuntimeError(saved_files[0] if saved_files else "ComfyUI returned no files")
        wall_seconds = time.perf_counter() - started
        videos = [Path(item) for item in saved_files if Path(item).suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}]
        video = videos[0] if videos else None
        probe = _ffprobe(video) if video else None
        sheet = _contact_sheet(video, output_dir / "contact_sheet.jpg") if video else None
        return {
            "status": "passed",
            "prompt_id": prompt_id,
            "wall_seconds": round(wall_seconds, 3),
            "saved_files": saved_files,
            "video_path": str(video) if video else None,
            "ffprobe": probe,
            "contact_sheet": sheet,
            "system_stats_before": before_stats,
            "gpu_samples": sampler.stop(),
        }
    except Exception as exc:
        if prompt_id:
            communicator.cancel_prompt(prompt_id)
        return {
            "status": "failed",
            "prompt_id": prompt_id,
            "wall_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "system_stats_before": before_stats,
            "gpu_samples": sampler.stop(),
        }


def _max_metric(samples: list[dict[str, object]], key: str) -> object | None:
    values = [item[key] for item in samples if isinstance(item.get(key), (int, float))]
    return max(values) if values else None


def _case_summary(case_result: dict[str, object]) -> dict[str, object]:
    runs = case_result.get("runs")
    if not isinstance(runs, list):
        return {}
    passed = [run for run in runs if isinstance(run, dict) and run.get("status") == "passed"]
    times = [float(run["wall_seconds"]) for run in passed if isinstance(run.get("wall_seconds"), (int, float))]
    gpu_samples = [sample for run in passed for sample in run.get("gpu_samples", []) if isinstance(sample, dict)]
    return {
        "passed_runs": len(passed),
        "median_wall_seconds": round(statistics.median(times), 3) if times else None,
        "min_wall_seconds": round(min(times), 3) if times else None,
        "max_wall_seconds": round(max(times), 3) if times else None,
        "peak_gpu_memory_used_mib": _max_metric(gpu_samples, "memory_used_mib"),
        "peak_gpu_utilization_percent": _max_metric(gpu_samples, "utilization_gpu_percent"),
    }


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# FastH3 benchmark",
        "",
        f"- Prompt: `{report['prompt']}`",
        f"- Canvas: `{report['canvas']['width']}x{report['canvas']['height']}`, `{report['canvas']['length']} frames @ {report['canvas']['fps']} fps`",
        f"- Seed: `{report['seed']}`",
        "",
        "| Case | Runtime | Preflight | Passed | Median wall | Peak VRAM |",
        "|---|---|---|---:|---:|---:|",
    ]
    for case in report.get("cases", []):
        summary = case.get("summary", {})
        lines.append(
            f"| {case['name']} | {case['runtime']} | {case['preflight']['status']} | "
            f"{summary.get('passed_runs', 0)} | {summary.get('median_wall_seconds', 'n/a')}s | "
            f"{summary.get('peak_gpu_memory_used_mib', 'n/a')} MiB |"
        )
        runs = case.get("runs", [])
        passed_runs = [run for run in runs if isinstance(run, dict) and run.get("status") == "passed"]
        if passed_runs:
            run_details = ", ".join(
                "run {}: {}s".format(run.get("run_index"), run.get("wall_seconds"))
                for run in passed_runs
            )
            lines.append(f"  - Measured runs: {run_details}")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Wall time includes ComfyUI queue execution, model loading/offload, VAE decode, audio decode, and saving.",
        "The contact sheets are evidence for human visual review; speed and technical metadata do not prove creative quality.",
        "FastH3 V2 is a T2AV path and is not a replacement for the existing first/last-frame or Ref2VA routes.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8189)
    parser.add_argument("--comfy-root", type=Path, default=Path(r"D:\ComfyUI_windows_portable"))
    parser.add_argument("--model-root", type=Path, default=Path(r"E:\comfyui\_extra\models"))
    parser.add_argument("--output-root", type=Path, default=Path(r"E:\comfyui\_extra\benchmarks\fasth3"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--prompt-file", type=Path, help="JSON prompt matrix/object used to override the workflow prompt.")
    parser.add_argument("--prompt-name", help="Named prompt inside --prompt-file when the file contains a prompt matrix.")
    parser.add_argument("--case", action="append", choices=("baseline_gguf", "native_int8", "fasth3_v2"))
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--skip-asset-hash",
        action="store_true",
        help="Reuse the downloader's prior SHA-256 verification and only inspect asset sizes during this benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    model_root = args.model_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    prompt = PROMPT
    prompt_name = "kirby_seed_meadow"
    if args.prompt_file:
        prompt_payload = json.loads(args.prompt_file.expanduser().resolve().read_text(encoding="utf-8"))
        if isinstance(prompt_payload, dict) and isinstance(prompt_payload.get("prompt"), str):
            prompt = prompt_payload["prompt"]
            prompt_name = str(prompt_payload.get("name") or args.prompt_file.stem)
        elif isinstance(prompt_payload, dict) and args.prompt_name:
            selected_prompt = prompt_payload.get(args.prompt_name)
            if not isinstance(selected_prompt, dict) or not isinstance(selected_prompt.get("prompt"), str):
                raise SystemExit(f"Prompt name not found or invalid: {args.prompt_name}")
            prompt = selected_prompt["prompt"]
            prompt_name = args.prompt_name
        else:
            raise SystemExit("--prompt-file must contain an object with prompt, or a named matrix used with --prompt-name")
    cases = {
        "baseline_gguf": BenchmarkCase(
            "baseline_gguf", REPO_ROOT / "configs/workflow/minimax_h3_lowvram_t2v.json", 20, "ComfyUI current/staging", (
                "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf", "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", "minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"
            )),
        "native_int8": BenchmarkCase(
            "native_int8", REPO_ROOT / "configs/workflow/minimax_h3_native_int8_t2v.json", 20, "official MiniMax H3 INT8", tuple(asset.name for asset in FASTH3_ASSETS if asset.role != "fasth3_v2_diffusion")),
        "fasth3_v2": BenchmarkCase(
            "fasth3_v2", REPO_ROOT / "configs/workflow/minimax_h3_fasth3_v2_t2v.json", 8, "FastVideo FastH3 V2 INT8 + VSA", tuple(asset.name for asset in FASTH3_ASSETS if asset.role != "minimax_h3_native_diffusion")),
    }
    selected = [cases[name] for name in (args.case or list(cases))]
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "comfy_host": args.comfy_host,
        "comfy_port": args.comfy_port,
        "comfy_root": str(args.comfy_root.resolve()),
        "model_root": str(model_root),
        "output_root": str(output_root),
        "prompt_name": prompt_name,
        "prompt": prompt,
        "seed": args.seed,
        "canvas": {"width": 608, "height": 352, "length": 124, "fps": 24},
        "asset_verification": "download_sha256_reused" if args.skip_asset_hash else "runtime_sha256",
        "asset_status": inspect_fast_h3_assets(model_root, verify_hash=not args.skip_asset_hash),
        "cases": [],
    }

    object_info = _object_info(args.comfy_host, args.comfy_port)
    if object_info is None:
        raise SystemExit(f"ComfyUI is not reachable at {args.comfy_host}:{args.comfy_port}")

    for case in selected:
        case_result: dict[str, object] = {
            "name": case.name,
            "workflow": str(case.workflow_path),
            "runtime": case.runtime,
            "asset_status": _safe_case_assets(case, model_root, args.comfy_root.resolve()),
        }
        missing_assets = [item for item in case_result["asset_status"] if not item["ready"]]
        required_nodes = {str(node.get("class_type")) for node in _load_workflow(case.workflow_path).values() if isinstance(node, dict)}
        missing_nodes = sorted(node for node in required_nodes if node and node not in object_info)
        preflight = {
            "status": "passed" if not missing_assets and not missing_nodes else "blocked",
            "missing_assets": missing_assets,
            "missing_nodes": missing_nodes,
        }
        case_result["preflight"] = preflight
        case_result["runs"] = []
        if args.skip_preflight or preflight["status"] == "passed":
            communicator = AgenticComfyCommunicator(args.comfy_host, args.comfy_port, timeout=7200)
            communicator.connect_websocket()
            try:
                base_workflow = _load_workflow(case.workflow_path)
                for index in range(args.repeats):
                    run_dir = output_root / case.name / f"run_{index + 1:02d}"
                    workflow = _prepare_workflow(
                        base_workflow,
                        seed=args.seed,
                        run_prefix=f"fasth3_benchmark/{case.name}/run_{index + 1:02d}",
                        steps=case.steps,
                        prompt=prompt,
                    )
                    result = _render_one(communicator, workflow, run_dir, f"{case.name}_{index + 1:02d}", free_before=index == 0)
                    result["run_index"] = index + 1
                    case_result["runs"].append(result)
                    print(f"{case.name} run {index + 1}: {result['status']} {result.get('wall_seconds')}s", flush=True)
                    if result["status"] != "passed":
                        break
            finally:
                if communicator.ws and communicator.ws.connected:
                    communicator.ws.close()
        case_result["summary"] = _case_summary(case_result)
        report["cases"].append(case_result)
        (output_root / "fasth3_benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = output_root / "fasth3_benchmark.json"
    markdown_path = output_root / "fasth3_benchmark.md"
    _write_markdown(markdown_path, report)
    print(f"REPORT {report_path}")
    print(f"MARKDOWN {markdown_path}")
    return 0 if all(case.get("summary", {}).get("passed_runs", 0) > 0 for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
