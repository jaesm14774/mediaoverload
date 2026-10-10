"""Matched local ComfyUI / Wan2GP benchmark; preserves exact inputs and evidence."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]


def write(path, value):
    pathlib.Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def request(base, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + endpoint, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def telemetry(path, stop):
    with pathlib.Path(path).open("w", encoding="utf-8") as stream:
        stream.write("elapsed_s,memory_used_mib,gpu_percent,power_w,temperature_c\n")
        start = time.perf_counter()
        while not stop.is_set():
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            stream.write(f"{time.perf_counter()-start:.3f},{result.stdout.strip()}\n")
            stream.flush()
            stop.wait(2)


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    prompt = (
        "integrated_multimodal_description: [Shot 1] A five-second continuous hand-painted anime shot in a sunlit meadow. "
        "Kirby, a small pink spherical creature with stubby arms, red feet, oval eyes and rosy cheeks, notices a glowing yellow seed rolling away. "
        "Kirby leans forward, takes two quick waddling steps, reaches down and catches the seed with both hands. "
        "He lifts it to eye level and gives a delighted bounce. The seed stops moving when caught; grass bends under his feet. "
        "A locked medium-wide camera keeps Kirby's whole body, face and the seed in frame. Warm amber light, soft teal shadows, visible painterly texture.\n"
        "overall_soundscape: Gentle wind, two soft footsteps, a tiny chime as the seed is caught, and a joyful short chirp.\n"
        "non_diegetic_music: A quiet playful celesta phrase."
    )
    manifest = {"prompt": prompt, "width": 608, "height": 352, "length": 124, "fps": 24, "duration_seconds": 124/24,
                "steps": 16, "seed": 20261008, "sampler": "res_multistep", "shift_video": 12.0, "shift_audio": 3.0,
                "workflow_comparison": True, "spectrum_implementation_differs": True, "audio_vae_representation_differs": True,
                "diffusion": "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf", "text_encoder": "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"}
    write(directory / "manifest.json", manifest)
    graph = json.loads((ROOT / "scripts/fixtures/comfy_h3_t2va_20261008.json").read_text("utf-8"))
    graph["5"]["inputs"].update(prompt=prompt, width=manifest["width"], height=manifest["height"], length=manifest["length"])
    graph["7"]["inputs"]["steps"] = manifest["steps"]
    graph["9"]["inputs"]["noise_seed"] = manifest["seed"]
    graph["15"]["inputs"]["filename_prefix"] = "benchmarks/wan2gp_20261008/comfy_matched"
    write(directory / "comfy_graph.json", graph)
    settings = {"settings_version": 2.79, "model_type": "mediaoverload_h3_q4", "prompt": prompt,
                "resolution": f"{manifest['width']}x{manifest['height']}", "video_length": manifest["length"],
                "force_fps": str(manifest["fps"]), "num_inference_steps": manifest["steps"], "seed": manifest["seed"],
                "sample_solver": "res_multistep", "guidance_scale": 1.0, "flow_shift": 12.0,
                "guidance_phases": 1, "image_mode": 0, "prompt_enhancer": "", "activated_loras": [],
                "skip_steps_cache_type": "spectrum", "skip_steps_start_step_perc": 0,
                "sliding_window_size": 124, "sliding_window_overlap": 18, "repeat_generation": 1,
                "temporal_upsampling": "", "spatial_upsampling": "", "audio_prompt_type": "", "video_prompt_type": ""}
    write(directory / "wan_settings.json", settings)
    print(directory / "manifest.json", flush=True)


def comfy(directory, base):
    import websocket
    queue = request(base, "/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("ComfyUI queue is occupied; benchmark must run sequentially")
    write(directory / "comfy_system_stats.json", request(base, "/system_stats"))
    request(base, "/free", {"unload_models": True, "free_memory": True})
    client = str(uuid.uuid4())
    ws = websocket.create_connection(base.replace("http", "ws", 1) + "/ws?clientId=" + client, timeout=10)
    graph = json.loads((directory / "comfy_graph.json").read_text("utf-8"))
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(directory / "comfy_gpu.csv", stop), daemon=True)
    monitor.start()
    started = time.perf_counter()
    result = {"engine": "ComfyUI", "success": False}
    events = []
    try:
        receipt = request(base, "/prompt", {"prompt": graph, "client_id": client})
        write(directory / "comfy_receipt.json", receipt)
        prompt_id = receipt["prompt_id"]
        result["prompt_id"] = prompt_id
        print("Submitted", prompt_id, flush=True)
        deadline = started + 7200
        while time.perf_counter() < deadline:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                raw = None
            if isinstance(raw, str):
                event = json.loads(raw)
                data = event.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    event["elapsed_s"] = time.perf_counter() - started
                    events.append(event)
                    if event["type"] in ("executing", "progress", "execution_error", "execution_success"):
                        print(json.dumps(event, ensure_ascii=False), flush=True)
            history = request(base, "/history/" + prompt_id)
            if prompt_id in history:
                entry = history[prompt_id]
                write(directory / "comfy_history.json", entry)
                result["success"] = entry.get("status", {}).get("status_str") == "success"
                cached = [node for kind, info in entry["status"]["messages"] if kind == "execution_cached" for node in info["nodes"]]
                if "11" in cached:
                    raise RuntimeError("Sampler output was cached; this is not a valid render timing")
                result["cached_nodes"] = cached
                result["wall_seconds"] = time.perf_counter() - started
                outputs = entry.get("outputs", {}).get("15", {})
                files = outputs.get("images", []) + outputs.get("gifs", [])
                if not files:
                    files = outputs.get("videos", [])
                result["output_records"] = files
                for record in files:
                    path = pathlib.Path(r"D:\ComfyUI_windows_portable\ComfyUI\output") / record.get("subfolder", "") / record["filename"]
                    if path.suffix.lower() == ".mp4":
                        result["video_path"] = str(path)
                        result["probe"] = probe(path)
                if not result["success"] or "video_path" not in result:
                    raise RuntimeError("ComfyUI did not produce a successful video")
                print(json.dumps({k: result[k] for k in ("engine", "success", "wall_seconds", "video_path")}, ensure_ascii=False), flush=True)
                return
        raise TimeoutError("Benchmark exceeded two hours")
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result.setdefault("wall_seconds", time.perf_counter() - started)
        write(directory / "comfy_result.json", result)
        write(directory / "comfy_events.json", events)
        ws.close()
        stop.set()
        monitor.join(5)


def wan(directory, wan_root):
    for base in ("http://127.0.0.1:8188", "http://127.0.0.1:8189"):
        queue = request(base, "/queue")
        if queue["queue_running"] or queue["queue_pending"]:
            raise RuntimeError(f"GPU generation queue at {base} is occupied")
    python = wan_root / "venv/Scripts/python.exe"
    command = [str(python), "-u", str(pathlib.Path(__file__).resolve()), "wan-worker", "--directory", str(directory), "--wan-root", str(wan_root)]
    write(directory / "wan_command.json", command)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(directory / "wan_gpu.csv", stop), daemon=True)
    monitor.start()
    started = time.perf_counter()
    result = {"engine": "Wan2GP", "success": False}
    try:
        with (directory / "wan_process.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=wan_root, stdout=log, stderr=subprocess.STDOUT)
        if (directory / "wan_worker_result.json").exists():
            result.update(json.loads((directory / "wan_worker_result.json").read_text("utf-8")))
        result.update(exit_code=process.returncode, process_wall_seconds=time.perf_counter()-started)
        files = list((directory / "wan_outputs").rglob("*.mp4"))
        if process.returncode == 0 and files:
            result.update(success=True, video_path=str(files[-1]), probe=probe(files[-1]))
        print(json.dumps({k: result[k] for k in ("engine", "success", "wall_seconds", "startup_seconds", "video_path") if k in result}, ensure_ascii=False), flush=True)
    finally:
        write(directory / "wan_result.json", result)
        stop.set()
        monitor.join(5)
    if not result["success"]:
        raise RuntimeError("Wan2GP did not produce a successful video; see wan_process.log")


def wan_worker(directory, wan_root):
    sys.path.insert(0, str(wan_root))
    initialized = time.perf_counter()
    from shared.api import init
    session = init(root=wan_root, output_dir=directory / "wan_outputs", cli_args=["--profile", "4", "--attention", "sage2", "--verbose", "2"], console_isatty=False)
    startup_seconds = time.perf_counter() - initialized
    settings = json.loads((directory / "wan_settings.json").read_text("utf-8"))
    events = []
    started = time.perf_counter()
    job = session.submit_task(settings)
    for event in job.events.iter(timeout=0.2):
        if event.kind == "progress":
            progress = event.data
            item = {"kind": event.kind, "elapsed_s": time.perf_counter()-started}
            for field in ("phase", "progress", "current_step", "total_steps", "status"):
                value = getattr(progress, field, None)
                if isinstance(value, (str, int, float)) or value is None:
                    item[field] = value
            events.append(item)
            print(json.dumps(item), flush=True)
    generated = job.result()
    result = {"success": generated.success, "wall_seconds": time.perf_counter()-started, "startup_seconds": startup_seconds,
              "generated_files": [str(p) for p in generated.generated_files], "errors": [e.message for e in generated.errors]}
    write(directory / "wan_worker_result.json", result)
    write(directory / "wan_events.json", events)
    print(json.dumps(result), flush=True)
    if not generated.success:
        raise RuntimeError("Wan2GP generation failed: " + "; ".join(result["errors"]))


def summarize(directory):
    from fractions import Fraction
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    arms = [json.loads((directory / f"{engine}_result.json").read_text("utf-8")) for engine in ("comfy", "wan")]
    for arm in arms:
        if not arm["success"] or "probe" not in arm:
            raise RuntimeError("Both arms must generate real files before comparison")
        video = next(s for s in arm["probe"]["streams"] if s["codec_type"] == "video")
        actual = {"width": video["width"], "height": video["height"], "length": int(video["nb_read_frames"]), "fps": float(Fraction(video["avg_frame_rate"])), "duration_seconds": float(video["duration"])}
        arm["actual"] = actual
        for key in ("width", "height", "length", "fps"):
            if actual[key] != manifest[key]:
                raise RuntimeError(f"{arm['engine']} output {key}={actual[key]} differs from matched input {manifest[key]}")
        if abs(actual["duration_seconds"] - manifest["duration_seconds"]) > 1/manifest["fps"]:
            raise RuntimeError("Output duration differs by more than one frame")
    summary = {"manifest": manifest, "arms": arms, "speedup": arms[0]["wall_seconds"] / arms[1]["wall_seconds"], "sample_count_per_arm": 1}
    write(directory / "comparison.json", summary)
    print(json.dumps({"speedup": summary["speedup"], "sample_count_per_arm": 1, "arms": [{k: arm[k] for k in ("engine", "wall_seconds", "actual")} for arm in arms]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "comfy", "wan", "wan-worker", "summarize"))
    parser.add_argument("--directory", type=pathlib.Path, default=ROOT / "output/benchmarks/wan2gp_20261008")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--wan-root", type=pathlib.Path, default=pathlib.Path(r"D:\Wan2GP_benchmark"))
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.action == "prepare": prepare(directory)
    elif args.action == "comfy": comfy(directory, args.comfy_url)
    elif args.action == "wan": wan(directory, args.wan_root)
    elif args.action == "wan-worker": wan_worker(directory, args.wan_root)
    else: summarize(directory)
