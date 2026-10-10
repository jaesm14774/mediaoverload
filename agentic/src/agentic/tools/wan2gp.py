"""H3 input contracts and real rendering through the isolated WanGP API."""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import math
import os
import queue
import random
import shutil
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from agentic.assets.registry import AssetRegistry
from agentic.assets.wan2gp import h3_requirements, wan2gp_config
from agentic.h3_reference import build_reference_lineage, normalize_reference_manifest
from agentic.minimax_prompting import require_h3_prompt
from agentic.runtime.h3_modes import H3Mode, resolve_h3_mode, validate_h3_payload
from agentic.runtime.registry import ToolRegistry

LOG = logging.getLogger(__name__)
def _file_evidence(path: str, role: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"H3 {role} does not exist: {source}")
    try:
        if role in {"first_frame", "last_frame"} or role.startswith("<Picture"):
            with Image.open(source) as picture:
                picture.verify()
        else:
            probe_video(source)
    except (OSError, ValueError, KeyError, StopIteration, subprocess.CalledProcessError) as exc:
        raise ValueError(f"H3 {role} is not readable media: {source}") from exc
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(source), "role": role, "size_bytes": source.stat().st_size, "sha256": digest}


def build_h3_settings(mode: str | H3Mode, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = resolve_h3_mode(mode)
    prompt = require_h3_prompt(payload.get("prompt"))
    validate_h3_payload(mode, payload)
    width, height = int(payload.get("width") or 608), int(payload.get("height") or 352)
    length, steps = int(payload.get("length") or 124), int(payload.get("steps") or 16)
    fps = int(payload.get("frame_rate") or payload.get("fps") or 24)
    if not (256 <= width <= 1344 and 256 <= height <= 1344):
        raise ValueError("H3 canvas dimensions must be between 256 and 1344")
    if not (1 <= length <= 362 and 1 <= steps <= 32 and fps == 24):
        raise ValueError("H3 supports 1-362 delivered frames, 1-32 steps and 24 fps in this workflow")
    profile = str(payload.get("model_profile") or "q4")
    if profile not in {"q4", "q2"}:
        raise ValueError(f"Unsupported WanGP H3 model_profile: {profile}; choose q4 or q2")
    native_length = max(107, math.ceil((length - 5) / 17) * 17 + 5)
    architecture = "ref2va" if mode is H3Mode.REF2VA else "fl2va"
    settings: dict[str, Any] = {
        "model_type": f"mediaoverload_h3_{architecture}_{profile}",
        "prompt": prompt, "multi_prompts_gen_type": "FG", "resolution": f"{width // 32 * 32}x{height // 32 * 32}",
        "video_length": native_length, "force_fps": str(fps),
        "num_inference_steps": steps, "seed": int(payload.get("seed", random.randint(1, 999999999))),
        "sample_solver": "res_multistep", "guidance_scale": 1.0, "flow_shift": 12.0,
        "guidance_phases": 1, "image_mode": 0, "prompt_enhancer": "", "activated_loras": [],
        "skip_steps_cache_type": "spectrum", "skip_steps_start_step_perc": 0,
        "sliding_window_size": native_length, "sliding_window_overlap": 18,
        "repeat_generation": 1, "temporal_upsampling": "", "spatial_upsampling": "",
        "image_prompt_type": "", "video_prompt_type": "", "audio_prompt_type": "",
    }
    evidence: dict[str, Any] = {"h3_mode": mode.value, "conditioning_files": [], "delivery": {"width": width, "height": height, "length": length, "fps": fps}}
    for key, alternatives, flag, role in (
        ("image_start", ("image_path", "input_image_path"), "S", "first_frame"),
        ("image_end", ("last_image_path", "last_frame_path"), "E", "last_frame"),
    ):
        path = next((str(payload[name]) for name in alternatives if payload.get(name)), "")
        if path:
            info = _file_evidence(path, role)
            evidence["conditioning_files"].append(info)
            settings[key] = [info["path"]]
            settings["image_prompt_type"] += flag
    if mode is H3Mode.REF2VA:
        refs = normalize_reference_manifest(payload.get("reference_manifest"), image_paths=payload.get("reference_image_paths"), video_paths=payload.get("reference_video_paths"), require_files=True)
        evidence.update(reference_manifest=refs, reference_lineage=build_reference_lineage(refs))
        images, videos = [], []
        for ref in refs:
            info = _file_evidence(ref["path"], ref["prompt_label"])
            evidence["conditioning_files"].append(info)
            (images if ref["type"] == "image" else videos).append(info["path"])
        if images:
            settings.update(image_refs=images, video_prompt_type="I")
        if videos:
            settings["video_prompt_type"] += ("V-U", "V+-U", "V+*-U")[len(videos) - 1]
            for index, path in enumerate(videos):
                settings["video_guide" + (str(index + 1) if index else "")] = path
    return settings, evidence


class Wan2GPClient:
    """One JSON pipe into an isolated interpreter; WanGP owns generation."""
    def __init__(self, output_root: Path) -> None:
        self.config = wan2gp_config()
        self.output_root = output_root.resolve() / "wan2gp"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.process: subprocess.Popen | None = None
        self.messages: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.startup_seconds = 0.0
        self.log_path = self.output_root / "worker.log"
        self.log_handle = None
        atexit.register(self.close)

    def _start(self) -> None:
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError(f"WanGP worker exited; see {self.log_path}")
            return
        root = Path(self.config["root"])
        python = root / ("venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python")
        if not python.is_file() or not (root / "shared/api.py").is_file():
            raise RuntimeError(f"WanGP isolated installation is missing at {root}; see docs/wan2gp_h3.md")
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        command = [str(python), "-u", str(Path(__file__).with_name("wan2gp_worker.py")), "--root", str(root), "--output", str(self.output_root / "raw"), "--profile", str(self.config["profile"]), "--attention", str(self.config["attention"])]
        self.process = subprocess.Popen(command, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_handle, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

        def read() -> None:
            try:
                with (self.output_root / "worker_stdout.log").open("a", encoding="utf-8") as console:
                    for line in self.process.stdout:
                        if line.startswith("WAN2GP_EVENT "):
                            self.messages.put(json.loads(line[len("WAN2GP_EVENT "):]))
                        else:
                            console.write(line)
                            console.flush()
            except Exception as exc:
                self.messages.put({"kind": "fatal", "errors": [str(exc)]})
            finally:
                self.messages.put({"kind": "fatal", "errors": [f"WanGP worker closed; see {self.log_path}"]})
        threading.Thread(target=read, daemon=True).start()
        try:
            message = self.messages.get(timeout=300)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError(f"WanGP did not initialize within 300 seconds; see {self.log_path}") from exc
        if message.get("kind") != "ready":
            self.close()
            raise RuntimeError(str(message))
        self.startup_seconds = float(message["startup_seconds"])
        LOG.info("Powered by WanGP; initialized in %.2fs", self.startup_seconds)

    def request(self, payload: dict, events_path: Path | None = None) -> dict:
        with self.lock:
            self._start()
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + 7200
            handle = events_path.open("w", encoding="utf-8") if events_path else None
            try:
                phase = ""
                while True:
                    message = self.messages.get(timeout=max(0.1, deadline - time.monotonic()))
                    if handle:
                        handle.write(json.dumps(message, ensure_ascii=False) + "\n")
                        handle.flush()
                    if message.get("kind") in {"result", "fatal"}:
                        if not message.get("success"):
                            raise RuntimeError("WanGP generation failed: " + "; ".join(message.get("errors", [])))
                        return message
                    if message.get("phase") != phase:
                        phase = message.get("phase", "")
                        LOG.info("WanGP %s: %s", phase, message.get("status", ""))
            except (queue.Empty, KeyboardInterrupt):
                self.close()
                raise
            finally:
                if handle:
                    handle.close()

    def close(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.stdin.write('{"action":"close"}\n')
                process.stdin.flush()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                if os.name == "nt":
                    # Windows venv python may launch a child interpreter. Stop
                    # only the process tree this client created.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    process.terminate()
                process.wait(timeout=10)
        if self.log_handle:
            self.log_handle.close()


def probe_video(path: Path) -> dict[str, Any]:
    data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)], text=True, encoding="utf-8"))
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    return {"width": video["width"], "height": video["height"], "length": int(video["nb_read_frames"]), "fps": float(Fraction(video["avg_frame_rate"])), "duration_seconds": float(video.get("duration", data["format"]["duration"])), "audio": audio}


def deliver_video(source: Path, target: Path, delivery: dict[str, int]) -> dict[str, Any]:
    actual = probe_video(source)
    if actual["length"] < delivery["length"] or actual["audio"] is None:
        raise RuntimeError(f"WanGP returned insufficient frames or missing native audio: {actual}")
    if all(actual[key] == delivery[key] for key in ("width", "height", "length", "fps")):
        shutil.copy2(source, target)
    else:
        width, height, length, fps = (delivery[k] for k in ("width", "height", "length", "fps"))
        # Sample the complete generated timeline, retaining both anchors.
        frames = [round(i * (actual["length"] - 1) / (length - 1)) for i in range(length)] if length > 1 else [actual["length"] - 1]
        select = "+".join(f"eq(n,{index})" for index in frames)
        video_filter = f"select='{select}',setpts=N/({fps}*TB),scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        ratio = actual["length"] / length
        tempos = []
        while ratio > 2:
            tempos.append("atempo=2")
            ratio /= 2
        tempos.extend([f"atempo={ratio}", f"atrim=duration={length / fps}", "asetpts=PTS-STARTPTS"])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0", "-vf", video_filter, "-af", ",".join(tempos), "-r", str(fps), "-t", str(length / fps), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(target)], check=True, capture_output=True)
    result = probe_video(target)
    if any(result[key] != delivery[key] for key in ("width", "height", "length", "fps")) or result["audio"] is None:
        raise RuntimeError(f"WanGP delivery contract failed: {result}")
    return result


class Wan2GPToolset:
    def __init__(self, asset_registry: AssetRegistry, output_root: Path, comfy_host: str = "127.0.0.1", comfy_port: int = 8188) -> None:
        self.asset_registry = asset_registry
        self.client = Wan2GPClient(output_root)
        self.comfy_url = f"http://{comfy_host}:{comfy_port}"

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register("wan2gp.render_h3", self.execute, "Render H3 video and native audio locally. Powered by WanGP.")

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest = self.asset_registry.get_manifest(str(payload["workflow_name"]))
        if manifest.conditioning.get("provider") != "wan2gp":
            raise ValueError(f"Not a WanGP workflow: {manifest.name}")
        template = self.asset_registry.load_workflow_template(manifest)
        merged = {**manifest.recommended_defaults, **payload}
        mode = resolve_h3_mode(merged.get("h3_mode") or template["h3_mode"])
        allowed = template.get("allowed_modes", [template["h3_mode"]])
        if mode.value not in allowed:
            raise ValueError(f"{manifest.name} does not support H3 mode {mode.value}")
        settings, evidence = build_h3_settings(mode, merged)
        missing = []
        for asset in h3_requirements(mode.value, str(merged.get("model_profile") or "q4")):
            path = Path(asset["target_dir"]) / asset["name"]
            if not path.is_file() or (asset.get("expected_size") and path.stat().st_size != asset["expected_size"]):
                missing.append(str(path))
        if missing:
            raise RuntimeError("WanGP H3 assets missing or corrupt: " + "; ".join(missing))
        count = int(merged.get("video_count") or 1)
        if not 1 <= count <= 4:
            raise ValueError("video_count must be between 1 and 4")
        run_dir = Path(str(merged["run_dir"])).resolve()
        videos = run_dir / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        # Free idle Comfy models before WanGP needs the GPU. Never interrupt a queue.
        try:
            response = requests.get(self.comfy_url + "/queue", timeout=5)
        except requests.ConnectionError:
            response = None
        if response is not None:
            response.raise_for_status()
            current = response.json()
            if current.get("queue_running") or current.get("queue_pending"):
                raise RuntimeError("ComfyUI queue is occupied; WanGP will not interrupt another GPU job")
            requests.post(self.comfy_url + "/free", json={"unload_models": True, "free_memory": True}, timeout=30).raise_for_status()
        started = time.perf_counter()
        summary = {"provider": "wan2gp", "credit": "Powered by WanGP", "workflow_name": manifest.name, "workflow_path": str(self.asset_registry.materialize_workflow(manifest)), "payload": merged, **evidence, "render_attempts": [], "saved_files": []}
        summary_path = run_dir / "wan2gp_h3_summary.json"
        try:
            for index in range(count):
                effective = {**settings, "seed": settings["seed"] + index}
                settings_path = run_dir / f"wan2gp_settings_{index + 1:02d}.json"
                settings_path.write_text(json.dumps(effective, indent=2, ensure_ascii=False), encoding="utf-8")
                result = self.client.request({"settings": effective}, run_dir / f"wan2gp_events_{index + 1:02d}.jsonl")
                files = [Path(p) for p in result["generated_files"] if Path(p).suffix.lower() == ".mp4"]
                if len(files) != 1 or not files[0].is_file():
                    raise RuntimeError(f"WanGP did not produce one real MP4: {result['generated_files']}")
                target = videos / f"wan2gp_h3_{mode.value}_{index + 1:02d}.mp4"
                actual = deliver_video(files[0], target, evidence["delivery"])
                summary["saved_files"].append(str(target))
                summary["render_attempts"].append({"seed": effective["seed"], "settings_path": str(settings_path), "raw_video_path": str(files[0]), "wall_seconds": result["wall_seconds"], "actual": actual})
            summary["success"] = True
        except Exception as exc:
            summary.update(success=False, error=str(exc))
            raise
        finally:
            summary.update(wall_seconds=time.perf_counter() - started, worker_startup_seconds=self.client.startup_seconds, worker_log_path=str(self.client.log_path))
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return {"run_dir": str(run_dir), "saved_files": summary["saved_files"], "summary_path": str(summary_path), "workflow_name": manifest.name, "h3_mode": mode.value, "provider": "wan2gp", "credit": "Powered by WanGP", "wall_seconds": summary["wall_seconds"], "reference_manifest": evidence.get("reference_manifest", [])}


def register_wan2gp_tools(registry: ToolRegistry, assets: AssetRegistry, output_root: Path, comfy_host: str | None = None, comfy_port: int | None = None) -> None:
    Wan2GPToolset(assets, output_root, comfy_host or os.environ.get("COMFYUI_HOST") or "127.0.0.1", comfy_port or int(os.environ.get("COMFYUI_PORT") or 8188)).register_tools(registry)
