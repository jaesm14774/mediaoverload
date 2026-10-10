"""H3 conditioning and native audio/video through the shared ComfyUI service."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from fractions import Fraction
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from agentic.assets.minimax_h3 import DIFFUSION_MODELS, TEXT_ENCODERS, h3_requirements
from agentic.assets.registry import AssetRegistry, AssetRequirement
from agentic.h3_reference import build_reference_lineage, normalize_reference_manifest
from agentic.minimax_prompting import require_h3_prompt
from agentic.runtime.h3_modes import H3Mode, resolve_h3_mode, validate_h3_payload
from agentic.runtime.registry import ToolRegistry
from agentic.tools.comfy_adapter import ComfyAdapter


def probe_video(path: Path) -> dict[str, Any]:
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path)], text=True, encoding="utf-8"))
    video = next((stream for stream in data["streams"] if stream["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"H3 input has no video stream: {path}")
    audio = next((stream for stream in data["streams"] if stream["codec_type"] == "audio"), None)
    return {"width": video["width"], "height": video["height"], "length": int(video["nb_read_frames"]),
            "fps": float(Fraction(video["avg_frame_rate"])),
            "duration_seconds": float(video.get("duration", data["format"]["duration"])), "audio": audio}


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
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"H3 {role} is not readable media: {source}") from exc
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(source), "role": role, "size_bytes": source.stat().st_size, "sha256": digest}


def build_h3_workflow(template: dict[str, Any], mode: str | H3Mode,
                      payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the model recipe and retain the existing H3 input/delivery contract."""
    mode = resolve_h3_mode(mode)
    prompt = require_h3_prompt(payload.get("prompt"))
    validate_h3_payload(mode, payload)
    width, height = int(payload.get("width") or 608), int(payload.get("height") or 352)
    length, steps = int(payload.get("length") or 124), int(payload.get("steps") or 16)
    fps = int(payload.get("fps") or payload.get("frame_rate") or 24)
    if not (256 <= width <= 1344 and 256 <= height <= 1344):
        raise ValueError("H3 canvas dimensions must be between 256 and 1344")
    if not (1 <= length <= 362 and 1 <= steps <= 32 and fps == 24):
        raise ValueError("H3 supports 1-362 delivered frames, 1-32 steps and 24 fps in this workflow")
    profile = str(payload.get("model_profile") or "q4")
    requirements = h3_requirements(mode.value, profile)
    graph = json.loads(json.dumps(template))
    graph["1"]["inputs"]["unet_name"] = DIFFUSION_MODELS["ref2va" if mode is H3Mode.REF2VA else "fl2va"][0]
    graph["2"]["inputs"]["clip_name"] = TEXT_ENCODERS[profile][0]
    inputs = graph["5"]["inputs"]
    inputs.update(prompt=prompt, width=width // 32 * 32, height=height // 32 * 32,
                  length=max(107, math.ceil((length - 5) / 17) * 17 + 5))
    shift = float(graph["6"]["inputs"]["shift_video"])
    # The model's shifted flow schedule is supplied to Comfy's ManualSigmas node.
    sigmas = [shift * t / (1 + (shift - 1) * t) for t in ((steps - i) / steps for i in range(steps + 1))]
    graph["7"]["inputs"]["sigmas"] = ", ".join(format(value, ".12g") for value in sigmas)
    graph["9"]["inputs"]["noise_seed"] = int(payload.get("seed", random.randint(1, 999999999)))
    graph["14"]["inputs"]["fps"] = float(fps)
    evidence: dict[str, Any] = {"h3_mode": mode.value, "model_profile": profile, "required_assets": requirements,
                               "conditioning_files": [], "delivery": {"width": width, "height": height, "length": length, "fps": fps}}
    next_id = 30

    def bind_file(path: str, role: str, input_key: str, *, video: bool = False) -> None:
        nonlocal next_id
        info = _file_evidence(path, role)
        node_id = str(next_id)
        if video:
            graph[node_id] = {"class_type": "LoadVideo", "inputs": {"file": Path(info["path"]).name}}
            component_id = str(next_id + 1)
            graph[component_id] = {"class_type": "GetVideoComponents", "inputs": {"video": [node_id, 0]}}
            inputs[input_key] = [component_id, 0]
            next_id += 2
        else:
            graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": Path(info["path"]).name}}
            inputs[input_key] = [node_id, 0]
            next_id += 1
        evidence["conditioning_files"].append({**info, "upload_node": node_id})

    for key, alternatives, role in (
        ("first_frame", ("image_path", "input_image_path"), "first_frame"),
        ("last_frame", ("last_image_path", "last_frame_path"), "last_frame"),
    ):
        path = next((str(payload[name]) for name in alternatives if payload.get(name)), "")
        if path:
            bind_file(path, role, key)
    if mode is H3Mode.REF2VA:
        references = normalize_reference_manifest(payload.get("reference_manifest"),
            image_paths=payload.get("reference_image_paths"), video_paths=payload.get("reference_video_paths"), require_files=True)
        evidence.update(reference_manifest=references, reference_lineage=build_reference_lineage(references))
        inputs["ref_image_size"] = str(payload.get("reference_image_size") or "match")
        image_index = video_index = 0
        for reference in references:
            video = reference["type"] == "video"
            key = f"ref_videos.ref_video_{video_index}" if video else f"ref_images.ref_image_{image_index}"
            bind_file(reference["path"], reference["prompt_label"], key, video=video)
            if video:
                video_index += 1
            else:
                image_index += 1
    return graph, evidence


def deliver_video(source: Path, target: Path, delivery: dict[str, int]) -> dict[str, Any]:
    actual = probe_video(source)
    if actual["length"] < delivery["length"] or actual["audio"] is None:
        raise RuntimeError(f"ComfyUI H3 returned insufficient frames or missing native audio: {actual}")
    if all(actual[key] == delivery[key] for key in ("width", "height", "length", "fps")):
        shutil.copy2(source, target)
    else:
        width, height, length, fps = (delivery[key] for key in ("width", "height", "length", "fps"))
        # Retain the complete generated timeline and both endpoint anchors.
        frames = [round(i * (actual["length"] - 1) / (length - 1)) for i in range(length)] if length > 1 else [actual["length"] - 1]
        select = "+".join(f"eq(n,{index})" for index in frames)
        video_filter = f"select='{select}',setpts=N/({fps}*TB),scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        ratio = actual["length"] / length
        tempos = []
        while ratio > 2:
            tempos.append("atempo=2")
            ratio /= 2
        tempos.extend([f"atempo={ratio}", f"atrim=duration={length / fps}", "asetpts=PTS-STARTPTS"])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0", "-vf", video_filter,
                        "-af", ",".join(tempos), "-r", str(fps), "-t", str(length / fps), "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(target)], check=True, capture_output=True)
    result = probe_video(target)
    if any(result[key] != delivery[key] for key in ("width", "height", "length", "fps")) or result["audio"] is None or result["audio"].get("channels") != 2:
        raise RuntimeError(f"ComfyUI H3 delivery contract failed: {result}")
    return result


class ComfyH3Toolset:
    def __init__(self, asset_registry: AssetRegistry, output_root: Path,
                 comfy_host: str | None = None, comfy_port: int | None = None) -> None:
        self.asset_registry = asset_registry
        self.output_root = output_root
        self.comfy_host = comfy_host or os.environ.get("COMFYUI_HOST") or "127.0.0.1"
        self.comfy_port = comfy_port or int(os.environ.get("COMFYUI_PORT") or 8188)
        self.adapter = ComfyAdapter()

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register("comfy.render_h3", self.execute, "Render H3 video and native audio through ComfyUI")

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest = self.asset_registry.get_manifest(str(payload["workflow_name"]))
        merged = {**manifest.recommended_defaults, **payload}
        mode = resolve_h3_mode(merged.get("h3_mode") or manifest.conditioning.get("h3_mode"))
        allowed = manifest.conditioning.get("allowed_modes", [manifest.conditioning.get("h3_mode")])
        if mode.value not in allowed:
            raise ValueError(f"{manifest.name} does not support H3 mode {mode.value}")
        template = self.asset_registry.load_workflow_template(manifest)
        graph, evidence = build_h3_workflow(template, mode, merged)
        requirements = [AssetRequirement.from_dict(item) for item in evidence["required_assets"]]
        readiness = self.asset_registry.ensure_requirements(replace(manifest, required_assets=requirements))
        missing = [item["target_path"] for item in readiness if item["status"] != "ready"]
        if missing:
            raise RuntimeError("ComfyUI H3 assets missing or corrupt: " + "; ".join(missing))
        count = int(merged.get("video_count") or 1)
        if not 1 <= count <= 4:
            raise ValueError("video_count must be between 1 and 4")
        run_dir = Path(str(merged["run_dir"])).resolve()
        videos, raw = run_dir / "videos", run_dir / "raw"
        videos.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        summary = {"provider": "comfyui", "workflow_name": manifest.name, "payload": merged,
                   **evidence, "render_attempts": [], "saved_files": []}
        summary_path = run_dir / "comfy_h3_summary.json"
        started = time.perf_counter()
        generator = None
        try:
            generator = self.adapter.build_generator(host=self.comfy_host, port=self.comfy_port)
            node_info = generator.get_object_info()
            missing_nodes = sorted({node["class_type"] for node in graph.values()} - node_info.keys())
            if missing_nodes:
                raise RuntimeError("ComfyUI is missing required H3 node type(s): " + ", ".join(missing_nodes))
            for item in evidence["conditioning_files"]:
                filename = generator.upload_image(item["path"])
                node = graph[item["upload_node"]]
                node["inputs"]["file" if node["class_type"] == "LoadVideo" else "image"] = filename
            base_seed = graph["9"]["inputs"]["noise_seed"]
            for index in range(count):
                graph["9"]["inputs"]["noise_seed"] = (base_seed + index) % (2**64)
                graph_path = run_dir / f"comfy_h3_request_{index + 1:02d}.json"
                graph_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
                paths = generator.generate(workflow_path=str(graph_path), workflow=graph, updates=[],
                                           output_dir=str(raw), file_prefix=f"h3_{mode.value}_{index + 1:02d}")
                files = [Path(path) for path in paths if Path(path).suffix.lower() == ".mp4"]
                if len(files) != 1 or not files[0].is_file():
                    raise RuntimeError(f"ComfyUI H3 did not produce one real MP4: {paths}")
                target = videos / f"h3_{mode.value}_{index + 1:02d}.mp4"
                actual = deliver_video(files[0], target, evidence["delivery"])
                summary["saved_files"].append(str(target))
                summary["render_attempts"].append({"seed": graph["9"]["inputs"]["noise_seed"],
                    "request_path": str(graph_path), "raw_video_path": str(files[0]), "actual": actual})
            summary["success"] = True
        except Exception as exc:
            summary.update(success=False, error=str(exc))
            raise
        finally:
            summary["wall_seconds"] = time.perf_counter() - started
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            if generator is not None and generator.communicator.ws is not None:
                generator.communicator.ws.close()
        return {"run_dir": str(run_dir), "saved_files": summary["saved_files"], "summary_path": str(summary_path),
                "workflow_name": manifest.name, "h3_mode": mode.value, "provider": "comfyui",
                "wall_seconds": summary["wall_seconds"], "reference_manifest": evidence.get("reference_manifest", [])}


def register_comfy_h3_tools(registry: ToolRegistry, assets: AssetRegistry, output_root: Path,
                           comfy_host: str | None = None, comfy_port: int | None = None) -> None:
    ComfyH3Toolset(assets, output_root, comfy_host, comfy_port).register_tools(registry)
