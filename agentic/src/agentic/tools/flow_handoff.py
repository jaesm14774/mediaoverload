"""Google Flow render backend driven by a human/agent operator through a file handoff.

Enabled with ``AGENTIC_RENDER_BACKEND=flow_handoff``. Instead of ComfyUI, the image and
video render tools write ``NNNN.request.json|md`` into ``FLOW_HANDOFF_DIR`` and block until
the operator (Claude driving Google Flow in a browser) drops the generated files into
``NNNN.out/`` and creates ``NNNN.done``. Every later pipeline stage (Discord review,
publishing) is untouched because the tools return the same result shape as the Comfy ones.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from agentic.runtime.registry import ToolRegistry

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
FLOW_MAX_SECONDS = 15
FLOW_FPS = 24
PAD_FRAMES = 12  # a 15 s Flow clip is 360 frames; delivery asks for 362


def handoff_dir(output_root: Path | None = None) -> Path:
    raw = os.environ.get("FLOW_HANDOFF_DIR")
    path = Path(raw).expanduser().resolve() if raw else (Path(output_root or Path.cwd()) / "flow_handoff").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timeout_seconds() -> float:
    return float(os.environ.get("FLOW_HANDOFF_TIMEOUT_SECONDS") or 3 * 3600)


def _next_sequence(directory: Path) -> int:
    """Reserve the next sequence number atomically so concurrent renders never share one."""
    while True:
        existing = [int(p.name[:4]) for p in directory.glob("????.*") if p.name[:4].isdigit()]
        seq = max(existing, default=0) + 1
        try:
            os.close(os.open(directory / f"{seq:04d}.reserve", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return seq
        except FileExistsError:
            continue


def aspect_for(width: int, height: int) -> str:
    ratio = width / height
    if ratio >= 1.2:
        return "16:9"
    return "9:16" if ratio <= 0.8 else "1:1 (Flow may only offer 16:9/9:16; pad or crop afterwards)"


def _probe_video(path: Path) -> dict[str, Any]:
    """Real width/height/fps of a video, so the pass-through result has the shape the Comfy tool returns."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    num, _, den = str(stream["r_frame_rate"]).partition("/")
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "fps": float(num) / float(den or 1)}


def _wait_for_done(directory: Path, seq: int) -> Path:
    done = directory / f"{seq:04d}.done"
    out = directory / f"{seq:04d}.out"
    deadline = time.monotonic() + _timeout_seconds()
    failed = directory / f"{seq:04d}.failed"
    while time.monotonic() < deadline:
        if failed.exists():
            raise RuntimeError(f"Flow handoff {seq:04d} failed: {failed.read_text(encoding='utf-8')}")
        if done.exists():
            return out
        time.sleep(2.0)
    raise TimeoutError(f"Flow handoff {seq:04d} was not completed within {_timeout_seconds():.0f}s")


def _write_request(directory: Path, seq: int, request: dict[str, Any], instructions: str) -> None:
    (directory / f"{seq:04d}.out").mkdir(parents=True, exist_ok=True)
    (directory / f"{seq:04d}.request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / f"{seq:04d}.request.md").write_text(instructions, encoding="utf-8")


class FlowHandoffToolset:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register("asset.ensure_workflow_ready", self.ensure_ready,
                          "Flow handoff: no local workflow assets are needed")
        registry.register("comfy.nvidia_vfx_postprocess", self.vfx_passthrough,
                          "Flow handoff: no GPU post-process, pass the video through unchanged")
        registry.register("comfy.workflow.text_to_image", self.text_to_image,
                          "Generate keyframe images in Google Flow (operator handoff)")
        registry.register("comfy.render_h3", self.render_video,
                          "Generate image-to-video with audio in Google Flow (operator handoff)")

    @staticmethod
    def ensure_ready(payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("workflow_name") or "flow_handoff")
        return {"workflow_name": name, "workflow_path": "", "asset_status": []}

    @staticmethod
    def vfx_passthrough(payload: dict[str, Any]) -> dict[str, Any]:
        source = Path(str(payload.get("video_path") or "")).expanduser().resolve()
        target = Path(str(payload.get("output_path") or "")).expanduser().resolve()
        if not source.is_file() or not str(payload.get("output_path") or "").strip() or source == target:
            raise ValueError("VFX passthrough needs an existing video_path and a different output_path")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        # media.video.nvidia_vfx_postprocess reads result["fps"|"width"|"height"], so report the real stream.
        info = _probe_video(target)
        return {"video_path": str(target), "final_video_path": str(target), "saved_files": [str(target)],
                "summary_path": "", "source": {}, "result": {"passthrough": True, **info}, "settings": {},
                "chunk_count": 0, "wall_seconds": 0.0, "provider": "flow_handoff"}

    def text_to_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        directory = handoff_dir(self.output_root)
        seq = _next_sequence(directory)
        run_dir = Path(str(payload["run_dir"])).resolve()
        images = run_dir / "images"
        images.mkdir(parents=True, exist_ok=True)
        width, height = int(payload.get("width") or 1024), int(payload.get("height") or 1024)
        count = max(1, int(payload.get("image_count") or 1))
        request = {
            "kind": "image", "seq": seq, "prompt": payload.get("prompt"),
            "negative_prompt": payload.get("negative_prompt") or "", "count": count,
            "width": width, "height": height, "aspect": aspect_for(width, height),
            "model": "Nano Banana 2.1 (0 credits, never Nano Banana Pro)",
            "out_dir": str((directory / f"{seq:04d}.out").resolve()),
        }
        _write_request(directory, seq, request, (
            f"# Flow IMAGE request {seq:04d}\n\n- generate {count} image(s), aspect {request['aspect']}, model {request['model']}\n"
            f"- download 1K originals into: {request['out_dir']}\n- then create the file {seq:04d}.done next to the .out folder\n\n"
            f"## Prompt\n\n{request['prompt']}\n"))
        out = _wait_for_done(directory, seq)
        files = sorted(p for p in out.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(files) < count:
            raise RuntimeError(f"Flow handoff {seq:04d} returned {len(files)} image(s), expected {count}, in {out}")
        saved: list[str] = []
        for index, source in enumerate(files[:count], start=1):
            target = images / f"flow_{seq:04d}_{index:02d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            saved.append(str(target))
        summary_path = run_dir / "flow_image_summary.json"
        summary_path.write_text(json.dumps({"provider": "flow_handoff", "request": request, "saved_files": saved},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
        return {"run_dir": str(run_dir), "saved_files": saved, "summary_path": str(summary_path),
                "workflow_name": str(payload.get("workflow_name") or "flow_nano_banana"),
                "provider": "flow_handoff"}

    def render_video(self, payload: dict[str, Any]) -> dict[str, Any]:
        from agentic.tools.comfy_h3 import deliver_video  # lazy: pulls in PIL/ffmpeg helpers

        directory = handoff_dir(self.output_root)
        width, height = int(payload.get("width") or 608), int(payload.get("height") or 352)
        length = int(payload.get("length") or 124)
        fps = int(payload.get("fps") or payload.get("frame_rate") or FLOW_FPS)
        if length > FLOW_MAX_SECONDS * fps + PAD_FRAMES:
            # Fail before the operator spends credits on a clip that can never satisfy the delivery contract.
            raise ValueError(f"Flow clips are at most {FLOW_MAX_SECONDS}s; delivery needs {length} frames at {fps} fps")
        seq = _next_sequence(directory)
        run_dir = Path(str(payload["run_dir"])).resolve()
        videos = run_dir / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        seconds = min(FLOW_MAX_SECONDS, max(3, math.ceil(length / fps)))
        first = next((str(payload[k]) for k in ("image_path", "input_image_path") if payload.get(k)), "")
        last = next((str(payload[k]) for k in ("last_image_path", "last_frame_path") if payload.get(k)), "")
        # Ref2VA segments (serial_video segment 1) carry reference images instead of a start frame. Flow has
        # no equivalent, so the first reference image becomes the start frame and the rest are listed for the operator.
        references = [item for item in (payload.get("reference_manifest") or []) if isinstance(item, dict)]
        reference_images = [{"path": str(item.get("path")), "label": str(item.get("prompt_label") or "")}
                            for item in references if item.get("type") != "video" and item.get("path")]
        if not first and reference_images:
            first = reference_images[0]["path"]
        request = {
            "kind": "video", "seq": seq, "prompt": payload.get("prompt"), "start_frame": first,
            "reference_images": reference_images,
            "end_frame": last, "seconds": seconds, "resolution": "360p", "model": "Omni 1.2 Flash",
            "aspect": aspect_for(width, height), "delivery": {"width": width, "height": height, "length": length, "fps": fps},
            "credits": seconds if seconds <= 10 else 10 + 2 * (seconds - 10),
            "out_dir": str((directory / f"{seq:04d}.out").resolve()),
        }
        _write_request(directory, seq, request, (
            f"# Flow VIDEO request {seq:04d}\n\n- model {request['model']}, {seconds}s, 360p, aspect {request['aspect']}, "
            f"~{request['credits']} credits (ask the user first if over budget)\n"
            f"- start frame: {first or '(none)'}\n- end frame: {last or '(none)'}\n"
            + "".join(f"- reference image {r['label'] or ''}: {r['path']}\n" for r in reference_images) +
            f"- download the 360p original mp4 into: {request['out_dir']}\n- then create {seq:04d}.done\n\n"
            f"## Prompt\n\n{request['prompt']}\n"))
        out = _wait_for_done(directory, seq)
        files = sorted(out.glob("*.mp4"))
        if len(files) != 1:
            raise RuntimeError(f"Flow handoff {seq:04d} must return exactly one mp4, got {[f.name for f in files]}")
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                "-of", "csv=p=0", str(files[0])], capture_output=True, text=True)
        if not probe.stdout.strip():
            raise RuntimeError(f"Flow video {files[0].name} has no audio track; the native H3 contract needs audio")
        # Flow clips are exactly N*24 frames while the delivery contract asks for a few extra frames.
        # Hold the last frame and pad silence, then trim to exactly the delivery length so nothing is sped up.
        stereo = directory / f"{seq:04d}.stereo.mp4"
        pad = PAD_FRAMES / fps
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(files[0]),
                        "-vf", f"tpad=stop_mode=clone:stop_duration={pad}", "-af", f"apad=pad_dur={pad}",
                        "-frames:v", str(length), "-t", f"{length / fps:.6f}",
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(fps),
                        "-c:a", "aac", "-ac", "2", str(stereo)], check=True, capture_output=True)
        target = videos / "h3_flow_01.mp4"
        actual = deliver_video(stereo, target, request["delivery"])
        summary_path = run_dir / "comfy_h3_summary.json"
        summary_path.write_text(json.dumps({"provider": "flow_handoff", "request": request, "saved_files": [str(target)],
                                            "actual": actual, "success": True}, indent=2, ensure_ascii=False, default=str),
                                encoding="utf-8")
        return {"run_dir": str(run_dir), "saved_files": [str(target)], "summary_path": str(summary_path),
                "workflow_name": str(payload.get("workflow_name") or "flow_omni_1_2_flash"),
                "h3_mode": str(payload.get("h3_mode") or "i2va"), "provider": "flow_handoff",
                "wall_seconds": 0.0, "phase_seconds": {}, "reference_manifest": []}


def register_flow_handoff_tools(registry: ToolRegistry, output_root: Path) -> None:
    FlowHandoffToolset(output_root).register_tools(registry)
