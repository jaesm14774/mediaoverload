"""Final video upscaling and frame generation through NVIDIA ComfyUI nodes."""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

from agentic.runtime.registry import ToolRegistry
from agentic.tools.comfy_adapter import ComfyAdapter


CHUNK_FRAMES = 32
UPSCALE_FACTOR = 2
FRAME_RATE_MULTIPLIER = 2


def build_nvidia_vfx_chunk_workflow(
    video_name: str,
    *,
    frame_count: int,
    skip_first_frames: int,
    output_fps: float,
    output_prefix: str,
) -> dict[str, dict[str, Any]]:
    """Build one bounded API graph using ComfyUI's flattened dynamic inputs."""
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": frame_count,
                "skip_first_frames": skip_first_frames,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "2": {
            "class_type": "RTXVideoSuperResolution",
            "inputs": {
                "images": ["1", 0],
                "resize_type": "scale by multiplier",
                "resize_type.scale": float(UPSCALE_FACTOR),
                "quality": "ULTRA",
                "strength": 1.0,
                "image_encoding": "8-bit RGB",
            },
        },
        "3": {
            "class_type": "RTXVideoFrameGeneration",
            "inputs": {
                "images": ["2", 0],
                "generation_type": "frame rate multiplier",
                "generation_type.multiplier": FRAME_RATE_MULTIPLIER,
                "mode": "MEDIUM",
                "automatic_shot_change_detection": True,
                "shot_change": False,
                "image_encoding": "8-bit RGB",
            },
        },
        "4": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["3", 0],
                "fps": output_fps,
                "bit_depth": 8,
            },
        },
        "5": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["4", 0],
                "filename_prefix": output_prefix,
                "format": "mp4",
                "codec": "h264",
                "codec.encoding": "re-encode",
                "codec.encoding.crf": 18.0,
            },
        },
    }


class ComfyNvidiaVfxToolset:
    """Run NVIDIA VSR and frame generation on bounded video batches."""

    REQUIRED_NODES = {
        "VHS_LoadVideo",
        "RTXVideoSuperResolution",
        "RTXVideoFrameGeneration",
        "CreateVideo",
        "SaveVideo",
    }

    def __init__(
        self,
        comfy_host: str | None = None,
        comfy_port: int | None = None,
        adapter: ComfyAdapter | None = None,
    ) -> None:
        self.comfy_host = comfy_host or os.environ.get("COMFYUI_HOST") or "127.0.0.1"
        self.comfy_port = comfy_port or int(os.environ.get("COMFYUI_PORT") or 8188)
        self.adapter = adapter or ComfyAdapter()

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(
            "comfy.nvidia_vfx_postprocess",
            self.execute,
            "Upscale and interpolate a final video with NVIDIA VFX nodes",
        )

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = Path(str(payload.get("video_path") or "")).expanduser().resolve()
        target = Path(str(payload.get("output_path") or "")).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"NVIDIA VFX source video does not exist: {source}")
        if not str(payload.get("output_path") or "").strip():
            raise ValueError("NVIDIA VFX post-process requires output_path")
        if source == target:
            raise ValueError("NVIDIA VFX output_path must differ from video_path")
        if target.suffix.lower() != ".mp4":
            raise ValueError("NVIDIA VFX post-process output_path must end in .mp4")
        if target.exists():
            raise FileExistsError(f"NVIDIA VFX output already exists: {target}")

        source_info = _probe_video(source)
        _validate_source_video(source_info)
        output_fps = source_info["fps"] * FRAME_RATE_MULTIPLIER
        expected_frames = source_info["frame_count"] * FRAME_RATE_MULTIPLIER
        target.parent.mkdir(parents=True, exist_ok=True)

        run_id = uuid.uuid4().hex[:12]
        summary_path = target.with_name(f"{target.stem}_nvidia_vfx_summary.json")
        if summary_path.exists():
            raise FileExistsError(f"NVIDIA VFX summary already exists: {summary_path}")
        comfy_output_subfolder = f"MediaOverload/NVIDIA_VFX/{run_id}"
        summary: dict[str, Any] = {
            "success": False,
            "source_video": str(source),
            "output_video": str(target),
            "source": source_info,
            "settings": {
                "upscale_factor": UPSCALE_FACTOR,
                "upscale_quality": "ULTRA",
                "frame_rate_multiplier": FRAME_RATE_MULTIPLIER,
                "frame_generation_mode": "MEDIUM",
                "automatic_shot_change_detection": True,
                "chunk_frames": CHUNK_FRAMES,
                "order": ["RTXVideoSuperResolution", "RTXVideoFrameGeneration"],
            },
            "expected": {
                "width": source_info["width"] * UPSCALE_FACTOR,
                "height": source_info["height"] * UPSCALE_FACTOR,
                "fps": output_fps,
                "frame_count": expected_frames,
                "duration_seconds": expected_frames / output_fps,
                "audio_codec": source_info["audio_codec"],
                "audio_channels": source_info["audio_channels"],
            },
            "chunk_results": [],
            "comfy_output_subfolder": comfy_output_subfolder,
        }
        started = time.perf_counter()
        generator = None
        try:
            generator = self.adapter.build_generator(
                host=self.comfy_host,
                port=self.comfy_port,
            )
            node_info = generator.get_object_info()
            missing = sorted(self.REQUIRED_NODES - node_info.keys())
            if missing:
                raise RuntimeError(
                    "ComfyUI is missing NVIDIA VFX post-process node(s): "
                    + ", ".join(missing)
                    + ". Install Comfy-Org/ComfyUI_Nvidia_VFX_Nodes in this ComfyUI "
                    "custom_nodes folder, install nvidia-vfx in ComfyUI's Python, "
                    "and restart ComfyUI on an RTX 40-series or newer GPU."
                )

            uploaded_video = generator.upload_image(str(source))
            chunk_paths: list[Path] = []
            with tempfile.TemporaryDirectory(
                prefix="nvidia-vfx-",
                dir=str(target.parent),
            ) as temporary:
                chunk_dir = Path(temporary)
                starts = _chunk_starts(source_info["frame_count"])
                for index, start in enumerate(starts):
                    count = min(
                        CHUNK_FRAMES,
                        source_info["frame_count"] - start,
                    )
                    graph = build_nvidia_vfx_chunk_workflow(
                        uploaded_video,
                        frame_count=count,
                        skip_first_frames=start,
                        output_fps=output_fps,
                        output_prefix=(
                            f"{comfy_output_subfolder}/chunk_{index + 1:04d}"
                        ),
                    )
                    paths = generator.generate(
                        workflow_path=str(source),
                        workflow=graph,
                        updates=[],
                        output_dir=str(chunk_dir),
                        file_prefix=f"chunk_{index + 1:04d}",
                    )
                    files = [
                        Path(path)
                        for path in paths
                        if Path(path).suffix.lower() == ".mp4"
                    ]
                    if len(files) != 1 or not files[0].is_file():
                        raise RuntimeError(
                            "ComfyUI NVIDIA VFX chunk did not produce one MP4: "
                            f"{paths}"
                        )
                    chunk_info = _probe_video(files[0])
                    expected_chunk_frames = (
                        (count - 1) * FRAME_RATE_MULTIPLIER + 1
                        if count > 1
                        else count
                    )
                    if chunk_info["frame_count"] != expected_chunk_frames:
                        raise RuntimeError(
                            "ComfyUI NVIDIA VFX chunk returned "
                            f"{chunk_info['frame_count']} frames, expected "
                            f"{expected_chunk_frames}"
                        )
                    chunk_paths.append(files[0])
                    summary["chunk_results"].append(
                        {
                            "index": index + 1,
                            "first_source_frame": start,
                            "source_frame_count": count,
                            "output_frame_count": chunk_info["frame_count"],
                            "comfy_output_filename": files[0].name,
                        }
                    )

                staged_output = chunk_dir / "final.mp4"
                _assemble_chunks(
                    chunk_paths,
                    source,
                    staged_output,
                    source_info=source_info,
                    output_fps=output_fps,
                    expected_frames=expected_frames,
                )
                result_info = _probe_video(staged_output)
                _validate_result(source_info, result_info, expected_frames, output_fps)
                os.replace(staged_output, target)
            summary.update(
                success=True,
                result=result_info,
                chunk_count=len(chunk_paths),
            )
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            summary["wall_seconds"] = time.perf_counter() - started
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            if generator is not None:
                websocket = getattr(generator.communicator, "ws", None)
                if websocket is not None:
                    websocket.close()

        return {
            "video_path": str(target),
            "final_video_path": str(target),
            "saved_files": [str(target)],
            "summary_path": str(summary_path),
            "source": source_info,
            "result": summary["result"],
            "settings": summary["settings"],
            "chunk_count": summary["chunk_count"],
            "wall_seconds": summary["wall_seconds"],
        }


def _chunk_starts(frame_count: int) -> list[int]:
    starts: list[int] = []
    start = 0
    while start < frame_count:
        starts.append(start)
        end = min(start + CHUNK_FRAMES, frame_count)
        if end == frame_count:
            break
        start = end - 1
    return starts


def _probe_video(path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is not installed or is not available in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffprobe could not read {path}: {(exc.stderr or '').strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out while reading {path}") from exc

    data = json.loads(completed.stdout or "{}")
    streams = data.get("streams", [])
    video = next(
        (item for item in streams if item.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise ValueError(f"Video has no readable video stream: {path}")
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    audio = audio_streams[0] if audio_streams else None
    rate = _parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    if rate <= 0:
        raise ValueError(f"Video has no usable frame rate: {path}")
    format_info = data.get("format", {})
    duration = _parse_duration(video.get("duration"))
    if duration is None:
        duration = _parse_duration(format_info.get("duration"))
    audio_duration = _parse_duration(audio.get("duration")) if audio else None
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": float(rate),
        "fps_fraction": str(rate),
        "frame_count": int(video.get("nb_read_frames") or video.get("nb_frames") or 0),
        "duration_seconds": duration or 0.0,
        "video_codec": str(video.get("codec_name") or ""),
        "pixel_format": str(video.get("pix_fmt") or ""),
        "color_transfer": str(video.get("color_transfer") or ""),
        "audio_codec": str(audio.get("codec_name") or "") if audio else "",
        "audio_channels": int(audio.get("channels") or 0) if audio else 0,
        "audio_duration_seconds": audio_duration or 0.0,
        "audio_stream_count": len(audio_streams),
    }


def _parse_duration(raw: object) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _parse_rate(raw: object) -> Fraction:
    try:
        value = str(raw or "0/1")
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return Fraction(0, 1)


def _validate_source_video(source: dict[str, Any]) -> None:
    if source["frame_count"] < 1 or source["duration_seconds"] <= 0:
        raise ValueError("NVIDIA VFX post-process requires a non-empty video")
    if source["width"] < 64 or source["height"] < 64:
        raise ValueError("NVIDIA VFX post-process requires video dimensions of at least 64 pixels")
    if source["width"] % 4 or source["height"] % 4:
        raise ValueError(
            "NVIDIA VFX post-process requires source dimensions divisible by 4 "
            "for exact 2x output dimensions"
        )
    if source["width"] > 4096 or source["height"] > 4096:
        raise ValueError("2x NVIDIA VFX upscaling supports source dimensions up to 4096 pixels")
    if source["fps"] > 60:
        raise ValueError("NVIDIA VFX 2x frame generation requires a source frame rate of 60 FPS or less")
    if source["audio_stream_count"] > 1:
        raise ValueError("NVIDIA VFX post-process currently accepts one audio stream per video")
    if source["pixel_format"].endswith(("10le", "12le", "14le", "16le")):
        raise ValueError("NVIDIA VFX post-process is configured for 8-bit SDR video")
    if source["color_transfer"].lower() in {"smpte2084", "arib-std-b67"}:
        raise ValueError("NVIDIA VFX post-process is configured for SDR video, not HDR")
    if source["audio_duration_seconds"] and abs(
        source["audio_duration_seconds"] - source["duration_seconds"]
    ) > 1 / source["fps"]:
        raise ValueError("Source audio and video durations differ by more than one source frame")


def _assemble_chunks(
    chunks: list[Path],
    source: Path,
    target: Path,
    *,
    source_info: dict[str, Any],
    output_fps: float,
    expected_frames: int,
) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for chunk in chunks:
        command.extend(["-i", str(chunk)])
    source_input = len(chunks)
    command.extend(["-i", str(source)])

    filters: list[str] = []
    for index in range(len(chunks)):
        trim = "trim=start_frame=1," if index else ""
        filters.append(
            f"[{index}:v:0]{trim}setpts=PTS-STARTPTS[chunk{index}]"
        )
    if len(chunks) > 1:
        joined_inputs = "".join(f"[chunk{index}]" for index in range(len(chunks)))
        filters.append(
            f"{joined_inputs}concat=n={len(chunks)}:v=1:a=0[joined]"
        )
        final_input = "[joined]"
    else:
        final_input = "[chunk0]"
    filters.append(
        f"{final_input}tpad=stop_mode=clone:stop_duration={1 / output_fps:.12g},"
        f"trim=end_frame={expected_frames},"
        f"setpts=N/({output_fps:.12g}*TB)[final_video]"
    )
    command.extend(["-filter_complex", ";".join(filters), "-map", "[final_video]"])
    command.extend(["-map_metadata", str(source_input)])
    if source_info["audio_stream_count"]:
        command.extend(["-map", f"{source_input}:a:0"])
        if source_info["audio_codec"] == "aac":
            command.extend(["-c:a", "copy"])
        else:
            command.extend(["-c:a", "aac", "-b:a", "192k", "-af", "asetpts=PTS-STARTPTS"])
    else:
        command.append("-an")
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    try:
        subprocess.run(
            command,
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or is not available in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not join NVIDIA VFX chunks: {(exc.stderr or '').strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out while joining NVIDIA VFX chunks") from exc


def _validate_result(
    source: dict[str, Any],
    result: dict[str, Any],
    expected_frames: int,
    expected_fps: float,
) -> None:
    for key, expected in (
        ("width", source["width"] * UPSCALE_FACTOR),
        ("height", source["height"] * UPSCALE_FACTOR),
        ("frame_count", expected_frames),
    ):
        if result[key] != expected:
            raise RuntimeError(
                f"NVIDIA VFX output {key} was {result[key]}, expected {expected}"
            )
    if abs(result["fps"] - expected_fps) > 0.01:
        raise RuntimeError(
            f"NVIDIA VFX output frame rate was {result['fps']}, expected {expected_fps}"
        )
    tolerance = 1 / expected_fps
    if abs(result["duration_seconds"] - source["duration_seconds"]) > tolerance:
        raise RuntimeError(
            "NVIDIA VFX output duration differs from its source by more than one output frame"
        )
    if result["audio_stream_count"] != source["audio_stream_count"]:
        raise RuntimeError("NVIDIA VFX output did not preserve the source audio stream")
    if source["audio_stream_count"]:
        if result["audio_channels"] != source["audio_channels"]:
            raise RuntimeError("NVIDIA VFX output changed the source audio channel count")
        if abs(result["audio_duration_seconds"] - source["audio_duration_seconds"]) > tolerance:
            raise RuntimeError("NVIDIA VFX output audio duration differs from its source")


def register_comfy_nvidia_vfx_tools(
    registry: ToolRegistry,
    comfy_host: str | None = None,
    comfy_port: int | None = None,
) -> None:
    ComfyNvidiaVfxToolset(comfy_host, comfy_port).register_tools(registry)
