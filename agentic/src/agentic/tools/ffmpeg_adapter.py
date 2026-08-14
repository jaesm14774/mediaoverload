from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal


class FFmpegAdapter:
    """Agentic-native wrapper for the FFmpeg operations used by the runtime."""

    def __init__(self) -> None:
        self._checked = False

    def extract_last_frame(self, video_path: str, output_path: str) -> str:
        self._ensure_binaries()
        self._ensure_parent(output_path)
        self._run(
            [
                "ffmpeg",
                "-sseof",
                "-0.1",
                "-i",
                video_path,
                "-update",
                "1",
                "-q:v",
                "2",
                "-y",
                output_path,
            ]
        )
        return output_path

    def probe_media(self, media_path: str) -> dict[str, object]:
        self._ensure_binaries()
        raw = self._run_capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                media_path,
            ]
        )
        payload = json.loads(raw or "{}")
        streams = payload.get("streams", []) if isinstance(payload, dict) else []
        if not isinstance(streams, list):
            streams = []
        video_stream = next((stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"), {})
        audio_stream = next((stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"), {})
        format_info = payload.get("format", {}) if isinstance(payload, dict) else {}
        if not isinstance(format_info, dict):
            format_info = {}
        duration_raw = format_info.get("duration") or video_stream.get("duration") or audio_stream.get("duration")
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError):
            duration = 0.0
        return {
            "path": str(media_path),
            "duration": duration,
            "format": str(format_info.get("format_name") or ""),
            "has_video": bool(video_stream),
            "has_audio": bool(audio_stream),
            "width": int(video_stream.get("width") or 0),
            "height": int(video_stream.get("height") or 0),
            "frame_rate": self._parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
            "video_duration": self._parse_float(video_stream.get("duration")),
            "video_codec": str(video_stream.get("codec_name") or ""),
            "audio_codec": str(audio_stream.get("codec_name") or ""),
            "audio_duration": self._parse_float(audio_stream.get("duration")),
            "sample_rate": int(audio_stream.get("sample_rate") or 0),
            "channels": int(audio_stream.get("channels") or 0),
            "channel_layout": str(audio_stream.get("channel_layout") or ""),
            "stream_count": len(streams),
        }

    def analyze_audio(self, media_path: str, *, silence_threshold_db: float = -50.0, silence_min_seconds: float = 0.4) -> dict[str, object]:
        """Measure loudness and sustained silence without decoding audio in Python."""

        self._ensure_binaries()
        command = [
            "ffmpeg",
            "-hide_banner",
            "-i",
            media_path,
            "-vn",
            "-af",
            f"volumedetect,silencedetect=noise={silence_threshold_db}dB:d={silence_min_seconds}",
            "-f",
            "null",
            "-",
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is not installed or not available in PATH.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"Audio analysis failed: {stderr}") from exc

        log = completed.stderr or ""
        mean_match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", log)
        max_match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", log)
        silence_durations = [
            float(value)
            for value in re.findall(r"silence_duration:\s*(\d+(?:\.\d+)?)", log)
        ]
        probe = self.probe_media(media_path)
        duration = float(probe.get("duration") or 0.0)
        silence_seconds = min(duration, sum(silence_durations))
        return {
            "mean_volume_db": float(mean_match.group(1)) if mean_match else None,
            "max_volume_db": float(max_match.group(1)) if max_match else None,
            "silence_threshold_db": silence_threshold_db,
            "silence_min_seconds": silence_min_seconds,
            "silence_seconds": silence_seconds,
            "silence_ratio": (silence_seconds / duration) if duration > 0 else 1.0,
        }

    def make_contact_sheet(
        self,
        video_path: str,
        output_path: str,
        *,
        frame_count: int = 6,
        columns: int = 3,
        scale_width: int = 320,
        duration_seconds: float | None = None,
    ) -> str:
        self._ensure_binaries()
        self._ensure_parent(output_path)
        frame_count = max(1, int(frame_count))
        columns = max(1, min(int(columns), frame_count))
        rows = max(1, (frame_count + columns - 1) // columns)
        duration = float(duration_seconds or 0.0)
        interval = max(0.5, (duration if duration > 0 else 15.0) / frame_count)
        self._run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                f"fps=1/{interval:.3f},scale={int(scale_width)}:-2,tile={columns}x{rows}",
                "-frames:v",
                "1",
                "-q:v",
                "3",
                "-y",
                output_path,
            ]
        )
        return output_path

    def concat_videos(
        self,
        video_paths: list[str],
        output_path: str,
        method: Literal["demuxer", "filter"] = "demuxer",
    ) -> str:
        self._ensure_binaries()
        if not video_paths:
            raise ValueError("video_paths cannot be empty")
        self._ensure_parent(output_path)
        if len(video_paths) == 1:
            shutil.copy2(video_paths[0], output_path)
            return output_path
        if method == "demuxer":
            return self._concat_demuxer(video_paths, output_path)
        if method == "filter":
            return self._concat_filter(video_paths, output_path)
        raise ValueError(f"Unknown method: {method}")

    def video_to_gif(
        self,
        video_path: str,
        output_path: str,
        fps: int = 12,
        max_colors: int = 256,
        scale_width: int = 512,
    ) -> str:
        self._ensure_binaries()
        self._ensure_parent(output_path)
        palette_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_palette.png"))
        try:
            self._run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vf",
                    f"fps={fps},scale={scale_width}:-1:flags=lanczos,palettegen=max_colors={max_colors}",
                    "-y",
                    palette_path,
                ]
            )
            self._run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-i",
                    palette_path,
                    "-lavfi",
                    f"fps={fps},scale={scale_width}:-1:flags=lanczos[x];[x][1:v]paletteuse",
                    "-y",
                    output_path,
                ]
            )
        finally:
            if os.path.exists(palette_path):
                os.unlink(palette_path)
        return output_path

    def merge_audio_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        self._ensure_binaries()
        self._ensure_parent(output_path)
        self._run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                "-y",
                output_path,
            ]
        )
        return output_path

    def concat_audio(self, audio_paths: list[str], output_path: str) -> str:
        self._ensure_binaries()
        if not audio_paths:
            raise ValueError("audio_paths cannot be empty")
        self._ensure_parent(output_path)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            list_path = handle.name
            for audio_path in audio_paths:
                escaped = os.path.abspath(audio_path).replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        try:
            self._run(
                [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    list_path,
                    "-c:a",
                    "libmp3lame",
                    "-y",
                    output_path,
                ]
            )
        finally:
            if os.path.exists(list_path):
                os.unlink(list_path)
        return output_path

    def create_video_from_image(self, image_path: str, audio_path: str, output_path: str, fps: int = 30) -> str:
        self._ensure_binaries()
        self._ensure_parent(output_path)
        self._run(
            [
                "ffmpeg",
                "-loop",
                "1",
                "-i",
                image_path,
                "-i",
                audio_path,
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-shortest",
                "-r",
                str(fps),
                "-y",
                output_path,
            ]
        )
        return output_path

    def gif_to_mp4(self, gif_path: str, output_path: str, fps: float | None = None) -> str:
        self._ensure_binaries()
        self._ensure_parent(output_path)
        effective_fps = fps or 10
        self._run(
            [
                "ffmpeg",
                "-i",
                gif_path,
                "-vf",
                f"fps={effective_fps}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "faststart",
                "-y",
                output_path,
            ]
        )
        return output_path

    def pad_video_to_aspect(
        self,
        video_path: str,
        output_path: str,
        *,
        target_width: int,
        target_height: int,
        background: str = "#15151f",
    ) -> str:
        """Fit a video inside a platform canvas without cropping the story."""
        self._ensure_binaries()
        self._ensure_parent(output_path)
        self._run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                (
                    f"scale={int(target_width)}:{int(target_height)}:force_original_aspect_ratio=decrease,"
                    f"pad={int(target_width)}:{int(target_height)}:(ow-iw)/2:(oh-ih)/2:color={background},"
                    "format=yuv420p"
                ),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-y",
                output_path,
            ]
        )
        return output_path

    def _concat_demuxer(self, video_paths: list[str], output_path: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            list_path = handle.name
            for video_path in video_paths:
                escaped = os.path.abspath(video_path).replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        try:
            self._run(
                [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    list_path,
                    "-c",
                    "copy",
                    "-y",
                    output_path,
                ]
            )
        finally:
            if os.path.exists(list_path):
                os.unlink(list_path)
        return output_path

    def _concat_filter(self, video_paths: list[str], output_path: str) -> str:
        inputs: list[str] = []
        filter_parts: list[str] = []
        for index, path in enumerate(video_paths):
            inputs.extend(["-i", path])
            filter_parts.append(f"[{index}:v][{index}:a]")
        filter_complex = f"{''.join(filter_parts)}concat=n={len(video_paths)}:v=1:a=1[v][a]"
        self._run(
            [
                "ffmpeg",
                *inputs,
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-y",
                output_path,
            ]
        )
        return output_path

    def _ensure_binaries(self) -> None:
        if self._checked:
            return
        for binary in ("ffmpeg", "ffprobe"):
            self._run([binary, "-version"])
        self._checked = True

    @staticmethod
    def _ensure_parent(output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _parse_rate(cls, value: object) -> float:
        text = str(value or "").strip()
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = cls._parse_float(denominator)
            return cls._parse_float(numerator) / denominator_value if denominator_value else 0.0
        return cls._parse_float(text)

    @staticmethod
    def _run(command: list[str]) -> None:
        try:
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            binary = command[0] if command else "ffmpeg"
            raise RuntimeError(f"{binary} is not installed or not available in PATH.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            joined = " ".join(command)
            raise RuntimeError(f"Command failed: {joined}\n{stderr}") from exc

    @staticmethod
    def _run_capture(command: list[str]) -> str:
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return completed.stdout
        except FileNotFoundError as exc:
            binary = command[0] if command else "ffprobe"
            raise RuntimeError(f"{binary} is not installed or not available in PATH.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            joined = " ".join(command)
            raise RuntimeError(f"Command failed: {joined}\n{stderr}") from exc
