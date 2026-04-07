from __future__ import annotations

import os
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
