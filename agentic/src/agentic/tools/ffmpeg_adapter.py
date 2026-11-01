from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import IO, Iterable, Literal

from PIL import Image, ImageChops, ImageStat

from agentic.editorial_pacing import EDITORIAL_PACING_SPEEDS, motion_pacing_labels


FFMPEG_COMMAND_TIMEOUT_SECONDS = 600
MAX_EDITORIAL_AUDIO_VIDEO_DRIFT_SECONDS = 0.032


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

    def extract_frames_at(
        self, video_path: str, output_paths: list[str], timestamps_seconds: list[float],
    ) -> list[str]:
        """Decode once and sample the same frames as independent output seeks.

        Each split branch trims at its requested timestamp. Separate outputs
        also preserve repeated samples when a short clip has few source frames.
        All destinations must share a PNG or JPEG suffix that matches the
        encoded image format.
        Materialize only after every requested frame exists, so an EOF cannot
        make a stale destination look like a successful extraction.
        """
        timestamps, destinations, suffix = self._validate_frame_batch_request(
            video_path,
            output_paths,
            timestamps_seconds,
        )
        self._ensure_binaries()
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="frame_batch_", dir=destinations[0].parent) as temporary:
            staged = [Path(temporary) / f"frame-{index}{suffix}" for index in range(len(timestamps))]
            count = len(timestamps)
            labels = "".join(f"[branch{index}]" for index in range(count))
            filters = [f"[0:v:0]format=yuvj420p,split={count}{labels}"]
            for index, timestamp in enumerate(timestamps):
                filters.append(
                    f"[branch{index}]trim=start={timestamp:.6f},setpts=PTS-STARTPTS[frame{index}]"
                )
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
                       "-filter_complex", ";".join(filters)]
            for index, path in enumerate(staged):
                command.extend(["-map", f"[frame{index}]", "-frames:v", "1", "-threads", "1",
                                "-q:v", "2", "-y", str(path)])
            self._run(command)
            if any(not path.is_file() or path.stat().st_size == 0 for path in staged):
                raise RuntimeError("Video did not produce every requested frame; check timestamps against its duration")
            for source, destination in zip(staged, destinations):
                shutil.copyfile(source, destination)
        return output_paths

    @staticmethod
    def _validate_frame_batch_request(
        video_path: str,
        output_paths: list[str],
        timestamps_seconds: list[float],
    ) -> tuple[list[float], list[Path], str]:
        timestamps = [float(value) for value in timestamps_seconds]
        if not timestamps or len(timestamps) != len(output_paths):
            raise ValueError("Provide one output path for each requested timestamp")
        if any(not math.isfinite(value) or value < 0 for value in timestamps):
            raise ValueError("timestamps_seconds must be finite and non-negative")
        destinations = [Path(path).resolve() for path in output_paths]
        if len(set(destinations)) != len(destinations):
            raise ValueError("Frame output paths must be distinct")
        suffixes = {path.suffix.lower() for path in destinations}
        if len(suffixes) != 1 or not suffixes.issubset({".png", ".jpg", ".jpeg"}):
            raise ValueError("Frame output paths must share a .png, .jpg, or .jpeg extension")
        suffix = next(iter(suffixes))
        if Path(video_path).resolve() in destinations:
            raise ValueError("Frame output must be different from source video")
        return timestamps, destinations, suffix

    def extract_frame_at(self, video_path: str, output_path: str, timestamp_seconds: float) -> str:
        """Extract one deterministic review frame at a bounded timestamp."""

        self._ensure_binaries()
        timestamp = float(timestamp_seconds)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        self._ensure_parent(output_path)
        self._run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-ss",
                f"{timestamp:.6f}",
                "-frames:v",
                "1",
                "-vf",
                "format=yuvj420p",
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
        video_tags = video_stream.get("tags", {}) if isinstance(video_stream, dict) else {}
        if not isinstance(video_tags, dict):
            video_tags = {}
        video_side_data = video_stream.get("side_data_list", []) if isinstance(video_stream, dict) else []
        if not isinstance(video_side_data, list):
            video_side_data = []
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
            "pixel_format": str(video_stream.get("pix_fmt") or ""),
            "sample_aspect_ratio": str(video_stream.get("sample_aspect_ratio") or ""),
            "has_rotation_metadata": "rotate" in video_tags
            or any(item.get("side_data_type") == "Display Matrix" for item in video_side_data if isinstance(item, dict)),
            "audio_codec": str(audio_stream.get("codec_name") or ""),
            "audio_duration": self._parse_float(audio_stream.get("duration")),
            "sample_rate": int(audio_stream.get("sample_rate") or 0),
            "channels": int(audio_stream.get("channels") or 0),
            "channel_layout": str(audio_stream.get("channel_layout") or ""),
            "stream_count": len(streams),
        }

    @staticmethod
    def count_video_frames(media_path: str) -> int:
        """Count video frames without decoding unrelated audio streams."""
        progress = FFmpegAdapter._run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "error",
                "-progress",
                "pipe:1",
                "-i",
                media_path,
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ]
        )
        frame_counts = [
            int(value)
            for line in progress.splitlines()
            if line.startswith("frame=")
            and (value := line.partition("=")[2].strip()).isdigit()
        ]
        if not frame_counts or frame_counts[-1] < 1:
            raise RuntimeError(f"FFmpeg did not report a decoded frame count for {media_path}")
        return frame_counts[-1]

    def analyze_audio(
        self,
        media_path: str,
        *,
        silence_threshold_db: float = -50.0,
        silence_min_seconds: float = 0.4,
        duration_seconds: float | None = None,
    ) -> dict[str, object]:
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
                timeout=FFMPEG_COMMAND_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is not installed or not available in PATH.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"Audio analysis failed: {stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Audio analysis exceeded the ffmpeg command timeout.") from exc

        log = completed.stderr or ""
        mean_match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", log)
        max_match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", log)
        silence_durations = [
            float(value)
            for value in re.findall(r"silence_duration:\s*(\d+(?:\.\d+)?)", log)
        ]
        if duration_seconds is None:
            probe = self.probe_media(media_path)
            duration_seconds = float(probe.get("duration") or 0.0)
        duration = float(duration_seconds)
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
        interval = (duration if duration > 0 else 15.0) / frame_count
        self._run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                f"fps=1/{interval:.9f},scale={int(scale_width)}:-2,tile={columns}x{rows}:nb_frames={frame_count}",
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

    def analyze_editorial_pacing(self, video_path: str) -> dict[str, object]:
        """Infer per-beat pace from changes in the rendered frames."""
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Input video does not exist: {video_path}")
        self._ensure_binaries()
        probe = self.probe_media(video_path)
        duration = float(probe.get("video_duration") or probe.get("duration") or 0.0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("input video duration must be available and greater than zero")

        sample_fps = 4
        width, height = 64, 36
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
            "-vf", f"fps={sample_fps},scale={width}:{height}:flags=area,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        try:
            sampled = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=FFMPEG_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.CalledProcessError as exc:
            error = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg motion sampling failed: {error}") from exc
        frame_size = width * height
        if len(sampled.stdout) % frame_size:
            raise RuntimeError("FFmpeg motion sampling returned a partial grayscale frame")
        # Share FFmpeg's buffer and calculate each pixel difference in Pillow's C code.
        sampled_frames = memoryview(sampled.stdout)
        frame_count = len(sampled_frames) // frame_size
        window_count = min(8, max(2, math.ceil(duration / 1.5)))
        window_count = min(window_count, max(1, frame_count - 1))
        if window_count == 0:
            window_count = 1
        motion_totals = [0.0] * window_count
        motion_counts = [0] * window_count
        for index in range(1, frame_count):
            previous_start = (index - 1) * frame_size
            current_start = index * frame_size
            previous = Image.frombuffer(
                "L",
                (width, height),
                sampled_frames[previous_start:current_start],
                "raw",
                "L",
                0,
                1,
            )
            current = Image.frombuffer(
                "L",
                (width, height),
                sampled_frames[current_start:current_start + frame_size],
                "raw",
                "L",
                0,
                1,
            )
            timestamp = (index - 0.5) / sample_fps
            window = min(window_count - 1, int(timestamp / duration * window_count))
            difference = ImageStat.Stat(ImageChops.difference(previous, current)).sum[0]
            motion_totals[window] += difference / (frame_size * 255)
            motion_counts[window] += 1
        motion_scores = [
            total / count if count else 0.0
            for total, count in zip(motion_totals, motion_counts)
        ]
        labels = motion_pacing_labels(motion_scores)
        raw_segments = []
        for index, label in enumerate(labels):
            start = duration * index / window_count
            end = duration if index == window_count - 1 else duration * (index + 1) / window_count
            segment = {
                "start_seconds": start,
                "end_seconds": end,
                "pacing": label,
                "speed_factor": EDITORIAL_PACING_SPEEDS[label],
                "motion_score": motion_scores[index],
            }
            if raw_segments and raw_segments[-1]["pacing"] == label:
                raw_segments[-1]["end_seconds"] = end
                raw_segments[-1]["motion_score"] = (
                    float(raw_segments[-1]["motion_score"]) + motion_scores[index]
                ) / 2
            else:
                raw_segments.append(segment)
        return {
            "method": "relative_motion",
            "source_duration": duration,
            "segments": raw_segments,
        }

    @staticmethod
    def _parse_editorial_segments(
        segments: list[dict[str, object]],
    ) -> list[dict[str, float]]:
        if not isinstance(segments, list) or not segments:
            raise ValueError("segments must contain a complete editorial timeline")
        parsed: list[dict[str, float]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("segments must contain timeline objects")
            try:
                start = float(segment["start_seconds"])
                end = float(segment["end_seconds"])
                speed = float(segment["speed_factor"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("segments must define start, end, and speed_factor") from exc
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or not math.isfinite(speed)
                or start < 0
                or end <= start
                or not 0.5 <= speed <= 1.5
            ):
                raise ValueError("segments must define finite, ordered times and a 0.5-1.5 speed_factor")
            parsed.append({"start_seconds": start, "end_seconds": end, "speed_factor": speed})
        if abs(parsed[0]["start_seconds"]) > 1e-6 or any(
            abs(left["end_seconds"] - right["start_seconds"]) > 1e-6
            for left, right in zip(parsed, parsed[1:])
        ):
            raise ValueError("segments must cover the source timeline without gaps or overlaps")
        return parsed

    @staticmethod
    def _validate_editorial_segments_against_media(
        parsed: list[dict[str, float]],
        duration: float,
        frame_rate: float,
    ) -> None:
        frame_tolerance = 1 / frame_rate
        cursor = 0.0
        for segment in parsed:
            if abs(segment["start_seconds"] - cursor) > frame_tolerance:
                raise ValueError("segments must cover the source timeline without gaps or overlaps")
            segment["start_seconds"] = cursor
            cursor = segment["end_seconds"]
            if cursor > duration + frame_tolerance:
                raise ValueError("segments must end at the source video duration")
        if abs(cursor - duration) > frame_tolerance:
            raise ValueError("segments must cover the source timeline through its duration")
        parsed[-1]["end_seconds"] = duration

    @staticmethod
    def _editorial_video_filter(
        index: int,
        segment: dict[str, float],
        frame_rate: float,
    ) -> str:
        start = segment["start_seconds"]
        end = segment["end_seconds"]
        speed = segment["speed_factor"]
        video_chain = f"trim=start={start:.9f}:end={end:.9f},setpts=PTS-STARTPTS"
        if speed < 1.0:
            video_chain += f",minterpolate=fps={frame_rate / speed:.12g}:mi_mode=mci"
        video_chain += f",setpts=(PTS-STARTPTS)/{speed:.12g},fps={frame_rate:.12g}"
        return f"[vsrc{index}]{video_chain}[v{index}]"

    def _editorial_audio_filter(
        self,
        index: int,
        segment: dict[str, float],
    ) -> str:
        start = segment["start_seconds"]
        end = segment["end_seconds"]
        speed = segment["speed_factor"]
        atempo = ",".join(self._atempo_filters(speed))
        return (
            f"[asrc{index}]atrim=start={start:.9f}:end={end:.9f},"
            f"asetpts=PTS-STARTPTS,{atempo},asetpts=PTS-STARTPTS[a{index}]"
        )

    def _build_editorial_filter(
        self,
        parsed: list[dict[str, float]],
        frame_rate: float,
        has_audio: bool,
    ) -> str:
        count = len(parsed)
        filters = [
            f"[0:v:0]split={count}"
            + "".join(f"[vsrc{index}]" for index in range(count))
        ]
        if has_audio:
            filters.append(
                f"[0:a:0]asplit={count}"
                + "".join(f"[asrc{index}]" for index in range(count))
            )
        for index, segment in enumerate(parsed):
            filters.append(self._editorial_video_filter(index, segment, frame_rate))
            if has_audio:
                filters.append(self._editorial_audio_filter(index, segment))
        if has_audio:
            labels = "".join(
                label
                for pair in zip(
                    (f"[v{index}]" for index in range(count)),
                    (f"[a{index}]" for index in range(count)),
                )
                for label in pair
            )
            filters.append(f"{labels}concat=n={count}:v=1:a=1[v][a]")
        else:
            labels = "".join(f"[v{index}]" for index in range(count))
            filters.append(f"{labels}concat=n={count}:v=1:a=0[v]")
        return ";".join(filters)

    def _write_aligned_editorial_audio(
        self,
        intermediate: Path,
        candidate: Path,
    ) -> None:
        paced_media = self.probe_media(str(intermediate))
        target_duration = float(paced_media.get("video_duration") or 0.0)
        if not math.isfinite(target_duration) or target_duration <= 0:
            raise RuntimeError("Editorial pacing produced a video without a valid duration")
        normalize_audio = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(intermediate),
            "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
            "-af", f"apad,atrim=duration={target_duration:.9f},asetpts=PTS-STARTPTS",
            "-c:a", "aac", "-movflags", "+faststart", "-y", str(candidate),
        ]
        self._run(normalize_audio)
        final_media = self.probe_media(str(candidate))
        audio_duration = float(final_media.get("audio_duration") or 0.0)
        video_duration = float(final_media.get("video_duration") or 0.0)
        drift = abs(audio_duration - video_duration)
        if drift > MAX_EDITORIAL_AUDIO_VIDEO_DRIFT_SECONDS:
            raise RuntimeError(
                "Editorial pacing could not align audio and video within 32 ms "
                f"(measured {drift * 1000:.3f} ms)"
            )

    @staticmethod
    def _predict_editorial_video_duration(
        parsed: list[dict[str, float]],
        frame_rate: float,
    ) -> float:
        output_frames = sum(
            math.floor(
                (segment["end_seconds"] - segment["start_seconds"])
                * frame_rate
                / segment["speed_factor"]
                + 0.5
            )
            for segment in parsed
        )
        return max(1, output_frames) / frame_rate

    def _render_editorial_output(
        self,
        video_path: str,
        output_path: str,
        filter_complex: str,
        has_audio: bool,
        predicted_duration: float | None = None,
    ) -> None:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
            "-filter_complex", filter_complex, "-map", "[v]",
        ]
        if has_audio:
            command.extend(["-map", "[a]"])
        else:
            command.append("-an")
        command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        if has_audio:
            with tempfile.TemporaryDirectory(
                prefix="editorial-pacing-", dir=str(Path(output_path).parent)
            ) as temporary:
                if predicted_duration is not None:
                    aligned_filter = (
                        f"{filter_complex};[a]aformat=sample_fmts=s16,apad,"
                        f"atrim=duration={predicted_duration:.9f},"
                        "asetpts=PTS-STARTPTS[a_aligned]"
                    )
                    aligned_command = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
                        "-filter_complex", aligned_filter, "-map", "[v]", "-map", "[a_aligned]",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-movflags", "+faststart", "-y",
                    ]
                    aligned_candidate = Path(temporary) / "aligned-single-pass.mp4"
                    self._run([*aligned_command, str(aligned_candidate)])
                    final_media = self.probe_media(str(aligned_candidate))
                    audio_duration = float(final_media.get("audio_duration") or 0.0)
                    video_duration = float(final_media.get("video_duration") or 0.0)
                    drift = abs(audio_duration - video_duration)
                    if (
                        audio_duration > 0
                        and video_duration > 0
                        and math.isfinite(drift)
                        and drift <= MAX_EDITORIAL_AUDIO_VIDEO_DRIFT_SECONDS
                    ):
                        os.replace(aligned_candidate, output_path)
                        return

                intermediate = Path(temporary) / "paced.mov"
                command.extend(["-c:a", "pcm_s16le", "-y", str(intermediate)])
                self._run(command)
                candidate = Path(temporary) / "aligned.mp4"
                self._write_aligned_editorial_audio(intermediate, candidate)
                os.replace(candidate, output_path)
        else:
            command.extend(["-movflags", "+faststart", "-y", output_path])
            self._run(command)

    def apply_editorial_pacing(
        self,
        video_path: str,
        output_path: str,
        segments: list[dict[str, object]],
    ) -> str:
        """Retiming a complete, time-coded editorial plan and keeping audio aligned."""
        parsed = self._parse_editorial_segments(segments)

        self._ensure_binaries()
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Input video does not exist: {video_path}")
        self._ensure_parent(output_path)
        if os.path.abspath(video_path) == os.path.abspath(output_path):
            raise ValueError("output_path must be different from video_path")

        probe = self.probe_media(video_path)
        duration = float(probe.get("video_duration") or probe.get("duration") or 0.0)
        frame_rate = float(probe.get("frame_rate") or 0.0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("input video duration must be available and greater than zero")
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("input video frame rate must be available and greater than zero")
        self._validate_editorial_segments_against_media(parsed, duration, frame_rate)
        if all(abs(segment["speed_factor"] - 1.0) < 1e-9 for segment in parsed):
            shutil.copy2(video_path, output_path)
            return output_path

        has_audio = bool(probe.get("has_audio"))
        filter_complex = self._build_editorial_filter(parsed, frame_rate, has_audio)
        predicted_duration = (
            self._predict_editorial_video_duration(parsed, frame_rate)
            if has_audio
            else None
        )
        self._render_editorial_output(
            video_path,
            output_path,
            filter_complex,
            has_audio,
            predicted_duration,
        )
        return output_path

    def normalize_video_canvas(
        self,
        video_path: str,
        output_path: str,
        *,
        target_width: int,
        target_height: int,
        background: str = "#15151f",
    ) -> str:
        """Fit a video inside an exact canvas while preserving its aspect ratio and FPS."""

        self._ensure_binaries()
        width = int(target_width)
        height = int(target_height)
        if width <= 0 or height <= 0:
            raise ValueError("target_width and target_height must be greater than zero")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?", str(background)):
            raise ValueError("background must be a six- or eight-digit hex color")
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Input video does not exist: {video_path}")
        self._ensure_parent(output_path)
        if os.path.abspath(video_path) == os.path.abspath(output_path):
            raise ValueError("output_path must be different from video_path")

        probe = self.probe_media(video_path)
        duration = float(probe.get("video_duration") or probe.get("duration") or 0.0)
        if not math.isfinite(duration) or duration <= 0:
            duration = 0.0
        audio_codec = str(probe.get("audio_codec") or "").lower()
        remux_compatible = (
            Path(output_path).suffix.lower() in {".mp4", ".mov", ".m4v"}
            and int(probe.get("width") or 0) == width
            and int(probe.get("height") or 0) == height
            and str(probe.get("video_codec") or "").lower() == "h264"
            and str(probe.get("pixel_format") or "").lower() == "yuv420p"
            and str(probe.get("sample_aspect_ratio") or "") in {"", "1:1"}
            and not probe.get("has_rotation_metadata")
        )

        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
        if remux_compatible:
            command.extend(["-c:v", "copy"])
        else:
            command.extend(
                [
                    "-vf",
                    (
                        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={background},"
                        "setsar=1,format=yuv420p"
                    ),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        if probe.get("has_audio"):
            # Preserve AAC packet timing; a second encode can push its endpoint past video.
            command.extend(["-c:a", "copy" if audio_codec == "aac" else "aac"])
        command.extend(["-movflags", "+faststart"])
        if duration:
            command.extend(["-t", f"{duration:.12g}"])
        command.extend(["-y", output_path])
        self._run(command)
        return output_path

    def overlay_png_sequence(
        self,
        *,
        video_path: str,
        frame_pattern: str,
        output_path: str,
        fps: float,
        frame_count: int,
    ) -> str:
        """Composite an RGBA PNG sequence over a video while copying its audio stream."""

        self._ensure_binaries()
        frame_rate = float(fps)
        count = int(frame_count)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("fps must be finite and greater than zero")
        if count <= 0:
            raise ValueError("frame_count must be greater than zero")
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Input video does not exist: {video_path}")
        if not Path(frame_pattern).parent.is_dir():
            raise FileNotFoundError(f"PNG sequence directory does not exist: {Path(frame_pattern).parent}")
        self._ensure_parent(output_path)
        if os.path.abspath(video_path) == os.path.abspath(output_path):
            raise ValueError("output_path must be different from video_path")
        self._run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                f"{frame_rate:.12g}",
                "-start_number",
                "0",
                "-i",
                frame_pattern,
                "-i",
                video_path,
                "-filter_complex",
                "[1:v:0][0:v:0]overlay=eof_action=pass:shortest=1:format=auto[v]",
                "-map",
                "[v]",
                "-map",
                "1:a?",
                "-frames:v",
                str(count),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-vsync",
                "cfr",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                "-y",
                output_path,
            ]
        )
        return output_path

    def overlay_rgba_frames(
        self,
        *,
        video_path: str,
        frames: Iterable[bytes],
        output_path: str,
        width: int,
        height: int,
        fps: float,
        frame_count: int,
    ) -> str:
        """Composite streamed RGBA frames over a video while copying its audio."""
        self._ensure_binaries()
        frame_rate = float(fps)
        count = int(frame_count)
        canvas_width = int(width)
        canvas_height = int(height)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("fps must be finite and greater than zero")
        if count <= 0 or canvas_width <= 0 or canvas_height <= 0:
            raise ValueError("frame_count and RGBA canvas dimensions must be positive")
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Input video does not exist: {video_path}")
        self._ensure_parent(output_path)
        if os.path.abspath(video_path) == os.path.abspath(output_path):
            raise ValueError("output_path must be different from video_path")
        destination = Path(output_path)
        temporary_output = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}{destination.suffix}"
        )
        command = self._rgba_overlay_command(
            video_path,
            str(temporary_output),
            canvas_width,
            canvas_height,
            frame_rate,
            count,
        )
        try:
            self._run_with_rgba_frames(
                command,
                frames,
                canvas_width * canvas_height * 4,
                count,
                str(temporary_output),
            )
            os.replace(temporary_output, destination)
        finally:
            temporary_output.unlink(missing_ok=True)
        return output_path

    @staticmethod
    def _rgba_overlay_command(
        video_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        frame_count: int,
    ) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
            "-i",
            video_path,
            "-filter_complex",
            "[1:v:0][0:v:0]overlay=eof_action=pass:shortest=1:format=auto[v]",
            "-map",
            "[v]",
            "-map",
            "1:a?",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "cfr",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-y",
            output_path,
        ]

    @staticmethod
    def _run_with_rgba_frames(
        command: list[str],
        frames: Iterable[bytes],
        bytes_per_frame: int,
        frame_count: int,
        output_path: str,
    ) -> None:
        joined_command = " ".join(command)
        with tempfile.TemporaryFile() as stderr_file:
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    bufsize=0,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("ffmpeg is not installed or not available in PATH.") from exc
            if process.stdin is None:
                process.kill()
                process.wait()
                Path(output_path).unlink(missing_ok=True)
                raise RuntimeError("FFmpeg raw video input pipe is unavailable")

            writer_result: list[tuple[int, Exception | None]] = []
            writer = threading.Thread(
                target=FFmpegAdapter._write_rgba_frame_stream,
                args=(process.stdin, frames, bytes_per_frame, frame_count, writer_result),
                name="ffmpeg-rgba-frame-writer",
                daemon=True,
            )
            writer.start()
            try:
                return_code = process.wait(timeout=FFMPEG_COMMAND_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                FFmpegAdapter._stop_rgba_stream(process, writer)
                Path(output_path).unlink(missing_ok=True)
                raise RuntimeError(
                    f"Command timed out after {FFMPEG_COMMAND_TIMEOUT_SECONDS} seconds: {joined_command}"
                ) from exc
            except BaseException:
                FFmpegAdapter._stop_rgba_stream(process, writer)
                Path(output_path).unlink(missing_ok=True)
                raise
            writer.join()
            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace").strip()

        if return_code != 0:
            Path(output_path).unlink(missing_ok=True)
            raise RuntimeError(f"Command failed: {joined_command}\n{stderr}")
        if not writer_result:
            Path(output_path).unlink(missing_ok=True)
            raise RuntimeError("RGBA frame writer did not report a result")
        written_frames, writer_error = writer_result[0]
        if writer_error is not None:
            Path(output_path).unlink(missing_ok=True)
            raise RuntimeError(f"Could not stream RGBA frames to FFmpeg: {writer_error}") from writer_error
        if written_frames != frame_count:
            Path(output_path).unlink(missing_ok=True)
            raise RuntimeError(f"RGBA frame writer streamed {written_frames} frames; expected {frame_count}")

    @staticmethod
    def _write_rgba_frame_stream(
        stream: IO[bytes],
        frames: Iterable[bytes],
        bytes_per_frame: int,
        expected_count: int,
        result: list[tuple[int, Exception | None]],
    ) -> None:
        written_count = 0
        try:
            for frame in frames:
                if written_count >= expected_count:
                    raise ValueError("RGBA frame input contained more frames than frame_count")
                if not isinstance(frame, bytes) or len(frame) != bytes_per_frame:
                    raise ValueError(f"RGBA frames must be bytes with exactly {bytes_per_frame} bytes")
                remaining = memoryview(frame)
                while remaining:
                    written = stream.write(remaining)
                    if written is None or written <= 0:
                        raise BrokenPipeError("FFmpeg stopped accepting RGBA frame data")
                    remaining = remaining[written:]
                written_count += 1
            if written_count != expected_count:
                raise ValueError(f"RGBA frame input contained {written_count} frames; expected {expected_count}")
            result.append((written_count, None))
        except Exception as exc:
            result.append((written_count, exc))
        finally:
            try:
                stream.close()
            except OSError:
                pass

    @staticmethod
    def _stop_rgba_stream(process: subprocess.Popen[bytes], writer: threading.Thread) -> None:
        if process.poll() is None:
            process.kill()
        process.wait()
        writer.join()

    def trim_video(
        self,
        video_path: str,
        output_path: str,
        duration_seconds: float,
        *,
        normalize_audio: bool = False,
    ) -> str:
        """Trim a packaged video to an explicit duration while keeping its streams."""

        self._ensure_binaries()
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_seconds must be a finite number greater than zero")
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Input video does not exist: {video_path}")
        self._ensure_parent(output_path)
        if os.path.abspath(video_path) == os.path.abspath(output_path):
            raise ValueError("output_path must be different from video_path")
        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-t",
            f"{duration:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
        ]
        if normalize_audio:
            command.extend(
                [
                    "-af",
                    "aresample=48000,loudnorm=I=-20:TP=-1.5:LRA=11",
                    "-ar",
                    "48000",
                ]
            )
        command.extend(
            [
                "-movflags",
                "+faststart",
                "-shortest",
                "-y",
                output_path,
            ]
        )
        self._run(command)
        return output_path

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
                    "format=yuv420p,fps=30"
                ),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "128k",
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

    @staticmethod
    def _atempo_filters(speed: float) -> list[str]:
        """Build an atempo chain that also works outside FFmpeg's 0.5-2 range."""

        remaining = speed
        filters: list[str] = []
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.12g}")
        return filters

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
                timeout=FFMPEG_COMMAND_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            binary = command[0] if command else "ffmpeg"
            raise RuntimeError(f"{binary} is not installed or not available in PATH.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            joined = " ".join(command)
            raise RuntimeError(f"Command failed: {joined}\n{stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            joined = " ".join(command)
            raise RuntimeError(f"Command timed out after {FFMPEG_COMMAND_TIMEOUT_SECONDS} seconds: {joined}") from exc

    @staticmethod
    def _run_capture(command: list[str]) -> str:
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
                timeout=FFMPEG_COMMAND_TIMEOUT_SECONDS,
            )
            return completed.stdout
        except FileNotFoundError as exc:
            binary = command[0] if command else "ffprobe"
            raise RuntimeError(f"{binary} is not installed or not available in PATH.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            joined = " ".join(command)
            raise RuntimeError(f"Command failed: {joined}\n{stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            joined = " ".join(command)
            raise RuntimeError(f"Command timed out after {FFMPEG_COMMAND_TIMEOUT_SECONDS} seconds: {joined}") from exc
