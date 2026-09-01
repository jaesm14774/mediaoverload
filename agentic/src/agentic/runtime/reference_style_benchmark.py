from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from agentic.assets.registry import AssetRegistry
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.observability import RunRecorder
from agentic.tools.comfy_workflow_tool import ComfyWorkflowToolset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
SCREENSHOT_MARKERS = ("螢幕擷取畫面", "screenshot", "screen_capture")
MAX_IMAGE_BYTES = 250 * 1024 * 1024
MAX_VIDEO_BYTES = 1 * 1024 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_VIDEO_SECONDS = 30 * 60


@dataclass(frozen=True, slots=True)
class ReferenceStyleBenchmarkConfig:
    repo_root: Path
    collection_root: Path
    output_root: Path
    logs_root: Path
    config_path: Path
    max_attempts: int = 5
    score_threshold: int = 80
    seed_base: int = 20260830
    width: int = 1024
    height: int = 576
    steps: int = 8
    limit: int = 10
    use_img2img_rescue: bool = True
    seed_probe: bool = False
    execute: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_slug(value: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.casefold()).strip("-._")
    return slug[:80] or fallback


def _is_screenshot(path: Path) -> bool:
    name = path.name.casefold()
    return any(marker.casefold() in name for marker in SCREENSHOT_MARKERS)


def _probe_video_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    duration = max(0.1, float(completed.stdout.strip() or 0.1))
    if duration > MAX_VIDEO_SECONDS:
        raise ValueError(f"video exceeds the {MAX_VIDEO_SECONDS}s benchmark limit: {path}")
    return duration


def _extract_video_midframe(path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _probe_video_duration(path) / 2
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            "scale=768:-2",
            "-q:v",
            "3",
            str(output_path),
        ],
        check=True,
        timeout=90,
    )
    if not output_path.is_file():
        raise RuntimeError(f"ffmpeg did not produce {output_path}")
    return output_path


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _write_contact_sheet(paths: list[Path], output_path: Path, *, columns: int = 5) -> Path:
    if not paths:
        raise ValueError("cannot write a contact sheet without images")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h, label_h = 300, 250, 30
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), (247, 242, 232))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, path in enumerate(paths):
        x = (index % columns) * cell_w
        y = (index // columns) * (cell_h + label_h)
        try:
            with Image.open(path) as source:
                image = ImageOps.contain(source.convert("RGB"), (cell_w - 18, cell_h - 18))
            sheet.paste(image, (x + (cell_w - image.width) // 2, y + (cell_h - image.height) // 2))
        except Exception as exc:
            draw.text((x + 8, y + 8), f"ERROR: {exc}", fill=(180, 30, 30), font=font)
        draw.rectangle((x, y + cell_h, x + cell_w, y + cell_h + label_h), fill=(34, 34, 42))
        draw.text((x + 8, y + cell_h + 8), path.name[:42], fill=(250, 245, 232), font=font)
    sheet.save(output_path, quality=90)
    return output_path


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"benchmark config must be an object: {path}")
    return loaded


def collect_reference_items(collection_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    if not collection_root.is_dir():
        raise FileNotFoundError(f"collection root does not exist: {collection_root}")
    root = collection_root.resolve(strict=True)
    images: list[Path] = []
    videos: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            size = resolved.stat().st_size
        except (OSError, RuntimeError, ValueError):
            continue
        suffix = resolved.suffix.casefold()
        if suffix in IMAGE_SUFFIXES and size <= MAX_IMAGE_BYTES:
            images.append(resolved)
        elif suffix in VIDEO_SUFFIXES and size <= MAX_VIDEO_BYTES:
            videos.append(resolved)
    items: list[dict[str, Any]] = []
    for index, path in enumerate(images, start=1):
        width, height = _image_size(path)
        if width * height > MAX_IMAGE_PIXELS:
            continue
        items.append(
            {
                "item_id": f"image-{index:03d}-{_safe_slug(path.stem)}",
                "source_path": str(path.resolve()),
                "source_type": "image",
                "source_name": path.name,
                "source_sha256": _sha256(path),
                "width": width,
                "height": height,
                "likely_screenshot": _is_screenshot(path),
                "img2img_eligible": not _is_screenshot(path),
            }
        )
    return items, videos


def select_reference_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(items):
        return list(items)
    if limit == 1:
        return [items[0]]
    indices = sorted({round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)})
    return [items[index] for index in indices]


def stable_seed(item_id: str, seed_base: int) -> int:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % 900_000_000
    return max(1, min(2_147_483_646, int(seed_base) + offset))


def effective_k_sampler_seed(workflow_path: Path, requested_seed: int) -> int:
    graph = json.loads(workflow_path.read_text(encoding="utf-8"))
    sampler_nodes = [node_id for node_id, node in graph.items() if node.get("class_type") == "KSampler"]
    if not sampler_nodes:
        return int(requested_seed)
    # AgenticNodeManager indexes matching nodes from zero before applying
    # seed + node_index; the Comfy node ID is not part of the seed formula.
    return int(requested_seed) + sampler_nodes.index(sampler_nodes[0])


class ReferenceStyleBenchmark:
    def __init__(self, config: ReferenceStyleBenchmarkConfig) -> None:
        self.config = config
        self.config_data = _load_config(config.config_path)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + hashlib.sha1(str(config.collection_root).encode("utf-8")).hexdigest()[:8]
        self.run_output = config.output_root / self.run_id
        self.run_output.mkdir(parents=True, exist_ok=True)
        self.recorder = RunRecorder(config.logs_root / "runs", self.run_id)
        self.engine = LLMPromptEngine(mode="llm", recorder=self.recorder)
        asset_registry = AssetRegistry(config.repo_root / "agentic", asset_root=config.repo_root)
        self.comfy = ComfyWorkflowToolset(
            asset_registry=asset_registry,
            output_root=self.run_output,
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        self.style_analysis: dict[str, Any] = {}
        model_config = self.config_data.get("model", {})
        model_config = model_config if isinstance(model_config, dict) else {}
        self.image_workflow_name = str(model_config.get("workflow") or "krea2_turbo")
        self.refine_workflow_name = str(model_config.get("refine_workflow") or "krea2_turbo_img2img")
        benchmark_config = self.config_data.get("benchmark", {})
        benchmark_config = benchmark_config if isinstance(benchmark_config, dict) else {}
        configured_weights = benchmark_config.get("image_score_weights", {})
        configured_weights = configured_weights if isinstance(configured_weights, dict) else {}
        self.score_weights = {
            str(key): int(value)
            for key, value in configured_weights.items()
            if str(key).strip()
        }

    def _allow_external_images(self) -> None:
        allowed = [self.config.collection_root.resolve(), self.run_output.resolve(), self.config.repo_root.resolve()]
        existing = [Path(item.strip()).resolve() for item in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",") if item.strip()]
        merged: list[Path] = []
        for path in [*existing, *allowed]:
            if path not in merged:
                merged.append(path)
        os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(str(path) for path in merged)

    def prepare_references(self, items: list[dict[str, Any]], videos: list[Path]) -> dict[str, Any]:
        reference_root = self.run_output / "references"
        reference_root.mkdir(parents=True, exist_ok=True)
        video_frames: list[Path] = []
        for index, video in enumerate(videos, start=1):
            frame_path = reference_root / "video_frames" / f"video_{index:02d}_{_safe_slug(video.stem)}.jpg"
            try:
                video_frames.append(_extract_video_midframe(video, frame_path))
            except Exception as exc:
                self.recorder.record_event("reference.video_frame_failed", source=str(video), error=f"{type(exc).__name__}: {exc}")
        clean_images = [Path(item["source_path"]) for item in items if not item["likely_screenshot"]]
        board_paths = [*clean_images, *video_frames]
        style_board = _write_contact_sheet(board_paths, reference_root / "style_board.jpg", columns=5)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collection_root": str(self.config.collection_root),
            "images_discovered": len(items),
            "videos_discovered": len(videos),
            "screenshot_like_images": sum(bool(item["likely_screenshot"]) for item in items),
            "items": items,
            "video_midframes": [str(path) for path in video_frames],
            "style_board_path": str(style_board),
        }
        (reference_root / "reference_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self.recorder.record_event("reference.prepared", manifest_path=str(reference_root / "reference_manifest.json"))
        return manifest

    def analyze_style(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self._allow_external_images()
        clean_paths = [str(item["source_path"]) for item in manifest["items"] if not item["likely_screenshot"]]
        selected = [manifest["style_board_path"], *clean_paths[:4], *manifest.get("video_midframes", [])[:2]]
        self.style_analysis = self.engine.analyze_reference_style(reference_images=selected, reference_kind="images_plus_video_midframes")
        path = self.run_output / "style_analysis.json"
        path.write_text(json.dumps(self.style_analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.style_analysis

    def _render_attempt(self, item: dict[str, Any], attempt_dir: Path, prompt_data: dict[str, Any], requested_seed: int, use_img2img: bool) -> dict[str, Any]:
        workflow_name = self.refine_workflow_name if use_img2img else self.image_workflow_name
        prompt = str(prompt_data.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("cannot render an empty reference-style prompt")
        workflow_path = self.config.repo_root / "configs" / "workflow" / f"{workflow_name}.json"
        if not workflow_path.is_file():
            raise FileNotFoundError(f"configured Krea2 workflow does not exist: {workflow_path}")
        payload: dict[str, Any] = {
            "workflow_name": workflow_name,
            "run_dir": str(attempt_dir),
            "prompt": prompt,
            "negative_prompt": prompt_data["negative_prompt"],
            "seed": requested_seed,
            "width": self.config.width,
            "height": self.config.height,
            "steps": self.config.steps,
            "image_count": 1,
        }
        if use_img2img:
            payload["image_path"] = item["source_path"]
        result = self.comfy.execute(
            "comfy.workflow.image_to_image" if use_img2img else "comfy.workflow.text_to_image",
            payload,
        )
        saved_files = [str(path) for path in result.get("saved_files", []) if str(path).strip()]
        if not saved_files:
            raise RuntimeError(f"Krea2 returned no image for {workflow_name}")
        candidate = Path(saved_files[0]).resolve()
        return {
            "workflow_name": workflow_name,
            "candidate_path": str(candidate),
            "render_result": result,
            "requested_seed": requested_seed,
            "effective_seed": effective_k_sampler_seed(workflow_path, requested_seed),
            "negative_prompt_requested": str(prompt_data.get("negative_prompt") or "").strip(),
            "negative_prompt_applied": False,
        }

    def _write_attempt(self, item_dir: Path, attempt: int, prompt_data: dict[str, Any], render: dict[str, Any], review: dict[str, Any] | None, error: str = "") -> None:
        attempt_dir = item_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "prompt.json").write_text(json.dumps(prompt_data, indent=2, ensure_ascii=False), encoding="utf-8")
        if review is not None:
            (attempt_dir / "review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        record = {"attempt": attempt, "render": render, "review": review, "error": error}
        (attempt_dir / "attempt.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    def _run_seed_probe(self, item: dict[str, Any], item_dir: Path, winner: dict[str, Any]) -> dict[str, Any]:
        prompt_data = dict(winner.get("prompt") or {})
        render = winner.get("render") or {}
        workflow_name = str(render.get("workflow_name") or self.image_workflow_name)
        use_img2img = workflow_name == self.refine_workflow_name
        requested_seed = int(render.get("requested_seed") or stable_seed(item["item_id"], self.config.seed_base))
        probe_renders: list[dict[str, Any]] = []
        errors: list[str] = []
        for index in (1, 2):
            try:
                probe_renders.append(
                    self._render_attempt(
                        item,
                        item_dir / f"seed_probe_{index:02d}",
                        prompt_data,
                        requested_seed,
                        use_img2img,
                    )
                )
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        hashes = []
        for render_result in probe_renders:
            candidate_path = Path(str(render_result.get("candidate_path") or ""))
            hashes.append(_sha256(candidate_path) if candidate_path.is_file() else "")
        probe = {
            "enabled": True,
            "workflow_name": workflow_name,
            "requested_seed": requested_seed,
            "effective_seed": render.get("effective_seed"),
            "prompt_sha256": hashlib.sha256(str(prompt_data.get("prompt") or "").encode("utf-8")).hexdigest(),
            "candidate_paths": [str(render_result.get("candidate_path") or "") for render_result in probe_renders],
            "candidate_sha256": hashes,
            "same_bytes": len(hashes) == 2 and bool(hashes[0]) and hashes[0] == hashes[1],
            "errors": errors,
        }
        (item_dir / "seed_probe.json").write_text(json.dumps(probe, indent=2, ensure_ascii=False), encoding="utf-8")
        return probe

    def run_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item_dir = self.run_output / "items" / item["item_id"]
        item_dir.mkdir(parents=True, exist_ok=True)
        requested_seed = stable_seed(item["item_id"], self.config.seed_base)
        previous_prompt = ""
        previous_review: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        winner: dict[str, Any] | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            use_img2img = bool(self.config.use_img2img_rescue and attempt >= 4 and item["img2img_eligible"])
            mode = "img2img_rescue" if use_img2img else "text_to_image"
            prompt_data: dict[str, Any] = {"attempt": attempt, "generation_mode": mode}
            render: dict[str, Any] = {}
            try:
                prompt_data = self.engine.generate_reference_style_prompt(
                    reference_image=item["source_path"],
                    style_analysis=self.style_analysis,
                    attempt=attempt,
                    generation_mode=mode,
                    previous_prompt=previous_prompt,
                    previous_review=previous_review,
                )
                prompt_data["source_item_id"] = item["item_id"]
                prompt_data["requested_seed"] = requested_seed
                render = self._render_attempt(item, item_dir / f"attempt_{attempt:02d}", prompt_data, requested_seed, use_img2img)
                review = self.engine.evaluate_reference_style_match(
                    reference_image=item["source_path"],
                    candidate_image=render["candidate_path"],
                    prompt=prompt_data["prompt"],
                    style_analysis=self.style_analysis,
                    attempt=attempt,
                    threshold=self.config.score_threshold,
                    score_weights=self.score_weights,
                )
                record = {"attempt": attempt, "mode": mode, "prompt": prompt_data, "render": render, "review": review}
                attempts.append(record)
                self._write_attempt(item_dir, attempt, prompt_data, render, review)
                self.recorder.record_node(
                    node_id=f"reference-style-{item['item_id']}",
                    skill_name="krea2.reference_style_benchmark",
                    status="success" if review["passed"] else "failed_quality_gate",
                    attempt=attempt,
                    outputs=render,
                    metrics={"score": review["score"], "passed": review["passed"]},
                    logs=list(review.get("issues", [])),
                )
                previous_prompt = prompt_data["prompt"]
                previous_review = review
                if review["passed"]:
                    winner = record
                    break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempts.append({"attempt": attempt, "mode": mode, "prompt": prompt_data, "render": render, "error": error})
                self._write_attempt(item_dir, attempt, prompt_data, render, None, error=error)
                self.recorder.record_event("reference.item_attempt_failed", item_id=item["item_id"], attempt=attempt, error=error)

        result = {
            "item_id": item["item_id"],
            "source_path": item["source_path"],
            "source_sha256": item["source_sha256"],
            "requested_seed": requested_seed,
            "attempt_count": len(attempts),
            "passed": winner is not None,
            "winner": winner,
            "attempts": attempts,
        }
        (item_dir / "item_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if winner:
            if self.config.seed_probe:
                result["seed_probe"] = self._run_seed_probe(item, item_dir, winner)
                (item_dir / "item_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            self._append_success_prompt(result)
        return result

    def _append_success_prompt(self, result: dict[str, Any]) -> None:
        winner = result.get("winner") or {}
        prompt_data = winner.get("prompt") or {}
        library_path = self.config.repo_root / "configs" / "krea2_reference_style_prompts.jsonl"
        source_path = Path(result["source_path"]).resolve()
        try:
            source_relative_path = source_path.relative_to(self.config.collection_root.resolve()).as_posix()
        except ValueError:
            source_relative_path = source_path.name
        candidate_path = Path(str((winner.get("render") or {}).get("candidate_path") or "")).resolve()
        try:
            candidate_relative_path = candidate_path.relative_to(self.run_output.resolve()).as_posix()
        except ValueError:
            candidate_relative_path = candidate_path.name
        entry = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "item_id": result["item_id"],
            "source_name": source_path.name,
            "source_relative_path": source_relative_path,
            "source_sha256": result["source_sha256"],
            "prompt": prompt_data.get("prompt", ""),
            "negative_prompt": prompt_data.get("negative_prompt", ""),
            "creative_intent": prompt_data.get("creative_intent", ""),
            "generation_mode": prompt_data.get("generation_mode", ""),
            "requested_seed": result["requested_seed"],
            "effective_seed": (winner.get("render") or {}).get("effective_seed"),
            "negative_prompt_applied": (winner.get("render") or {}).get("negative_prompt_applied", False),
            "score": (winner.get("review") or {}).get("score"),
            "candidate_relative_path": candidate_relative_path,
        }
        with library_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def run(self) -> dict[str, Any]:
        try:
            report = self._run()
        except Exception as exc:
            self.recorder.finalize(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "output_root": str(self.run_output),
                }
            )
            raise
        self.recorder.finalize(
            {
                "status": "completed" if self.config.execute else "planned",
                "report_path": str(self.run_output / "benchmark_report.json"),
                "selected_count": report["selected_item_count"],
                "passed_count": report["passed_count"],
                "acceptance_passed": report["acceptance_passed"],
            }
        )
        return report

    def _run(self) -> dict[str, Any]:
        self._allow_external_images()
        items, videos = collect_reference_items(self.config.collection_root)
        selected_items = select_reference_items(items, self.config.limit)
        manifest = self.prepare_references(items, videos)
        self.analyze_style(manifest)
        results: list[dict[str, Any]] = []
        if self.config.execute:
            for item in selected_items:
                results.append(self.run_item(item))
        report = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collection_root": str(self.config.collection_root),
            "selected_item_count": len(selected_items),
            "discovered_image_count": len(items),
            "discovered_video_count": len(videos),
            "score_threshold": self.config.score_threshold,
            "max_attempts": self.config.max_attempts,
            "score_weights": self.score_weights,
            "model_recipe": {
                "workflow": self.image_workflow_name,
                "refine_workflow": self.refine_workflow_name,
                "width": self.config.width,
                "height": self.config.height,
                "steps": self.config.steps,
                "negative_prompt_applied": False,
            },
            "style_analysis": self.style_analysis,
            "selected_items": [item["item_id"] for item in selected_items],
            "results": results,
            "passed_count": sum(bool(item.get("passed")) for item in results),
            "acceptance_passed": bool(results) and sum(bool(item.get("passed")) for item in results) >= math.ceil(len(results) * 0.8),
            "llm_backend": self.engine.backend_info(),
            "output_root": str(self.run_output),
            "logs_run_root": str(self.recorder.run_dir),
        }
        report_path = self.run_output / "benchmark_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        self.recorder.record_event("reference.benchmark_completed", report_path=str(report_path), passed_count=report["passed_count"], selected_count=len(results))
        return report
