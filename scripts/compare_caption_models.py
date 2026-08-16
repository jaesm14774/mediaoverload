from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.llm_manager_adapter import static_openrouter_model_modes
from agentic.runtime.model_backends import (
    AgenticLLMManager,
    ModelConfig,
    OpenRouterModelCatalog,
    OpenRouterRotatingModel,
    static_openrouter_models,
)
from agentic.runtime.observability import RunRecorder
from agentic.tools.context_services import DiscordRunNotificationService

MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm", ".mkv", ".m4v"}
TEXT_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
FILLER_PATTERNS = (
    r"in a world where",
    r"dive into",
    r"seamlessly",
    r"captivating",
    r"vibrant",
    r"magical",
    r"epic",
    r"unleash",
    r"witness",
    r"journey",
    r"bring(?:ing)? .* to life",
    r"testament to",
    r"unlock(?:s|ing)?",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OpenRouter caption models on generated media.")
    parser.add_argument("--media", action="append", dest="media_paths", help="Generated image/video; repeatable")
    parser.add_argument(
        "--media-root",
        default=r"D:\MediaOverload\output",
        help="Search root when --media is omitted",
    )
    parser.add_argument("--model", action="append", dest="models", help="Exact OpenRouter model ID; repeatable")
    parser.add_argument("--max-models", type=int, default=10)
    parser.add_argument(
        "--live-catalog",
        action="store_true",
        help="Use the current OpenRouter free vision/text catalog instead of the checked-in pool",
    )
    parser.add_argument("--modality", choices=("vision", "text"), default="vision")
    parser.add_argument("--style", default="polished 2D anime")
    parser.add_argument("--character", default="Kirby")
    parser.add_argument("--prompt", default="", help="Override the rendered story prompt for every media item")
    parser.add_argument("--hashtag", action="append", dest="hashtags")
    parser.add_argument("--platform", action="append", dest="platforms")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-model HTTP read timeout in seconds")
    parser.add_argument("--output-root", default=r"D:\MediaOverload\caption_compare")
    parser.add_argument("--no-discord", action="store_true", help="Only write the comparison report")
    return parser.parse_args()


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")[:100] or "model"


def _discover_media(root: Path) -> list[str]:
    if not root.exists():
        return []
    found = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS]
    latest_by_kind: dict[str, Path] = {}
    for path in sorted(found, key=lambda item: item.stat().st_mtime, reverse=True):
        kind = "video" if path.suffix.lower() in TEXT_EXTENSIONS else "image"
        latest_by_kind.setdefault(kind, path)
    return [str(path) for path in latest_by_kind.values()]


def _load_render_prompt(media_path: Path, override: str) -> str:
    if override.strip():
        return override.strip()
    summary_path = media_path.with_name(f"{media_path.stem}_summary.json")
    candidates = [summary_path]
    if media_path.suffix.lower() in TEXT_EXTENSIONS:
        candidates.append(media_path.parent / "agentic_i2v_summary.json")
    else:
        candidates.append(media_path.parent / "agentic_image_summary.json")
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            nested = payload.get("payload") if isinstance(payload, dict) else {}
            prompt = nested.get("prompt") if isinstance(nested, dict) else ""
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
        except (OSError, ValueError, TypeError):
            continue
    return f"Create a concise social post for the attached generated {media_path.suffix.lower().lstrip('.')} featuring {media_path.stem}."


def _ffmpeg_path() -> str:
    configured = os.environ.get("FFMPEG_BINARY", "").strip()
    candidates = [
        configured,
        shutil.which("ffmpeg") or "",
        r"D:\ComfyUI_windows_portable\python_embeded\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return ""


def _visual_evidence(media_path: Path, output_dir: Path) -> list[str]:
    if media_path.suffix.lower() not in TEXT_EXTENSIONS:
        return [str(media_path)]
    contact_sheet = media_path.with_name(f"{media_path.stem}_contact_sheet.jpg")
    if contact_sheet.exists():
        return [str(contact_sheet)]
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError(f"Cannot inspect video; ffmpeg was not found for {media_path}.")
    frame_dir = output_dir / "visual_evidence" / _safe_slug(media_path.stem)
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[str] = []
    for index, timestamp in enumerate((0, 5, 10), start=1):
        frame_path = frame_dir / f"frame_{index:02d}.jpg"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(media_path),
            "-frames:v",
            "1",
            str(frame_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and frame_path.exists():
            frames.append(str(frame_path))
    if not frames:
        raise RuntimeError(f"ffmpeg could not extract visual evidence from {media_path}.")
    return frames


def _quality_flags(caption: str, hashtags: str) -> dict[str, Any]:
    normalized = " ".join(str(caption or "").split())
    lowered = normalized.lower()
    filler_matches = [pattern for pattern in FILLER_PATTERNS if re.search(pattern, lowered)]
    words = re.findall(r"[A-Za-z0-9']+|[\u4e00-\u9fff]", normalized)
    tokens = [token.lower() for token in words]
    repeated_tokens = sorted({token for token in tokens if len(token) > 4 and tokens.count(token) >= 3})
    encoding_suspect = any(
        marker in normalized or marker in str(hashtags or "")
        for marker in ("\ufffd", "??", "嚙")
    )
    return {
        "character_count": len(normalized),
        "word_count": len(words),
        "hashtag_count": len([token for token in str(hashtags or "").split() if token.startswith("#")]),
        "filler_matches": filler_matches,
        "repeated_tokens": repeated_tokens,
        "encoding_suspect": encoding_suspect,
        "overlong": len(normalized) > 240,
        "needs_prompt_review": bool(
            filler_matches or repeated_tokens or encoding_suspect or len(normalized) > 240
        ),
    }


def _caption_request(
    *,
    model_id: str,
    model_mode: str,
    goal: GoalRequest,
    media_paths: list[str],
    visual_paths: list[str],
    platforms: list[str],
    hashtags: list[str],
    recorder: RunRecorder,
) -> dict[str, Any]:
    model = OpenRouterRotatingModel(
        ModelConfig(model_name=model_id, temperature=0.3),
        [model_id],
        model_modes={model_id: model_mode},
        random_each_call=False,
    )
    manager = AgenticLLMManager(text_model=model, vision_model=model)
    engine = LLMPromptEngine(mode="llm", manager=manager, recorder=recorder)
    started = time.perf_counter()
    try:
        bundle = engine.prepare_publish_caption(
            goal,
            prefix="",
            hashtags=hashtags,
            platforms=platforms,
            media_paths=media_paths,
            review_notes="Write only the final social copy. Avoid generic hype, scene-description padding, and repeated adjectives.",
            visual_paths=visual_paths,
        )
        caption = str(bundle.get("caption") or "").strip()
        hashtags_text = str(bundle.get("hashtags") or "").strip()
        return {
            "status": "success",
            "model_id": str(bundle.get("llm_model") or model.last_success_model or model_id),
            "caption": caption,
            "hashtags": hashtags_text,
            "platform_captions": bundle.get("platform_captions", {}),
            "metrics": _quality_flags(caption, hashtags_text),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "model_id": str(model.last_attempt_model or model_id),
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": {},
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }


def _discord_text(result: dict[str, Any]) -> str:
    model_id = str(result.get("model_id") or "unknown")
    lines = [model_id]
    if result.get("status") == "success":
        lines.extend(["", str(result.get("caption", "")).strip(), str(result.get("hashtags", "")).strip()])
    else:
        lines.extend(["", "No usable post was generated for this model."])
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    os.environ["AGENTIC_PUBLISH_CAPTION_TIMEOUT_SECONDS"] = str(max(1.0, args.timeout))
    output_root = Path(args.output_root).resolve()
    comparison_id = datetime.now(timezone.utc).strftime("caption_compare_%Y%m%dT%H%M%SZ")
    comparison_dir = output_root / comparison_id
    comparison_dir.mkdir(parents=True, exist_ok=True)

    media_paths = [str(Path(path).resolve()) for path in (args.media_paths or [])]
    if not media_paths:
        media_paths = _discover_media(Path(args.media_root).resolve())
    media_paths = [path for path in media_paths if Path(path).exists() and Path(path).suffix.lower() in MEDIA_EXTENSIONS]
    if not media_paths:
        raise SystemExit("No generated image/video media found.")

    if args.models:
        models = list(args.models)
    elif args.live_catalog:
        models = OpenRouterModelCatalog.candidates(args.modality, limit=max(1, args.max_models), force_refresh=True)
    else:
        models = static_openrouter_models(args.modality)
    models = models[: max(1, args.max_models)]
    modes = static_openrouter_model_modes(args.modality)
    platforms = list(args.platforms or ["instagram_graph", "facebook", "youtube"])
    # Do not seed model comparison with project or brand tags. The caption
    # model must select hashtags from the article and the attached media;
    # callers can still provide explicit, content-relevant hints via --hashtag.
    hashtags = list(args.hashtags or [])
    report: dict[str, Any] = {
        "comparison_id": comparison_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "media_paths": media_paths,
        "models": models,
        "timeout_seconds": max(1.0, args.timeout),
        "results": [],
    }

    for media_path in media_paths:
        prompt = _load_render_prompt(Path(media_path), args.prompt)
        visual_paths = _visual_evidence(Path(media_path), comparison_dir)
        caption_goal = GoalRequest(
            prompt=(
                "Use the attached generated media as the only evidence. "
                "Describe what is visibly present; do not infer or reuse a hidden production prompt."
            ),
            media_type="publish_review",
            style=args.style,
            constraints={"character": args.character, "visual_grounding": {}, "hashtags": hashtags},
        )
        media_result: dict[str, Any] = {
            "media_path": media_path,
            "visual_paths": visual_paths,
            "prompt": prompt,
            "models": [],
        }
        for model_id in models:
            model_dir = comparison_dir / _safe_slug(Path(media_path).stem) / _safe_slug(model_id)
            recorder = RunRecorder(model_dir, "run")
            result = _caption_request(
                model_id=model_id,
                model_mode=modes.get(model_id, "structured"),
                goal=caption_goal,
                media_paths=[media_path],
                visual_paths=visual_paths,
                platforms=platforms,
                hashtags=hashtags,
                recorder=recorder,
            )
            result["requested_model_id"] = model_id
            media_result["models"].append(result)
        report["results"].append(media_result)

    report_path = comparison_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    discord = {"status": "skipped", "messages": []}
    if not args.no_discord:
        service = DiscordRunNotificationService()
        discord["messages"] = []
        for media_result in report["results"]:
            media_path = str(media_result["media_path"])
            models_for_media = list(media_result["models"])
            for index, result in enumerate(models_for_media, start=1):
                receipt = service.notify(
                    _discord_text(result),
                    media_paths=[media_path] if index == 1 else [],
                )
                discord["messages"].append({"media_path": media_path, "model_id": result.get("model_id"), "receipt": receipt})
        discord["status"] = "sent" if any(item["receipt"].get("status") == "sent" for item in discord["messages"]) else "failed"
    report["discord"] = discord
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"comparison_id": comparison_id, "report": str(report_path), "discord": discord}, ensure_ascii=False, indent=2))
    return 0 if any(result.get("status") == "success" for item in report["results"] for result in item["models"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
