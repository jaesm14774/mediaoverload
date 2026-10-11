"""Character Publishing responsibilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from agentic.app.character_config import MEDIA_EXTENSIONS, VIDEO_EXTENSIONS


def collect_media_paths_from_run_result(run_result: dict[str, Any]) -> list[str]:
    """Return final media artifacts for the publish review boundary.

    Generation nodes intentionally expose intermediate assets (opening/ending
    frames, segment clips, and previews) for downstream workflow steps. Those
    are not publish candidates. Prefer explicit package-node media paths and
    use package-node media when available, then collect the current route's
    explicit media outputs. Images and videos are both valid final media.
    """
    state = dict(run_result.get("state", {}) or {})
    node_outputs = dict(state.get("node_outputs", {}) or {})
    collected: list[str] = []

    def append_media(value: object) -> None:
        values = value if isinstance(value, list) else [value]
        for item in values:
            path = str(item or "")
            if Path(path).suffix.lower() in MEDIA_EXTENSIONS and path not in collected:
                collected.append(path)

    # Optional post-speed motion graphics produce the authoritative final
    # video; the last candidate represents the final render/retry branch.
    motion_graphics_candidates: list[str] = []
    for outputs in node_outputs.values():
        if (
            not isinstance(outputs, dict)
            or outputs.get("motion_graphics_applied") is not True
        ):
            continue
        candidates = outputs.get("final_video_path") or outputs.get("video_path") or []
        if isinstance(candidates, str):
            candidates = [candidates]
        motion_graphics_candidates.extend(
            path
            for path in candidates
            if isinstance(path, str) and Path(path).suffix.lower() in VIDEO_EXTENSIONS
        )
    if motion_graphics_candidates:
        return [motion_graphics_candidates[-1]]

    speed_candidates: list[str] = []
    for outputs in node_outputs.values():
        if not isinstance(outputs, dict) or outputs.get("speed") in {None, ""}:
            continue
        candidates = outputs.get("video_path") or outputs.get("saved_files") or []
        if isinstance(candidates, str):
            candidates = [candidates]
        speed_candidates.extend(
            path
            for path in candidates
            if isinstance(path, str) and Path(path).suffix.lower() in VIDEO_EXTENSIONS
        )
    if speed_candidates:
        return [speed_candidates[-1]]

    # A final transform node is authoritative even when the route has no
    # package node. This prevents raw and normalized copies from both reaching
    # the final Discord review.
    final_video_candidates: list[str] = []
    for outputs in node_outputs.values():
        if not isinstance(outputs, dict):
            continue
        candidates = outputs.get("final_video_path") or []
        if isinstance(candidates, str):
            candidates = [candidates]
        final_video_candidates.extend(
            path
            for path in candidates
            if isinstance(path, str) and Path(path).suffix.lower() in VIDEO_EXTENSIONS
        )
    if final_video_candidates:
        return [final_video_candidates[-1]]

    preferred_package_nodes = (
        "story-card-compose",
        "native-h3-package",
        "package-outputs",
        "collect-longvideo-outputs",
        "collect-text2img2video-outputs",
        "collect-text2video-outputs",
    )
    for node_id in preferred_package_nodes:
        outputs = node_outputs.get(node_id)
        if not isinstance(outputs, dict):
            continue
        # Explicit final paths take precedence over package saved_files. A
        # video package may also expose opening/ending PNGs in saved_files;
        # those are continuity artifacts, not publish candidates.
        if outputs.get("final_video_path"):
            append_media(outputs.get("final_video_path"))
            if collected:
                return collected
        package_keys = (
            "media_paths",
            "final_video_path",
            "video_path",
            "image_path",
            "final_image_path",
        )
        if node_id == "story-card-compose":
            package_keys = (*package_keys, "saved_files")
        for key in package_keys:
            append_media(outputs.get(key))
        if collected:
            return collected

    for outputs in node_outputs.values():
        if not isinstance(outputs, dict):
            continue
        for key in ("video_path", "final_video_path", "saved_files", "media_paths"):
            append_media(outputs.get(key))
    # Once a real video is present, opening frames and GIF previews are
    # continuity artifacts, not publish candidates. If the route only
    # produced a GIF, keep that as the final animated asset.
    video_paths = [
        path for path in collected if Path(path).suffix.lower() in VIDEO_EXTENSIONS
    ]
    if video_paths:
        return video_paths
    gif_paths = [path for path in collected if Path(path).suffix.lower() == ".gif"]
    if gif_paths:
        return gif_paths
    return collected


def _publish_review_scope(media_paths: list[str]) -> str:
    """Choose a review contract based on the final media, not the producer."""
    if media_paths and all(
        Path(path).suffix.lower() in VIDEO_EXTENSIONS for path in media_paths
    ):
        return "final_video"
    return "final_media"


def _publish_selection_limit(
    media_paths: list[str],
    configured_limit: Any,
    *,
    source_generation_type: str = "",
) -> int:
    """Resolve the publish-review cap without splitting a story-card carousel."""
    if _publish_review_scope(media_paths) == "final_video":
        return 1
    if str(source_generation_type).strip().lower() == "story_card":
        return len(media_paths)
    try:
        limit = int(configured_limit or 0)
    except (TypeError, ValueError):
        limit = 0
    return max(1, min(len(media_paths), limit or len(media_paths)))


def _review_stage_status(run_result: dict[str, Any]) -> str:
    records = run_result.get("records") if isinstance(run_result, dict) else None
    review_records = [
        record
        for record in (records if isinstance(records, list) else [])
        if isinstance(record, dict)
        and (
            str(record.get("skill_name") or "").startswith("review.")
            or "review" in str(record.get("node_id") or "").lower()
        )
    ]
    if not review_records:
        return "not_required"
    if any(
        str(record.get("status") or "").lower() not in {"success", "skipped"}
        for record in review_records
    ):
        return "failed"
    return "success"


def _publish_stage_status(
    publish_result: dict[str, Any] | None,
    *,
    requested: bool,
    generation_status: str,
    media_paths: list[str],
    platform_configs: Any,
) -> str:
    if not requested:
        return "not_requested"
    if generation_status != "success":
        return "not_run_generation_failed"
    if not media_paths:
        return "failed_missing_final_media"
    if not isinstance(platform_configs, dict) or not platform_configs:
        return "failed_missing_platform_config"
    if not publish_result:
        return "not_run"
    result = publish_result.get("result") if isinstance(publish_result, dict) else None
    if isinstance(result, dict):
        if str(result.get("dispatch_status") or "").lower() == "skipped":
            return "skipped_no_compatible_platform"
        return str(result.get("status") or "unknown")
    return "unknown"


def _effective_publish_after_generate(*, requested: bool, stage_probe: bool) -> bool:
    """Never publish an automatically selected stage-probe artifact."""
    return bool(requested and not stage_probe)


def _resolve_publish_prompt(
    generation_result: dict[str, Any], *, fallback_prompt: str
) -> tuple[str, str]:
    """Use the story actually rendered by generation when preparing publish copy."""
    state = generation_result.get("state") or {}
    node_outputs = state.get("node_outputs") if isinstance(state, dict) else {}
    if isinstance(node_outputs, dict):
        comic = node_outputs.get("motion-comic-write")
        if isinstance(comic, dict):
            lines = [f"Comic title: {comic['title']}"]
            for panel in comic["panels"]:
                lines.append(
                    f"{panel['role']}: {panel['caption']}\n"
                    f"Visible state: {panel['visible_state']}\nCause: {panel['cause']}"
                )
            lines.append("Write publish copy about this rendered comic, using the contact sheet to verify visible claims.")
            return "\n".join(lines), "motion_comic"
        native_story = node_outputs.get("native-story-prompt")
        if isinstance(native_story, dict):
            generated_storyboard = native_story.get("generated_storyboard")
            if isinstance(generated_storyboard, dict):
                compact_context = _build_publish_story_context(
                    generated_storyboard,
                    news_context=native_story.get("news_context"),
                )
                if compact_context:
                    return compact_context, "native_h3_story"
            native_prompt = str(native_story.get("prompt") or "").strip()
            if native_prompt:
                return native_prompt, "native_h3_story"
        native_render = node_outputs.get("native-h3-render")
        if isinstance(native_render, dict):
            native_prompt = str(native_render.get("native_h3_prompt") or "").strip()
            if native_prompt:
                return native_prompt, "native_h3_render"
    return str(fallback_prompt or "").strip(), "goal_prompt"


def _extract_publish_visual_grounding(run_result: dict[str, Any]) -> dict[str, Any]:
    """Expose only compact, visible-content QA evidence to caption generation."""
    state = run_result.get("state") or {}
    node_outputs = state.get("node_outputs") if isinstance(state, dict) else {}
    qa = None
    if isinstance(node_outputs, dict):
        for node_id in ("motion-comic-qa", "native-h3-qa"):
            if isinstance(node_outputs.get(node_id), dict):
                qa = node_outputs[node_id]
                break
    contact_sheet_path = (
        str(qa.get("contact_sheet_path") or "").strip() if isinstance(qa, dict) else ""
    )
    return {"contact_sheet_path": contact_sheet_path} if contact_sheet_path else {}


def _build_publish_story_context(
    storyboard: dict[str, Any],
    *,
    news_context: Any = None,
) -> str:
    """Give caption generation the story/news contract, not the full render prompt."""
    spine = (
        storyboard.get("story_spine")
        if isinstance(storyboard.get("story_spine"), dict)
        else {}
    )
    trace = (
        storyboard.get("news_trace")
        if isinstance(storyboard.get("news_trace"), dict)
        else {}
    )
    news = news_context if isinstance(news_context, dict) else {}
    lines = [
        f"Video story: {str(storyboard.get('name') or '').strip()}",
        f"Premise: {str(spine.get('premise') or '').strip()}",
        f"Objective: {str(spine.get('objective') or '').strip()}",
        f"Resolution: {str(spine.get('resolution') or '').strip()}",
        f"News headline: {str(news.get('title') or trace.get('source_title') or '').strip()}",
        f"News visual translation: {str(trace.get('visual_translation') or '').strip()}",
        f"News visual anchors: {', '.join(str(item).strip() for item in trace.get('visual_anchors', []) if str(item).strip())}",
        "Write publish copy about the rendered video. Do not paste the full production prompt or unrelated news text.",
    ]
    return "\n".join(line for line in lines if line.split(": ", 1)[-1].strip())
