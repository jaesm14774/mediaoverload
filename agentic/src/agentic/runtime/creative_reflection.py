"""Evidence-first creative self-reflection for recent MediaOverload runs.

The first loop is deliberately offline and deterministic.  It reads run
manifests, story records, review sessions, QA records, and locally available
video artifacts, then produces one root-cause hypothesis per run plus one
recommended next experiment.  It does not call an LLM or mutate generation
configuration, which keeps the first feedback loop auditable and reversible.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUN_COUNT = 20
DEFAULT_OUTPUT_DIR = Path("logs/reflections")

_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9'-]{2,}|[\u4e00-\u9fff]{2,}", re.IGNORECASE)
_STOPWORDS = {
    "about", "after", "again", "also", "anime", "around", "before", "being",
    "character", "clear", "cinematic", "could", "during", "every", "from",
    "full", "into", "keep", "kirby", "large", "main", "must", "only", "polished",
    "protagonist", "round", "same", "scene", "short", "small", "style", "that",
    "their", "these", "this", "through", "with", "video", "visual", "where", "while",
    "will", "one", "single", "story", "prompt", "make", "using", "shows", "showing",
    "rendered", "motion", "action", "expressive", "expressively", "soft", "dynamic",
}
_LOCATION_TERMS = {
    "living room", "tokyo", "street", "trading floor", "park", "meadow", "sky", "cloud",
    "data center", "atrium", "courtroom", "garden", "portal", "island", "hallway",
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value: Any, limit: int = 600) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _meaningful_source_text(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text or text.isdigit():
        return ""
    return text


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_timestamp(value: Any, fallback: datetime) -> datetime:
    raw = str(value or "").strip()
    if raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return fallback


def _record_value(record: dict[str, Any], key: str, default: Any = None) -> Any:
    outputs = record.get("outputs")
    if isinstance(outputs, dict) and key in outputs:
        return outputs.get(key)
    return default


def _records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    generation = manifest.get("generation")
    result = generation.get("result") if isinstance(generation, dict) else None
    records = result.get("records") if isinstance(result, dict) else None
    return [item for item in _as_list(records) if isinstance(item, dict)]


def _first_record(records: Iterable[dict[str, Any]], node_id: str) -> dict[str, Any]:
    return next((item for item in records if str(item.get("node_id") or "") == node_id), {})


def _resolve_path(repo_root: Path, raw_path: Any) -> Path | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    candidates: list[Path] = []
    direct = Path(raw)
    if direct.is_absolute():
        candidates.append(direct)
    if raw.startswith("/app/"):
        candidates.append(repo_root / raw[5:])
    if "/output/" in raw:
        candidates.append(repo_root / "output" / raw.split("/output/", 1)[1])
    if "\\output\\" in raw.lower():
        suffix = re.split(r"[\\/]output[\\/]", raw, maxsplit=1, flags=re.IGNORECASE)[-1]
        candidates.append(repo_root / "output" / suffix)
    candidates.append(repo_root / raw.lstrip("/\\"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _media_paths(repo_root: Path, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for record in records:
        outputs = record.get("outputs")
        if not isinstance(outputs, dict):
            continue
        raw_values: list[Any] = []
        for key in ("video_path", "gif_path", "saved_files", "media_paths"):
            raw_values.extend(_as_list(outputs.get(key)))
        for raw in raw_values:
            if not str(raw or "").lower().endswith((".mp4", ".gif", ".png", ".jpg", ".jpeg", ".webp")):
                continue
            resolved = _resolve_path(repo_root, raw)
            item = {"declared_path": str(raw), "local_path": str(resolved) if resolved else "", "exists": bool(resolved)}
            if item not in paths:
                paths.append(item)
    return paths


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(str(value or ""))
        if token.casefold() not in _STOPWORDS and len(token) > 2
    }


def _story_text(segments: list[dict[str, Any]]) -> str:
    fields = ("visual", "narration", "action", "camera", "start_state", "end_state", "cause", "effect")
    return " ".join(str(segment.get(field) or "") for segment in segments for field in fields)


def _continuity_breaks(segments: list[dict[str, Any]]) -> list[str]:
    breaks: list[str] = []
    for previous, current in zip(segments, segments[1:]):
        previous_terms = _tokens(previous.get("end_state"))
        current_terms = _tokens(current.get("start_state"))
        if previous_terms and current_terms and not previous_terms.intersection(current_terms):
            breaks.append(
                f"{_text(previous.get('segment_id'), 80)} end_state does not hand off to "
                f"{_text(current.get('segment_id'), 80)} start_state"
            )
    return breaks


def _review_session(repo_root: Path, manifest: dict[str, Any], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = list(records)
    generation = manifest.get("generation")
    prompt_summary = generation.get("prompt_summary", {}) if isinstance(generation, dict) else {}
    lineage = prompt_summary.get("prompt_lineage", []) if isinstance(prompt_summary, dict) else []
    candidates.extend(item for item in _as_list(lineage) if isinstance(item, dict))
    for record in reversed(candidates):
        raw = _record_value(record, "review_session_path", "")
        session_path = _resolve_path(repo_root, raw)
        if session_path:
            session = _load_json(session_path) or {}
            return {
                "path": str(session_path),
                "status": str(session.get("normalized_status") or session.get("status") or ""),
                "reviewer": _text(session.get("reviewer"), 80),
                "summary_missing": "未提供故事摘要" in str(session.get("text") or ""),
                "text": _text(session.get("text"), 900),
                "selected_count": len(_as_list(session.get("selected_paths"))),
            }
    return {}


def _story_data(manifest: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    idea = _first_record(records, "idea-brief")
    script = _first_record(records, "script-plan")
    native = _first_record(records, "native-story-prompt")
    segments = [item for item in _as_list(_record_value(script, "segments")) if isinstance(item, dict)]
    native_storyboard = _record_value(native, "generated_storyboard", {})
    if not isinstance(native_storyboard, dict):
        native_storyboard = {}
    spine = _record_value(native, "story_spine", {})
    if not isinstance(spine, dict):
        spine = native_storyboard.get("story_spine") if isinstance(native_storyboard.get("story_spine"), dict) else {}
    goal = manifest.get("plan", {}).get("goal", {}) if isinstance(manifest.get("plan"), dict) else {}
    constraints = goal.get("constraints", {}) if isinstance(goal, dict) else {}
    routing = manifest.get("routing_summary", {})
    if not isinstance(routing, dict):
        routing = {}
    strategy = str(manifest.get("source_generation_type") or routing.get("strategy") or "")
    native_story_source = _meaningful_source_text(_record_value(native, "story_source", ""))
    native_news_context = _record_value(native, "news_context", {})
    news_title = (
        _meaningful_source_text(native_news_context.get("title", ""))
        if isinstance(native_news_context, dict)
        else ""
    )
    source_candidates = (
        native_story_source,
        news_title,
        _meaningful_source_text(routing.get("prompt")),
        _meaningful_source_text(goal.get("prompt")),
        _meaningful_source_text(_record_value(idea, "prompt", "")),
        _meaningful_source_text(spine.get("premise", "")),
    )
    source_prompt = next((candidate for candidate in source_candidates if candidate), "")
    # Do not include the original idea prompt in the generated-story side of
    # the comparison. Otherwise the source terms always overlap and a static
    # preset can silently replace the user's premise without being detected.
    story_prompt = " ".join(
        [
            _record_value(native, "prompt", ""),
            json.dumps(spine, ensure_ascii=False),
            json.dumps(native_storyboard, ensure_ascii=False),
            _story_text(segments),
        ]
    )
    source_terms = _tokens(source_prompt)
    story_terms = _tokens(story_prompt)
    missing_terms = sorted(source_terms - story_terms, key=lambda item: (-len(item), item))[:12]
    storyboard_key = ""
    if strategy == "text2longvideo":
        storyboard_key = "storyboard_path"
    elif strategy.startswith("native_h3") or strategy == "text2image2native_h3_ref2va":
        storyboard_key = "native_h3_storyboard_path"
    storyboard_path = str(constraints.get(storyboard_key) or "").strip()
    drift = bool(
        segments
        and missing_terms
        and storyboard_path
        and any(term in storyboard_path.casefold() for term in ("meadow", "preset", "storyboard"))
    )
    segment_actions_missing = [
        str(segment.get("segment_id") or f"segment-{index + 1}")
        for index, segment in enumerate(segments)
        if not str(segment.get("action") or "").strip()
    ]
    locations = sorted(term for term in _LOCATION_TERMS if term in source_prompt.casefold())
    return {
        "source_prompt": _text(source_prompt, 700),
        "creative_brief": _text(_record_value(idea, "creative_brief", "") or _record_value(native, "creative_brief", ""), 500),
        "storyboard_path": storyboard_path,
        "story_spine": {key: _text(value, 300) for key, value in spine.items()} if isinstance(spine, dict) else {},
        "segments": [
            {
                "segment_id": _text(segment.get("segment_id"), 80),
                "action": _text(segment.get("action"), 260),
                "visual": _text(segment.get("visual"), 320),
                "start_state": _text(segment.get("start_state"), 220),
                "end_state": _text(segment.get("end_state"), 220),
            }
            for segment in segments
        ],
        "segment_count": len(segments),
        "missing_action_segments": segment_actions_missing,
        "continuity_breaks": _continuity_breaks(segments),
        "source_anchor_terms_missing_from_story": missing_terms,
        "source_location_terms": locations,
        "storyboard_drift": drift,
        "prompt_word_count": len(source_prompt.split()),
    }


def _qa_data(records: list[dict[str, Any]]) -> dict[str, Any]:
    qa = _first_record(records, "native-h3-qa")
    outputs = qa.get("outputs") if isinstance(qa, dict) else {}
    if not isinstance(outputs, dict):
        return {}
    semantic = outputs.get("semantic_qa") if isinstance(outputs.get("semantic_qa"), dict) else {}
    technical = outputs.get("technical_qa") if isinstance(outputs.get("technical_qa"), dict) else {}
    return {
        "node_status": str(qa.get("status") or ""),
        "technical_passed": technical.get("passed"),
        "semantic_enabled": semantic.get("enabled"),
        "semantic_status": str(semantic.get("status") or ""),
        "semantic_passed": semantic.get("passed"),
        "semantic_score": semantic.get("score"),
        "semantic_checks": semantic.get("checks") if isinstance(semantic.get("checks"), dict) else {},
        "observed_story": _text(semantic.get("observed_story"), 500),
        "issues": [_text(item, 240) for item in _as_list(semantic.get("issues")) if str(item).strip()],
        "technical_errors": [_text(item, 240) for item in _as_list(technical.get("errors")) if str(item).strip()],
    }


def _cause(category: str, severity: str, evidence: list[str], recommendation: str) -> dict[str, Any]:
    return {
        "category": category,
        "severity": severity,
        "evidence": [_text(item, 320) for item in evidence if str(item).strip()],
        "recommendation": recommendation,
    }


def _root_causes(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    story: dict[str, Any],
    review: dict[str, Any],
    qa: dict[str, Any],
) -> list[dict[str, Any]]:
    causes: list[dict[str, Any]] = []
    failure_node = str(manifest.get("failure_node") or "")
    failure_reason = str(manifest.get("failure_reason") or "")
    routing = manifest.get("routing_summary")
    routing = routing if isinstance(routing, dict) else {}
    strategy = str(manifest.get("source_generation_type") or routing.get("strategy") or "")

    if story.get("storyboard_drift"):
        causes.append(
            _cause(
                "storyboard_drift",
                "blocker",
                [
                    f"Configured storyboard: {story.get('storyboard_path')}",
                    f"Source terms absent from generated story: {', '.join(story.get('source_anchor_terms_missing_from_story', []))}",
                    "Segment state is therefore anchored to a different premise than the requested prompt.",
                ],
                "Do not apply a static character storyboard to a news-driven longvideo run; use the generated brief and require source-anchor overlap before rendering.",
            )
        )
    if story.get("missing_action_segments"):
        causes.append(
            _cause(
                "segment_contract_gap",
                "high",
                [f"Segments without an explicit physical action: {', '.join(story['missing_action_segments'])}"],
                "Make action, camera, start_state, end_state, cause, and effect required fields and fail before keyframe generation when any are empty.",
            )
        )
    if story.get("continuity_breaks"):
        causes.append(
            _cause(
                "continuity_break",
                "high",
                story["continuity_breaks"],
                "Carry the previous end_state into the next segment prompt and reject a segment pair that changes location or prop without a declared transition.",
            )
        )
    if story.get("prompt_word_count", 0) > 150:
        causes.append(
            _cause(
                "prompt_overload",
                "medium",
                [f"Source prompt contains about {story['prompt_word_count']} words.", "The run requests multiple locations, transformations, and plot devices in a short clip."],
                "Compress the idea into one dominant news mechanism, one setback, and one payoff; preserve source context, active mechanism, and visible consequence instead of routing every headline through a prop.",
            )
        )
    if review.get("summary_missing"):
        causes.append(
            _cause(
                "review_context_missing",
                "high",
                ["Discord review text contains '未提供故事摘要'.", "The reviewer is asked to judge a candidate without the actual story objective."],
                "Use a shared story-summary builder for every route, including text2longvideo and sticker routes, and include objective, obstacle, and payoff in review text.",
            )
        )
    if failure_node == "native-story-prompt":
        causes.append(
            _cause(
                "story_generation_gate",
                "high",
                [failure_reason or "Native story prompt was rejected before rendering."],
                "Turn the failed quality error into a bounded revision instruction and retry only the story generation step; do not spend GPU time on a known incomplete gag card.",
            )
        )
    if failure_node in {"native-h3-render", "native-ref2va-reference-review"} or "ComfyUI" in failure_reason:
        causes.append(
            _cause(
                "render_or_runtime_boundary",
                "boundary",
                [failure_reason or f"Failure at {failure_node}."],
                "Keep this separate from creative scoring; inspect ComfyUI/runtime artifacts and retry the same story only after the runtime boundary is healthy.",
            )
        )
    if failure_node == "dispatch-publish":
        causes.append(
            _cause(
                "publish_boundary",
                "boundary",
                [failure_reason or "Run reached publication and failed there."],
                "Do not change the creative prompt for this run; isolate platform/OAuth/API evidence from content quality and retry publication independently.",
            )
        )
    if failure_node in {"review-select", "stage-review-01", "native-opening-review", "native-l2va-ending-review"}:
        status = str(review.get("status") or "")
        if status in {"rejected", "reject"}:
            causes.append(
                _cause(
                    "human_review_rejection_without_reason",
                    "high",
                    ["Human review rejected the candidate set, but the stored session has no structured rejection reason."],
                    "Require one or more rejection tags (story drift, weak first action, weak silhouette, no payoff, identity drift, technical artifact) before Reject is accepted; feed only the selected tag back to the next experiment.",
                )
            )
        elif status in {"timeout", "failed", "blocked"}:
            causes.append(
                _cause(
                    "human_review_availability",
                    "boundary",
                    [f"Review ended with status '{status}'."],
                    "Keep the run blocked, but classify the event as review availability rather than creative failure; preserve the candidate set for later review.",
                )
            )
    if strategy.startswith("native_h3") and qa and qa.get("semantic_enabled") is False:
        causes.append(
            _cause(
                "semantic_qa_bypassed",
                "high",
                [f"{strategy} produced a video while semantic QA was disabled."],
                "Enable blocking semantic QA for every native H3 route so a technically valid but visually generic clip cannot proceed to human review or publish.",
            )
        )
    semantic_checks = qa.get("semantic_checks") if isinstance(qa.get("semantic_checks"), dict) else {}
    news_contract_failures = [
        key
        for key in (
            "news_mechanism_present",
            "news_consequence_present",
            "news_anchor_roles_complete",
            "news_anchor_diversity",
            "news_anchor_not_default_object_loop",
            "news_mechanism_reaches_story",
            "news_consequence_reaches_payoff",
        )
        if semantic_checks.get(key) is False
    ]
    if news_contract_failures:
        causes.append(
            _cause(
                "news_mechanism_collapse",
                "blocker",
                [
                    "Native H3 semantic QA rejected the news mechanism contract: "
                    + ", ".join(news_contract_failures),
                    "The story is not allowed to pass when the headline has collapsed into a reusable prop loop.",
                ],
                "Regenerate the Native H3 story with contract_version=2 and three distinct roles: source context, active news mechanism, and visible consequence; use a prop only when the event itself requires it.",
            )
        )
    semantic_score = qa.get("semantic_score")
    try:
        high_semantic_score = float(semantic_score) >= 95
    except (TypeError, ValueError):
        high_semantic_score = False
    if qa.get("semantic_passed") is True and high_semantic_score and str(review.get("status") or "") in {"rejected", "reject"}:
        causes.append(
            _cause(
                "qa_calibration_gap",
                "medium",
                [f"Semantic QA reported pass/{qa.get('semantic_score')}, but human review rejected the final candidate."],
                "Record human rejection tags and add them to a small calibration set; do not raise the automated score threshold without knowing which visual criterion was missed.",
            )
        )
    if not causes:
        causes.append(
            _cause(
                "insufficient_evidence",
                "medium",
                ["No authoritative story, QA, review, or runtime failure evidence was available for a narrower diagnosis."],
                "Preserve the run artifact and add the missing evidence surface before changing prompts.",
            )
        )
    return causes


def _next_experiment(batch_counts: Counter[str]) -> str:
    if batch_counts.get("news_mechanism_collapse"):
        return "Experiment 1: rerun one Native H3 news case with the v2 mechanism contract and compare anchor diversity, causal payoff, and human review before allowing render/publish."
    if batch_counts.get("storyboard_drift"):
        return "Experiment 1: rerun one text2longvideo case with the static storyboard default removed and the story-anchor/segment-contract gates enabled; compare drift, action completeness, and human review."
    if batch_counts.get("review_context_missing"):
        return "Experiment 1: populate every Discord review with the actual premise/objective/payoff, then compare rejection rate before changing generation prompts."
    if batch_counts.get("semantic_qa_bypassed"):
        return "Experiment 1: enable blocking semantic QA on the bypassed native route and measure how many generic clips are stopped before human review."
    return "Experiment 1: collect structured human rejection tags for the next five runs before making another creative prompt change."


def inspect_recent_runs(repo_root: Path, count: int = DEFAULT_RUN_COUNT) -> dict[str, Any]:
    run_root = repo_root / "logs" / "runs"
    candidates: list[tuple[datetime, Path, dict[str, Any]]] = []
    for run_dir in run_root.iterdir() if run_root.is_dir() else []:
        manifest_path = run_dir / "run_manifest.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            continue
        manifest = _load_json(manifest_path)
        if manifest is None:
            continue
        timestamp = _parse_timestamp(manifest.get("updated_at"), datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc))
        candidates.append((timestamp, run_dir, manifest))
    candidates.sort(key=lambda item: item[0], reverse=True)
    reports: list[dict[str, Any]] = []
    for timestamp, run_dir, manifest in candidates[: max(1, int(count))]:
        records = _records(manifest)
        story = _story_data(manifest, records)
        review = _review_session(repo_root, manifest, records)
        qa = _qa_data(records)
        media = _media_paths(repo_root, records)
        causes = _root_causes(manifest, records, story, review, qa)
        reports.append(
            {
                "run_id": str(manifest.get("run_id") or run_dir.name),
                "updated_at": timestamp.isoformat(),
                "status": str(manifest.get("status") or ""),
                "strategy": str(manifest.get("source_generation_type") or manifest.get("routing_summary", {}).get("strategy") or ""),
                "failure_node": str(manifest.get("failure_node") or ""),
                "failure_reason": _text(manifest.get("failure_reason"), 400),
                "run_dir": str(run_dir),
                "media": media,
                "story": story,
                "review": review,
                "qa": qa,
                "root_causes": causes,
                "next_step": causes[0]["recommendation"],
            }
        )
    counts = Counter(
        cause["category"]
        for report in reports
        for cause in report.get("root_causes", [])
        if cause.get("severity") != "boundary"
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "run_count_requested": int(count),
        "run_count_inspected": len(reports),
        "loop": [
            "observe: read manifest, story records, review session, QA, and local media evidence",
            "diagnose: classify the earliest workflow cause, not only the final failure node",
            "change: choose one bounded prompt, story, route, QA, or review-context experiment",
            "rerun: generate a new artifact batch with the same evidence surfaces",
            "compare: use human tags and semantic/technical QA as separate signals",
            "retain_or_rollback: keep the change only when the target failure rate improves without a new hard failure",
        ],
        "batch_counts": dict(counts.most_common()),
        "recommended_next_experiment": _next_experiment(counts),
        "runs": reports,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MediaOverload creative self-reflection",
        "",
        f"- Generated: `{report.get('generated_at', '')}`",
        f"- Runs inspected: `{report.get('run_count_inspected', 0)}` / requested `{report.get('run_count_requested', 0)}`",
        f"- Next experiment: {report.get('recommended_next_experiment', '')}",
        "",
        "## Loop contract",
        "",
        " → ".join(str(item) for item in report.get("loop", [])),
        "",
        "## Batch diagnosis",
        "",
    ]
    for category, count in (report.get("batch_counts") or {}).items():
        lines.append(f"- `{category}`: {count}")
    for item in report.get("runs", []):
        lines.extend(
            [
                "",
                f"## `{item.get('run_id')}` — {item.get('strategy')} — {item.get('status')}",
                f"- Updated: `{item.get('updated_at')}`",
                f"- Failure boundary: `{item.get('failure_node') or 'none'}` — {item.get('failure_reason') or 'none'}",
                f"- Source prompt: {item.get('story', {}).get('source_prompt', '')}",
                f"- Storyboard: `{item.get('story', {}).get('storyboard_path') or 'generated/no static storyboard'}`",
                f"- Local media found: `{sum(1 for media in item.get('media', []) if media.get('exists'))}` / `{len(item.get('media', []))}`",
            ]
        )
        story = item.get("story", {})
        if story.get("missing_action_segments"):
            lines.append(f"- Missing actions: `{', '.join(story['missing_action_segments'])}`")
        if story.get("continuity_breaks"):
            lines.append(f"- Continuity breaks: {'; '.join(story['continuity_breaks'])}")
        if story.get("source_anchor_terms_missing_from_story"):
            lines.append(f"- Source terms absent from story: `{', '.join(story['source_anchor_terms_missing_from_story'])}`")
        review = item.get("review", {})
        if review:
            lines.append(f"- Review: `{review.get('status')}`; summary missing=`{review.get('summary_missing')}`; selected=`{review.get('selected_count')}`")
        qa = item.get("qa", {})
        if qa:
            lines.append(f"- QA: technical=`{qa.get('technical_passed')}` semantic=`{qa.get('semantic_status')}/{qa.get('semantic_score')}` enabled=`{qa.get('semantic_enabled')}`")
            if qa.get("observed_story"):
                lines.append(f"- QA observed story: {qa['observed_story']}")
        lines.append("- Root causes:")
        for cause in item.get("root_causes", []):
            lines.append(f"  - **{cause.get('severity')} / {cause.get('category')}**: {'; '.join(cause.get('evidence', []))}")
            lines.append(f"    - Change: {cause.get('recommendation')}")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"creative_reflection_{stamp}.json"
    markdown_path = output_dir / f"creative_reflection_{stamp}.md"
    memory_path = output_dir / "reflection_memory.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    memory = {
        "schema_version": 1,
        "updated_at": report.get("generated_at", ""),
        "source_report": str(json_path),
        "rules": [
            {"rule": "Never treat technical QA as creative approval.", "evidence": "technical, semantic, human review, and publish boundaries are reported separately"},
            {"rule": "Reject story drift before rendering.", "evidence": "source prompt terms and segment state are compared against the selected storyboard"},
            {"rule": "One experiment changes one workflow lever.", "evidence": "the report emits one recommended next experiment for the batch"},
            {"rule": "Human rejection needs a structured reason.", "evidence": "unexplained Discord Reject events are classified as a review calibration gap"},
        ],
        "batch_counts": report.get("batch_counts", {}),
        "recommended_next_experiment": report.get("recommended_next_experiment", ""),
    }
    memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json_path, markdown_path, memory_path
