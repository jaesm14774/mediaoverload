"""Render ten matched, news-grounded Native H3 story-arc A/B pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.app.character_requests import (  # noqa: E402
    CharacterGenerationOptions,
    CharacterReviewOptions,
    CharacterRuntimeOptions,
    CharacterWorkflowRequest,
)
from agentic.app.character_workflow import run_character_workflow  # noqa: E402

EXPERIMENT_TEXT_PROVIDER = "ollama"
EXPERIMENT_TEXT_MODEL = "qwen3.8-27b-ud-q2xl-local:latest"
EXPERIMENT_VISION_MODEL = "google/gemma-4-26b-a4b-it:free"
EXPERIMENT_LLM_REQUEST_TIMEOUT_SECONDS = 600
EXPERIMENT_LLM_TOTAL_TIMEOUT_SECONDS = 660
DURATION_SECONDS = 15
DATA_PATH = REPO_ROOT / "docs" / "research" / "vox_director_news_arc_cases.json"
RUNS_ROOT = REPO_ROOT / "logs" / "runs"
FRAME_FRACTIONS = (0.03, 0.20, 0.38, 0.57, 0.76, 0.95)
SCORE_FIELDS = (
    "source_fact_alignment",
    "causal_progression",
    "opening_readability",
    "arc_fit",
    "payoff",
    "identity_geography_continuity",
    "overall_visual_quality",
)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_resume_manifest_hash(path: Path, expected_hash: str) -> str:
    manifest_text = path.read_text(encoding="utf-8")
    if not expected_hash or _sha256(manifest_text) != expected_hash:
        raise ValueError("Resume news manifest hash does not match the saved run controls.")
    return manifest_text


def stable_seed(case_id: str, seed_base: int) -> int:
    offset = int(_sha256(case_id)[:8], 16) % 100_000
    return max(1, min(2_147_483_646, int(seed_base) + offset))


def _load_cases(
    case_ids: set[str] | None = None,
    *,
    data_path: Path = DATA_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("The news-arc benchmark manifest must contain exactly ten cases.")
    ids = [str(case.get("case_id") or "") for case in cases]
    arcs = [str(case.get("arc") or "") for case in cases]
    urls = [str((case.get("article") or {}).get("url") or "") for case in cases]
    if len(set(ids)) != 10 or len(set(arcs)) != 10 or len(set(urls)) != 10:
        raise ValueError("Each benchmark case must have a unique id, arc, and article URL.")
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", case_id) for case_id in ids):
        raise ValueError("Each benchmark case id must be a safe relative path component.")
    if case_ids:
        unknown = case_ids - set(ids)
        if unknown:
            raise ValueError(f"Unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["case_id"] in case_ids]
    return data, cases


def _redact(value: object) -> str:
    import re

    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "sk-[REDACTED]", str(value or ""))


def _queue_state(host: str, port: int) -> tuple[int, int]:
    try:
        with urlopen(f"http://{host}:{port}/queue", timeout=3) as response:
            queue = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ComfyUI queue endpoint is unavailable: {type(exc).__name__}") from exc
    return len(queue.get("queue_running", [])), len(queue.get("queue_pending", []))


def _require_idle_comfy(host: str, port: int) -> None:
    running, pending = _queue_state(host, port)
    if running or pending:
        raise RuntimeError(f"ComfyUI is busy (running={running}, pending={pending}); no benchmark job was submitted")


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    probe = json.loads(result.stdout)
    return {
        "duration_seconds": round(float(probe.get("format", {}).get("duration") or 0), 3),
        "streams": probe.get("streams", []),
    }


def _final_video_probe_pass(path: Path | None, probe: Any) -> bool:
    if path is None or not path.is_file() or not isinstance(probe, dict) or probe.get("error"):
        return False
    try:
        duration = float(probe.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        return False
    if duration <= 0 or not isinstance(probe.get("streams"), list):
        return False
    for stream in probe["streams"]:
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        try:
            if int(stream.get("width") or 0) > 0 and int(stream.get("height") or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _extract_evidence(
    result: dict[str, Any], variant_dir: Path, run_dir: Path | None,
    elapsed: float, expected_article_title: str, expected_arc_instruction: str,
) -> dict[str, Any]:
    generation = result.get("generation") if isinstance(result.get("generation"), dict) else {}
    generation_result = generation.get("result") if isinstance(generation.get("result"), dict) else {}
    state = generation_result.get("state") if isinstance(generation_result.get("state"), dict) else {}
    outputs = state.get("node_outputs") if isinstance(state.get("node_outputs"), dict) else {}
    qa = next(
        (outputs.get(name) for name in ("native-h3-qa", "video-qa") if isinstance(outputs.get(name), dict)),
        {},
    )
    story = outputs.get("native-story-prompt") if isinstance(outputs.get("native-story-prompt"), dict) else {}
    llm_calls: list[dict[str, Any]] = []
    if run_dir and (run_dir / "llm").is_dir():
        for path in sorted((run_dir / "llm").glob("*.json")):
            try:
                call = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            llm_calls.append({
                "schema_name": call.get("schema_name", ""),
                "status": call.get("status", ""),
                "model_role": call.get("model_role", ""),
                "model_id": call.get("model_id", ""),
                "error": _redact(call.get("error", "")),
                "path": str(path),
            })
    story_call = next((item for item in llm_calls if item["schema_name"] == "native_h3_storyboard"), {})
    story_prompt_text = ""
    if story_call.get("path"):
        try:
            story_call_record = json.loads(Path(story_call["path"]).read_text(encoding="utf-8"))
            story_prompt_text = "\n".join(
                str(message.get("content") or "")
                for message in story_call_record.get("messages", [])
                if isinstance(message, dict)
            )
        except (OSError, json.JSONDecodeError):
            story_prompt_text = ""
    arc_guidance_prefix = "Story arc guidance (structure only; the article remains authoritative for facts):"
    arc_guidance_pass = (
        expected_arc_instruction in story_prompt_text
        if expected_arc_instruction
        else arc_guidance_prefix not in story_prompt_text
    )
    selected_character = ""
    if run_dir and (run_dir / "lifecycle.log").is_file():
        lifecycle = (run_dir / "lifecycle.log").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"character\.group\.selected \| group=[^|]+ \| selected=([^\s|]+)", lifecycle)
        if match:
            selected_character = match.group(1)
    video_paths = sorted(path.resolve() for path in variant_dir.rglob("*.mp4") if path.is_file())
    final_candidates = [path for path in video_paths if "video_speed" in str(path.parent).lower()]
    if not final_candidates:
        final_candidates = [path for path in video_paths if "video_canvas" in str(path.parent).lower()]
    final_video = max(final_candidates or video_paths, key=lambda item: item.stat().st_mtime) if video_paths else None
    probes: dict[str, Any] = {}
    for path in video_paths:
        try:
            probes[str(path)] = _ffprobe(path)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as exc:
            probes[str(path)] = {"error": f"{type(exc).__name__}: {_redact(exc)}"}
    qa_passed = bool(qa.get("passed"))
    story_passed = str(story.get("story_source") or story.get("source") or "") == "native_h3_llm"
    story_source_title = str((story.get("news_context") or {}).get("title") or "")
    source_title_match_pass = story_source_title == expected_article_title
    model_id = str(story_call.get("model_id") or "")
    model_pin_pass = bool(story_call.get("status") == "success" and EXPERIMENT_TEXT_MODEL in model_id)
    workflow_status = str(result.get("status") or "failed")
    final_probe = probes.get(str(final_video), {}) if final_video else {}
    final_video_probe_pass = _final_video_probe_pass(final_video, final_probe)
    return {
        "workflow_status": workflow_status,
        "technical_pass": bool(
            workflow_status == "success" and qa_passed and story_passed and model_pin_pass
            and source_title_match_pass and selected_character == "Kirby" and final_video
            and final_video_probe_pass
            and arc_guidance_pass
        ),
        "qa_passed": qa_passed,
        "qa_errors": [_redact(item) for item in qa.get("errors", []) if str(item).strip()],
        "story_source": str(story.get("story_source") or story.get("source") or ""),
        "story_source_title": story_source_title,
        "source_title_match_pass": source_title_match_pass,
        "resolved_character_name": selected_character,
        "native_storyboard": story.get("generated_storyboard", {}),
        "story_llm_call": story_call,
        "llm_calls": llm_calls,
        "model_pin_pass": model_pin_pass,
        "model_id_observed": model_id,
        "arc_guidance_expected": bool(expected_arc_instruction),
        "arc_guidance_pass": arc_guidance_pass,
        "contact_sheet_path": str(qa.get("contact_sheet_path") or ""),
        "video_paths": [str(path) for path in video_paths],
        "final_video_path": str(final_video) if final_video else "",
        "final_video_probe_pass": final_video_probe_pass,
        "ffprobe": probes,
        "run_observability_dir": str(run_dir) if run_dir else "",
        "elapsed_seconds": round(elapsed, 3),
        "failure_reason": _redact(result.get("failure_reason") or ""),
    }


def _start_vram_release_watcher(host: str, port: int) -> tuple[threading.Event, threading.Thread]:
    """Unload the pinned planner after it submits its Comfy render, as in prior local runs."""
    stop_event = threading.Event()

    def watch() -> None:
        deadline = time.monotonic() + 3600
        while not stop_event.wait(0.5) and time.monotonic() < deadline:
            try:
                running, pending = _queue_state(host, port)
            except RuntimeError:
                continue
            if running or pending:
                subprocess.run(
                    ["ollama", "stop", EXPERIMENT_TEXT_MODEL],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                )
                return

    thread = threading.Thread(target=watch, name="benchmark-planner-vram-release", daemon=True)
    thread.start()
    return stop_event, thread


def _next_variant_attempt_dir(case_dir: Path, variant: str) -> tuple[Path, int]:
    variant_root = case_dir / variant
    if not variant_root.exists():
        variant_root.mkdir(parents=True)
        return variant_root, 1
    if not any(variant_root.iterdir()):
        return variant_root, 1
    attempt = 2
    while (variant_root / f"attempt-{attempt:02d}").exists():
        attempt += 1
    attempt_dir = variant_root / f"attempt-{attempt:02d}"
    attempt_dir.mkdir(parents=True)
    return attempt_dir, attempt


def _run_variant(
    *, case: dict[str, Any], variant: str, seed: int, case_dir: Path,
    args: argparse.Namespace, render_signature: str, shared_brief: str,
) -> dict[str, Any]:
    variant_dir, attempt_number = _next_variant_attempt_dir(case_dir, variant)
    article = case["article"]
    article_context = {
        "title": article["title"],
        "keyword": article["keyword"],
        "category": "news",
        "created_at": f"{article['published']}T12:00:00Z",
        "content": article["summary"],
        "url": article["url"],
        "source": article["publisher"],
    }
    arc_instruction = str(case["arc_instruction"] or "") if variant == "B" else ""
    brief = shared_brief
    record: dict[str, Any] = {
        "case_id": case["case_id"],
        "variant": variant,
        "attempt": attempt_number,
        "article": article,
        "changed_variable": "separate_topic_matched_story_arc_guidance",
        "narrative_arc": "current_native_h3_news_contract_plus_shared_brief" if variant == "A" else case["arc"],
        "shared_story_brief": shared_brief,
        "native_h3_creative_brief": brief,
        "native_h3_arc_instruction": arc_instruction,
        "news_context": article_context,
        "creative_brief_sha256": _sha256(brief),
        "render_signature": render_signature,
        "controls": {
            "generation_type": "native_h3_t2v_story",
            "news_driven": True,
            "subject_mode": "single",
            "selected_character_name": "Kirby",
            "duration_seconds": DURATION_SECONDS,
            "seed": seed,
            "comfy_host": args.comfy_host,
            "comfy_port": int(args.comfy_port),
            "text_model_provider": EXPERIMENT_TEXT_PROVIDER,
            "text_model": EXPERIMENT_TEXT_MODEL,
            "vision_model_provider": "openrouter",
            "vision_model": EXPERIMENT_VISION_MODEL,
            "model_fallback": False,
        },
        "manual_review_scores": {**{name: None for name in SCORE_FIELDS}, "notes": "pending human video review"},
    }
    _json_write(variant_dir / "variant_config.json", record)
    _require_idle_comfy(args.comfy_host, int(args.comfy_port))
    started = time.perf_counter()
    watcher_stop, watcher = _start_vram_release_watcher(args.comfy_host, int(args.comfy_port))
    try:
        result = run_character_workflow(
            CharacterWorkflowRequest(
                repo_root=REPO_ROOT,
                config_path=REPO_ROOT / "configs" / "characters" / "kirby.yaml",
                generation=CharacterGenerationOptions(
                    preferred_generation_type="native_h3_t2v_story",
                    duration_seconds=DURATION_SECONDS,
                    output_dir=str(variant_dir),
                    rng=random.Random(seed),
                    seed=seed,
                    subject_mode="single",
                    selected_character_name="Kirby",
                    news_driven=True,
                    news_context=article_context,
                    native_h3_creative_brief=brief,
                    native_h3_arc_instruction=arc_instruction,
                ),
                review=CharacterReviewOptions(
                    publish_after_generate=False,
                    no_review=True,
                    stage_probe=True,
                    enable_review_loop=False,
                ),
                runtime=CharacterRuntimeOptions(
                    comfy_host=args.comfy_host,
                    comfy_port=int(args.comfy_port),
                    comfy_root=Path(args.comfy_root).resolve(),
                    auto_download_assets=False,
                ),
            )
        )
        result_dict = result if isinstance(result, dict) else dict(result)
        workflow_run_id = str(result_dict.get("run_id") or "").strip()
        observability_dir = (
            RUNS_ROOT / workflow_run_id
            if workflow_run_id and (RUNS_ROOT / workflow_run_id).is_dir()
            else None
        )
        evidence = _extract_evidence(
            result_dict,
            variant_dir,
            observability_dir,
            time.perf_counter() - started,
            str(article["title"]),
            arc_instruction,
        )
    except Exception as exc:
        evidence = {
            "workflow_status": "failed",
            "technical_pass": False,
            "qa_passed": False,
            "qa_errors": [],
            "llm_calls": [],
            "video_paths": [],
            "final_video_path": "",
            "ffprobe": {},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "failure_reason": f"{type(exc).__name__}: {_redact(exc)}",
        }
    finally:
        watcher_stop.set()
        watcher.join(timeout=1)
    record["evidence"] = evidence
    _json_write(variant_dir / "variant.json", record)
    return record


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _comparison_sheet(case: dict[str, Any], variants: dict[str, dict[str, Any]], case_dir: Path) -> str:
    from PIL import ImageDraw

    thumbs: dict[tuple[str, int], Image.Image] = {}
    source_paths = {
        variant: Path(variants[variant]["evidence"]["final_video_path"])
        for variant in ("A", "B")
    }
    durations = {
        variant: float(_ffprobe(path).get("duration_seconds") or 0)
        for variant, path in source_paths.items()
    }
    if min(durations.values()) <= 0:
        raise RuntimeError(f"Cannot build comparison sheet for invalid durations: {durations}")
    comparison_duration = min(durations.values())
    frame_times = tuple(comparison_duration * fraction for fraction in FRAME_FRACTIONS)
    for variant in ("A", "B"):
        source = source_paths[variant]
        for index, timestamp in enumerate(frame_times):
            target = case_dir / f"frame_{variant}_{index + 1}.jpg"
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-ss", str(timestamp), "-i", str(source),
                    "-frames:v", "1", "-vf",
                    "scale=480:270:force_original_aspect_ratio=decrease,pad=480:270:(ow-iw)/2:(oh-ih)/2",
                    "-y", str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            with Image.open(target) as image:
                thumbs[(variant, index)] = image.convert("RGB")
    width, image_height, row_gap, top, margin = 480, 270, 18, 132, 20
    sheet = Image.new("RGB", (margin * 2 + width * 2, top + len(frame_times) * (image_height + row_gap) + margin), "#111827")
    draw = ImageDraw.Draw(sheet)
    title_font, label_font, time_font = _font(22), _font(18), _font(14)
    article = case["article"]
    draw.text((margin, 18), article["title"], font=title_font, fill="#f9fafb")
    draw.text((margin, 52), f"Arc tested: {case['arc']}  |  Source: {article['publisher']} ({article['published']})", font=label_font, fill="#cbd5e1")
    draw.text((margin, 88), "A — Current + shared brief", font=label_font, fill="#93c5fd")
    draw.text((margin + width, 88), f"B — Added {case['arc']} arc", font=label_font, fill="#86efac")
    for index, timestamp in enumerate(frame_times):
        y = top + index * (image_height + row_gap)
        draw.text((margin, y - 17), f"{timestamp:g}s", font=time_font, fill="#cbd5e1")
        sheet.paste(thumbs[("A", index)], (margin, y))
        sheet.paste(thumbs[("B", index)], (margin + width, y))
        draw.rectangle((margin, y, margin + width * 2, y + image_height), outline="#334155", width=1)
    target = case_dir / "A_vs_B_contact_sheet.jpg"
    sheet.save(target, quality=92)
    return str(target)


def _comparison_gallery(run_dir: Path, cases: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    pair_by_id = {pair["case_id"]: pair for pair in pairs}
    blocks: list[str] = []
    review_rows: list[list[str]] = [[
        "case_id", "arc", "article_url", "A_video", "B_video", *[f"A_{key}" for key in SCORE_FIELDS],
        *[f"B_{key}" for key in SCORE_FIELDS], "winner", "notes",
    ]]
    for case in cases:
        pair = pair_by_id.get(case["case_id"], {})
        variants = pair.get("variants", {})
        article = case["article"]
        card_parts = [
            f"<h2>{html.escape(article['title'])}</h2>",
            f"<p><a href=\"{html.escape(article['url'], quote=True)}\">{html.escape(article['publisher'])}, {html.escape(article['published'])}</a> · tested arc: <code>{html.escape(case['arc'])}</code></p>",
        ]
        video_names: dict[str, str] = {"A": "", "B": ""}
        if all(name in variants for name in ("A", "B")) and pair.get("technical_both_passed"):
            for name, heading, color in (("A", "A — 現行 Native H3 新聞編排", "#2563eb"), ("B", f"B — 加入 {case['arc']} 弧線", "#16a34a")):
                evidence = variants[name]["evidence"]
                file_path = Path(evidence["final_video_path"])
                rel_video = file_path.relative_to(run_dir).as_posix()
                video_names[name] = rel_video
                duration = (
                    (evidence.get("ffprobe") or {}).get(str(file_path), {}).get("duration_seconds")
                )
                duration_label = f" · {float(duration):.2f}s final output" if duration else ""
                card_parts.append(
                    f"<figure><figcaption style=\"color:{color}\"><strong>{heading}{duration_label}</strong></figcaption>"
                    f"<video controls preload=\"none\" src=\"{quote(rel_video, safe='/')}\"></video></figure>"
                )
                source_candidates = [
                    Path(item) for item in evidence.get("video_paths", [])
                    if Path(item) != file_path and Path(item).is_file()
                ]
                if source_candidates:
                    source_path = max(
                        source_candidates,
                        key=lambda item: float(
                            (evidence.get("ffprobe") or {}).get(str(item), {}).get("duration_seconds", 0)
                        ),
                    )
                    source_rel = source_path.relative_to(run_dir).as_posix()
                    source_duration = (
                        (evidence.get("ffprobe") or {}).get(str(source_path), {}).get("duration_seconds")
                    )
                    source_label = f" ({float(source_duration):.2f}s)" if source_duration else ""
                    card_parts.append(
                        f"<p>{name} 原始 H3 片{source_label}: "
                        f"<a href=\"{quote(source_rel, safe='/')}\">另開影片</a></p>"
                    )
            sheet = Path(pair.get("comparison_contact_sheet") or "")
            if sheet.is_file():
                rel_sheet = sheet.relative_to(run_dir).as_posix()
                card_parts.append(f"<a class=\"sheet\" href=\"{quote(rel_sheet, safe='/')}\"><img src=\"{quote(rel_sheet, safe='/')}\" alt=\"A on the left and B on the right, sampled at six matching timestamps\"></a>")
        else:
            card_parts.append("<p class=\"failed\">Incomplete pair: see pair.json for the technical failure. No creative winner assigned.</p>")
            for name in ("A", "B"):
                if name in variants:
                    video_path = str(variants[name].get("evidence", {}).get("final_video_path") or "")
                    if video_path:
                        rel_video = Path(video_path).relative_to(run_dir).as_posix()
                        video_names[name] = rel_video
                        card_parts.append(f"<p>{name}: <a href=\"{quote(rel_video, safe='/')}\">open rendered video</a></p>")
        blocks.append(f"<section id=\"{html.escape(case['case_id'])}\">{''.join(card_parts)}</section>")
        review_rows.append([
            case["case_id"], case["arc"], article["url"], video_names["A"], video_names["B"],
            *["" for _ in range(len(SCORE_FIELDS) * 2)], "", "",
        ])
    gallery = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Vox Director × MediaOverload 新聞弧線 A/B</title>
<style>body{font:16px/1.55 system-ui,'Microsoft JhengHei',sans-serif;background:#0b1220;color:#e5e7eb;margin:0;padding:24px}main{max-width:1500px;margin:auto}h1{margin:0}p{color:#cbd5e1}section{padding:22px 0;border-top:1px solid #334155}section h2{font-size:1.2rem}section>figure{display:inline-block;vertical-align:top;width:calc(50% - 18px);margin:10px 12px 14px 0}figcaption{padding:6px 0}video{width:100%;max-height:580px;background:#000;border-radius:8px}.sheet img{display:block;max-width:100%;height:auto;margin-top:12px;border:1px solid #334155}.failed{color:#fca5a5}code{color:#bbf7d0}</style><main><h1>新聞故事弧線 A/B 比較</h1><p>每列 A/B 共用來源文章、Kirby、seed、15 秒 Native H3 路由、共用文章範圍 brief 與技術 QA。A 是既有新聞契約加共用 brief；B 只增加本篇指定的弧線指令。主影片是套用既有輸出速度後的結果，實際秒數標在影片標題；另附原始 H3 片便於檢查未加速節奏。影片左右並排播放，縮圖表固定 A 在左、B 在右。</p><p>人工評分建議 0–4：來源事實吻合、因果推進、開場可讀性、弧線適配、payoff、角色／場景連續性、畫面品質。這 10 對是專家／人工可視判讀，不代表觀眾留存或偏好。</p>""" + "\n".join(blocks) + "</main></html>"
    (run_dir / "comparison_gallery.html").write_text(gallery, encoding="utf-8")
    with (run_dir / "manual_review_template.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(review_rows)


def _pin_experiment_models() -> None:
    os.environ["AGENTIC_LLM_MODE"] = "llm"
    os.environ["AGENTIC_TEXT_MODEL_PROVIDER"] = EXPERIMENT_TEXT_PROVIDER
    os.environ["AGENTIC_TEXT_MODEL"] = EXPERIMENT_TEXT_MODEL
    os.environ["AGENTIC_OLLAMA_NUM_CTX"] = "8192"
    os.environ["AGENTIC_OLLAMA_NUM_PREDICT"] = "2048"
    # Both engine-level limits must be pinned too; the Ollama-only timeout is
    # capped by the explicit request timeout and the engine's total deadline.
    os.environ["AGENTIC_LLM_REQUEST_TIMEOUT_SECONDS"] = str(EXPERIMENT_LLM_REQUEST_TIMEOUT_SECONDS)
    os.environ["AGENTIC_LLM_TOTAL_TIMEOUT_SECONDS"] = str(EXPERIMENT_LLM_TOTAL_TIMEOUT_SECONDS)
    os.environ["AGENTIC_OLLAMA_REQUEST_TIMEOUT_SECONDS"] = str(EXPERIMENT_LLM_REQUEST_TIMEOUT_SECONDS)
    os.environ["AGENTIC_OLLAMA_THINK"] = "false"
    os.environ["AGENTIC_VISION_MODEL_PROVIDER"] = "openrouter"
    os.environ["AGENTIC_VISION_MODEL"] = EXPERIMENT_VISION_MODEL
    os.environ["AGENTIC_OPENROUTER_ROTATE_TEXT_MODELS"] = "false"
    os.environ["AGENTIC_OPENROUTER_ROTATE_VISION_MODELS"] = "false"
    os.environ["AGENTIC_RANDOM_MODELS"] = "false"
    os.environ["AGENTIC_PROVIDER_FALLBACK_ENABLED"] = "false"
    os.environ["AGENTIC_TEXT_ALLOW_FALLBACK"] = "false"
    os.environ["AGENTIC_VISION_ALLOW_FALLBACK"] = "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ten matched news-grounded Vox-inspired Native H3 story-arc pairs")
    parser.add_argument("--output-root", default=r"E:\comfyui\_extra\benchmarks\vox_director_news_arc_ab")
    parser.add_argument("--comfy-root", default=r"D:\ComfyUI_windows_portable")
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8188)
    parser.add_argument("--seed-base", type=int, default=20260925)
    parser.add_argument("--case-id", action="append", help="Run selected manifest case(s); omit to run all ten")
    parser.add_argument("--resume-dir", help="Resume a previously interrupted run directory after verifying its controls")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root == REPO_ROOT or REPO_ROOT in output_root.parents:
        raise ValueError("Benchmark output must stay outside the repository")
    if not Path(args.comfy_root).is_dir():
        raise FileNotFoundError(f"ComfyUI root does not exist: {args.comfy_root}")
    resume_dir = Path(args.resume_dir).expanduser().resolve() if args.resume_dir else None
    if resume_dir:
        if not resume_dir.is_dir() or not (resume_dir / "run_config.json").is_file():
            raise FileNotFoundError(f"No benchmark run_config.json in resume directory: {resume_dir}")
        if resume_dir.parent != output_root:
            raise ValueError("--resume-dir must be a direct child of --output-root")
        prior_config = json.loads((resume_dir / "run_config.json").read_text(encoding="utf-8"))
        prior_case_ids = set((prior_config.get("case_arc_map") or {}).keys())
        if args.case_id:
            prior_case_ids &= set(args.case_id)
        _validate_resume_manifest_hash(
            resume_dir / "news_cases.json",
            str(prior_config.get("case_manifest_sha256") or ""),
        )
        dataset, selected_cases = _load_cases(
            prior_case_ids,
            data_path=resume_dir / "news_cases.json",
        )
    else:
        dataset, selected_cases = _load_cases(set(args.case_id or []))
    if not selected_cases:
        raise ValueError("No benchmark cases were selected.")
    _pin_experiment_models()
    _require_idle_comfy(args.comfy_host, int(args.comfy_port))
    run_id = resume_dir.name if resume_dir else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = resume_dir or (output_root / run_id)
    if not resume_dir:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "news_cases.json").write_text(DATA_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    existing_roots = [
        Path(item.strip()).expanduser().resolve()
        for item in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",")
        if item.strip()
    ]
    os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(
        str(path) for path in dict.fromkeys([*existing_roots, REPO_ROOT.resolve(), run_dir.resolve()])
    )
    fixed_controls = {
        "route": "native_h3_t2v_story -> MiniMax H3 T2V -> existing hard media QA",
        "generation_type": "native_h3_t2v_story",
        "duration_seconds": DURATION_SECONDS,
        "character": "Kirby",
        "subject_mode": "single",
        "text_model_provider": EXPERIMENT_TEXT_PROVIDER,
        "text_model": EXPERIMENT_TEXT_MODEL,
        "vision_model": EXPERIMENT_VISION_MODEL,
        "text_model_rotation": False,
        "vision_model_rotation": False,
        "ollama_num_ctx": 8192,
        "ollama_num_predict": 2048,
        "ollama_request_timeout_seconds": 600,
        "ollama_think": False,
        "random_models": False,
        "provider_fallback": False,
        "comfy_host": args.comfy_host,
        "comfy_port": int(args.comfy_port),
        "seed_base": int(args.seed_base),
        "shared_story_brief": dataset["shared_story_brief"],
        "arc_guidance_channel": "native_h3_arc_instruction_v1",
    }
    render_signature = _sha256(json.dumps(fixed_controls, sort_keys=True))
    run_config = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_repo": dataset["reference_repository"],
        "variant_a": "existing_news_contract_plus_shared_article_scope_brief",
        "variant_b": "same_contract_and_brief_plus_topic_matched_arc_guidance",
        "changed_variable": "separate_topic_matched_story_arc_guidance",
        "pair_count_target": len(selected_cases),
        "case_arc_map": {case["case_id"]: case["arc"] for case in selected_cases},
        "case_source_map": {case["case_id"]: case["article"]["url"] for case in selected_cases},
        "case_manifest_sha256": _sha256(DATA_PATH.read_text(encoding="utf-8")),
        "fixed_controls": fixed_controls,
        "render_signature": render_signature,
        "manual_review_required": True,
        "review_rubric": list(SCORE_FIELDS),
        "publish_after_generate": False,
        "dispatch": False,
        "production_prompt_changed": False,
        "model_execution_timeouts": {
            "llm_request_seconds": EXPERIMENT_LLM_REQUEST_TIMEOUT_SECONDS,
            "llm_total_seconds": EXPERIMENT_LLM_TOTAL_TIMEOUT_SECONDS,
            "ollama_request_seconds": EXPERIMENT_LLM_REQUEST_TIMEOUT_SECONDS,
        },
    }
    pairs: list[dict[str, Any]] = []
    if resume_dir:
        stored_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if stored_config.get("render_signature") != render_signature:
            raise ValueError("Resume controls do not match the saved run's render signature.")
        expected_timeouts = {
            "llm_request_seconds": EXPERIMENT_LLM_REQUEST_TIMEOUT_SECONDS,
            "llm_total_seconds": EXPERIMENT_LLM_TOTAL_TIMEOUT_SECONDS,
            "ollama_request_seconds": EXPERIMENT_LLM_REQUEST_TIMEOUT_SECONDS,
        }
        if stored_config.get("model_execution_timeouts") != expected_timeouts:
            raise ValueError("Resume model execution timeouts do not match the saved run controls.")
        run_config = stored_config
        _json_write(run_dir / "run_config.json", run_config)
        for case in selected_cases:
            pair_path = run_dir / "cases" / case["case_id"] / "pair.json"
            if not pair_path.is_file():
                continue
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            variants = pair.get("variants", {})
            complete = bool(
                pair.get("technical_both_passed")
                and all(variants.get(name, {}).get("evidence", {}).get("technical_pass") for name in ("A", "B"))
            )
            if not complete:
                print(f"Resuming incomplete pair: {case['case_id']}", flush=True)
                continue
            if not Path(pair.get("comparison_contact_sheet") or "").is_file():
                pair["comparison_contact_sheet"] = _comparison_sheet(
                    case,
                    variants,
                    run_dir / "cases" / case["case_id"],
                )
                _json_write(pair_path, pair)
            pairs.append(pair)
        _comparison_gallery(run_dir, selected_cases, pairs)
    else:
        _json_write(run_dir / "run_config.json", run_config)
    for index, case in enumerate(selected_cases, start=1):
        if any(pair["case_id"] == case["case_id"] for pair in pairs):
            continue
        seed = stable_seed(case["case_id"], int(args.seed_base))
        case_dir = run_dir / "cases" / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        pair_path = case_dir / "pair.json"
        existing_pair = json.loads(pair_path.read_text(encoding="utf-8")) if pair_path.is_file() else {}
        order = tuple(existing_pair.get("execution_order") or (("B", "A") if index % 2 else ("A", "B")))
        variants: dict[str, dict[str, Any]] = dict(existing_pair.get("variants") or {})
        print(f"[{index}/{len(selected_cases)}] {case['case_id']} arc={case['arc']} seed={seed} order={order}", flush=True)
        for variant in order:
            if variants.get(variant, {}).get("evidence", {}).get("technical_pass"):
                print(f"  {variant}: keeping prior successful attempt", flush=True)
                continue
            variants[variant] = _run_variant(
                case=case,
                variant=variant,
                seed=seed,
                case_dir=case_dir,
                args=args,
                render_signature=render_signature,
                shared_brief=dataset["shared_story_brief"],
            )
            partial = {
                "case_id": case["case_id"],
                "article": case["article"],
                "narrative_arc": case["arc"],
                "changed_variable": "separate_topic_matched_story_arc_guidance",
                "seed": seed,
                "execution_order": list(order),
                "control_signature": render_signature,
                "technical_both_passed": bool(
                    variants.get("A", {}).get("evidence", {}).get("technical_pass")
                    and variants.get("B", {}).get("evidence", {}).get("technical_pass")
                ),
                "creative_winner": "undecided_until_human_video_review",
                "variants": variants,
            }
            _json_write(pair_path, partial)
            print(
                f"  {variant}: technical_pass={variants[variant].get('evidence', {}).get('technical_pass')} "
                f"elapsed={variants[variant].get('evidence', {}).get('elapsed_seconds')}s",
                flush=True,
            )
            if not variants[variant].get("evidence", {}).get("technical_pass"):
                pairs.append(partial)
                _comparison_gallery(run_dir, selected_cases, pairs)
                _json_write(run_dir / "benchmark_summary.json", {
                    **run_config,
                    "pairs_completed": sum(item["technical_both_passed"] for item in pairs),
                    "technical_all_pairs_passed": False,
                    "pairs": pairs,
                    "adoption_decision": "not_adopt_due_to_technical_failure",
                })
                print(f"Evidence: {run_dir}", flush=True)
                print("Stopped after a technical failure; no remaining variants were submitted.", flush=True)
                return 2
        partial["comparison_contact_sheet"] = _comparison_sheet(case, variants, case_dir)
        pairs.append(partial)
        _json_write(case_dir / "pair.json", partial)
        _comparison_gallery(run_dir, selected_cases, pairs)
        _json_write(run_dir / "benchmark_summary.json", {
            **run_config,
            "pairs_completed": len(pairs),
            "technical_all_pairs_passed": True,
            "pairs": pairs,
            "adoption_decision": "pending_human_video_review",
        })
        _require_idle_comfy(args.comfy_host, int(args.comfy_port))
    _comparison_gallery(run_dir, selected_cases, pairs)
    _json_write(run_dir / "benchmark_summary.json", {
        **run_config,
        "pairs_completed": len(pairs),
        "technical_all_pairs_passed": all(pair["technical_both_passed"] for pair in pairs),
        "pairs": pairs,
        "adoption_decision": "pending_human_video_review",
    })
    print(f"Evidence: {run_dir}", flush=True)
    print(f"Technical A/B pairs passed: {sum(pair['technical_both_passed'] for pair in pairs)}/{len(pairs)}", flush=True)
    print(f"Review gallery: {run_dir / 'comparison_gallery.html'}", flush=True)
    return 0 if all(pair["technical_both_passed"] for pair in pairs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
