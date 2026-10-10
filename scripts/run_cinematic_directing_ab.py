"""Render a matched camera/story/style prompt experiment through Krea2 and MiniMax H3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont
import yaml


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
EXPERIMENT_VISION_MODEL = EXPERIMENT_TEXT_MODEL
EXPERIMENT_OLLAMA_URL = "http://127.0.0.1:11434"
EXPERIMENT_LLM_TIMEOUT_SECONDS = 600
DURATION_SECONDS = 6
SINGLE_PROTAGONIST_CONTRACT = (
    "Exactly one visible selected protagonist, Kirby; no duplicate, clone, reflection, miniature copy, "
    "background character, or second protagonist. Keep one dominant physical idea."
)
CAMERA_STYLE_STORY_TREATMENT = (
    "Directing notes: Preserve the requested action and style. For a causal story, keep one visible goal, one obstacle, "
    "and one turn that changes the outcome; each beat must change a visible state and cause the next, and the ending "
    "must resolve or reframe the opening. If the request is one action, do not invent a second problem or subplot. "
    "For a character-led beat, show anticipation in the face and body, the physical action with follow-through, and a "
    "brief personality-specific reaction to the result. Give the character a clear expression and changing silhouette; "
    "use a blink, bounce, squash/recoil, or accessory lag only when it fits that character and medium. Add at most one or "
    "two small secondary motions tied to the action. Vary the action's speed around the reaction and payoff so each beat "
    "can be read. For each shot, name its starting framing and angle, one continuous motivated camera move or an "
    "intentional lock, its direction and relation to the subject's path, what the move reveals, and its ending composition. "
    "If the camera needs a second operation, describe a separate shot. Keep camera mechanics coherent: pan and tilt "
    "rotate from a fixed point and do not change subject scale; dolly, truck, and arc travel through space and show "
    "parallax; follow/tracking keeps pace with the subject; zoom changes focal length without camera travel. A scale "
    "change must follow from camera travel, zoom, or subject movement. Keep the character's expression and the visible "
    "result together at the payoff; show the result before a reaction close-up and retain enough context. Keep the face "
    "and every story-critical object within frame with clear edge margins during and after the move. If a close-up would "
    "crop the reaction or result, stop wider or cut to a separate shot. Keep the "
    "face, goal, and action readable while the camera supports rather than competes with the performance. "
    "Describe the selected style with stable, visible rules for medium and surface, palette and light, and motion or "
    "transitions, then keep those rules consistent across shots. Use a medium-specific reveal only when it strengthens "
    "the story turn. Keep the character, important props, action, and faces readable."
)
SCORE_FIELDS = (
    "brief_and_source_fidelity",
    "story_causality",
    "camera_purpose_and_mechanics",
    "character_performance_and_charm",
    "motion_variety_and_pacing",
    "style_consistency",
    "identity_and_geography_continuity",
    "opening_and_payoff_fidelity",
    "overall_visual_quality",
)

CASES: tuple[dict[str, str], ...] = (
    {
        "case_id": "watercolor_kite_release",
        "style_profile": "storybook_watercolor",
        "style_label": "tactile storybook watercolor",
        "brief": "Kirby wants to fly a paper kite on a breezy hill. Its tail catches on a low branch and the kite drops. A gust frees it and lifts it into the sky while Kirby holds the line and watches.",
    },
    {
        "case_id": "watercolor_flower_recovery",
        "style_profile": "storybook_watercolor",
        "style_label": "tactile storybook watercolor",
        "brief": "Kirby wants to water one wilted flower with a nearly empty can. The first drop lands beside it. Kirby tilts the can carefully until the last drop reaches the soil, and the flower lifts one fresh leaf.",
    },
    {
        "case_id": "cel_puddle_crossing",
        "style_profile": "clean_cel_action",
        "style_label": "clean cel-shaded action",
        "brief": "Kirby wants to cross one wide puddle. His first jump falls short and splashes his feet. He spots a springy leaf at the edge, uses it for one second jump, and lands safely on the other side.",
    },
    {
        "case_id": "cel_runaway_hat",
        "style_profile": "clean_cel_action",
        "style_label": "clean cel-shaded action",
        "brief": "Kirby chases one small hat rolling down a grassy slope. A gust pushes it toward the edge, but the hat catches on a low bush. Kirby reaches it and settles the hat on his head.",
    },
    {
        "case_id": "painterly_glow_seed",
        "style_profile": "painterly_atmosphere",
        "style_label": "painterly atmospheric adventure",
        "brief": "Kirby wants to keep one tiny glowing seed alight in a windy meadow. The wind dims it. Kirby turns a broad stone into a windbreak, and the seed glows again, lighting the path immediately around it.",
    },
    {
        "case_id": "painterly_cloudlight_wait",
        "style_profile": "painterly_atmosphere",
        "style_label": "painterly atmospheric adventure",
        "brief": "Kirby follows one warm patch of sunlight across a quiet field. A cloud covers the sun and the patch disappears. Kirby waits beside one closed flower as the wind moves the cloud away; the light returns and the flower opens.",
    },
    {
        "case_id": "graphic_spinning_top",
        "style_profile": "graphic_color_pop",
        "style_label": "graphic color-pop fantasy",
        "brief": "Kirby wants to keep one bright spinning top upright on a smooth floor. It begins to wobble as it slows. Kirby gives the floor one careful tap, the top spins upright again, and Kirby holds a surprised pose.",
    },
    {
        "case_id": "graphic_paper_ring",
        "style_profile": "graphic_color_pop",
        "style_label": "graphic color-pop fantasy",
        "brief": "Kirby rolls one paper ring toward a small gap in the path. The ring stops against a stone. Kirby tilts one flat board, the ring rolls around the stone and returns to his feet, and he catches it upright.",
    },
    {
        "case_id": "tabletop_block_bridge",
        "style_profile": "tactile_tabletop",
        "style_label": "warm tactile miniature-world illustration",
        "brief": "Kirby stacks two wooden blocks to bridge a narrow gap. The top block tilts when he steps onto it. Kirby shifts the lower block until the bridge settles, then crosses and looks back at the stable bridge.",
    },
    {
        "case_id": "tabletop_marble_track",
        "style_profile": "tactile_tabletop",
        "style_label": "warm tactile miniature-world illustration",
        "brief": "Kirby rolls one glass marble down a short wooden track toward a hole. A leaf blocks the hole and sends the marble into a small cup beside the track. Kirby leans over the cup as the marble settles.",
    },
)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_seed(case_id: str, repeat: int, seed_base: int) -> int:
    offset = int(_sha256(f"{case_id}:{repeat}")[:8], 16) % 100_000
    return max(1, min(2_147_483_646, int(seed_base) + offset))


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
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(result.stdout)
    return {"duration_seconds": round(float(probe.get("format", {}).get("duration") or 0), 3), "streams": probe.get("streams", [])}


def _probe_contains_video(probe: dict[str, Any]) -> bool:
    if probe.get("error"):
        return False
    try:
        duration = float(probe.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(duration) or duration <= 0:
        return False
    for stream in probe.get("streams", []):
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        try:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if width > 0 and height > 0:
            return True
    return False


def _pin_experiment_models() -> None:
    os.environ["AGENTIC_LLM_MODE"] = "llm"
    os.environ["AGENTIC_TEXT_MODEL_PROVIDER"] = EXPERIMENT_TEXT_PROVIDER
    os.environ["AGENTIC_TEXT_MODEL"] = EXPERIMENT_TEXT_MODEL
    os.environ["AGENTIC_VISION_MODEL_PROVIDER"] = EXPERIMENT_TEXT_PROVIDER
    os.environ["AGENTIC_VISION_MODEL"] = EXPERIMENT_VISION_MODEL
    os.environ["OLLAMA_API_BASE_URL"] = EXPERIMENT_OLLAMA_URL
    os.environ["AGENTIC_LLM_REQUEST_TIMEOUT_SECONDS"] = str(EXPERIMENT_LLM_TIMEOUT_SECONDS)
    os.environ["AGENTIC_OLLAMA_REQUEST_TIMEOUT_SECONDS"] = str(EXPERIMENT_LLM_TIMEOUT_SECONDS)
    os.environ["AGENTIC_LLM_TOTAL_TIMEOUT_SECONDS"] = str(EXPERIMENT_LLM_TIMEOUT_SECONDS)
    os.environ["AGENTIC_RANDOM_MODELS"] = "false"
    os.environ["AGENTIC_PROVIDER_FALLBACK_ENABLED"] = "false"
    os.environ["AGENTIC_TEXT_ALLOW_FALLBACK"] = "false"
    os.environ["AGENTIC_VISION_ALLOW_FALLBACK"] = "false"


def _pinned_config(run_dir: Path, style_profile: str) -> Path:
    source = REPO_ROOT / "configs" / "characters" / "kirby.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    creative = payload["generation"]["creative_profile"]
    creative["canvas"]["mode"] = "fixed"
    creative["canvas"]["fixed"] = "landscape_16_9"
    creative["style"]["mode"] = "fixed"
    creative["style"]["fixed"] = style_profile
    payload["generation"]["subject_mode"] = "single"
    target = run_dir / "resolved_configs" / f"kirby_{style_profile}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def _extract_evidence(result: dict[str, Any], variant_dir: Path, elapsed_seconds: float) -> dict[str, Any]:
    generation = result.get("generation") if isinstance(result.get("generation"), dict) else {}
    generation_result = generation.get("result") if isinstance(generation.get("result"), dict) else {}
    state = generation_result.get("state") if isinstance(generation_result.get("state"), dict) else {}
    node_outputs = state.get("node_outputs") if isinstance(state.get("node_outputs"), dict) else {}
    qa = node_outputs.get("video-qa") if isinstance(node_outputs.get("video-qa"), dict) else {}
    lineage = state.get("prompt_lineage") if isinstance(state.get("prompt_lineage"), list) else []
    idea = next((item for item in lineage if isinstance(item, dict) and item.get("node_id") == "idea-brief"), {})
    backend = idea.get("llm_backend") if isinstance(idea.get("llm_backend"), dict) else {}
    paths = sorted(path.resolve() for path in variant_dir.rglob("*.mp4") if path.is_file())
    probes: dict[str, Any] = {}
    for path in paths:
        try:
            probes[str(path)] = _ffprobe(path)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
            probes[str(path)] = {"error": f"{type(exc).__name__}: {exc}"}
    probe_pass = bool(paths) and len(probes) == len(paths) and all(
        _probe_contains_video(probe) for probe in probes.values()
    )
    prompt_pass = bool(idea) and not bool(idea.get("fallback_reason") or idea.get("manager_error"))
    model_pass = (
        backend.get("text_provider") == EXPERIMENT_TEXT_PROVIDER
        and backend.get("text_model_raw") == EXPERIMENT_TEXT_MODEL
        and backend.get("vision_provider") == EXPERIMENT_TEXT_PROVIDER
        and backend.get("vision_model_raw") == EXPERIMENT_VISION_MODEL
    )
    return {
        "workflow_status": str(result.get("status") or "failed"),
        "technical_pass": bool(qa.get("passed")) and probe_pass and prompt_pass and model_pass,
        "qa_passed": bool(qa.get("passed")),
        "probe_pass": probe_pass,
        "prompt_generation_pass": prompt_pass,
        "model_pin_pass": model_pass,
        "text_provider": str(backend.get("text_provider") or ""),
        "effective_text_model": str(backend.get("text_model_raw") or ""),
        "vision_provider": str(backend.get("vision_provider") or ""),
        "effective_vision_model": str(backend.get("vision_model_raw") or ""),
        "qa_errors": [str(item) for item in qa.get("errors", []) if str(item).strip()],
        "contact_sheet_path": str(qa.get("contact_sheet_path") or ""),
        "video_paths": [str(path) for path in paths],
        "primary_video_path": str(max(paths, key=lambda path: path.stat().st_size)) if paths else "",
        "ffprobe": probes,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "failure_reason": str(result.get("failure_reason") or ""),
    }


def _run_variant(
    *, case: dict[str, str], variant: str, repeat: int, seed: int, case_dir: Path,
    args: argparse.Namespace, render_signature: str, config_path: Path,
) -> dict[str, Any]:
    variant_dir = case_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=False)
    base_prompt = f"Creative request: {case['brief']}\nRequested visual style: {case['style_label']}\n\n{SINGLE_PROTAGONIST_CONTRACT}"
    prompt = base_prompt if variant == "A" else f"{base_prompt}\n\n{CAMERA_STYLE_STORY_TREATMENT}"
    record: dict[str, Any] = {
        "case_id": case["case_id"],
        "repeat": repeat,
        "variant": variant,
        "changed_variable": "cinematic_camera_story_style_direction_package",
        "style_profile": case["style_profile"],
        "base_prompt_sha256": _sha256(base_prompt),
        "variant_prompt_sha256": _sha256(prompt),
        "render_signature": render_signature,
        "controls": {
            "generation_type": "text2image2video",
            "selected_character_name": "Kirby",
            "subject_mode": "single",
            "duration_seconds": DURATION_SECONDS,
            "temperature": 0.2,
            "seed": seed,
            "style_profile": case["style_profile"],
            "canvas_profile": "landscape_16_9",
            "config_sha256": _sha256(config_path.read_text(encoding="utf-8")),
            "comfy_host": args.comfy_host,
            "comfy_port": int(args.comfy_port),
            "text_model_provider": EXPERIMENT_TEXT_PROVIDER,
            "text_model": EXPERIMENT_TEXT_MODEL,
        },
        "prompt": prompt,
        "manual_review_scores": {name: None for name in SCORE_FIELDS},
    }
    _json_write(variant_dir / "variant_config.json", record)
    _require_idle_comfy(args.comfy_host, int(args.comfy_port))
    started = time.perf_counter()
    try:
        result = run_character_workflow(
            CharacterWorkflowRequest(
                repo_root=REPO_ROOT,
                config_path=config_path,
                generation=CharacterGenerationOptions(
                    prompt=prompt,
                    temperature=0.2,
                    preferred_generation_type="text2image2video",
                    duration_seconds=DURATION_SECONDS,
                    output_dir=str(variant_dir),
                    rng=random.Random(seed),
                    seed=seed,
                    subject_mode="single",
                    selected_character_name="Kirby",
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
        evidence = _extract_evidence(result_dict, variant_dir, time.perf_counter() - started)
        _json_write(variant_dir / "workflow_result.json", result_dict)
    except Exception as exc:
        evidence = {
            "workflow_status": "failed", "technical_pass": False, "qa_passed": False, "probe_pass": False,
            "prompt_generation_pass": False, "model_pin_pass": False, "qa_errors": [],
            "contact_sheet_path": "", "video_paths": [], "primary_video_path": "", "ffprobe": {},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    record["evidence"] = evidence
    _json_write(variant_dir / "variant.json", record)
    return record


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pair_sheet(pair_dir: Path, variants: dict[str, dict[str, Any]], x_variant: str) -> str:
    labeled: list[tuple[str, Image.Image]] = []
    for label, variant in (("X", x_variant), ("Y", "B" if x_variant == "A" else "A")):
        source = Path(variants[variant]["evidence"].get("contact_sheet_path") or "")
        if not source.is_file():
            continue
        image = Image.open(source).convert("RGB")
        labeled.append((label, image))
    if len(labeled) != 2:
        return ""
    target_height = max(image.height for _, image in labeled)
    resized: list[tuple[str, Image.Image]] = []
    for label, image in labeled:
        width = round(image.width * target_height / image.height)
        resized.append((label, image.resize((width, target_height), Image.Resampling.LANCZOS)))
    pad, header = 20, 56
    canvas = Image.new("RGB", (sum(image.width for _, image in resized) + pad * 3, target_height + header + pad * 2), "#101722")
    draw = ImageDraw.Draw(canvas)
    cursor = pad
    for label, image in resized:
        draw.text((cursor, pad), f"{label} · blinded variant", fill="#f1f5f9", font=_font(28))
        canvas.paste(image, (cursor, pad + header))
        cursor += image.width + pad
    path = pair_dir / "blind_contact_sheet.png"
    canvas.save(path)
    return str(path)


def _write_blind_gallery(run_dir: Path, pairs: list[dict[str, Any]], blind_key: list[dict[str, Any]]) -> None:
    _json_write(run_dir / "unblinding_key.json", {"pairs": blind_key})
    rows: list[dict[str, str]] = []
    sections: list[str] = []
    for pair in pairs:
        x_variant = str(pair["blind_order"][0])
        y_variant = "B" if x_variant == "A" else "A"
        case = pair["case"]
        pair_dir = run_dir / "cases" / str(pair["case_id"]) / f"repeat-{int(pair['repeat']):02d}"
        blind_pair_id = f"{pair['case_id']}__r{int(pair['repeat']):02d}"
        blind_assets_dir = run_dir / "blind_assets" / blind_pair_id
        blind_assets_dir.mkdir(parents=True, exist_ok=True)
        contact = _pair_sheet(pair_dir, pair["variants"], x_variant)
        headings: list[str] = []
        for label, variant in (("X", x_variant), ("Y", y_variant)):
            evidence = pair["variants"][variant]["evidence"]
            video = Path(evidence.get("primary_video_path") or "")
            blind_video = blind_assets_dir / f"{label}_video.mp4"
            if video.is_file():
                shutil.copyfile(video, blind_video)
            video_ref = blind_video.relative_to(run_dir).as_posix() if blind_video.is_file() else ""
            contact_sheet = Path(evidence.get("contact_sheet_path") or "")
            blind_sheet = blind_assets_dir / f"{label}_contact_sheet.jpg"
            if contact_sheet.is_file():
                shutil.copyfile(contact_sheet, blind_sheet)
            sheet_ref = blind_sheet.relative_to(run_dir).as_posix() if blind_sheet.is_file() else ""
            video_markup = (
                f'<video controls preload="none" src="{html.escape(video_ref)}"></video>'
                if video_ref else "<p>Video unavailable</p>"
            )
            sheet_markup = (
                f'<img class="individual" src="{html.escape(sheet_ref)}">'
                if sheet_ref else ""
            )
            headings.append(f'<div><h3>Variant {label}</h3>{video_markup}{sheet_markup}</div>')
        pair_sheet_ref = (
            f'cases/{html.escape(str(pair["case_id"]))}/repeat-{int(pair["repeat"]):02d}/blind_contact_sheet.png'
            if contact else ""
        )
        pair_sheet_markup = (
            f'<img class="pair-sheet" src="{pair_sheet_ref}">' if pair_sheet_ref else ""
        )
        sections.append(
            f'<section><h2>{html.escape(blind_pair_id)}</h2><p>{html.escape(case["brief"])}</p>'
            f'{pair_sheet_markup}'
            f'<div class="videos">{"".join(headings)}</div></section>'
        )
        rows.append({"pair_id": blind_pair_id, "case_id": str(pair["case_id"]), "repeat": str(pair["repeat"])})
    gallery = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind cinematic direction review</title>
<style>body{font:15px/1.5 system-ui,sans-serif;background:#0b1220;color:#e5e7eb;margin:0;padding:24px}main{max-width:1600px;margin:auto}section{border-top:1px solid #334155;padding:22px 0}h1,h2,h3{margin:0 0 8px}p{color:#cbd5e1}.pair-sheet{display:block;width:min(100%,1400px);height:auto;margin:14px 0;border:1px solid #334155}.videos{display:grid;grid-template-columns:1fr 1fr;gap:20px}video{width:100%;max-height:520px;background:#000}.individual{display:block;width:100%;height:auto;margin-top:8px}@media(max-width:850px){.videos{grid-template-columns:1fr}}</style>
<main><h1>Blind X/Y review · camera, story, and style</h1><p>Review the same brief in both columns. Score X and Y before opening the separate unblinding key. The contact sheets are for action and style; watch the full clip for timing and motion.</p>__SECTIONS__</main></html>""".replace("__SECTIONS__", "\n".join(sections))
    (run_dir / "blind_review.html").write_text(gallery, encoding="utf-8")
    with (run_dir / "blind_review_scores.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["pair_id", "case_id", "repeat", *[f"X_{field}" for field in SCORE_FIELDS], *[f"Y_{field}" for field in SCORE_FIELDS], "preference", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **{key: "" for key in [*[f"X_{field}" for field in SCORE_FIELDS], *[f"Y_{field}" for field in SCORE_FIELDS], "preference", "notes"]}})


def _selected_cases(case_ids: list[str] | None) -> list[dict[str, str]]:
    if not case_ids:
        return list(CASES)
    by_id = {case["case_id"]: case for case in CASES}
    unknown = set(case_ids) - set(by_id)
    if unknown:
        raise ValueError(f"Unknown case id(s): {', '.join(sorted(unknown))}")
    return [case for case in CASES if case["case_id"] in set(case_ids)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 10 creative briefs × 2 matched seeds through the real Krea2 → MiniMax H3 route")
    parser.add_argument("--output-root", default=r"E:\comfyui\_extra\benchmarks\cinematic_directing_ab")
    parser.add_argument("--comfy-root", default=r"D:\ComfyUI_windows_portable")
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8188)
    parser.add_argument("--seed-base", type=int, default=20261004)
    parser.add_argument("--repeats", type=int, default=2, choices=(1, 2), help="Matched independent seeds per creative brief")
    parser.add_argument("--case-id", action="append", choices=[case["case_id"] for case in CASES])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root == REPO_ROOT or REPO_ROOT in output_root.parents:
        raise ValueError("Benchmark output must stay outside the repository")
    if not Path(args.comfy_root).is_dir():
        raise FileNotFoundError(f"ComfyUI root does not exist: {args.comfy_root}")
    cases = _selected_cases(args.case_id)
    _pin_experiment_models()
    _require_idle_comfy(args.comfy_host, int(args.comfy_port))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_paths: dict[str, Path] = {}
    for style_profile in sorted({case["style_profile"] for case in cases}):
        config_paths[style_profile] = _pinned_config(run_dir, style_profile)
    os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(
        str(path.resolve())
        for path in dict.fromkeys(
            [*(Path(item.strip()).expanduser().resolve() for item in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",") if item.strip()), REPO_ROOT.resolve(), run_dir.resolve()]
        )
    )
    fixed_controls = {
        "route": "text2image2video -> Krea2 keyframe -> MiniMax H3 I2V -> existing hard media QA",
        "duration_seconds": DURATION_SECONDS,
        "character": "Kirby",
        "subject_mode": "single",
        "temperature": 0.2,
        "llm_request_timeout_seconds": EXPERIMENT_LLM_TIMEOUT_SECONDS,
        "canvas_profile": "landscape_16_9",
        "text_model_provider": EXPERIMENT_TEXT_PROVIDER,
        "text_model": EXPERIMENT_TEXT_MODEL,
        "vision_model_provider": EXPERIMENT_TEXT_PROVIDER,
        "vision_model": EXPERIMENT_VISION_MODEL,
        "ollama_url": EXPERIMENT_OLLAMA_URL,
        "provider_fallback": False,
        "comfy_host": args.comfy_host,
        "comfy_port": int(args.comfy_port),
        "seed_base": int(args.seed_base),
        "repeats_per_case": int(args.repeats),
        "styles": {profile: _sha256(path.read_text(encoding="utf-8")) for profile, path in config_paths.items()},
    }
    render_signature = _sha256(json.dumps(fixed_controls, sort_keys=True))
    run_config = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variant_a": "baseline_prompt",
        "variant_b": "camera_story_style_direction_package",
        "changed_variable": "cinematic_camera_story_style_direction_package",
        "prompt_count": len(cases),
        "repeat_count_per_prompt": int(args.repeats),
        "pair_count_target": len(cases) * int(args.repeats),
        "treatment": CAMERA_STYLE_STORY_TREATMENT,
        "cases": [{key: value for key, value in case.items()} for case in cases],
        "fixed_controls": fixed_controls,
        "render_signature": render_signature,
        "manual_review_required": True,
        "blind_review": True,
        "review_rubric": list(SCORE_FIELDS),
        "publish_after_generate": False,
        "dispatch": False,
        "production_prompt_changed": False,
    }
    _json_write(run_dir / "run_config.json", run_config)

    pairs: list[dict[str, Any]] = []
    blind_key: list[dict[str, Any]] = []
    pair_index = 0
    blind_rng = random.SystemRandom()
    for case in cases:
        config_path = config_paths[case["style_profile"]]
        for repeat in range(1, int(args.repeats) + 1):
            pair_index += 1
            seed = stable_seed(case["case_id"], repeat, int(args.seed_base))
            pair_id = f"{case['case_id']}__r{repeat:02d}"
            pair_dir = run_dir / "cases" / case["case_id"] / f"repeat-{repeat:02d}"
            pair_dir.mkdir(parents=True, exist_ok=False)
            order = ("A", "B") if pair_index % 2 else ("B", "A")
            variants: dict[str, dict[str, Any]] = {}
            print(f"[{pair_index}/{len(cases) * args.repeats}] {pair_id} | style={case['style_profile']} | seed={seed} | order={order}", flush=True)
            for variant in order:
                variants[variant] = _run_variant(
                    case=case, variant=variant, repeat=repeat, seed=seed,
                    case_dir=pair_dir, args=args, render_signature=render_signature,
                    config_path=config_path,
                )
                if not variants[variant]["evidence"].get("technical_pass"):
                    pair = {
                        "pair_id": pair_id, "case_id": case["case_id"], "repeat": repeat,
                        "seed": seed, "execution_order": list(order), "blind_order": [],
                        "control_signature": render_signature, "technical_both_passed": False,
                        "creative_winner": "not_scored_technical_failure", "case": case, "variants": variants,
                    }
                    pairs.append(pair)
                    _json_write(run_dir / "benchmark_summary.json", {**run_config, "pairs_completed": len(pairs), "pairs": pairs})
                    print(f"Evidence: {run_dir}", flush=True)
                    print("Stopped after a technical failure; no remaining variants were submitted.", flush=True)
                    return 2
                _require_idle_comfy(args.comfy_host, int(args.comfy_port))
            blind_order = ("A", "B") if blind_rng.randrange(2) == 0 else ("B", "A")
            pair = {
                "pair_id": pair_id, "case_id": case["case_id"], "repeat": repeat,
                "seed": seed, "execution_order": list(order), "blind_order": list(blind_order),
                "control_signature": render_signature, "technical_both_passed": True,
                "creative_winner": "undecided_until_blind_manual_review", "case": case, "variants": variants,
            }
            pairs.append(pair)
            blind_key.append({"pair_id": pair_id, "X": blind_order[0], "Y": blind_order[1]})
            _json_write(run_dir / "benchmark_summary.json", {**run_config, "pairs_completed": len(pairs), "pairs": pairs})
            print(f"[{pair_index}/{len(cases) * args.repeats}] complete | A/B technical pass", flush=True)

    _write_blind_gallery(run_dir, pairs, blind_key)
    summary = {
        **run_config,
        "pairs_completed": len(pairs),
        "technical_both_passed_pairs": sum(bool(pair["technical_both_passed"]) for pair in pairs),
        "pairs": pairs,
        "adoption_decision": "pending_blind_creative_review",
    }
    _json_write(run_dir / "benchmark_summary.json", summary)
    print(f"Evidence: {run_dir}", flush=True)
    print(f"Technical matched pairs passed: {summary['technical_both_passed_pairs']}/{len(pairs)}", flush=True)
    return 0 if len(pairs) == len(cases) * int(args.repeats) else 1


if __name__ == "__main__":
    raise SystemExit(main())
