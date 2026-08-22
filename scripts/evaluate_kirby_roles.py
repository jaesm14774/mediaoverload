"""Evaluate every Kirby anime role through the local Krea and Nova workflows.

This is an isolated data-quality runner. It reads the Kirby role group, renders
the same role contract through both text-to-image workflows, asks the local
vision model to judge identity, and writes resumable evidence. It deliberately
does not update MySQL; the separate apply step must consume a complete report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.assets.registry import AssetRegistry
from agentic.runtime.model_backends import ModelConfig, OllamaModel
from agentic.runtime.registry import ToolRegistry
from agentic.tools.comfy_workflow_tool import register_comfy_workflow_tools


ROLE_GROUP = "Kirby"
TABLE_NAME = "anime.anime_roles"
KREA_WORKFLOW = "krea2_turbo"
NOVA_WORKFLOW = "nova-anime-xl"
DEFAULT_COMFY_ROOT = Path(r"D:\ComfyUI_windows_portable")
DEFAULT_NOVA_ASSET_ROOT = Path(r"E:\comfyui_extra")
DEFAULT_OUTPUT_ROOT = DEFAULT_COMFY_ROOT / "ComfyUI" / "output" / "mediaoverload_kirby_role_eval" / "20260822_clean_prompt"
DEFAULT_VISION_MODEL = "qwen3.8-27b-ud-q2xl-local"
WIKI_API = "https://kirby.fandom.com/api.php"
WIKI_BASE = "https://kirby.fandom.com/wiki/"
WIKI_USER_AGENT = "MediaOverloadRoleResearch/1.0 (local visual identity audit)"
IDENTITY_PASS_SCORE = 80

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "krea": {
            "type": "object",
            "properties": {
                "identity_match": {"type": "boolean"},
                "identity_score": {"type": "integer"},
            },
            "required": ["identity_match", "identity_score"],
        },
        "nova": {
            "type": "object",
            "properties": {
                "candidate_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate": {"type": "string"},
                            "identity_match": {"type": "boolean"},
                            "identity_score": {"type": "integer"},
                        },
                        "required": ["candidate", "identity_match", "identity_score"],
                    },
                }
            },
            "required": ["candidate_results"],
        },
        "notes": {"type": "string"},
    },
    "required": ["krea", "nova", "notes"],
}

KREA_ONLY_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "krea": JUDGE_SCHEMA["properties"]["krea"],
        "notes": {"type": "string"},
    },
    "required": ["krea", "notes"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")[:100] or "role"


def _load_database_rows() -> list[dict[str, Any]]:
    import pymysql

    connection = pymysql.connect(
        host=os.getenv("mysql_host"),
        port=int(os.getenv("mysql_port", "3306")),
        user=os.getenv("mysql_user"),
        password=os.getenv("mysql_password"),
        database=os.getenv("mysql_db_name"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        read_timeout=30,
        write_timeout=30,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id, role_name_zh, role_name_en, role_description, keywords, group_name, status, weight "
                f"FROM {TABLE_NAME} WHERE group_name=%s ORDER BY id",
                (ROLE_GROUP,),
            )
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def _role_prompt(role: dict[str, Any]) -> str:
    english = str(role.get("role_name_en") or "").strip()
    chinese = str(role.get("role_name_zh") or "").strip()
    description = str(role.get("role_description") or "").strip()
    keywords = str(role.get("keywords") or "").strip()
    return (
        f"Character identity: {english} ({chinese}). "
        f"Role description: {description}. "
        f"Visual identity keywords: {keywords}. "
        "Create exactly one full-body depiction of this named role, centered on a plain light background, "
        "clean 2D anime game illustration, readable silhouette, no other characters, no collage, "
        "no character sheet, no turnaround, no text, no watermark. Preserve only this role's defining "
        "species, body shape, colors, face, clothing, weapon, and accessories from the supplied description."
    )


def _negative_prompt() -> str:
    return (
        "multiple characters, collage, character sheet, turnaround, character lineup, duplicate, "
        "generic mascot, wrong identity, text, watermark, logo, speech bubble, blurry, deformed, "
        "extra limbs, cropped body"
    )


def _load_source_records() -> dict[str, dict[str, Any]]:
    path = REPO_ROOT / "artifacts" / "kirby_role_research" / "20260822" / "source_harvest.json"
    if not path.is_file():
        raise RuntimeError(f"Kirby source harvest is missing: {path}")
    payload = _read_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise RuntimeError(f"Kirby source harvest has no records: {path}")
    return {
        str(record.get("role_name_en") or "").strip(): record
        for record in records
        if isinstance(record, dict) and record.get("resolution") == "exact_or_alias"
    }


def _ensure_web_reference(
    role: dict[str, Any],
    role_dir: Path,
    source_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    english = str(role.get("role_name_en") or "").strip()
    source = source_records.get(english)
    if not source:
        return {"status": "unavailable", "error": f"no exact web source for {english}"}
    source_title = str(source.get("source_title") or "").strip()
    reference_dir = role_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = reference_dir / "web_reference.json"
    image_path = reference_dir / "web_reference.jpg"
    if image_path.is_file() and metadata_path.is_file():
        metadata = _read_json(metadata_path) or {}
        return {"status": "ready", "path": str(image_path), **metadata}

    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "pageimages",
            "piprop": "original|thumbnail",
            "pithumbsize": "1024",
            "titles": source_title,
            "redirects": "1",
        }
    )
    request = urllib.request.Request(
        f"{WIKI_API}?{params}",
        headers={"User-Agent": WIKI_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        pages = payload.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        image_info = page.get("original") or page.get("thumbnail") or {}
        image_url = str(image_info.get("source") or "").strip()
        if not image_url:
            return {"status": "unavailable", "error": f"source page has no image: {source_title}"}
        image_request = urllib.request.Request(image_url, headers={"User-Agent": WIKI_USER_AGENT})
        with urllib.request.urlopen(image_request, timeout=60) as response:
            image_path.write_bytes(response.read())
        metadata = {
            "source_title": source_title,
            "source_url": f"{WIKI_BASE}{urllib.parse.quote(source_title.replace(' ', '_'))}",
            "image_url": image_url,
            "downloaded_at": _now(),
        }
        _write_json(metadata_path, metadata)
        return {"status": "ready", "path": str(image_path), **metadata}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def _free_comfy_memory(host: str, port: int) -> dict[str, Any]:
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}/free",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return {"status": "success", "http_status": response.status}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def _render(
    tools: ToolRegistry,
    *,
    workflow_name: str,
    role: dict[str, Any],
    role_dir: Path,
    seed: int,
) -> dict[str, Any]:
    model_dir = role_dir / workflow_name
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        result = tools.call(
            "comfy.workflow.text_to_image",
            {
                "workflow_name": workflow_name,
                "run_dir": str(model_dir),
                "prompt": _role_prompt(role),
                "negative_prompt": _negative_prompt(),
                "width": 1024,
                "height": 576,
                "image_count": 1,
                "seed": seed,
            },
        )
        files = [
            str(Path(str(item)))
            for item in (result.get("saved_files", []) if isinstance(result, dict) else [])
            if Path(str(item)).is_file()
        ]
        payload = {
            "status": "success" if files else "failed",
            "workflow_name": workflow_name,
            "files": files,
            "result": result,
            "duration_seconds": round(time.perf_counter() - started, 2),
            "completed_at": _now(),
        }
        if not files:
            payload["error"] = "workflow returned no image files"
        _write_json(model_dir / "render_result.json", payload)
        return payload
    except Exception as exc:
        payload = {
            "status": "unavailable",
            "workflow_name": workflow_name,
            "files": [],
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
            "completed_at": _now(),
        }
        _write_json(model_dir / "render_result.json", payload)
        return payload


def _make_contact_sheet(
    role_dir: Path,
    render_results: dict[str, dict[str, Any]],
    reference: dict[str, Any] | None = None,
    workflow_names: tuple[str, ...] = (KREA_WORKFLOW, NOVA_WORKFLOW),
) -> Path | None:
    from PIL import Image, ImageDraw, ImageFont

    tiles: list[tuple[str, Path]] = []
    reference_path = Path(str(reference.get("path"))) if isinstance(reference, dict) and reference.get("path") else None
    if reference_path and reference_path.is_file():
        tiles.append(("WEB REFERENCE", reference_path))
    for model_name in workflow_names:
        result = render_results.get(model_name)
        if not isinstance(result, dict):
            continue
        for index, raw_path in enumerate(result.get("files", []), start=1):
            path = Path(str(raw_path))
            if path.is_file():
                tiles.append((f"{model_name} candidate {index}", path))
    if not tiles:
        return None

    tile_width, tile_height, label_height = 512, 288, 34
    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, path) in enumerate(tiles):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_width, tile_height))
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        image_x = x + (tile_width - image.width) // 2
        image_y = y + label_height + (tile_height - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        draw.rectangle((x, y, x + tile_width, y + label_height), fill=(30, 30, 30))
        draw.text((x + 8, y + 8), label, fill="white", font=font)
    output = role_dir / "comparison_contact_sheet.jpg"
    sheet.save(output, quality=90, optimize=True)
    return output


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("vision judge returned a non-object JSON value")
    return payload


def _judge(
    judge_model: OllamaModel,
    *,
    role: dict[str, Any],
    contact_sheet: Path,
    reference: dict[str, Any],
    krea_only: bool = False,
) -> dict[str, Any]:
    english = str(role.get("role_name_en") or "").strip()
    chinese = str(role.get("role_name_zh") or "").strip()
    description = str(role.get("role_description") or "").strip()
    keywords = str(role.get("keywords") or "").strip()
    if krea_only:
        candidate_instruction = (
            "The only generated candidate to evaluate is the Krea2 Turbo tile. Do not evaluate or infer any Nova result; "
            "Nova is intentionally disabled for this run."
        )
    else:
        candidate_instruction = "Evaluate Krea as one candidate and every Nova candidate separately."
    prompt = (
        "Judge the attached contact sheet as a strict character identity test against the WEB REFERENCE tile. "
        f"The requested role is {english} ({chinese}). Description: {description}. Keywords: {keywords}. "
        f"The WEB REFERENCE comes from the online source page {reference.get('source_title', '')}. "
        "Ignore any text or labels inside the reference image itself. "
        f"{candidate_instruction} Identity means the image depicts this exact named role, not merely a generic creature. "
        "Compare the generated image to the web reference first, then use the description only to resolve small visual ambiguity. "
        "Compare species/body, colors, face, clothing, weapon, and accessories. "
        f"Use identity_match=true only when identity_score is at least {IDENTITY_PASS_SCORE}. "
        "A multi-view tile may still pass identity if all visible views are the same requested role; mention "
        "composition problems in reasons but do not confuse them with identity. Return JSON only."
    )
    messages = [
        {"role": "system", "content": "You are a strict visual character-identity evaluator. Return JSON only."},
        {"role": "user", "content": prompt},
    ]
    started = time.perf_counter()
    try:
        raw = judge_model.chat_completion(
            messages=messages,
            images=[str(contact_sheet)],
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": KREA_ONLY_JUDGE_SCHEMA if krea_only else JUDGE_SCHEMA},
            },
            request_timeout=180,
        )
        return {
            "status": "success",
            "payload": _parse_json_response(raw),
            "duration_seconds": round(time.perf_counter() - started, 2),
            "completed_at": _now(),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
            "completed_at": _now(),
        }


def _krea_pass(judgement: dict[str, Any]) -> bool:
    value = judgement.get("krea") if isinstance(judgement, dict) else None
    return bool(isinstance(value, dict) and value.get("identity_match") is True and int(value.get("identity_score", 0)) >= IDENTITY_PASS_SCORE)


def _nova_pass(judgement: dict[str, Any]) -> bool:
    value = judgement.get("nova") if isinstance(judgement, dict) else None
    candidates = value.get("candidate_results") if isinstance(value, dict) else None
    if not isinstance(candidates, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("identity_match") is True
        and int(item.get("identity_score", 0)) >= IDENTITY_PASS_SCORE
        for item in candidates
    )


def _decision(
    role: dict[str, Any],
    renders: dict[str, dict[str, Any]],
    judge: dict[str, Any],
    reference: dict[str, Any],
    krea_only: bool = False,
) -> dict[str, Any]:
    before = int(role.get("status", -1))
    required_workflows = (KREA_WORKFLOW,) if krea_only else (KREA_WORKFLOW, NOVA_WORKFLOW)
    render_ready = all(
        isinstance(renders.get(workflow_name), dict)
        and renders[workflow_name].get("status") == "success"
        for workflow_name in required_workflows
    )
    judged = judge.get("status") == "success"
    reference_ready = reference.get("status") == "ready"
    krea_pass = _krea_pass(judge.get("payload", {})) if judged else False
    nova_pass = _nova_pass(judge.get("payload", {})) if judged else False
    qualified = reference_ready and render_ready and judged and krea_pass if krea_only else reference_ready and render_ready and judged and krea_pass and nova_pass
    recommended = 1 if qualified else -1
    if qualified and krea_only:
        reason = "online reference exists and Krea2 Turbo passed the strict identity gate"
    elif qualified:
        reason = "online reference exists and both Krea2 Turbo and Nova passed the strict identity gate"
    elif not reference_ready:
        reason = "online reference unavailable; strict gate fails"
    elif not render_ready or not judged:
        reason = "render or visual judgement incomplete; strict gate fails"
    else:
        reason = "the Krea2 Turbo result failed the strict identity gate against the online reference" if krea_only else "at least one model failed the strict identity gate against the online reference"
    return {
        "status_before": before,
        "recommended_status": recommended,
        "reference_ready": reference_ready,
        "krea_identity_pass": krea_pass,
        "nova_identity_pass": nova_pass,
        "both_models_pass": bool(qualified and not krea_only),
        "krea_only": krea_only,
        "reason": reason,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all roles in anime.anime_roles group Kirby.")
    parser.add_argument("--comfy-root", default=str(DEFAULT_COMFY_ROOT))
    parser.add_argument("--nova-asset-root", default=str(DEFAULT_NOVA_ASSET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8188)
    parser.add_argument("--limit", type=int, default=0, help="Optional bounded smoke limit; zero means all current Kirby roles.")
    parser.add_argument("--role-id", type=int, action="append", default=[])
    parser.add_argument(
        "--skip-nova",
        action="store_true",
        help="Render only Krea2 Turbo; do not submit any Nova workflow task.",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "render", "judge"),
        default="all",
        help="all renders then judges in two phases; render and judge can be resumed separately",
    )
    return parser.parse_args()


def _role_dir(output_root: Path, role: dict[str, Any]) -> Path:
    return output_root / "roles" / f"{int(role['id'])}_{_safe_slug(str(role.get('role_name_en') or role.get('role_name_zh') or 'role'))}"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _existing_renders(
    role_dir: Path,
    required_workflows: tuple[str, ...] = (KREA_WORKFLOW, NOVA_WORKFLOW),
) -> dict[str, dict[str, Any]] | None:
    render_record = _read_json(role_dir / "render_evidence.json")
    if render_record and isinstance(render_record.get("renders"), dict):
        renders = render_record["renders"]
        if all(isinstance(renders.get(name), dict) and renders[name].get("status") == "success" for name in required_workflows):
            return renders
    evaluation = _read_json(role_dir / "evaluation.json")
    if evaluation and isinstance(evaluation.get("renders"), dict):
        renders = evaluation["renders"]
        if all(isinstance(renders.get(name), dict) and renders[name].get("status") == "success" for name in required_workflows):
            return renders
    return None


def main() -> int:
    args = _parse_args()
    phase = str(args.phase)
    load_dotenv(REPO_ROOT / "media_overload.env")
    output_root = Path(args.output_root).expanduser().resolve()
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    nova_asset_root = Path(args.nova_asset_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(
        [str(REPO_ROOT), str(output_root), str(comfy_root / "ComfyUI" / "output")]
    )

    roles = _load_database_rows()
    if len(roles) != 75:
        raise RuntimeError(f"Expected 75 remaining Kirby roles, found {len(roles)}")
    source_records = _load_source_records()
    if args.role_id:
        roles = [role for role in roles if int(role["id"]) in set(args.role_id)]
    if args.limit > 0:
        roles = roles[: args.limit]

    snapshot_path = output_root / "kirby_roles_db_snapshot.json"
    if not snapshot_path.exists():
        _write_json(snapshot_path, {"captured_at": _now(), "group_name": ROLE_GROUP, "table": TABLE_NAME, "rows": _load_database_rows()})

    krea_assets: list[dict[str, Any]] = []
    nova_assets = [
        nova_asset_root / "models" / "checkpoints" / "sdxl" / "novaAnimeXL_ilV180.safetensors",
        nova_asset_root / "models" / "checkpoints" / "sdxl" / "novaAnimeXL_ilV140.safetensors",
        nova_asset_root / "models" / "checkpoints" / "sdxl" / "waiIllustriousSDXL_v150.safetensors",
        nova_asset_root / "models" / "loras" / "sdxl" / "reiXL_NB11.safetensors",
    ]
    missing_nova = [str(path) for path in nova_assets if not path.is_file()]
    if missing_nova and not args.skip_nova:
        raise RuntimeError(f"Nova workflow assets are missing: {missing_nova}")

    vision_model_name = os.getenv("AGENTIC_VISION_MODEL", "").strip() or DEFAULT_VISION_MODEL
    progress_path = output_root / "progress.json"
    progress: dict[str, Any] = {
        "started_at": _now(),
        "group_name": ROLE_GROUP,
        "total_roles": len(roles),
        "rendered_role_ids": [],
        "judged_role_ids": [],
        "reference_ready_role_ids": [],
        "krea_assets": krea_assets,
        "nova_assets": [str(path) for path in nova_assets],
        "nova_disabled": bool(args.skip_nova),
        "vision_model": vision_model_name,
    }
    previous = _read_json(progress_path)
    if previous:
        progress.update(previous)

    if phase in {"all", "render"}:
        # Krea assets are verified against D; Nova is verified from the
        # workflow's actual loader names under E because its legacy manifest
        # has no declarations.
        registry = AssetRegistry(REPO_ROOT / "agentic", asset_root=comfy_root)
        krea_manifest = registry.get_manifest(KREA_WORKFLOW)
        krea_assets = registry.ensure_requirements(krea_manifest, auto_download=False)
        if not krea_assets or not all(item.get("status") == "ready" for item in krea_assets):
            raise RuntimeError(f"Krea2 Turbo assets are not ready: {krea_assets}")
        progress["krea_assets"] = krea_assets
        tools = ToolRegistry()
        register_comfy_workflow_tools(tools, registry, output_root, comfy_host=args.comfy_host, comfy_port=args.comfy_port)
        render_workflows = (KREA_WORKFLOW,) if args.skip_nova else (KREA_WORKFLOW, NOVA_WORKFLOW)

        rendered_ids = {int(value) for value in progress.get("rendered_role_ids", [])}
        reference_ready_ids = {int(value) for value in progress.get("reference_ready_role_ids", [])}
        for index, role in enumerate(roles, start=1):
            role_id = int(role["id"])
            role_dir = _role_dir(output_root, role)
            reference = _ensure_web_reference(role, role_dir, source_records)
            if reference.get("status") == "ready":
                reference_ready_ids.add(role_id)
            existing = _existing_renders(role_dir, render_workflows)
            if existing is not None:
                renders = existing
            else:
                role_dir.mkdir(parents=True, exist_ok=True)
                seed = 64000000 + role_id
                renders = {
                    workflow_name: _render(tools, workflow_name=workflow_name, role=role, role_dir=role_dir, seed=seed)
                    for workflow_name in render_workflows
                }
            render_record = {
                "completed": all(result.get("status") == "success" for result in renders.values()),
                "completed_at": _now(),
                "sequence": index,
                "role": role,
                "reference": reference,
                "prompt": _role_prompt(role),
                "negative_prompt": _negative_prompt(),
                "seed": 64000000 + role_id,
                "renders": renders,
            }
            _write_json(role_dir / "render_evidence.json", render_record)
            if render_record["completed"]:
                rendered_ids.add(role_id)
            progress["rendered_role_ids"] = sorted(rendered_ids)
            progress["reference_ready_role_ids"] = sorted(reference_ready_ids)
            progress["last_role_id"] = role_id
            progress["last_role_name"] = role.get("role_name_en")
            progress["updated_at"] = _now()
            _write_json(progress_path, progress)
            print(
                json.dumps(
                    {"phase": "render", "sequence": index, "total": len(roles), "role_id": role_id, "role_name_en": role.get("role_name_en"), "rendered": render_record["completed"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        progress["comfy_memory_release"] = _free_comfy_memory(args.comfy_host, args.comfy_port)
        _write_json(progress_path, progress)
        if phase == "render":
            progress["render_complete"] = len(rendered_ids) == len(roles)
            _write_json(progress_path, progress)
            return 0

    judge_model = OllamaModel(ModelConfig(model_name=vision_model_name, temperature=0.0, max_tokens=900))
    judged_ids = {int(value) for value in progress.get("judged_role_ids", [])}
    judge_workflows = (KREA_WORKFLOW,) if args.skip_nova else (KREA_WORKFLOW, NOVA_WORKFLOW)
    for index, role in enumerate(roles, start=1):
        role_id = int(role["id"])
        role_dir = _role_dir(output_root, role)
        if role_id in judged_ids and _read_json(role_dir / "evaluation.json"):
            continue
        renders = _existing_renders(role_dir, judge_workflows)
        if renders is None:
            print(json.dumps({"phase": "judge", "role_id": role_id, "status": "missing_render_evidence"}, ensure_ascii=False), flush=True)
            continue
        role_started = time.perf_counter()
        render_record = _read_json(role_dir / "render_evidence.json") or {}
        reference = render_record.get("reference") if isinstance(render_record.get("reference"), dict) else _ensure_web_reference(role, role_dir, source_records)
        contact_sheet = _make_contact_sheet(role_dir, renders, reference, judge_workflows)
        judge = _judge(judge_model, role=role, contact_sheet=contact_sheet, reference=reference, krea_only=args.skip_nova) if contact_sheet and reference.get("status") == "ready" else {"status": "unavailable", "error": "web reference or rendered images unavailable"}
        decision = _decision(role, renders, judge, reference, krea_only=args.skip_nova)
        record = {
            "completed": True,
            "completed_at": _now(),
            "sequence": index,
            "duration_seconds": round(time.perf_counter() - role_started, 2),
            "role": role,
            "reference": reference,
            "prompt": _role_prompt(role),
            "negative_prompt": _negative_prompt(),
            "seed": 64000000 + role_id,
            "renders": renders,
            "contact_sheet": str(contact_sheet) if contact_sheet else None,
            "judge": judge,
            "decision": decision,
        }
        _write_json(role_dir / "evaluation.json", record)
        judged_ids.add(role_id)
        progress["judged_role_ids"] = sorted(judged_ids)
        progress["last_role_id"] = role_id
        progress["last_role_name"] = role.get("role_name_en")
        progress["updated_at"] = _now()
        _write_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "phase": "judge",
                    "sequence": index,
                    "total": len(roles),
                    "role_id": role_id,
                    "role_name_en": role.get("role_name_en"),
                    "decision": decision,
                    "duration_seconds": record["duration_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    progress["completed_at"] = _now()
    progress["complete"] = len(judged_ids) == len(roles)
    _write_json(progress_path, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
