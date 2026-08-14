"""Run the canonical MiniMax H3 modes through real ComfyUI workflows.

This runner intentionally creates every conditioning asset through ComfyUI
before the H3 render. It does not use mock media, reference audio, or a
prompt-only shortcut. All generated artifacts and the report are written to
the caller-selected D/E-drive output root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.assets.registry import AssetRegistry
from agentic.runtime.h3_modes import H3Mode, mode_contract
from agentic.runtime.registry import ToolRegistry
from agentic.tools.comfy_workflow_tool import register_comfy_workflow_tools
from agentic.tools.media_services import register_media_service_tools


IMAGE_WORKFLOW_CANDIDATES = ("kirby_keyframe_anima", "anima_anime", "nova_model_plus_z_image_anime")
CANVAS = {"width": 608, "height": 352, "frame_rate": 24}
MODE_PROFILES: dict[H3Mode, dict[str, Any]] = {
    H3Mode.T2VA: {"length": 362, "steps": 16, "duration": 362 / 24},
    H3Mode.I2VA: {"length": 124, "steps": 20, "duration": 124 / 24},
    H3Mode.FL2VA: {"length": 362, "steps": 16, "duration": 362 / 24},
    H3Mode.L2VA: {"length": 362, "steps": 16, "duration": 362 / 24},
    H3Mode.REF2VA: {"length": 124, "steps": 20, "duration": 124 / 24},
}


def _first_saved(result: dict[str, object], suffixes: tuple[str, ...]) -> str:
    for item in result.get("saved_files", []) if isinstance(result.get("saved_files"), list) else []:
        path = Path(str(item))
        if path.suffix.lower() in suffixes and path.is_file():
            return str(path)
    raise RuntimeError(f"ComfyUI returned no usable artifact with suffixes {suffixes}: {result}")


def _select_image_workflow(registry: AssetRegistry) -> str:
    for name in IMAGE_WORKFLOW_CANDIDATES:
        try:
            registry.get_manifest(name)
            return name
        except KeyError:
            continue
    raise RuntimeError(f"None of the image workflows are available under configs/workflow: {IMAGE_WORKFLOW_CANDIDATES}")


def _check_assets(registry: AssetRegistry, workflow_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    status: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for name in workflow_names:
        manifest = registry.get_manifest(name)
        entries = registry.ensure_requirements(manifest, auto_download=False)
        status[name] = entries
        missing.extend(f"{name}: {entry['asset']} -> {entry['target_path']}" for entry in entries if entry["status"] != "ready")
    if missing:
        raise RuntimeError("Missing/corrupt models on the configured D/E drive:\n" + "\n".join(missing))
    return status


def _render_image(tools: ToolRegistry, *, workflow_name: str, run_dir: Path, prompt: str) -> str:
    result = tools.call(
        "comfy.workflow.text_to_image",
        {
            "workflow_name": workflow_name,
            "run_dir": str(run_dir),
            "prompt": prompt,
            "negative_prompt": "blur, deformed, duplicate character, text, watermark, low detail",
            "width": CANVAS["width"],
            "height": CANVAS["height"],
            "image_count": 1,
        },
    )
    return _first_saved(result, (".png", ".jpg", ".jpeg", ".webp"))


def _render_video(tools: ToolRegistry, *, mode: H3Mode, run_dir: Path, payload: dict[str, Any]) -> tuple[str, dict[str, object]]:
    contract = mode_contract(mode.value)
    tool_name = {
        H3Mode.T2VA: "comfy.workflow.text_to_video",
        H3Mode.I2VA: "comfy.workflow.image_to_video",
        H3Mode.FL2VA: "comfy.workflow.image_to_video",
        H3Mode.L2VA: "comfy.workflow.image_to_video",
        H3Mode.REF2VA: "comfy.workflow.reference_to_video",
    }[mode]
    request = {
        "workflow_name": contract.workflow_name,
        "run_dir": str(run_dir),
        "h3_mode": mode.value,
        "prompt": payload["prompt"],
        "negative_prompt": payload.get("negative_prompt", "保持角色身份、連續動作、穩定鏡頭、清楚音場"),
        "width": CANVAS["width"],
        "height": CANVAS["height"],
        "length": payload["length"],
        "steps": payload["steps"],
        "video_count": 1,
        "model_profile": "q4",
    }
    request.update({key: value for key, value in payload.items() if key not in {"prompt", "length", "steps"} and value is not None})
    result = tools.call(tool_name, request)
    return _first_saved(result, (".mp4", ".webm", ".mkv", ".mov")), result


def _qa_video(tools: ToolRegistry, *, video_path: str, run_dir: Path, duration: float) -> dict[str, object]:
    return tools.call(
        "media.video_qa",
        {
            "video_path": video_path,
            "target_duration": duration,
            "duration_tolerance": 0.9,
            "expected_width": CANVAS["width"],
            "expected_height": CANVAS["height"],
            "expected_fps": CANVAS["frame_rate"],
            "require_audio": True,
            "require_stereo_audio": True,
            "analyze_audio": True,
            "warn_if_no_audio": True,
            "contact_sheet_path": str(run_dir / "qa" / "contact_sheet.jpg"),
            "frame_count": 8,
            "columns": 4,
            "scale_width": 320,
        },
    )


def _run_mode(
    tools: ToolRegistry,
    registry: AssetRegistry,
    *,
    mode: H3Mode,
    output_root: Path,
    image_workflow: str,
    smoke: bool = False,
) -> dict[str, object]:
    profile = dict(MODE_PROFILES[mode])
    if smoke:
        # Keep the exact workflow and conditioning contract while bounding
        # GPU time for a strategy-matrix regression run.
        profile["length"] = min(int(profile["length"]), 124)
        profile["duration"] = float(profile["length"]) / CANVAS["frame_rate"]
    contract = mode_contract(mode.value)
    run_dir = output_root / mode.value
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        "A polished 2D anime story of Kirby discovering a glowing seed in a windy meadow, "
        "clear subject silhouette, deliberate camera movement, visible action progression, "
        "strong beginning-middle-payoff, native environmental sound, no dialogue."
    )
    result: dict[str, object] = {
        "mode": mode.value,
        "generation_type": contract.generation_type,
        "workflow_name": contract.workflow_name,
        "render_mode": contract.render_mode,
        "output_dir": str(run_dir),
        "reference_audio_enabled": False,
        "smoke": smoke,
        "length": profile["length"],
    }

    if mode is H3Mode.T2VA:
        video_path, render = _render_video(
            tools,
            mode=mode,
            run_dir=run_dir,
            payload={"prompt": prompt, "length": profile["length"], "steps": profile["steps"]},
        )
    elif mode is H3Mode.I2VA:
        opening = _render_image(
            tools,
            workflow_name=image_workflow,
            run_dir=run_dir / "generated_opening",
            prompt="Kirby stands in a windy meadow beside a tiny glowing seed, polished 2D anime keyframe, wide composition",
        )
        video_path, render = _render_video(
            tools,
            mode=mode,
            run_dir=run_dir,
            payload={
                "prompt": prompt,
                "image_path": opening,
                "use_first_frame": True,
                "use_last_frame": False,
                "length": profile["length"],
                "steps": profile["steps"],
            },
        )
        result["generated_opening_image"] = opening
    elif mode is H3Mode.FL2VA:
        opening = _render_image(
            tools,
            workflow_name=image_workflow,
            run_dir=run_dir / "generated_opening",
            prompt="Kirby finds a tiny glowing seed in a windy meadow, polished 2D anime opening keyframe, clear silhouette",
        )
        landing = _render_image(
            tools,
            workflow_name=image_workflow,
            run_dir=run_dir / "generated_landing",
            prompt="Kirby holds the glowing seed above a meadow path now lit with warm light, polished 2D anime landing keyframe, clear payoff",
        )
        video_path, render = _render_video(
            tools,
            mode=mode,
            run_dir=run_dir,
            payload={
                "prompt": prompt,
                "image_path": opening,
                "last_image_path": landing,
                "use_first_frame": True,
                "use_last_frame": True,
                "length": profile["length"],
                "steps": profile["steps"],
            },
        )
        result.update({"generated_opening_image": opening, "generated_landing_image": landing})
    elif mode is H3Mode.L2VA:
        landing = _render_image(
            tools,
            workflow_name=image_workflow,
            run_dir=run_dir / "generated_landing",
            prompt="Kirby holds the glowing seed above a meadow path now lit with warm light, polished 2D anime landing keyframe, clear payoff",
        )
        video_path, render = _render_video(
            tools,
            mode=mode,
            run_dir=run_dir,
            payload={
                "prompt": prompt,
                "last_image_path": landing,
                "use_first_frame": False,
                "use_last_frame": True,
                "length": profile["length"],
                "steps": profile["steps"],
            },
        )
        result["generated_landing_image"] = landing
    else:
        reference_image = _render_image(
            tools,
            workflow_name=image_workflow,
            run_dir=run_dir / "generated_reference_image",
            prompt="Kirby in a windy meadow beside a glowing seed, polished 2D anime identity and composition reference",
        )
        reference_video, _ = _render_video(
            tools,
            mode=H3Mode.I2VA,
            run_dir=run_dir / "generated_reference_video",
            payload={
                "prompt": "Kirby turns toward a glowing seed as meadow grass moves in the wind, polished 2D anime motion reference",
                "image_path": reference_image,
                "use_first_frame": True,
                "use_last_frame": False,
                "length": MODE_PROFILES[H3Mode.I2VA]["length"],
                "steps": MODE_PROFILES[H3Mode.I2VA]["steps"],
            },
        )
        manifest = [
            {"path": reference_image, "type": "image", "role": "identity", "weight": 1.0},
            {"path": reference_video, "type": "video", "role": "motion", "weight": 1.0},
        ]
        video_path, render = _render_video(
            tools,
            mode=mode,
            run_dir=run_dir,
            payload={
                "prompt": prompt,
                "reference_manifest": manifest,
                "ref_image_size": "match",
                "length": profile["length"],
                "steps": profile["steps"],
            },
        )
        result.update({"reference_manifest": manifest, "generated_reference_video": reference_video})

    qa = _qa_video(tools, video_path=video_path, run_dir=run_dir, duration=profile["duration"])
    result.update({"video_path": video_path, "render": render, "qa": qa, "passed": bool(qa.get("passed"))})
    if not result["passed"]:
        raise RuntimeError(f"{mode.value} technical QA failed: {json.dumps(qa, ensure_ascii=False)}")
    return result


def _write_report(report_path: Path, report: dict[str, object]) -> None:
    """Persist progress after every mode so long GPU runs are resumable."""
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all canonical MiniMax H3 modes through real ComfyUI")
    parser.add_argument("--comfy-root", default=r"D:\ComfyUI_windows_portable", help="Portable ComfyUI root on D/E drive")
    parser.add_argument("--output-root", default=r"D:\ComfyUI_windows_portable\ComfyUI\output\mediaoverload_h3_p2_e2e")
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8188)
    parser.add_argument("--mode", choices=[mode.value for mode in H3Mode], action="append", dest="modes")
    parser.add_argument("--smoke", action="store_true", help="Run the same graphs at a bounded 5-second length")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    registry = AssetRegistry(REPO_ROOT / "agentic", asset_root=comfy_root)
    image_workflow = _select_image_workflow(registry)
    selected_modes = [H3Mode(value) for value in (args.modes or [mode.value for mode in H3Mode])]
    workflow_names = [image_workflow, *(mode_contract(mode.value).workflow_name for mode in selected_modes)]
    if H3Mode.REF2VA in selected_modes:
        workflow_names.append(mode_contract(H3Mode.I2VA.value).workflow_name)
    asset_status = _check_assets(registry, list(dict.fromkeys(workflow_names)))
    tools = ToolRegistry()
    register_comfy_workflow_tools(tools, registry, output_root, comfy_host=args.comfy_host, comfy_port=args.comfy_port)
    register_media_service_tools(tools, output_root)
    report_path = output_root / "h3_modes_e2e_report.json"
    existing_report: dict[str, object] = {}
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_report = loaded
        except (OSError, json.JSONDecodeError):
            existing_report = {}
    existing_modes = existing_report.get("modes")
    if not isinstance(existing_modes, dict):
        existing_modes = {}
    report: dict[str, object] = {
        **existing_report,
        "comfy_root": str(comfy_root),
        "output_root": str(output_root),
        "image_workflow": image_workflow,
        "reference_audio_enabled": False,
        "asset_status": asset_status,
        "modes": dict(existing_modes),
    }
    _write_report(report_path, report)
    for mode in selected_modes:
        previous = report["modes"].get(mode.value) if isinstance(report.get("modes"), dict) else None
        if (
            isinstance(previous, dict)
            and bool(previous.get("passed"))
            and Path(str(previous.get("video_path") or "")).is_file()
        ):
            print(f"RESUME {mode.value}: already passed")
            continue
        try:
            report["modes"][mode.value] = _run_mode(
                tools,
                registry,
                mode=mode,
                output_root=output_root,
                image_workflow=image_workflow,
                smoke=args.smoke,
            )
            print(f"PASS {mode.value}: {report['modes'][mode.value]['video_path']}")
        except Exception as exc:
            report["modes"][mode.value] = {"mode": mode.value, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"FAIL {mode.value}: {exc}", file=sys.stderr)
        _write_report(report_path, report)
    passed = all(bool(item.get("passed")) for item in report["modes"].values())
    print(f"REPORT {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
