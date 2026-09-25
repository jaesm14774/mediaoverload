from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.editing import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    EditPlan,
    build_edit_plan,
)
from agentic.runtime.drama import DramaPlan, DramaPlanError, compile_drama_plan
from agentic.runtime.registry import SkillRegistry, ToolRegistry


class EditingSkills:
    """Agent-facing timeline composition skills."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        output_root: Path,
    ) -> None:
        self.tools = tool_registry
        self.output_root = output_root

    def compose_timeline(self, context: SkillContext) -> SkillResult:
        run_dir = self.output_root / "editing" / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:10]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        raw_drama_plan = context.node.inputs.get("drama_plan")
        drama_plan_path: str | None = None
        if raw_drama_plan is not None and context.node.inputs.get("edit_plan") is not None:
            raise ValueError("compose_timeline accepts either drama_plan or edit_plan, not both")
        if raw_drama_plan is not None:
            if isinstance(raw_drama_plan, DramaPlan):
                drama_plan = raw_drama_plan.validate(require_assets=True)
            elif isinstance(raw_drama_plan, dict):
                drama_plan = DramaPlan.from_dict(raw_drama_plan).validate(require_assets=True)
            elif isinstance(raw_drama_plan, str):
                drama_source = Path(raw_drama_plan).expanduser().resolve()
                if not drama_source.is_file():
                    raise DramaPlanError(f"DramaPlan file does not exist: {drama_source}")
                payload = json.loads(drama_source.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise DramaPlanError("DramaPlan JSON must contain an object")
                drama_plan = DramaPlan.from_dict(payload).validate(require_assets=True)
            else:
                raise DramaPlanError("drama_plan must be an object or a JSON file path")
            plan = compile_drama_plan(drama_plan)
            persisted_drama_plan = self._resolve_output_path(
                context.node.inputs.get("drama_plan_path"),
                run_dir / "drama_plan.json",
            )
            self._write_json(persisted_drama_plan, drama_plan.to_dict())
            drama_plan_path = str(persisted_drama_plan)
        else:
            raw_plan = context.node.inputs.get("edit_plan")
            if isinstance(raw_plan, dict):
                plan = EditPlan.from_dict(raw_plan)
            else:
                paths = self._input_paths(context)
                plan = build_edit_plan(
                    paths,
                    profile=str(context.node.inputs.get("profile") or "xfade_clean_v1"),
                    output_width=int(context.node.inputs.get("output_width") or 576),
                    output_height=int(context.node.inputs.get("output_height") or 1024),
                    fps=float(context.node.inputs.get("fps") or 24),
                    target_duration_seconds=self._optional_float(context.node.inputs.get("target_duration_seconds")),
                    variant_seed=int(context.node.inputs.get("variant_seed") or 0),
                    transition_duration_seconds=float(context.node.inputs.get("transition_duration_seconds") or 0.10),
                )
        output_path = self._resolve_output_path(
            context.node.inputs.get("output_path"),
            run_dir / "edited.mp4",
        )
        contact_sheet_path = self._resolve_output_path(
            context.node.inputs.get("contact_sheet_path"),
            run_dir / "contact_sheet.jpg",
        )
        manifest_path = self._resolve_output_path(
            context.node.inputs.get("manifest_path"),
            run_dir / "edit_manifest.json",
        )
        result = self._render_candidate(plan, run_dir / "candidate_01", review_evidence=True)
        technical_qa = self._technical_qa(plan, result, context.node.inputs)
        if not isinstance(technical_qa, dict):
            raise RuntimeError("media.video_qa returned a non-object inspection result")
        technical_qa = {**technical_qa, "automatic_gate_applied": False}
        result = self.tools.call(
            "media.materialize_edit",
            {
                "result": result,
                "output_path": str(output_path),
                "contact_sheet_path": str(contact_sheet_path),
                "manifest_path": str(manifest_path),
            },
        )
        return self._success_result(
            {**result, "technical_qa": technical_qa},
            run_dir=run_dir,
            plan=plan,
            drama_plan_path=drama_plan_path,
            logs=[
                f"Rendered {plan.profile} timeline with {len(plan.clips)} clips; Discord owns review and automatic DQ is disabled."
            ],
        )

    def _render_candidate(
        self,
        plan: EditPlan,
        candidate_dir: Path,
        *,
        review_evidence: bool,
    ) -> dict[str, object]:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        return self.tools.call(
            "media.compose_edit",
            {
                "edit_plan": plan.to_dict(),
                "output_path": str(candidate_dir / "edited.mp4"),
                "contact_sheet_path": str(candidate_dir / "contact_sheet.jpg"),
                "manifest_path": str(candidate_dir / "edit_manifest.json"),
                "review_evidence_dir": str(candidate_dir / "review_frames") if review_evidence else "",
            },
        )

    def _technical_qa(
        self,
        plan: EditPlan,
        result: dict[str, object],
        inputs: dict[str, Any],
    ) -> dict[str, object]:
        video_path = str(result.get("video_path") or "")
        # DramaPlan currently carries dialogue/SFX as declarative cues. Until
        # the audio compositor is enabled, visual-only drama renders must not
        # silently claim that their generated filler track is real audio.
        default_require_audio = plan.profile != "baseline_concat" and inputs.get("drama_plan") is None
        return self.tools.call(
            "media.video_qa",
            {
                "video_path": video_path,
                "target_duration": plan.target_duration_seconds,
                "duration_tolerance": 0.35,
                "expected_width": plan.output_width,
                "expected_height": plan.output_height,
                "expected_fps": plan.fps,
                "require_audio": bool(inputs.get("require_audio", default_require_audio)),
                "require_stereo_audio": bool(inputs.get("require_stereo_audio", default_require_audio)),
                "analyze_audio": bool(inputs.get("analyze_audio", False)),
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    @staticmethod
    def _success_result(
        result: dict[str, object],
        *,
        run_dir: Path,
        plan: EditPlan,
        drama_plan_path: str | None,
        logs: list[str],
    ) -> SkillResult:
        return SkillResult(
            status="success",
            outputs={
                **result,
                "run_dir": str(run_dir),
                **({"drama_plan_path": drama_plan_path} if drama_plan_path else {}),
                "saved_files": [
                    str(result.get("video_path") or ""),
                    str(result.get("manifest_path") or ""),
                    str(result.get("contact_sheet_path") or ""),
                ],
            },
            metrics={"clip_count": len(plan.clips), "transition_count": len(plan.transitions)},
            logs=logs,
        )

    @staticmethod
    def _input_paths(context: SkillContext) -> list[str]:
        raw = context.node.inputs.get("input_paths") or context.node.inputs.get("clip_paths") or []
        if isinstance(raw, str):
            return [raw]
        if raw:
            return [str(item) for item in raw if str(item).strip()]
        paths: list[str] = []
        for dependency in context.node.depends_on:
            outputs = context.state[dependency]
            saved_files = outputs.get("saved_files")
            if isinstance(saved_files, list):
                paths.extend(
                    str(item)
                    for item in saved_files
                    if Path(str(item)).suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
                )
            video_path = outputs.get("video_path")
            if isinstance(video_path, str) and video_path:
                paths.append(video_path)
        return list(dict.fromkeys(paths))

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    def _resolve_output_path(self, raw_path: Any, default_path: Path) -> Path:
        candidate = Path(str(raw_path or default_path)).expanduser().resolve()
        root = self.output_root.expanduser().resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Edit output must stay under the configured output root: {candidate}")
        return candidate


def register_editing_skills(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    output_root: Path,
) -> None:
    skills = EditingSkills(tool_registry, output_root)
    skill_registry.register(
        "media.video.compose_timeline",
        skills.compose_timeline,
        "Compose an agent-controlled OpenCut-inspired timeline from images or video segments",
        stage="package",
        tags=("media", "editing", "timeline"),
        tool_names=("media.compose_edit",),
    )
