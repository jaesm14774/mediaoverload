from __future__ import annotations

from dataclasses import replace
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
    EditTransition,
    build_edit_plan,
)
from agentic.runtime.drama import DramaPlan, DramaPlanError, compile_drama_plan
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.registry import SkillRegistry, ToolRegistry


class EditingSkills:
    """Agent-facing timeline composition skills."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        output_root: Path,
        prompt_engine: PromptEngine | None = None,
    ) -> None:
        self.tools = tool_registry
        self.output_root = output_root
        self.prompt_engine = prompt_engine or PromptEngine()

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
        benchmark_mode = bool(context.node.inputs.get("benchmark_mode", False))
        creative_review_enabled = bool(
            context.node.inputs.get("creative_review", context.node.inputs.get("creative_review_enabled", False))
        ) or (plan.profile == "editorial_kinetic_v1" and not benchmark_mode)
        if benchmark_mode:
            result = self._render_candidate(
                plan,
                run_dir / "benchmark_candidate",
                review_evidence=True,
            )
            technical_qa = self._technical_qa(plan, result, context.node.inputs)
            if not isinstance(technical_qa, dict) or not technical_qa.get("passed"):
                return SkillResult(
                    status="failed",
                    outputs={
                        "run_dir": str(run_dir),
                        "technical_qa": technical_qa,
                        "benchmark_mode": True,
                    },
                    metrics={"creative_review_attempts": 0, "benchmark_mode": 1},
                    logs=["Benchmark candidate failed production technical QA before external review."],
                )
            materialized = self.tools.call(
                "media.materialize_edit",
                {
                    "result": result,
                    "output_path": str(output_path),
                    "contact_sheet_path": str(contact_sheet_path),
                    "manifest_path": str(manifest_path),
                },
            )
            return self._success_result(
                materialized,
                run_dir=run_dir,
                plan=plan,
                creative_review={
                    "enabled": False,
                    "required": True,
                    "status": "external_benchmark_review_required",
                },
                drama_plan_path=drama_plan_path,
                logs=[
                    f"Rendered fixed {plan.profile} benchmark candidate with production QA; external review required.",
                ],
            )
        if not creative_review_enabled:
            result = self._render_candidate(
                plan,
                run_dir / "candidate_01",
                review_evidence=False,
            )
            technical_qa = self._technical_qa(plan, result, context.node.inputs)
            if not isinstance(technical_qa, dict) or not technical_qa.get("passed"):
                return SkillResult(
                    status="failed",
                    outputs={"run_dir": str(run_dir), "technical_qa": technical_qa},
                    metrics={"creative_review_attempts": 0},
                    logs=["Edit candidate failed technical QA before materialization."],
                )
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
                result,
                run_dir=run_dir,
                plan=plan,
                creative_review={"enabled": False, "required": False, "status": "not_requested"},
                drama_plan_path=drama_plan_path,
                logs=[f"Rendered {plan.profile} timeline with {len(plan.clips)} clips."],
            )

        max_attempts = self._bounded_attempts(context.node.inputs.get("creative_review_max_attempts", 3))
        attempts: list[dict[str, Any]] = []
        previous_review: dict[str, Any] | None = None
        current_plan = plan
        best: tuple[float, int, EditPlan, dict[str, object], dict[str, Any]] | None = None
        for attempt in range(1, max_attempts + 1):
            candidate_dir = run_dir / "candidates" / f"candidate_{attempt:02d}"
            candidate_result = self._render_candidate(current_plan, candidate_dir, review_evidence=True)
            technical_qa = self._technical_qa(current_plan, candidate_result, context.node.inputs)
            contact_path = str(candidate_result.get("contact_sheet_path") or "")
            evidence_paths = [
                str(path)
                for path in (candidate_result.get("review_evidence_paths") or [])
                if str(path).strip()
            ]
            if not isinstance(technical_qa, dict) or not technical_qa.get("passed"):
                attempt_record = {
                    "attempt": attempt,
                    "profile": current_plan.profile,
                    "variant_seed": current_plan.variant_seed,
                    "video_path": str(candidate_result.get("video_path") or ""),
                    "manifest_path": str(candidate_result.get("manifest_path") or ""),
                    "contact_sheet_path": contact_path,
                    "review_evidence_paths": evidence_paths,
                    "technical_qa": technical_qa,
                    "review": {
                        "enabled": True,
                        "required": True,
                        "passed": False,
                        "status": "technical_failed",
                        "next_change": "change_variant",
                        "reason": "Candidate failed technical QA before visual review.",
                    },
                }
                review_path = candidate_dir / "creative_review.json"
                self._write_json(review_path, attempt_record["review"])
                attempt_record["creative_review_path"] = str(review_path)
                attempts.append(attempt_record)
                creative_review = self._creative_review_bundle(
                    attempts,
                    status="technical_failed",
                    selected=None,
                )
                receipt_path = self._write_json(run_dir / "creative_review.json", creative_review)
                return SkillResult(
                    status="failed",
                    outputs={
                        "run_dir": str(run_dir),
                        "creative_review": creative_review,
                        "creative_review_path": str(receipt_path),
                    },
                    metrics={"creative_review_attempts": attempt},
                    logs=["Edit candidate failed technical QA before visual review; no candidate was published."],
                )
            review = self.prompt_engine.evaluate_edit_contact_sheet(
                contact_sheet_path=contact_path,
                evidence_paths=evidence_paths,
                goal=str(context.plan.goal.prompt),
                style=str(context.plan.goal.style),
                plan=current_plan.to_dict(),
                candidate_attempt=attempt,
                previous_review=previous_review,
            )
            if not isinstance(review, dict):
                review = {
                    "enabled": True,
                    "required": True,
                    "passed": None,
                    "status": "unavailable",
                    "reason": "creative review returned a non-object",
                }
            attempt_record = {
                "attempt": attempt,
                "profile": current_plan.profile,
                "variant_seed": current_plan.variant_seed,
                "video_path": str(candidate_result.get("video_path") or ""),
                "manifest_path": str(candidate_result.get("manifest_path") or ""),
                "contact_sheet_path": contact_path,
                "review_evidence_paths": evidence_paths,
                "technical_qa": technical_qa,
                "review": review,
            }
            review_path = candidate_dir / "creative_review.json"
            self._write_json(review_path, review)
            attempt_record["creative_review_path"] = str(review_path)
            attempts.append(attempt_record)
            if review.get("status") == "unavailable" or review.get("passed") is None:
                creative_review = self._creative_review_bundle(
                    attempts,
                    status="unavailable",
                    selected=None,
                )
                receipt_path = self._write_json(run_dir / "creative_review.json", creative_review)
                return SkillResult(
                    status="failed",
                    outputs={"run_dir": str(run_dir), "creative_review": creative_review, "creative_review_path": str(receipt_path)},
                    metrics={"creative_review_attempts": attempt},
                    logs=[str(review.get("reason") or "Creative visual review was unavailable; edit blocked.")],
                )
            if review.get("passed") is True:
                score = float(review.get("score") or 0.0)
                if best is None or score > best[0]:
                    best = (score, attempt, current_plan, candidate_result, review)
            previous_review = review
            if review.get("passed") is True and str(review.get("next_change") or "keep") == "keep":
                break
            if attempt < max_attempts:
                next_plan = self._next_candidate_plan(
                    current_plan,
                    str(review.get("next_change") or "change_variant"),
                    attempt,
                )
                if next_plan.to_dict() == current_plan.to_dict():
                    next_plan = self._next_candidate_plan(current_plan, "change_variant", attempt)
                current_plan = next_plan

        if best is None:
            creative_review = self._creative_review_bundle(attempts, status="rejected", selected=None)
            receipt_path = self._write_json(run_dir / "creative_review.json", creative_review)
            return SkillResult(
                status="failed",
                outputs={"run_dir": str(run_dir), "creative_review": creative_review, "creative_review_path": str(receipt_path)},
                metrics={"creative_review_attempts": len(attempts)},
                logs=["All edit candidates failed the required visual creative review; no candidate was published."],
            )

        selected_score, selected_attempt, selected_plan, selected_result, selected_review = best
        creative_review = self._creative_review_bundle(
            attempts,
            status="accepted",
            selected={
                "attempt": selected_attempt,
                "score": selected_score,
                "profile": selected_plan.profile,
                "variant_seed": selected_plan.variant_seed,
                "review": selected_review,
            },
        )
        receipt_path = self._write_json(run_dir / "creative_review.json", creative_review)
        result = self.tools.call(
            "media.materialize_edit",
            {
                "result": selected_result,
                "output_path": str(output_path),
                "contact_sheet_path": str(contact_sheet_path),
                "manifest_path": str(manifest_path),
                "creative_review": creative_review,
            },
        )
        return SkillResult(
            status="success",
            outputs={
                **result,
                "run_dir": str(run_dir),
                "creative_review": creative_review,
                "creative_review_path": str(receipt_path),
                "creative_review_required": True,
                "drama_plan_path": drama_plan_path,
                "saved_files": [str(output_path), str(manifest_path), str(contact_sheet_path), str(receipt_path)],
            },
            metrics={
                "clip_count": len(selected_plan.clips),
                "transition_count": len(selected_plan.transitions),
                "creative_review_attempts": len(attempts),
                "creative_review_score": selected_score,
            },
            logs=[
                f"Creative review accepted candidate {selected_attempt} ({selected_plan.profile}, score={selected_score:.0f}) after {len(attempts)} attempt(s).",
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
    def _bounded_attempts(value: Any) -> int:
        try:
            attempts = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("creative_review_max_attempts must be an integer") from exc
        if attempts < 1 or attempts > 4:
            raise ValueError("creative_review_max_attempts must be between 1 and 4")
        return attempts

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    @staticmethod
    def _transition_plan(
        plan: EditPlan,
        *,
        profile: str,
        duration: float,
        seed: int,
    ) -> EditPlan:
        count = max(0, len(plan.clips) - 1)
        if profile == "baseline_concat":
            transitions: tuple[EditTransition, ...] = ()
        elif profile == "xfade_clean_v1":
            transitions = tuple(EditTransition("fade", duration) for _ in range(count))
        elif profile == "chapter_dip_v1":
            transitions = tuple(EditTransition("fadeblack", duration) for _ in range(count))
        else:
            names = ("fade", "wipeleft", "wiperight", "smoothleft", "circleopen")
            transitions = tuple(
                EditTransition(names[(seed + index) % len(names)], duration)
                for index in range(count)
            )
        motion_names = ("slow_zoom_in", "pan_left", "slow_zoom_out", "pan_right", "drift_up", "drift_down")
        clips = tuple(
            replace(
                clip,
                motion=(
                    motion_names[(seed + index) % len(motion_names)]
                    if Path(clip.path).suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
                    else clip.motion
                ),
            )
            for index, clip in enumerate(plan.clips)
        )
        return replace(plan, clips=clips, profile=profile, transitions=transitions, variant_seed=seed)

    @classmethod
    def _next_candidate_plan(cls, plan: EditPlan, change: str, attempt: int) -> EditPlan:
        current_duration = (
            float(plan.transitions[0].duration_seconds)
            if plan.transitions
            else 0.10
        )
        seed = plan.variant_seed + max(1, attempt)
        if change == "shorter_fade":
            return cls._transition_plan(plan, profile="xfade_clean_v1", duration=0.07, seed=seed)
        if change == "clean_fade":
            return cls._transition_plan(plan, profile="xfade_clean_v1", duration=0.10, seed=seed)
        if change == "hard_cut" and plan.profile != "baseline_concat":
            return cls._transition_plan(plan, profile="baseline_concat", duration=0.0, seed=seed)
        if change == "hard_cut" and plan.profile == "baseline_concat":
            return cls._transition_plan(plan, profile="editorial_kinetic_v1", duration=0.10, seed=seed)
        if change == "try_editorial":
            return cls._transition_plan(plan, profile="editorial_kinetic_v1", duration=0.10, seed=seed)
        if change == "try_chapter_dip":
            return cls._transition_plan(plan, profile="chapter_dip_v1", duration=0.10, seed=seed)
        if change == "keep":
            return cls._next_candidate_plan(plan, "change_variant", attempt)
        if plan.profile == "editorial_kinetic_v1":
            return cls._transition_plan(plan, profile="editorial_kinetic_v1", duration=current_duration, seed=seed)
        if plan.profile == "xfade_clean_v1":
            if current_duration >= 0.10:
                return cls._transition_plan(plan, profile="xfade_clean_v1", duration=0.07, seed=seed)
            return cls._transition_plan(plan, profile="editorial_kinetic_v1", duration=0.10, seed=seed)
        return cls._transition_plan(plan, profile="editorial_kinetic_v1", duration=0.10, seed=seed)

    @staticmethod
    def _creative_review_bundle(
        attempts: list[dict[str, Any]],
        *,
        status: str,
        selected: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "enabled": True,
            "required": True,
            "status": status,
            "selected": selected,
            "attempts": attempts,
        }

    @staticmethod
    def _success_result(
        result: dict[str, object],
        *,
        run_dir: Path,
        plan: EditPlan,
        creative_review: dict[str, Any],
        drama_plan_path: str | None,
        logs: list[str],
    ) -> SkillResult:
        return SkillResult(
            status="success",
            outputs={
                **result,
                "run_dir": str(run_dir),
                "creative_review": creative_review,
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
    prompt_engine: PromptEngine | None = None,
) -> None:
    skills = EditingSkills(tool_registry, output_root, prompt_engine=prompt_engine)
    skill_registry.register(
        "media.video.compose_timeline",
        skills.compose_timeline,
        "Compose an agent-controlled OpenCut-inspired timeline from images or video segments",
        stage="package",
        tags=("media", "editing", "timeline"),
        tool_names=("media.compose_edit",),
    )
