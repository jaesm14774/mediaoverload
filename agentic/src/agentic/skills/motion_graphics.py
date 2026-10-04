from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.motion_graphics import MotionGraphicsPlan
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.skills.shared import build_run_dir


class MotionGraphicsSkills:
    def __init__(self, tools: ToolRegistry, output_root: Path, prompt_engine: Any | None = None) -> None:
        self.tools = tools
        self.output_root = output_root
        self.prompt_engine = prompt_engine or PromptEngine()

    def compose_and_review(self, context: SkillContext) -> SkillResult:
        if not context.node.depends_on:
            raise RuntimeError("Motion graphics composition requires the completed source-video node")
        source_outputs = context.state[context.node.depends_on[0]]
        saved_files = source_outputs.get("saved_files", [])
        if isinstance(saved_files, str):
            saved_files = [saved_files]
        source_video = str(
            source_outputs.get("video_path")
            or next(iter(saved_files), "")
        )
        if not source_video:
            raise RuntimeError("No source video is available for motion graphics composition")

        source_probe = self.tools.call("media.probe_video", {"video_path": source_video}).get("probe")
        if not isinstance(source_probe, dict):
            raise RuntimeError("Could not read the source video metadata for motion graphics planning")
        duration = float(source_probe.get("video_duration") or source_probe.get("duration") or 0.0)
        fps = float(source_probe.get("frame_rate") or 0.0)
        if duration <= 0 or fps <= 0:
            raise RuntimeError("Source video duration and frame rate are required for motion graphics planning")

        run_dir = build_run_dir(self.output_root, context.plan.goal.prompt, "motion_graphics", default_slug="kirby")
        run_dir.mkdir(parents=True, exist_ok=True)
        plan = MotionGraphicsPlan.from_dict(
            self.prompt_engine.build_motion_graphics_plan(
                context.plan.goal,
                duration_seconds=duration,
                fps=fps,
            )
        )
        rounds: list[dict[str, object]] = []
        latest: dict[str, object] | None = None
        final_review_status = "complete"
        review_warning = ""

        # Each candidate is rendered from the immutable, speed-adjusted source.
        # The second plan can therefore never inherit pixels from the first output.
        for round_number in range(1, 3):
            output_path = run_dir / f"motion_graphics_round_{round_number}.mp4"
            latest = self.tools.call(
                "media.compose_motion_graphics",
                {
                    "video_path": source_video,
                    "output_path": str(output_path),
                    "work_dir": str(run_dir / "frames"),
                    "motion_graphics_plan": plan.to_dict(),
                },
            )
            contact_sheet = str(latest.get("contact_sheet_path") or "")
            round_record: dict[str, object] = {
                "round": round_number,
                "source_video_path": source_video,
                "rendered_video_path": str(latest.get("video_path") or output_path),
                "contact_sheet_path": contact_sheet,
                "plan": plan.to_dict(),
                "plan_path": str(latest.get("plan_path") or ""),
                "critique": "",
                "satisfied": False,
            }
            revision: MotionGraphicsPlan | None = None
            try:
                critique = self.prompt_engine.review_motion_graphics_plan(
                    context.plan.goal,
                    plan=plan.to_dict(),
                    contact_sheet_path=contact_sheet,
                    round_number=round_number,
                )
                if not isinstance(critique, dict) or not isinstance(critique.get("satisfied"), bool):
                    raise ValueError("Motion graphics visual review returned an invalid response")
                round_record["critique"] = str(critique.get("critique") or "")
                round_record["satisfied"] = critique["satisfied"]
                if not critique["satisfied"] and round_number == 1:
                    revision = MotionGraphicsPlan.from_dict(critique.get("plan"))
            except Exception as exc:
                round_record["review_error"] = f"{type(exc).__name__}: {exc}"
                rounds.append(round_record)
                final_review_status = "unavailable"
                review_warning = str(exc)
                break
            rounds.append(round_record)
            if critique["satisfied"] or round_number == 2:
                break
            if revision is None:
                final_review_status = "unavailable"
                review_warning = "Visual review requested a revision but returned no valid revised plan"
                break
            plan = revision

        if latest is None:
            raise RuntimeError("Motion graphics did not produce a rendered video")
        final_video_path = str(latest.get("video_path") or "")
        if not final_video_path:
            raise RuntimeError("Motion graphics renderer did not return a video path")
        manifest_path = run_dir / "motion_graphics_manifest.json"
        manifest = {
            "source_video_path": source_video,
            "final_video_path": final_video_path,
            "motion_graphics_applied": True,
            "visual_review_status": final_review_status,
            "review_rounds": len(rounds),
            "rounds": rounds,
            "technical_contract": {
                "probe": latest.get("probe", {}),
                "source_probe": latest.get("source_probe", source_probe),
                "safe_area": latest.get("safe_area", {"passed": True}),
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        output: dict[str, object] = {
            "video_path": final_video_path,
            "final_video_path": final_video_path,
            "source_video_path": source_video,
            "motion_graphics_applied": True,
            "visual_review_status": final_review_status,
            "review_rounds": len(rounds),
            "manifest_path": str(manifest_path),
            "contact_sheet_path": str(latest.get("contact_sheet_path") or ""),
            "plan_path": str(latest.get("plan_path") or ""),
            "probe": latest.get("probe", {}),
            "saved_files": [
                path
                for path in (
                    final_video_path,
                    str(manifest_path),
                    str(latest.get("plan_path") or ""),
                    str(latest.get("contact_sheet_path") or ""),
                )
                if path
            ],
        }
        logs = [f"Rendered declarative motion graphics and completed {len(rounds)} review round(s)."]
        if review_warning:
            logs.append(f"Visual model review was unavailable; the last successful render remains available for Discord review: {review_warning}")
        return SkillResult(
            status="success",
            outputs=output,
            metrics={"review_rounds": len(rounds), "frame_count": int(latest.get("frame_count") or 0)},
            logs=logs,
        )


def register_motion_graphics_skills(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    output_root: Path,
    prompt_engine: Any | None = None,
) -> None:
    skill = MotionGraphicsSkills(tool_registry, output_root, prompt_engine=prompt_engine)
    skill_registry.register(
        "media.video.motion_graphics",
        skill.compose_and_review,
        "Render and visually review a declarative programmatic motion-graphics overlay",
    )
