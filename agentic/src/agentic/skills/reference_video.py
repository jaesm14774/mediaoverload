from __future__ import annotations

from pathlib import Path

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.reference_video import ReferenceVideoAnalyzer, ReferenceVideoError
from agentic.runtime.registry import SkillRegistry, ToolRegistry


class ReferenceVideoSkills:
    """Evidence-producing skills for the reference-video planning stage."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.analyzer = ReferenceVideoAnalyzer()

    def analyze_video(self, context: SkillContext) -> SkillResult:
        source = str(context.node.inputs.get("source") or "").strip()
        try:
            brief = self.analyzer.analyze(
                source,
                output_root=self.output_root,
                max_keyframes=int(context.node.inputs.get("max_keyframes") or 12),
                analysis_depth=str(context.node.inputs.get("analysis_depth") or "standard"),
            )
        except (ReferenceVideoError, ValueError) as exc:
            return SkillResult(
                status="failed",
                logs=[f"Reference video analysis failed: {type(exc).__name__}: {exc}"],
            )
        scenes = list((brief.get("structure_analysis") or {}).get("scenes") or [])
        keyframes = list(brief.get("keyframes") or [])
        return SkillResult(
            status="success",
            outputs=brief,
            metrics={
                "duration_seconds": float((brief.get("media") or {}).get("duration_seconds") or 0),
                "scene_count": len(scenes),
                "keyframe_count": len(keyframes),
            },
            logs=[
                f"Measured reference-video structure into {brief.get('brief_path')}",
                "Semantic interpretation remains attached to extracted frames for the existing story model.",
            ],
        )


def register_reference_video_skills(
    skill_registry: SkillRegistry,
    _tool_registry: ToolRegistry,
    output_root: Path,
) -> None:
    skills = ReferenceVideoSkills(output_root)
    skill_registry.register(
        "reference.video.analyze",
        skills.analyze_video,
        "Extract a structural reference-video brief and keyframe evidence",
        stage="analysis",
        tags=("reference-video", "ffmpeg", "evidence"),
    )
