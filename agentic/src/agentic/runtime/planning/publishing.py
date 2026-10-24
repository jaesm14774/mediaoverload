"""Publishing strategy graph assembly."""

from __future__ import annotations

from pathlib import Path
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest
from agentic.runtime.planning.common import PlanningContext


PUBLISH_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


def _infer_publish_review_scope(goal: GoalRequest) -> str:
    explicit_scope = str(goal.constraints.get("review_scope") or "").strip().lower()
    if explicit_scope:
        return explicit_scope
    media_paths = goal.constraints.get("media_paths") or []
    if isinstance(media_paths, str):
        media_paths = [media_paths]
    normalized_paths = [str(path) for path in media_paths if str(path).strip()]
    if not normalized_paths:
        input_dir = str(goal.constraints.get("input_dir") or "").strip()
        if input_dir:
            root = Path(input_dir)
            if root.is_dir():
                normalized_paths = [
                    str(path) for path in root.iterdir() if path.is_file()
                ]
    if normalized_paths and all(
        Path(path).suffix.lower() in PUBLISH_VIDEO_EXTENSIONS
        for path in normalized_paths
    ):
        return "final_video"
    return "final_media"


def build_publish_review(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    platforms = goal.constraints.get("platforms") or []
    dry_run = bool(goal.constraints.get("dry_run", True))
    selection_limit = int(goal.constraints.get("selection_limit") or 10)
    serial_publish = bool(goal.constraints.get("serial_mode"))
    nodes = [
        ExecutionNode(
            node_id="ingest-media",
            skill_name="publish.media.ingest",
            inputs={},
            tags=["publish", "review"],
            stage="ingest",
        ),
        ExecutionNode(
            node_id="review-select",
            skill_name="review.assets.select",
            inputs={
                "limit": selection_limit,
                "review_scope": _infer_publish_review_scope(goal),
                **(
                    {
                        "review_phase": "serial_final_publish",
                        "media_paths": goal.constraints.get("media_paths", []),
                        "platforms": list(platforms),
                        "require_human_review": True,
                    }
                    if serial_publish
                    else {}
                ),
                "review_all_candidates": bool(
                    goal.constraints.get("review_all_candidates", False)
                ),
            },
            depends_on=["ingest-media"],
            tags=["review", "publish"],
            stage="review",
        ),
        ExecutionNode(
            node_id="process-media",
            skill_name="publish.media.process",
            inputs={"output_dir": goal.constraints.get("output_dir")},
            depends_on=["review-select"],
            tags=["publish", "media"],
            tool_name="publish.process_media",
            stage="package",
        ),
        ExecutionNode(
            node_id="prepare-caption",
            skill_name="publish.caption.prepare",
            inputs={
                "prefix": "" if serial_publish else goal.constraints.get("caption_prefix", ""),
                "hashtags": goal.constraints.get("hashtags", []),
                "platforms": platforms,
            },
            depends_on=["review-select", "process-media"],
            tags=["publish", "caption"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="dispatch-publish",
            skill_name="publish.social.dispatch",
            inputs={
                "platforms": platforms,
                "platform_configs": goal.constraints.get("platform_configs", {}),
                "additional_params": goal.constraints.get("additional_params", {}),
                "publish_mode": goal.constraints.get("publish_mode", ""),
                "dry_run": dry_run,
            },
            depends_on=["process-media", "prepare-caption", "review-select"],
            tags=["publish", "dispatch"],
            tool_name="publish.social",
            stage="publish",
        ),
        ExecutionNode(
            node_id="persist-publish-review-summary",
            skill_name="agent.summary.persist",
            inputs={
                "summary_name": "publish_review_summary.json",
                "summary_scope": "publish_review",
            },
            depends_on=[
                "review-select",
                "process-media",
                "prepare-caption",
                "dispatch-publish",
            ],
            tags=["artifact", "summary", "publish"],
            stage="package",
        ),
    ]
    metadata = {
        "graph_overview": [node.node_id for node in nodes],
        "platforms": list(platforms),
        "dry_run": dry_run,
        "publish_mode": str(goal.constraints.get("publish_mode") or ""),
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="publish_review_v1",
        nodes=nodes,
        metadata=metadata,
        description=f"Agentic publish/review chain for goal '{goal.prompt}'",
    )
