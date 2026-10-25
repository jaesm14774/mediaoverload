"""Automatic three-beat story, illustrations and progressive-panel video."""

from agentic.runtime.contracts import (
    MIN_VIDEO_DURATION_SECONDS,
    ExecutionNode,
    ExecutionPlan,
    GoalRequest,
)
from agentic.runtime.motion_comics import MOTION_COMIC_DEFAULTS
from agentic.runtime.planning.common import PlanningContext


def build_motion_comic(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    config = {
        **MOTION_COMIC_DEFAULTS,
        **dict(goal.constraints.get("motion_comic") or {}),
    }
    config["duration_seconds"] = max(
        goal.duration_seconds,
        int(config.get("duration_seconds") or MIN_VIDEO_DURATION_SECONDS),
        MIN_VIDEO_DURATION_SECONDS,
    )
    manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("image_workflow_name", "workflow_name"),
        allowed_media_types={"image"},
    )
    nodes = [
        ExecutionNode(
            node_id="motion-comic-write",
            skill_name="agent.motion_comic.write",
            inputs={},
            tags=["creative", "motion-comic"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="motion-comic-assets",
            skill_name="media.ensure_workflow",
            inputs={"workflow_name": manifest.name},
            depends_on=["motion-comic-write"],
            stage="assets",
        ),
        ExecutionNode(
            node_id="motion-comic-panels",
            skill_name="media.motion_comic.panels",
            inputs={"workflow_name": manifest.name, "config": config},
            depends_on=["motion-comic-write", "motion-comic-assets"],
            stage="render",
        ),
        ExecutionNode(
            node_id="motion-comic-compose",
            skill_name="media.motion_comic.compose",
            inputs={"config": config},
            depends_on=["motion-comic-write", "motion-comic-panels"],
            stage="package",
        ),
    ]
    editorial_node = context.append_editorial_pacing_node(
        nodes,
        source_node="motion-comic-compose",
        node_id="motion-comic-editorial-pacing",
    )
    final_video_node = context.append_nvidia_vfx_postprocess_node(
        goal,
        nodes,
        source_node=editorial_node,
        node_id="motion-comic-nvidia-vfx",
    )
    nodes.append(
        ExecutionNode(
            node_id="motion-comic-qa",
            skill_name="media.video.inspect",
            inputs=context.scaled_video_qa_inputs(
                goal,
                {
                    "target_duration": config["duration_seconds"],
                    "expected_fps": config["fps"],
                    "frame_count": 3,
                    "columns": 3,
                },
            ),
            depends_on=[final_video_node],
            stage="qa",
        )
    )
    return ExecutionPlan(
        goal=goal,
        workflow_name="motion_comic_v1",
        nodes=nodes,
        metadata={
            "selected_workflows": [manifest.name],
            "required_assets": [asset.to_dict() for asset in manifest.required_assets],
            "graph_overview": [node.node_id for node in nodes],
            "story_source": "llm",
            "panel_count": 3,
            "config": config,
        },
        description=f"LLM-authored progressive-panel comic for '{goal.prompt}'",
    )
