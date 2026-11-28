"""Real-world intruder graph: a real-photo keyframe and a kid's sketch, one image-to-video clip, the sketch taped into its corner."""

from __future__ import annotations

import math

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest
from agentic.runtime.planning.common import PlanningContext
from agentic.runtime.prompt_tasks.real_world_intruder import CLIP_SECONDS

# H3 lengths are 17k+5 frames; this is the shortest one that reaches CLIP_SECONDS at 24 fps.
CLIP_LENGTH = math.ceil((CLIP_SECONDS * 24 - 5) / 17) * 17 + 5
KEYFRAME_SCALE = 2  # Krea draws at twice the video canvas; the video model downsizes it
SKETCH_SIZE = 1024
TAGS = ["real-world-intruder"]


def build_real_world_intruder(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    """One news item gives the real scene; one drawn character intrudes with one gag in one locked-camera clip."""
    image = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("keyframe_workflow_name", "image_workflow_name"),
        allowed_media_types={"image"},
    )
    video = context.manifest_from_goal_constraints(
        goal,
        "minimax_h3_i2va",
        constraint_keys=("video_workflow_name",),
        allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
    )
    render_config = context.native_h3_render_config(goal, video)
    width, height, length = render_config["width"], render_config["height"], CLIP_LENGTH  # the goal duration does not stretch the gag
    require_human_review = bool(goal.constraints.get("require_human_review", False))

    nodes = [
        ExecutionNode(
            node_id="intruder-story",
            skill_name="real_world_intruder.prepare_story",
            tags=["creative", "story", *TAGS],
            stage="prompting",
        )
    ]
    for node_id, manifest in (("intruder-image-asset-check", image), ("intruder-video-asset-check", video)):
        nodes.append(
            ExecutionNode(
                node_id=node_id,
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": manifest.name},
                depends_on=["intruder-story"],
                tags=["assets", *TAGS],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            )
        )

    def picture(node_id: str, prompt_key: str, picture_width: int, picture_height: int) -> ExecutionNode:
        return ExecutionNode(
            node_id=node_id,
            skill_name="real_world_intruder.render_image",
            inputs={
                "workflow_name": image.name,
                "story_node": "intruder-story",
                "prompt_key": prompt_key,
                "width": picture_width,
                "height": picture_height,
                "suffix": node_id.removeprefix("intruder-"),
            },
            depends_on=["intruder-story", "intruder-image-asset-check"],
            tags=["render", "image", *TAGS],
            tool_name="comfy.workflow.text_to_image",
            stage="render",
        )

    nodes.append(picture("intruder-keyframe", "keyframe_prompt", width * KEYFRAME_SCALE, height * KEYFRAME_SCALE))
    nodes.append(picture("intruder-sketch", "sketch_prompt", SKETCH_SIZE, SKETCH_SIZE))
    anchor = "intruder-keyframe"
    if require_human_review:
        anchor = "intruder-keyframe-review"
        nodes.append(
            ExecutionNode(
                node_id=anchor,
                skill_name="review.assets.select",
                inputs={
                    "limit": 1,
                    "review_all_candidates": True,
                    "review_scope": "first_frame",
                    "review_phase": "intruder_keyframe",
                    "require_human_review": True,
                    "review_notes": "Real-world intruder keyframe: approve before any video renders.",
                },
                depends_on=["intruder-keyframe"],
                tags=["review", *TAGS],
                stage="review",
            )
        )
    nodes.append(
        ExecutionNode(
            node_id="intruder-clip",
            skill_name="real_world_intruder.render_clip",
            inputs={
                "workflow_name": video.name,
                "story_node": "intruder-story",
                "anchor_node": anchor,
                "width": width,
                "height": height,
                "length": length,
            },
            depends_on=["intruder-story", "intruder-video-asset-check", anchor],
            tags=["render", "video", *TAGS],
            tool_name="comfy.render_h3",
            stage="render",
        )
    )
    nodes.append(
        ExecutionNode(
            node_id="intruder-compose",
            skill_name="real_world_intruder.compose",
            inputs={"clip_node": "intruder-clip", "sketch_node": "intruder-sketch"},
            depends_on=["intruder-clip", "intruder-sketch"],
            tags=["package", "video", *TAGS],
            tool_name="media.sketch_inset",
            stage="package",
        )
    )
    nodes.extend(
        context.native_h3_finalize_nodes(
            goal,
            tags=TAGS,
            include_keyframe_source_check=False,
            render_node="intruder-clip",
            source_node="intruder-compose",
            qa_inputs={"expected_fps": 24},
        )
    )
    return ExecutionPlan(
        goal=goal,
        workflow_name=video.name,
        nodes=nodes,
        metadata={
            "recipe": "real_world_intruder",
            "selected_workflow": video.name,
            "keyframe_workflow": image.name,
            "required_assets": [
                asset.to_dict() for manifest in (image, video) for asset in manifest.required_assets
            ],
            "real_world_intruder": {
                "width": width,
                "height": height,
                "clip_length": length,
                "require_human_review": require_human_review,
                "backend": "comfyui",
            },
            "graph_overview": [node.node_id for node in nodes],
        },
        description="Real-world intruder: a drawn character wanders into a real photograph, with a kid's taped sketch fading out of the corner",
    )
