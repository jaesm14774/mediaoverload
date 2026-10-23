"""Stickers strategy graph assembly."""

from __future__ import annotations

from agentic.runtime.contracts import (
    MIN_VIDEO_DURATION_SECONDS,
    ExecutionNode,
    ExecutionPlan,
    GoalRequest,
)
from agentic.runtime.planning.common import PlanningContext


def build_sticker_pack(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("image_workflow_name", "workflow_name"),
        allowed_media_types={"image"},
    )
    character = goal.constraints.get("character", "")
    expression_count = context.constraint_int(goal, "sticker_expression_count", 8)
    images_per_prompt = context.constraint_int(goal, "images_per_prompt", 1)
    nodes = [
        ExecutionNode(
            node_id="sticker-expressions",
            skill_name="agent.sticker.expressions",
            inputs={
                "prompt": goal.prompt,
                "character": character,
                "expression_count": expression_count,
            },
            tags=["creative", "sticker"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="sticker-prompts",
            skill_name="agent.sticker.prompt_set",
            inputs={"character": character},
            depends_on=["sticker-expressions"],
            tags=["creative", "sticker"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="image-asset-check",
            skill_name="image.ensure_workflow",
            inputs={
                "workflow_name": image_manifest.name,
            },
            depends_on=["sticker-prompts"],
            tags=["assets", "sticker"],
            stage="assets",
        ),
        ExecutionNode(
            node_id="render-stickers",
            skill_name="media.image.render_batch",
            inputs={
                "workflow_name": image_manifest.name,
                "width": context.canvas_dimensions(goal, image_manifest)[0],
                "height": context.canvas_dimensions(goal, image_manifest)[1],
                "images_per_prompt": images_per_prompt,
                "suffix": "stickers",
            },
            depends_on=["sticker-prompts", "image-asset-check"],
            tags=["render", "sticker", "batch"],
            stage="render",
        ),
        ExecutionNode(
            node_id="package-stickers",
            skill_name="agent.sticker.package",
            inputs={},
            depends_on=["render-stickers"],
            tags=["artifact", "sticker"],
            stage="package",
        ),
    ]
    metadata = {
        "selected_workflow": image_manifest.name,
        "required_assets": [
            asset.to_dict() for asset in image_manifest.required_assets
        ],
        "graph_overview": [node.node_id for node in nodes],
        "character": character,
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name=image_manifest.name,
        nodes=nodes,
        metadata=metadata,
        description=f"Agentic sticker pack chain for goal '{goal.prompt}'",
    )


def build_game_sprite(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    """Build one fully automatic, open-ended motion-to-sprite run.

    The LLM owns the motion concept. The planner owns only the technical
    output contract: one continuous clip sampled into a 4x4 atlas.
    """

    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("image_workflow_name",),
        allowed_media_types={"image"},
    )
    video_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_I2V_WORKFLOWS,
        constraint_keys=("video_workflow_name", "workflow_name"),
        allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
    )
    image_width, image_height = context.canvas_dimensions(goal, image_manifest)
    if goal.constraints.get("sprite_source_width") not in {None, ""}:
        image_width = int(goal.constraints["sprite_source_width"])
    if goal.constraints.get("sprite_source_height") not in {None, ""}:
        image_height = int(goal.constraints["sprite_source_height"])
    video_defaults = dict(video_manifest.recommended_defaults or {})
    video_width = int(
        goal.constraints.get("sprite_video_width") or video_defaults.get("width", 608)
    )
    video_height = int(
        goal.constraints.get("sprite_video_height") or video_defaults.get("height", 352)
    )
    target_duration_seconds = max(
        MIN_VIDEO_DURATION_SECONDS,
        int(goal.duration_seconds or MIN_VIDEO_DURATION_SECONDS),
    )
    video_length = max(
        round(MIN_VIDEO_DURATION_SECONDS * 24),
        int(
            goal.constraints.get("sprite_video_length")
            or round(target_duration_seconds * 24)
        ),
    )
    video_steps = int(goal.constraints.get("sprite_video_steps") or 16)
    cell_width = context.constraint_int(goal, "sprite_cell_width", 64)
    cell_height = context.constraint_int(goal, "sprite_cell_height", 64)
    nodes = [
        ExecutionNode(
            node_id="sprite-motion-plan",
            skill_name="agent.sprite.motion_plan",
            inputs={},
            tags=["creative", "sprite", "dynamic-motion"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="sprite-image-assets",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": image_manifest.name,
            },
            depends_on=["sprite-motion-plan"],
            tags=["assets", "sprite", "krea"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        ),
        ExecutionNode(
            node_id="sprite-master-image",
            skill_name="media.image.generate_keyframe",
            inputs={
                "workflow_name": image_manifest.name,
                "prompt_key": "image_prompt",
                "use_prior_frame": False,
                "width": image_width,
                "height": image_height,
                "image_count": 1,
                "suffix": "sprite_master",
            },
            depends_on=["sprite-motion-plan", "sprite-image-assets"],
            tags=["render", "image", "sprite", "krea"],
            tool_name="comfy.workflow.text_to_image",
            stage="render",
        ),
        ExecutionNode(
            node_id="sprite-video-assets",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": video_manifest.name,
            },
            depends_on=["sprite-master-image"],
            tags=["assets", "sprite", "h3"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        ),
        ExecutionNode(
            node_id="sprite-motion-video",
            skill_name="media.image.animate",
            inputs={
                "workflow_name": video_manifest.name,
                "prompt_key": "video_prompt",
                "width": video_width,
                "height": video_height,
                "length": video_length,
                "steps": video_steps,
                "model_profile": str(
                    goal.constraints.get("sprite_h3_model_profile") or "q2"
                ),
                "video_count": 1,
            },
            depends_on=[
                "sprite-motion-plan",
                "sprite-master-image",
                "sprite-video-assets",
            ],
            tags=["render", "video", "sprite", "h3"],
            tool_name="comfy.render_h3",
            stage="render",
        ),
    ]
    sprite_video_node = context.append_nvidia_vfx_postprocess_node(
        goal,
        nodes,
        source_node="sprite-motion-video",
        node_id="sprite-motion-nvidia-vfx",
    )
    nodes.append(
        ExecutionNode(
            node_id="sprite-package",
            skill_name="agent.sprite.package",
            inputs={
                "motion_node": "sprite-motion-plan",
                "image_node": "sprite-master-image",
                "video_node": sprite_video_node,
                "cell_width": cell_width,
                "cell_height": cell_height,
                "fps": float(goal.constraints.get("sprite_fps") or 12),
            },
            depends_on=[
                "sprite-motion-plan",
                "sprite-master-image",
                sprite_video_node,
            ],
            tags=["artifact", "sprite", "gif", "atlas"],
            stage="package",
        )
    )
    metadata = {
        "selected_workflows": [image_manifest.name, video_manifest.name],
        "required_assets": [
            *[asset.to_dict() for asset in image_manifest.required_assets],
            *[asset.to_dict() for asset in video_manifest.required_assets],
        ],
        "graph_overview": [node.node_id for node in nodes],
        "grid": {"rows": 4, "columns": 4},
        "motion_source": "llm_dynamic",
        "human_review": False,
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="game_sprite_v1",
        nodes=nodes,
        metadata=metadata,
        description=f"Automatic open-ended game sprite chain for goal '{goal.prompt}'",
    )


def build_animated_sticker(
    context: PlanningContext, goal: GoalRequest
) -> ExecutionPlan:
    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("image_workflow_name", "workflow_name"),
        allowed_media_types={"image"},
    )
    video_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_I2V_WORKFLOWS,
        constraint_keys=("video_workflow_name",),
        allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
    )
    character = goal.constraints.get("character", "")
    review_loop_enabled = context.review_loop_enabled(goal)
    review_notes = str(goal.constraints.get("review_notes", "") or "")
    selection_limit = context.review_selection_limit(
        goal, default=context.constraint_int(goal, "review_selection_limit", 2)
    )
    expression_count = context.constraint_int(goal, "sticker_expression_count", 8)
    images_per_prompt = context.constraint_int(goal, "images_per_prompt", 1)
    nodes = [
        ExecutionNode(
            node_id="sticker-expressions",
            skill_name="agent.sticker.expressions",
            inputs={
                "prompt": goal.prompt,
                "character": character,
                "expression_count": expression_count,
            },
            tags=["creative", "sticker", "animated"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="sticker-prompts",
            skill_name="agent.sticker.prompt_set",
            inputs={"character": character},
            depends_on=["sticker-expressions"],
            tags=["creative", "sticker"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="image-asset-check",
            skill_name="image.ensure_workflow",
            inputs={
                "workflow_name": image_manifest.name,
            },
            depends_on=["sticker-prompts"],
            tags=["assets", "sticker"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        ),
        ExecutionNode(
            node_id="render-stickers",
            skill_name="media.image.render_batch",
            inputs={
                "workflow_name": image_manifest.name,
                "width": context.canvas_dimensions(goal, image_manifest)[0],
                "height": context.canvas_dimensions(goal, image_manifest)[1],
                "images_per_prompt": images_per_prompt,
                "suffix": "animated_stickers",
            },
            depends_on=["sticker-prompts", "image-asset-check"],
            tags=["render", "sticker", "batch"],
            stage="render",
        ),
        ExecutionNode(
            node_id="motion-prompt",
            skill_name="agent.sticker.motion_prompt",
            inputs={},
            depends_on=["render-stickers"],
            tags=["creative", "sticker", "motion"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="video-asset-check",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": video_manifest.name,
            },
            depends_on=["motion-prompt"],
            tags=["assets", "sticker", "video"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        ),
        ExecutionNode(
            node_id="animate-sticker",
            skill_name="media.image.animate",
            inputs={
                "workflow_name": video_manifest.name,
                "width": context.canvas_dimensions(goal, image_manifest)[0],
                "height": context.canvas_dimensions(goal, image_manifest)[1],
                "video_count": context.constraint_int(goal, "video_count", 1),
            },
            depends_on=["motion-prompt", "video-asset-check"],
            tags=["render", "video", "sticker"],
            tool_name="comfy.render_h3",
            stage="render",
        ),
        ExecutionNode(
            node_id="gif-preview",
            skill_name="media.video.gif_preview",
            inputs={"fps": 12, "scale_width": 512},
            depends_on=["animate-sticker"],
            tags=["preview", "gif", "sticker"],
            tool_name="media.video_to_gif",
            stage="package",
        ),
        ExecutionNode(
            node_id="package-animated-sticker",
            skill_name="agent.sticker.animate.package",
            inputs={},
            depends_on=["animate-sticker", "gif-preview"],
            tags=["artifact", "sticker", "animated"],
            stage="package",
        ),
    ]
    if review_loop_enabled:
        nodes.extend(
            [
                ExecutionNode(
                    node_id="review-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_notes": review_notes},
                    depends_on=[
                        "render-stickers",
                        "animate-sticker",
                        "gif-preview",
                        "package-animated-sticker",
                    ],
                    tags=["review", "sticker", "retry"],
                    stage="review",
                ),
                ExecutionNode(
                    node_id="review-refine-prompt",
                    skill_name="agent.review.refine_prompt",
                    inputs={"review_notes": review_notes, "retry_count": 1},
                    depends_on=["motion-prompt", "review-select"],
                    tags=["review", "creative", "retry"],
                    stage="review",
                ),
                ExecutionNode(
                    node_id="review-animate-sticker",
                    skill_name="media.image.animate",
                    inputs={
                        "workflow_name": video_manifest.name,
                        "width": context.canvas_dimensions(goal, image_manifest)[0],
                        "height": context.canvas_dimensions(goal, image_manifest)[1],
                        "video_count": context.constraint_int(goal, "video_count", 1),
                    },
                    depends_on=[
                        "review-refine-prompt",
                        "render-stickers",
                        "video-asset-check",
                    ],
                    tags=["render", "video", "sticker", "retry"],
                    tool_name="comfy.render_h3",
                    stage="render",
                ),
                ExecutionNode(
                    node_id="review-gif-preview",
                    skill_name="media.video.gif_preview",
                    inputs={"fps": 12, "scale_width": 512},
                    depends_on=["review-animate-sticker"],
                    tags=["preview", "gif", "sticker", "retry"],
                    tool_name="media.video_to_gif",
                    stage="package",
                ),
                ExecutionNode(
                    node_id="review-package-animated-sticker",
                    skill_name="agent.sticker.animate.package",
                    inputs={},
                    depends_on=["review-animate-sticker", "review-gif-preview"],
                    tags=["artifact", "sticker", "animated", "retry"],
                    stage="package",
                ),
                ExecutionNode(
                    node_id="review-final-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_notes": review_notes},
                    depends_on=["review-package-animated-sticker"],
                    tags=["review", "sticker", "retry"],
                    stage="review",
                ),
            ]
        )
    metadata = {
        "selected_workflows": [image_manifest.name, video_manifest.name],
        "required_assets": [
            *[asset.to_dict() for asset in image_manifest.required_assets],
            *[asset.to_dict() for asset in video_manifest.required_assets],
        ],
        "graph_overview": [node.node_id for node in nodes],
        "character": character,
        "review_loop_enabled": review_loop_enabled,
        "review_notes": review_notes,
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="animated_sticker_v1",
        nodes=nodes,
        metadata=metadata,
        description=f"Agentic animated sticker chain for goal '{goal.prompt}'",
    )
