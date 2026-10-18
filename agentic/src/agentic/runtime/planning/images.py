"""Images strategy graph assembly."""

from __future__ import annotations

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest
from agentic.runtime.story_cards import (
    STORY_CARD_DEFAULT_HEIGHT,
    STORY_CARD_DEFAULT_WIDTH,
    STORY_CARD_PAGE_COUNT_DEFAULT,
    resolve_story_card_canvas_dimension,
    resolve_story_card_page_count,
)
from agentic.runtime.planning.common import PlanningContext


def build_image(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    workflow_manifest = context.pick_goal_workflow(goal)
    width, height = context.canvas_dimensions(goal, workflow_manifest)
    image_count = context.constraint_int(goal, "image_count", 1)
    bool(goal.constraints.get("stage_probe_auto_select", False))
    nodes = [
        ExecutionNode(
            node_id="idea-brief",
            skill_name="agent.goal.expand",
            inputs={
                "prompt": goal.prompt,
                "style": goal.style,
                "idea_variants": context.idea_director.generate_variations(goal),
            },
            tags=["creative"],
        ),
        ExecutionNode(
            node_id="asset-check",
            skill_name="image.ensure_workflow",
            inputs={
                "workflow_name": workflow_manifest.name,
            },
            depends_on=["idea-brief"],
            tags=["assets"],
        ),
        ExecutionNode(
            node_id="render-image",
            skill_name="image.render",
            inputs={
                "workflow_name": workflow_manifest.name,
                "width": width,
                "height": height,
                "image_count": image_count,
            },
            depends_on=["idea-brief", "asset-check"],
            tags=["render"],
        ),
    ]
    if bool(goal.constraints.get("stage_probe_auto_select", False)):
        nodes.append(
            ExecutionNode(
                node_id="stage-preview-select",
                skill_name="review.assets.select",
                inputs={
                    "limit": context.review_selection_limit(goal, default=1),
                    "review_all_candidates": True,
                    "review_scope": "reference",
                    "review_phase": "reference_selection",
                    "require_human_review": False,
                    "auto_select_for_probe": True,
                    "review_notes": "",
                },
                depends_on=["render-image"],
                tags=["review", "stage", "probe"],
                stage="review",
            )
        )
    metadata = {
        "selected_workflow": workflow_manifest.name,
        "required_assets": [
            asset.to_dict() for asset in workflow_manifest.required_assets
        ],
        "graph_overview": [node.node_id for node in nodes],
        "width": width,
        "height": height,
        "stage_probe_auto_select": bool(
            goal.constraints.get("stage_probe_auto_select", False)
        ),
    }
    description = f"Image workflow '{workflow_manifest.name}' for goal '{goal.prompt}'"
    return ExecutionPlan(
        goal=goal,
        workflow_name=workflow_manifest.name,
        nodes=nodes,
        metadata=metadata,
        description=description,
    )


def build_comfy_primitive(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    workflow_manifest = context.pick_goal_workflow(goal)
    constraints = goal.constraints
    input_image_path = constraints.get("input_image_path")
    if (
        goal.media_type in {"image_refine", "image_upscale", "image_to_video"}
        and not input_image_path
    ):
        raise ValueError(f"media_type '{goal.media_type}' requires --input-image")

    if goal.media_type == "image_refine":
        nodes = [
            ExecutionNode(
                node_id="prompt-prepare",
                skill_name="agent.prompt.compose",
                inputs={"prompt": goal.prompt, "style": goal.style},
                tags=["creative"],
            ),
            ExecutionNode(
                node_id="asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": workflow_manifest.name,
                },
                depends_on=["prompt-prepare"],
                tags=["assets"],
            ),
            ExecutionNode(
                node_id="img2img-render",
                skill_name="media.image.refine",
                inputs={
                    "image_path": input_image_path,
                },
                depends_on=["prompt-prepare", "asset-check"],
                tags=["render"],
            ),
        ]
    elif goal.media_type == "image_upscale":
        nodes = [
            ExecutionNode(
                node_id="asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": workflow_manifest.name,
                },
                tags=["assets"],
            ),
            ExecutionNode(
                node_id="upscale-render",
                skill_name="media.image.upscale",
                inputs={"image_path": input_image_path},
                depends_on=["asset-check"],
                tags=["render"],
            ),
        ]
    else:
        nodes = [
            ExecutionNode(
                node_id="prompt-prepare",
                skill_name="agent.prompt.compose",
                inputs={"prompt": goal.prompt, "style": goal.style},
                tags=["creative"],
            ),
            ExecutionNode(
                node_id="asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": workflow_manifest.name,
                },
                depends_on=["prompt-prepare"],
                tags=["assets"],
            ),
            ExecutionNode(
                node_id="i2v-render",
                skill_name="media.image.animate",
                inputs={
                    "image_path": input_image_path,
                    "prompt": constraints.get("text", "") or goal.prompt,
                    "length": context.native_h3_render_config(
                        goal, workflow_manifest
                    )["length"],
                    "width": constraints.get("canvas_width")
                    or constraints.get("width"),
                    "height": constraints.get("canvas_height")
                    or constraints.get("height"),
                },
                depends_on=["prompt-prepare", "asset-check"],
                tags=["render", "video"],
            ),
        ]

    metadata = {
        "selected_workflow": workflow_manifest.name,
        "required_assets": [
            asset.to_dict() for asset in workflow_manifest.required_assets
        ],
        "graph_overview": [node.node_id for node in nodes],
        "input_image_path": input_image_path,
    }
    description = (
        f"Workflow '{workflow_manifest.name}' for media_type '{goal.media_type}'"
    )
    return ExecutionPlan(
        goal=goal,
        workflow_name=workflow_manifest.name,
        nodes=nodes,
        metadata=metadata,
        description=description,
    )


def build_text2img2img(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("image_workflow_name", "workflow_name"),
        allowed_media_types={"image"},
    )
    refine_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_REFINE_WORKFLOWS,
        constraint_keys=("refine_workflow_name", "transition_workflow_name"),
        allowed_media_types={"image_refine"},
    )
    image_count = context.constraint_int(goal, "image_count", 1)
    stage_probe_auto_select = bool(
        goal.constraints.get("stage_probe_auto_select", False)
    )
    nodes = [
        ExecutionNode(
            node_id="idea-brief",
            skill_name="agent.goal.expand",
            inputs={
                "prompt": goal.prompt,
                "style": goal.style,
                "idea_variants": context.idea_director.generate_variations(goal),
            },
            tags=["creative"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="image-asset-check",
            skill_name="image.ensure_workflow",
            inputs={
                "workflow_name": image_manifest.name,
            },
            depends_on=["idea-brief"],
            tags=["assets"],
            stage="assets",
        ),
        ExecutionNode(
            node_id="render-image",
            skill_name="image.render",
            inputs={
                "workflow_name": image_manifest.name,
                "width": context.canvas_dimensions(goal, image_manifest)[0],
                "height": context.canvas_dimensions(goal, image_manifest)[1],
                "image_count": image_count,
            },
            depends_on=["idea-brief", "image-asset-check"],
            tags=["render", "image"],
            stage="render",
        ),
        ExecutionNode(
            node_id="refine-asset-check",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": refine_manifest.name,
            },
            depends_on=["render-image"],
            tags=["assets"],
            stage="assets",
        ),
        ExecutionNode(
            node_id="refine-image",
            skill_name="media.image.refine",
            inputs={"workflow_name": refine_manifest.name},
            depends_on=["idea-brief", "render-image", "refine-asset-check"],
            tags=["render", "image", "refine"],
            stage="render",
        ),
    ]
    if stage_probe_auto_select:
        nodes.insert(
            3,
            ExecutionNode(
                node_id="stage-preview-select",
                skill_name="review.assets.select",
                inputs={
                    "limit": context.review_selection_limit(goal, default=1),
                    "review_all_candidates": True,
                    "review_scope": "reference",
                    "review_phase": "reference_selection",
                    "require_human_review": False,
                    "auto_select_for_probe": True,
                    "review_notes": "",
                },
                depends_on=["render-image"],
                tags=["review", "stage", "probe"],
                stage="review",
            ),
        )
        refine_node = next(node for node in nodes if node.node_id == "refine-image")
        refine_node.depends_on = [
            "idea-brief",
            "stage-preview-select",
            "refine-asset-check",
        ]
    metadata = {
        "selected_workflows": [image_manifest.name, refine_manifest.name],
        "required_assets": [
            *[asset.to_dict() for asset in image_manifest.required_assets],
            *[asset.to_dict() for asset in refine_manifest.required_assets],
        ],
        "graph_overview": [node.node_id for node in nodes],
        "stage_probe_auto_select": stage_probe_auto_select,
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="text2img2img_v1",
        nodes=nodes,
        metadata=metadata,
        description=f"Agentic text2img2img chain for goal '{goal.prompt}'",
    )


def build_story_card(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    requested_page_count = goal.constraints.get(
        "story_card_page_count",
        STORY_CARD_PAGE_COUNT_DEFAULT,
    )
    page_count = resolve_story_card_page_count(requested_page_count)
    planned_page_count = (
        page_count if page_count is not None else STORY_CARD_PAGE_COUNT_DEFAULT
    )
    width = resolve_story_card_canvas_dimension(
        goal.constraints.get("story_card_width"),
        STORY_CARD_DEFAULT_WIDTH,
        name="story_card_width",
    )
    height = resolve_story_card_canvas_dimension(
        goal.constraints.get("story_card_height"),
        STORY_CARD_DEFAULT_HEIGHT,
        name="story_card_height",
    )
    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("image_workflow_name", "workflow_name"),
        allowed_media_types={"image"},
    )
    nodes = [
        ExecutionNode(
            node_id="story-card-write",
            skill_name="agent.story_card.write",
            inputs={"page_count": planned_page_count},
            tags=["creative", "story-card", "writing-first"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="story-card-image-asset-check",
            skill_name="image.ensure_workflow",
            inputs={
                "workflow_name": image_manifest.name,
            },
            depends_on=["story-card-write"],
            tags=["assets", "story-card"],
            stage="assets",
        ),
        ExecutionNode(
            node_id="story-card-backgrounds",
            skill_name="media.story_card.backgrounds",
            inputs={
                "workflow_name": image_manifest.name,
                "render_tool": "comfy.workflow.text_to_image",
                "width": width,
                "height": height,
                "story_node": "story-card-write",
            },
            depends_on=["story-card-write", "story-card-image-asset-check"],
            tags=["render", "story-card", "text-to-image", "background"],
            stage="render",
        ),
        ExecutionNode(
            node_id="story-card-compose",
            skill_name="media.story_card.compose",
            inputs={
                "width": width,
                "height": height,
                "story_node": "story-card-write",
                "background_node": "story-card-backgrounds",
                "overlay_opacity": int(
                    goal.constraints.get("story_card_overlay_opacity", 202)
                ),
                "brand_label": str(
                    goal.constraints.get("story_card_brand_label", "STORY NOTE")
                ),
                "font_path": str(goal.constraints.get("story_card_font_path", "")),
            },
            depends_on=["story-card-write", "story-card-backgrounds"],
            tags=["compose", "story-card", "text-primary"],
            stage="package",
        ),
    ]
    metadata = {
        "selected_workflows": [image_manifest.name],
        "required_assets": [
            *[asset.to_dict() for asset in image_manifest.required_assets],
        ],
        "graph_overview": [node.node_id for node in nodes],
        "page_count": planned_page_count,
        "width": width,
        "height": height,
        "text_primary": True,
        "background_generation": "style_locked_text_to_image",
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="story_card_v1",
        nodes=nodes,
        metadata=metadata,
        description=f"Writing-first story-card chain for goal '{goal.prompt}'",
    )


def build_carousel(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("image_workflow_name", "workflow_name"),
        allowed_media_types={"image"},
    )
    slide_count = context.constraint_int(
        goal, "segment_count", max(3, goal.duration_seconds // 10)
    )
    images_per_prompt = context.constraint_int(goal, "images_per_prompt", 1)
    nodes = [
        ExecutionNode(
            node_id="idea-brief",
            skill_name="agent.goal.expand",
            inputs={
                "prompt": goal.prompt,
                "style": goal.style,
                "idea_variants": context.idea_director.generate_variations(goal),
            },
            tags=["creative", "carousel"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="script-plan",
            skill_name="agent.story.segment",
            inputs={"segment_count": slide_count, "tone": "distinct carousel beat"},
            depends_on=["idea-brief"],
            tags=["story", "carousel"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="carousel-prompts",
            skill_name="agent.carousel.prompt_set",
            inputs={"style": goal.style},
            depends_on=["script-plan"],
            tags=["creative", "carousel"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="image-asset-check",
            skill_name="image.ensure_workflow",
            inputs={
                "workflow_name": image_manifest.name,
            },
            depends_on=["carousel-prompts"],
            tags=["assets", "carousel"],
            stage="assets",
        ),
        ExecutionNode(
            node_id="render-carousel",
            skill_name="media.image.render_batch",
            inputs={
                "workflow_name": image_manifest.name,
                "width": context.canvas_dimensions(goal, image_manifest)[0],
                "height": context.canvas_dimensions(goal, image_manifest)[1],
                "images_per_prompt": images_per_prompt,
                "suffix": "carousel",
            },
            depends_on=["carousel-prompts", "image-asset-check"],
            tags=["render", "carousel", "batch"],
            stage="render",
        ),
        ExecutionNode(
            node_id="package-carousel",
            skill_name="agent.carousel.package",
            inputs={},
            depends_on=["render-carousel"],
            tags=["artifact", "carousel"],
            stage="package",
        ),
    ]
    metadata = {
        "selected_workflow": image_manifest.name,
        "required_assets": [
            asset.to_dict() for asset in image_manifest.required_assets
        ],
        "graph_overview": [node.node_id for node in nodes],
        "slide_count": slide_count,
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="carousel_v1",
        nodes=nodes,
        metadata=metadata,
        description=f"Agentic carousel chain for goal '{goal.prompt}'",
    )
