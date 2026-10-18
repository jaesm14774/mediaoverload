"""Short Video strategy graph assembly."""

from __future__ import annotations

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest
from agentic.runtime.planning.common import PlanningContext


def build_text2img2video(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    pre_video_review = context.pre_video_review_enabled(goal)
    stage_probe_auto_select = bool(
        goal.constraints.get("stage_probe_auto_select", False)
    )
    reference_node = context.reference_video_analysis_node(goal)
    reference_video_source = str(
        goal.constraints.get("reference_video_source") or ""
    ).strip()
    if reference_video_source and reference_node is None:
        raise ValueError(
            "reference_video_source requires a readable reference analysis node"
        )
    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("image_workflow_name", "workflow_name"),
        allowed_media_types={"image"},
    )
    image_width, image_height = context.canvas_dimensions(goal, image_manifest)
    use_upscale_for_i2v = not bool(goal.constraints.get("skip_upscale_for_i2v", False))
    upscale_manifest = (
        context.manifest_from_goal_constraints(
            goal,
            *context.DEFAULT_UPSCALE_WORKFLOWS,
            constraint_keys=("upscale_workflow_name",),
            allowed_media_types={"image_upscale"},
        )
        if use_upscale_for_i2v
        else None
    )
    video_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_I2V_WORKFLOWS,
        constraint_keys=("video_workflow_name",),
        allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
    )
    review_loop_enabled = context.review_loop_enabled(goal)
    stage_review_enabled = context.stage_review_enabled(goal) or pre_video_review
    review_notes = str(goal.constraints.get("review_notes", "") or "")
    selection_limit = context.review_selection_limit(
        goal, default=context.constraint_int(goal, "review_selection_limit", 2)
    )
    image_count = (
        context.pre_video_candidate_count(goal)
        if pre_video_review
        else context.constraint_int(goal, "image_count", 1)
    )
    nodes: list[ExecutionNode] = []
    if reference_node:
        nodes.append(reference_node)
    nodes.extend(
        [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": context.idea_director.generate_variations(goal),
                },
                depends_on=[reference_node.node_id] if reference_node else [],
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
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-image",
                skill_name="image.render",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_width,
                    "height": image_height,
                    "image_count": image_count,
                },
                depends_on=["idea-brief", "image-asset-check"],
                tags=["render", "image"],
                tool_name="comfy.workflow.text_to_image",
                stage="render",
            ),
        ]
    )
    image_render_node = nodes[-1]
    if goal.constraints.get("seed") is not None:
        image_render_node.inputs["seed"] = int(goal.constraints["seed"])
    if use_upscale_for_i2v:
        nodes.extend(
            [
                ExecutionNode(
                    node_id="upscale-asset-check",
                    skill_name="media.ensure_workflow",
                    inputs={
                        "workflow_name": upscale_manifest.name,
                    },
                    depends_on=["render-image"],
                    tags=["assets"],
                    tool_name="asset.ensure_workflow_ready",
                    stage="assets",
                ),
                ExecutionNode(
                    node_id="upscale-image",
                    skill_name="image.upscale",
                    inputs={},
                    depends_on=["render-image", "upscale-asset-check"],
                    tags=["render", "upscale"],
                    tool_name="comfy.workflow.image_upscale",
                    stage="render",
                ),
            ]
        )
    nodes.append(
        ExecutionNode(
            node_id="video-asset-check",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": video_manifest.name,
            },
            depends_on=["upscale-image" if use_upscale_for_i2v else "render-image"],
            tags=["assets"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        )
    )
    if stage_review_enabled:
        nodes.append(
            ExecutionNode(
                node_id="stage-review-select",
                skill_name="review.assets.select",
                inputs={
                    "limit": 1,
                    "review_all_candidates": True,
                    "review_scope": "first_frame",
                    "review_phase": "opening_frame",
                    "require_human_review": context.pre_video_review_requires_human(
                        goal
                    ),
                    "auto_select_for_probe": stage_probe_auto_select,
                    "review_notes": review_notes,
                },
                depends_on=[
                    "render-image"
                    if pre_video_review or not use_upscale_for_i2v
                    else "upscale-image"
                ],
                tags=["review", "stage"],
                stage="review",
            )
        )
    animate_dependencies = ["idea-brief", "render-image", "video-asset-check"]
    if use_upscale_for_i2v and not stage_review_enabled:
        animate_dependencies.insert(2, "upscale-image")
    if stage_review_enabled:
        animate_dependencies = [
            "idea-brief",
            "render-image",
            "stage-review-select",
            "video-asset-check",
        ]
    nodes.extend(
        [
            ExecutionNode(
                node_id="animate-video",
                skill_name="image.animate",
                inputs={
                    "workflow_name": video_manifest.name,
                    "width": image_width,
                    "height": image_height,
                    "video_count": context.constraint_int(goal, "video_count", 1),
                    **(
                        {"length": context.text2img2video_length(goal)}
                        if context.text2img2video_length(goal) is not None
                        else {}
                    ),
                    **(
                        {"seed": int(goal.constraints["seed"])}
                        if goal.constraints.get("seed") is not None
                        else {}
                    ),
                },
                depends_on=animate_dependencies,
                tags=["render", "video"],
                tool_name="comfy.render_h3",
                stage="render",
            ),
            ExecutionNode(
                node_id="video-qa",
                skill_name="media.video.inspect",
                inputs={
                    **context.video_qa_inputs(goal, video_manifest),
                    "character": str(goal.constraints.get("character") or ""),
                    "subject_context": dict(
                        goal.constraints.get("subject_context") or {}
                    ),
                },
                depends_on=[
                    "animate-video",
                    "idea-brief",
                    *([reference_node.node_id] if reference_node else []),
                ],
                tags=["quality", "technical-qa", "video"],
                tool_name="media.video_qa",
                stage="quality",
            ),
            ExecutionNode(
                node_id="gif-preview",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=["animate-video"],
                tags=["preview", "gif"],
                tool_name="media.video_to_gif",
                stage="package",
            ),
        ]
    )
    video_canvas_node = context.append_video_canvas_node(
        nodes,
        source_node="animate-video",
        node_id="video-canvas",
        width=image_width,
        height=image_height,
    )
    video_output_node = context.append_video_speed_node(
        goal,
        nodes,
        source_node=video_canvas_node,
        node_id="video-speed",
    )
    motion_graphics_enabled = bool(
        goal.constraints.get("enable_motion_graphics", False)
    )
    final_video_node = video_output_node
    if motion_graphics_enabled:
        nodes.append(
            ExecutionNode(
                node_id="motion-graphics-review",
                skill_name="media.video.motion_graphics",
                inputs={"max_review_rounds": 2},
                depends_on=[video_output_node],
                tags=["render", "programmatic-graphics", "visual-review"],
                stage="quality",
            )
        )
        final_video_node = "motion-graphics-review"
    final_video_node = context.append_nvidia_vfx_postprocess_node(
        goal,
        nodes,
        source_node=final_video_node,
        node_id="video-nvidia-vfx",
    )
    video_qa_node = next(node for node in nodes if node.node_id == "video-qa")
    video_qa_node.inputs = context.scaled_video_qa_inputs(goal, video_qa_node.inputs)
    video_qa_node.depends_on = [
        final_video_node,
        "idea-brief",
        *([reference_node.node_id] if reference_node else []),
    ]
    gif_preview_node = next(node for node in nodes if node.node_id == "gif-preview")
    gif_preview_node.depends_on = [final_video_node]
    if review_loop_enabled:
        nodes.extend(
            [
                ExecutionNode(
                    node_id="review-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_notes": review_notes},
                    depends_on=[
                        "render-image",
                        *(["upscale-image"] if use_upscale_for_i2v else []),
                        final_video_node,
                        "gif-preview",
                    ],
                    tags=["review", "retry"],
                    stage="review",
                ),
                ExecutionNode(
                    node_id="review-refine-prompt",
                    skill_name="agent.review.refine_prompt",
                    inputs={"review_notes": review_notes, "retry_count": 1},
                    depends_on=["idea-brief", "review-select"],
                    tags=["review", "creative", "retry"],
                    stage="review",
                ),
                ExecutionNode(
                    node_id="review-render-image",
                    skill_name="image.render",
                    inputs={
                        "workflow_name": image_manifest.name,
                        "width": image_width,
                        "height": image_height,
                        "image_count": image_count,
                        **(
                            {"seed": int(goal.constraints["seed"])}
                            if goal.constraints.get("seed") is not None
                            else {}
                        ),
                    },
                    depends_on=["review-refine-prompt", "image-asset-check"],
                    tags=["render", "image", "retry"],
                    tool_name="comfy.workflow.text_to_image",
                    stage="render",
                ),
                *(
                    [
                        ExecutionNode(
                            node_id="review-upscale-image",
                            skill_name="image.upscale",
                            inputs={},
                            depends_on=["review-render-image", "upscale-asset-check"],
                            tags=["render", "upscale", "retry"],
                            tool_name="comfy.workflow.image_upscale",
                            stage="render",
                        )
                    ]
                    if use_upscale_for_i2v
                    else []
                ),
                ExecutionNode(
                    node_id="review-animate-video",
                    skill_name="image.animate",
                    inputs={
                        "workflow_name": video_manifest.name,
                        "width": context.canvas_dimensions(goal, image_manifest)[0],
                        "height": context.canvas_dimensions(goal, image_manifest)[1],
                        "video_count": context.constraint_int(goal, "video_count", 1),
                        **(
                            {"length": context.text2img2video_length(goal)}
                            if context.text2img2video_length(goal) is not None
                            else {}
                        ),
                        **(
                            {"seed": int(goal.constraints["seed"])}
                            if goal.constraints.get("seed") is not None
                            else {}
                        ),
                    },
                    depends_on=[
                        "review-refine-prompt",
                        *(
                            ["review-upscale-image"]
                            if use_upscale_for_i2v
                            else ["review-render-image"]
                        ),
                        "video-asset-check",
                    ],
                    tags=["render", "video", "retry"],
                    tool_name="comfy.render_h3",
                    stage="render",
                ),
                ExecutionNode(
                    node_id="review-gif-preview",
                    skill_name="media.video.gif_preview",
                    inputs={"fps": 12, "scale_width": 512},
                    depends_on=["review-animate-video"],
                    tags=["preview", "gif", "retry"],
                    tool_name="media.video_to_gif",
                    stage="package",
                ),
                ExecutionNode(
                    node_id="review-final-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_notes": review_notes},
                    depends_on=[
                        "review-render-image",
                        "review-upscale-image",
                        "review-animate-video",
                        "review-gif-preview",
                    ],
                    tags=["review", "retry"],
                    stage="review",
                ),
            ]
        )
        review_video_canvas_node = context.append_video_canvas_node(
            nodes,
            source_node="review-animate-video",
            node_id="review-video-canvas",
            width=image_width,
            height=image_height,
            retry=True,
        )
        review_video_output_node = context.append_video_speed_node(
            goal,
            nodes,
            source_node=review_video_canvas_node,
            node_id="review-video-speed",
            retry=True,
        )
        review_final_video_node = review_video_output_node
        if motion_graphics_enabled:
            nodes.append(
                ExecutionNode(
                    node_id="review-motion-graphics-review",
                    skill_name="media.video.motion_graphics",
                    inputs={"max_review_rounds": 2},
                    depends_on=[review_video_output_node],
                    tags=["render", "programmatic-graphics", "visual-review", "retry"],
                    stage="quality",
                )
            )
            review_final_video_node = "review-motion-graphics-review"
        review_final_video_node = context.append_nvidia_vfx_postprocess_node(
            goal,
            nodes,
            source_node=review_final_video_node,
            node_id="review-video-nvidia-vfx",
            retry=True,
        )
        review_gif_preview_node = next(
            node for node in nodes if node.node_id == "review-gif-preview"
        )
        review_gif_preview_node.depends_on = [review_final_video_node]
        review_final_select_node = next(
            node for node in nodes if node.node_id == "review-final-select"
        )
        review_final_select_node.depends_on = [
            "review-render-image",
            *(["review-upscale-image"] if use_upscale_for_i2v else []),
            review_final_video_node,
            "review-gif-preview",
        ]
    summary_dependencies = ["render-image", "animate-video", "video-qa", "gif-preview"]
    if use_upscale_for_i2v:
        summary_dependencies.insert(1, "upscale-image")
    if review_loop_enabled:
        summary_dependencies.append("review-final-select")
    nodes.append(
        ExecutionNode(
            node_id="persist-text2img2video-summary",
            skill_name="agent.summary.persist",
            inputs={
                "summary_name": "text2img2video_summary.json",
                "summary_scope": "text2img2video",
            },
            depends_on=summary_dependencies,
            tags=["artifact", "summary"],
            stage="package",
        )
    )
    metadata = {
        "selected_workflows": [
            image_manifest.name,
            *([upscale_manifest.name] if use_upscale_for_i2v else []),
            video_manifest.name,
        ],
        "required_assets": [
            *[asset.to_dict() for asset in image_manifest.required_assets],
            *(
                [asset.to_dict() for asset in upscale_manifest.required_assets]
                if use_upscale_for_i2v
                else []
            ),
            *[asset.to_dict() for asset in video_manifest.required_assets],
        ],
        "graph_overview": [node.node_id for node in nodes],
        "review_loop_enabled": review_loop_enabled,
        "video_speed": context.video_speed_config(goal),
        "motion_graphics_enabled": motion_graphics_enabled,
        "review_notes": review_notes,
        "reference_video_source": reference_video_source,
        **(
            {"reference_video": context.reference_video_metadata(goal)}
            if reference_node
            else {}
        ),
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="text2img2video_v1",
        nodes=nodes,
        metadata=metadata,
        description=f"Composed text2img2video chain for goal '{goal.prompt}'",
    )


def build_text2video(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    pre_video_review = context.pre_video_review_enabled(goal)
    stage_probe_auto_select = bool(
        goal.constraints.get("stage_probe_auto_select", False)
    )
    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("image_workflow_name",),
        allowed_media_types={"image"},
    )
    video_manifest = context.manifest_from_goal_constraints(
        goal,
        *(
            context.DEFAULT_I2V_WORKFLOWS
            if pre_video_review
            else context.DEFAULT_T2V_WORKFLOWS
        ),
        # A T2V manifest may advertise ``long_video`` as well, so merely
        # filtering by media type is not enough to guarantee that the
        # approved image becomes conditioning. Ignore the T2V override
        # whenever the shared image gate is active.
        constraint_keys=("video_workflow_name", "workflow_name")
        if not pre_video_review
        else (),
        allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
    )
    review_loop_enabled = context.review_loop_enabled(goal)
    stage_review_enabled = context.stage_review_enabled(goal) or pre_video_review
    review_notes = str(goal.constraints.get("review_notes", "") or "")
    selection_limit = context.review_selection_limit(
        goal, default=context.constraint_int(goal, "review_selection_limit", 2)
    )
    image_count = (
        context.pre_video_candidate_count(goal)
        if pre_video_review
        else context.constraint_int(goal, "image_count", 1)
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
            node_id="video-asset-check",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": video_manifest.name,
            },
            depends_on=["render-image"],
            tags=["assets"],
            stage="assets",
        ),
    ]
    if stage_review_enabled:
        nodes.append(
            ExecutionNode(
                node_id="stage-review-select",
                skill_name="review.assets.select",
                inputs={
                    "limit": 1,
                    "review_all_candidates": True,
                    "review_scope": "first_frame",
                    "review_phase": "opening_frame",
                    "require_human_review": context.pre_video_review_requires_human(
                        goal
                    ),
                    "auto_select_for_probe": stage_probe_auto_select,
                    "review_notes": review_notes,
                },
                depends_on=["render-image"],
                tags=["review", "stage"],
                stage="review",
            )
        )
    animate_dependencies = ["idea-brief", "render-image", "video-asset-check"]
    if stage_review_enabled:
        animate_dependencies = [
            "idea-brief",
            "stage-review-select",
            "video-asset-check",
        ]
    nodes.extend(
        [
            ExecutionNode(
                node_id="animate-video",
                skill_name="media.image.animate",
                inputs={
                    "workflow_name": video_manifest.name,
                    "width": context.canvas_dimensions(goal, image_manifest)[0],
                    "height": context.canvas_dimensions(goal, image_manifest)[1],
                    "video_count": context.constraint_int(goal, "video_count", 1),
                    "length": context.native_h3_render_config(
                        goal, video_manifest
                    )["length"],
                },
                depends_on=animate_dependencies,
                tags=["render", "video"],
                stage="render",
            ),
            ExecutionNode(
                node_id="video-qa",
                skill_name="media.video.inspect",
                inputs=context.video_qa_inputs(goal, video_manifest),
                depends_on=["animate-video"],
                tags=["quality", "technical-qa", "video"],
                tool_name="media.video_qa",
                stage="quality",
            ),
            ExecutionNode(
                node_id="gif-preview",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=["animate-video"],
                tags=["preview", "gif"],
                stage="package",
            ),
        ]
    )
    video_output_node = context.append_video_speed_node(
        goal,
        nodes,
        source_node="animate-video",
        node_id="video-speed",
    )
    video_output_node = context.append_nvidia_vfx_postprocess_node(
        goal,
        nodes,
        source_node=video_output_node,
        node_id="video-nvidia-vfx",
    )
    video_qa_node = next(node for node in nodes if node.node_id == "video-qa")
    video_qa_node.inputs = context.scaled_video_qa_inputs(goal, video_qa_node.inputs)
    video_qa_node.depends_on = [video_output_node]
    gif_preview_node = next(node for node in nodes if node.node_id == "gif-preview")
    gif_preview_node.depends_on = [video_output_node]
    if review_loop_enabled:
        nodes.extend(
            [
                ExecutionNode(
                    node_id="review-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_notes": review_notes},
                    depends_on=["render-image", video_output_node, "gif-preview"],
                    tags=["review", "retry"],
                    stage="review",
                ),
                ExecutionNode(
                    node_id="review-refine-prompt",
                    skill_name="agent.review.refine_prompt",
                    inputs={"review_notes": review_notes, "retry_count": 1},
                    depends_on=["idea-brief", "review-select"],
                    tags=["review", "creative", "retry"],
                    stage="review",
                ),
                ExecutionNode(
                    node_id="review-render-image",
                    skill_name="image.render",
                    inputs={
                        "workflow_name": image_manifest.name,
                        "width": context.canvas_dimensions(goal, image_manifest)[0],
                        "height": context.canvas_dimensions(goal, image_manifest)[1],
                        "image_count": image_count,
                    },
                    depends_on=["review-refine-prompt", "image-asset-check"],
                    tags=["render", "image", "retry"],
                    stage="render",
                ),
                ExecutionNode(
                    node_id="review-animate-video",
                    skill_name="media.image.animate",
                    inputs={
                        "workflow_name": video_manifest.name,
                        "width": context.canvas_dimensions(goal, image_manifest)[0],
                        "height": context.canvas_dimensions(goal, image_manifest)[1],
                        "video_count": context.constraint_int(goal, "video_count", 1),
                    },
                    depends_on=[
                        "review-refine-prompt",
                        "review-render-image",
                        "video-asset-check",
                    ],
                    tags=["render", "video", "retry"],
                    stage="render",
                ),
                ExecutionNode(
                    node_id="review-gif-preview",
                    skill_name="media.video.gif_preview",
                    inputs={"fps": 12, "scale_width": 512},
                    depends_on=["review-animate-video"],
                    tags=["preview", "gif", "retry"],
                    stage="package",
                ),
                ExecutionNode(
                    node_id="review-final-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_notes": review_notes},
                    depends_on=[
                        "review-render-image",
                        "review-animate-video",
                        "review-gif-preview",
                    ],
                    tags=["review", "retry"],
                    stage="review",
                ),
            ]
        )
        review_video_output_node = context.append_video_speed_node(
            goal,
            nodes,
            source_node="review-animate-video",
            node_id="review-video-speed",
            retry=True,
        )
        review_video_output_node = context.append_nvidia_vfx_postprocess_node(
            goal,
            nodes,
            source_node=review_video_output_node,
            node_id="review-video-nvidia-vfx",
            retry=True,
        )
        review_gif_preview_node = next(
            node for node in nodes if node.node_id == "review-gif-preview"
        )
        review_gif_preview_node.depends_on = [review_video_output_node]
        review_final_select_node = next(
            node for node in nodes if node.node_id == "review-final-select"
        )
        review_final_select_node.depends_on = [
            "review-render-image",
            review_video_output_node,
            "review-gif-preview",
        ]
    summary_dependencies = ["render-image", "animate-video", "video-qa", "gif-preview"]
    if review_loop_enabled:
        summary_dependencies.append("review-final-select")
    nodes.append(
        ExecutionNode(
            node_id="persist-text2video-summary",
            skill_name="agent.summary.persist",
            inputs={
                "summary_name": "text2video_summary.json",
                "summary_scope": "text2video",
            },
            depends_on=summary_dependencies,
            tags=["artifact", "summary"],
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
        "review_loop_enabled": review_loop_enabled,
        "video_speed": context.video_speed_config(goal),
        "review_notes": review_notes,
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="text2video_v1",
        nodes=nodes,
        metadata=metadata,
        description=f"Agentic text2video chain for goal '{goal.prompt}'",
    )
