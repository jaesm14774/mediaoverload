"""Native H3 strategy graph assembly."""

from __future__ import annotations

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest
from agentic.runtime.planning.common import PlanningContext


def build_native_h3_story(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    """Build one causal native H3 clip through the normal skill graph.

    The story prompt and both continuity anchors are generated as graph nodes,
    so scheduled runs can swap the storyboard or topic without introducing a
    second execution mechanism.
    """
    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=(
            "native_h3_keyframe_workflow_name",
            "keyframe_workflow_name",
            "image_workflow_name",
        ),
        allowed_media_types={"image"},
    )
    video_manifest = context.manifest_from_goal_constraints(
        goal,
        (
            "minimax_h3_fl2va"
            if goal.media_type == "native_h3_fl2va_story"
            or goal.constraints.get("native_h3_use_last_frame")
            else "minimax_h3_i2va"
        ),
        constraint_keys=(
            "video_workflow_name",
            "native_h3_workflow_name",
            "workflow_name",
        ),
        allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
    )
    render_config = context.native_h3_render_config(goal, video_manifest)
    width = render_config["width"]
    height = render_config["height"]
    length = int(
        goal.constraints.get("native_h3_fl2va_length") or render_config["length"]
    )
    steps = int(goal.constraints.get("native_h3_fl2va_steps") or render_config["steps"])
    video_count = render_config["video_count"]
    model_profile = str(goal.constraints.get("native_h3_model_profile") or "q4")
    pre_video_review = context.pre_video_review_enabled(goal)
    pre_video_requires_human = context.pre_video_review_requires_human(goal)
    stage_probe_auto_select = bool(
        goal.constraints.get("stage_probe_auto_select", False)
    )
    keyframe_candidate_count = (
        context.pre_video_candidate_count(goal)
        if pre_video_review and (pre_video_requires_human or stage_probe_auto_select)
        else context.constraint_int(goal, "native_h3_keyframe_candidate_count", 1)
    )
    require_human_review = bool(
        goal.constraints.get("require_human_review", False)
    ) or (pre_video_review and pre_video_requires_human)
    use_last_frame = bool(goal.constraints.get("native_h3_use_last_frame", False))
    frame_pair_strategy = str(
        goal.constraints.get("native_h3_frame_pair_strategy")
        or "opening_conditioned_image_to_image"
    ).strip()
    allowed_frame_pair_strategies = {
        "independent_text_to_image",
        "opening_conditioned_image_to_image",
    }
    if use_last_frame and frame_pair_strategy not in allowed_frame_pair_strategies:
        raise ValueError(
            "native_h3_frame_pair_strategy must be independent_text_to_image or opening_conditioned_image_to_image"
        )
    experiment_seed = goal.constraints.get("native_h3_experiment_seed")
    if experiment_seed is not None:
        experiment_seed = int(experiment_seed)
        if experiment_seed < 0 or experiment_seed > (2**64) - 3:
            raise ValueError(
                "native_h3_experiment_seed must allow three non-negative 64-bit seeds"
            )
    if (
        keyframe_candidate_count > 1
        and not require_human_review
        and not stage_probe_auto_select
    ):
        raise ValueError(
            "Native H3 multiple keyframe candidates require require_human_review=true; refusing to select one automatically."
        )
    if (
        use_last_frame
        and not require_human_review
        and not stage_probe_auto_select
        and keyframe_candidate_count > 1
    ):
        raise ValueError(
            "Native H3 use_last_frame=true with multiple ending candidates requires require_human_review=true; refusing to select one automatically."
        )
    storyboard_path = context.native_h3_storyboard_path(goal, "native_h3_story")
    reference_node = context.reference_video_analysis_node(goal)
    story_prompt_dependencies = [reference_node.node_id] if reference_node else []

    opening_inputs = {
        "workflow_name": image_manifest.name,
        "prompt_key": "opening_keyframe_prompt",
        "width": width,
        "height": height,
        "image_count": keyframe_candidate_count,
        "suffix": "native_h3_opening",
    }
    if experiment_seed is not None:
        opening_inputs["seed"] = experiment_seed
    nodes = [
        *([reference_node] if reference_node else []),
        ExecutionNode(
            node_id="native-story-prompt",
            skill_name="longvideo.prepare_native_h3_story",
            inputs={
                "storyboard_path": storyboard_path,
                "duration_seconds": int(goal.duration_seconds),
                "style": goal.style,
                "render_mode": (
                    "first_last_frame_to_video"
                    if goal.media_type == "native_h3_fl2va_story"
                    else "image_to_video"
                ),
            },
            depends_on=story_prompt_dependencies,
            tags=["creative", "story", "native-h3"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="native-image-asset-check",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": image_manifest.name,
            },
            depends_on=["native-story-prompt"],
            tags=["assets", "image", "native-h3"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        ),
        ExecutionNode(
            node_id="native-video-asset-check",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": video_manifest.name,
            },
            depends_on=["native-story-prompt"],
            tags=["assets", "video", "native-h3"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        ),
        ExecutionNode(
            node_id="native-opening-keyframe",
            skill_name="media.image.generate_keyframe",
            inputs=opening_inputs,
            depends_on=["native-story-prompt", "native-image-asset-check"],
            tags=["render", "image", "continuity", "native-h3"],
            tool_name="comfy.workflow.text_to_image",
            stage="render",
        ),
    ]
    opening_source_node = "native-opening-keyframe"
    if require_human_review or stage_probe_auto_select:
        opening_review_node = "native-opening-review"
        nodes.append(
            ExecutionNode(
                node_id=opening_review_node,
                skill_name="review.assets.select",
                inputs={
                    "limit": 1,
                    "review_all_candidates": True,
                    "review_scope": "first_frame",
                    "review_phase": "opening_frame",
                    "require_human_review": pre_video_requires_human,
                    "auto_select_for_probe": stage_probe_auto_select,
                    "review_notes": "stage: preview",
                },
                depends_on=["native-opening-keyframe"],
                tags=["review", "first-frame", "native-h3"],
                stage="review",
            )
        )
        opening_source_node = opening_review_node
    ending_source_node = ""
    if use_last_frame:
        refine_workflow_name = str(
            goal.constraints.get("identity_refine_workflow_name")
            or "krea2_turbo_img2img"
        ).strip()
        use_opening_reference = (
            frame_pair_strategy == "opening_conditioned_image_to_image"
        )
        ending_dependencies = [
            opening_source_node,
            "native-story-prompt",
            "native-image-asset-check",
        ]
        if use_opening_reference:
            nodes.append(
                ExecutionNode(
                    node_id="native-ending-image-refine-asset-check",
                    skill_name="media.ensure_workflow",
                    inputs={
                        "workflow_name": refine_workflow_name,
                    },
                    depends_on=["native-story-prompt"],
                    tags=["assets", "image", "continuity", "native-h3"],
                    tool_name="asset.ensure_workflow_ready",
                    stage="assets",
                )
            )
            ending_dependencies.append("native-ending-image-refine-asset-check")
        ending_inputs = {
            "workflow_name": image_manifest.name,
            "prompt_key": "ending_keyframe_prompt",
            "use_prior_frame": use_opening_reference,
            "identity_refine_workflow_name": refine_workflow_name,
            "width": width,
            "height": height,
            "image_count": keyframe_candidate_count,
            "suffix": "native_h3_ending",
        }
        if use_opening_reference:
            ending_inputs["denoise"] = 0.5
        if experiment_seed is not None:
            ending_inputs["seed"] = experiment_seed + 1
        nodes.append(
            ExecutionNode(
                node_id="native-ending-keyframe",
                skill_name="media.image.generate_keyframe",
                inputs=ending_inputs,
                depends_on=ending_dependencies,
                tags=["render", "image", "last-frame", "native-h3"],
                tool_name="comfy.workflow.text_to_image",
                stage="render",
            )
        )
        ending_source_node = "native-ending-keyframe"
        if require_human_review:
            nodes.append(
                ExecutionNode(
                    node_id="native-ending-review",
                    skill_name="review.assets.select",
                    inputs={
                        "limit": 1,
                        "review_all_candidates": True,
                        "review_scope": "last_frame",
                        "review_phase": "ending_frame",
                        "review_notes": "stage: preview",
                    },
                    depends_on=["native-ending-keyframe"],
                    tags=["review", "last-frame", "native-h3"],
                    stage="review",
                )
            )
            ending_source_node = "native-ending-review"

    source_depends_on = [opening_source_node]
    if ending_source_node:
        source_depends_on.append(ending_source_node)
    source_inputs = {
        "character": str(goal.constraints.get("character") or ""),
        "opening_node": opening_source_node,
        "ending_node": ending_source_node,
        "opening_prompt_key": "opening_keyframe_prompt",
        "ending_prompt_key": "ending_keyframe_prompt",
        "preserve_opening_frame": require_human_review,
        "preserve_ending_frame": bool(use_last_frame and require_human_review),
        "use_last_frame": use_last_frame,
        "workflow_name": image_manifest.name,
        "width": width,
        "height": height,
        "max_regenerations": 0,
    }
    render_inputs = {
        "workflow_name": video_manifest.name,
        "width": width,
        "height": height,
        "length": length,
        "steps": steps,
        "video_count": video_count,
        "model_profile": model_profile,
        "use_last_frame": use_last_frame,
        "h3_mode": "fl2va" if goal.media_type == "native_h3_fl2va_story" else "i2va",
    }
    if experiment_seed is not None:
        render_inputs["seed"] = experiment_seed + 2
    nodes.extend(
        [
            ExecutionNode(
                node_id="native-keyframe-source-check",
                skill_name="media.image.confirm_keyframe_sources",
                inputs=source_inputs,
                depends_on=source_depends_on,
                tags=["assets", "frame-source", "native-h3"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="native-h3-render",
                skill_name="longvideo.render_native_h3",
                inputs=render_inputs,
                depends_on=[
                    "native-story-prompt",
                    "native-video-asset-check",
                    "native-keyframe-source-check",
                ],
                tags=["render", "video", "native-h3"],
                tool_name="comfy.render_h3",
                stage="render",
            ),
            *context.native_h3_finalize_nodes(
                goal,
                tags=["native-h3"],
                include_keyframe_source_check=True,
                qa_inputs=(
                    {
                        "target_duration": length
                        / float(goal.constraints.get("native_h3_frame_rate") or 24),
                        "expected_width": width,
                        "expected_height": height,
                        "expected_fps": float(
                            goal.constraints.get("native_h3_frame_rate") or 24
                        ),
                    }
                ),
            ),
        ]
    )
    metadata = {
        "recipe": "native_h3_fl2va_story"
        if goal.media_type == "native_h3_fl2va_story"
        else "native_h3_story",
        "storyboard_path": storyboard_path,
        "selected_workflow": video_manifest.name,
        "keyframe_workflow": image_manifest.name,
        "required_assets": [
            asset.to_dict()
            for asset in (
                *image_manifest.required_assets,
                *video_manifest.required_assets,
            )
        ],
        "native_h3": {
            "width": width,
            "height": height,
            "length": length,
            "steps": steps,
            "target_duration": int(goal.duration_seconds),
            "keyframe_candidate_count": keyframe_candidate_count,
            "require_human_review": require_human_review,
            "stage_probe_auto_select": stage_probe_auto_select,
            "use_last_frame": use_last_frame,
            "frame_pair_strategy": frame_pair_strategy if use_last_frame else "",
            "experiment_seed": experiment_seed,
            "backend": "comfyui",
        },
        **(
            {"reference_video": context.reference_video_metadata(goal)}
            if reference_node
            else {}
        ),
        "graph_overview": [node.node_id for node in nodes],
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name=video_manifest.name,
        nodes=nodes,
        metadata=metadata,
        description=(
            f"Native H3 first/last-frame causal story from storyboard '{storyboard_path}'"
            if goal.media_type == "native_h3_fl2va_story"
            else f"Native H3 causal story from storyboard '{storyboard_path}'"
        ),
    )


def build_native_h3_t2v_story(
    context: PlanningContext, goal: GoalRequest
) -> ExecutionPlan:
    """Build one continuous native H3 text-to-video story without keyframes."""
    video_manifest = context.manifest_from_goal_constraints(
        goal,
        "minimax_h3_t2va",
        constraint_keys=(
            "video_workflow_name",
            "native_h3_workflow_name",
            "workflow_name",
        ),
        allowed_media_types={"text2video", "long_video"},
    )
    render_config = context.native_h3_render_config(goal, video_manifest)
    width = render_config["width"]
    height = render_config["height"]
    length = int(
        goal.constraints.get("native_h3_t2v_length")
        or goal.constraints.get("native_h3_length")
        or render_config["length"]
    )
    steps = int(
        goal.constraints.get("native_h3_t2v_steps")
        or goal.constraints.get("native_h3_steps")
        or render_config["steps"]
    )
    video_count = render_config["video_count"]
    frame_rate = float(goal.constraints.get("native_h3_frame_rate") or 24)
    render_duration = round(length / frame_rate, 3)
    storyboard_path = context.native_h3_storyboard_path(goal, "native_h3_t2v_story")
    reference_node = context.reference_video_analysis_node(goal)
    story_prompt_dependencies = [reference_node.node_id] if reference_node else []

    nodes = [
        *([reference_node] if reference_node else []),
        ExecutionNode(
            node_id="native-story-prompt",
            skill_name="longvideo.prepare_native_h3_story",
            inputs={
                "storyboard_path": storyboard_path,
                "duration_seconds": int(goal.duration_seconds),
                "style": goal.style,
                "render_mode": "text_to_video",
            },
            depends_on=story_prompt_dependencies,
            tags=["creative", "story", "native-h3", "t2v"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="native-video-asset-check",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": video_manifest.name,
            },
            depends_on=["native-story-prompt"],
            tags=["assets", "video", "native-h3", "t2v"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        ),
        ExecutionNode(
            node_id="native-h3-render",
            skill_name="longvideo.render_native_h3_t2v",
            inputs={
                "workflow_name": video_manifest.name,
                "width": width,
                "height": height,
                "length": length,
                "steps": steps,
                "video_count": video_count,
                "model_profile": str(
                    goal.constraints.get("native_h3_t2v_model_profile")
                    or goal.constraints.get("native_h3_model_profile")
                    or "q4"
                ),
            },
            depends_on=["native-story-prompt", "native-video-asset-check"],
            tags=["render", "video", "native-h3", "t2v"],
            tool_name="comfy.render_h3",
            stage="render",
        ),
        *context.native_h3_finalize_nodes(
            goal,
            tags=["native-h3", "t2v"],
            include_keyframe_source_check=False,
            qa_inputs={
                "target_duration": render_duration,
                "expected_width": width,
                "expected_height": height,
                "expected_fps": frame_rate,
            },
        ),
    ]
    metadata = {
        "recipe": "native_h3_t2v_story",
        "storyboard_path": storyboard_path,
        "selected_workflow": video_manifest.name,
        "required_assets": [
            asset.to_dict() for asset in video_manifest.required_assets
        ],
        "native_h3": {
            "width": width,
            "height": height,
            "length": length,
            "steps": steps,
            "target_duration": render_duration,
            "requested_duration": int(goal.duration_seconds),
            "backend": "comfyui",
        },
        "render_mode": "text_to_video",
        **(
            {"reference_video": context.reference_video_metadata(goal)}
            if reference_node
            else {}
        ),
        "graph_overview": [node.node_id for node in nodes],
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name=video_manifest.name,
        nodes=nodes,
        metadata=metadata,
        description=f"Native H3 text-to-video causal story from storyboard '{storyboard_path}'",
    )


def build_native_h3_l2va_story(
    context: PlanningContext, goal: GoalRequest
) -> ExecutionPlan:
    """Build a last-frame-only H3 causal clip.

    L2VA is deliberately a separate route so an approved ending frame can
    be treated as the immutable conditioning source rather than being
    accidentally paired with the opening-frame connection in the FL2VA
    template.
    """
    image_manifest = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=(
            "native_h3_keyframe_workflow_name",
            "keyframe_workflow_name",
            "image_workflow_name",
        ),
        allowed_media_types={"image"},
    )
    video_manifest = context.manifest_from_goal_constraints(
        goal,
        "minimax_h3_l2va",
        constraint_keys=(
            "video_workflow_name",
            "native_h3_workflow_name",
            "workflow_name",
        ),
        allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
    )
    render_config = context.native_h3_render_config(goal, video_manifest)
    # L2VA uses the configured 15-second H3 workflow in production. Do
    # not silently downgrade it to the 124-frame low-VRAM preview; the
    # post-render speed node is responsible for the requested 2x timing
    # transform after the full source clip has been rendered.
    width = render_config["width"]
    height = render_config["height"]
    length = int(
        goal.constraints.get("native_h3_l2va_length") or render_config["length"]
    )
    steps = int(goal.constraints.get("native_h3_l2va_steps") or render_config["steps"])
    video_count = render_config["video_count"]
    model_profile = str(goal.constraints.get("native_h3_model_profile") or "q4")
    storyboard_path = context.native_h3_storyboard_path(goal, "native_h3_l2va_story")
    reference_node = context.reference_video_analysis_node(goal)
    story_prompt_dependencies = [reference_node.node_id] if reference_node else []
    pre_video_review = context.pre_video_review_enabled(goal)
    pre_video_requires_human = context.pre_video_review_requires_human(goal)
    stage_probe_auto_select = bool(
        goal.constraints.get("stage_probe_auto_select", False)
    )
    require_human_review = bool(goal.constraints.get("require_human_review", True)) or (
        pre_video_review and pre_video_requires_human
    )
    if stage_probe_auto_select:
        require_human_review = False
    candidate_count = (
        context.pre_video_candidate_count(goal)
        if pre_video_review and (pre_video_requires_human or stage_probe_auto_select)
        else 1
    )
    ending_source_node = "native-l2va-ending-keyframe"
    nodes = [
        *([reference_node] if reference_node else []),
        ExecutionNode(
            node_id="native-story-prompt",
            skill_name="longvideo.prepare_native_h3_story",
            inputs={
                "storyboard_path": storyboard_path,
                "duration_seconds": int(goal.duration_seconds),
                "style": goal.style,
                "render_mode": "last_frame_to_video",
            },
            depends_on=story_prompt_dependencies,
            tags=["creative", "story", "native-h3", "l2va"],
            stage="prompting",
        ),
        ExecutionNode(
            node_id="native-image-asset-check",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": image_manifest.name,
            },
            depends_on=["native-story-prompt"],
            tags=["assets", "image", "native-h3", "l2va"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        ),
        ExecutionNode(
            node_id="native-video-asset-check",
            skill_name="media.ensure_workflow",
            inputs={
                "workflow_name": video_manifest.name,
            },
            depends_on=["native-story-prompt"],
            tags=["assets", "video", "native-h3", "l2va"],
            tool_name="asset.ensure_workflow_ready",
            stage="assets",
        ),
        ExecutionNode(
            node_id=ending_source_node,
            skill_name="media.image.generate_keyframe",
            inputs={
                "workflow_name": image_manifest.name,
                "prompt_key": "ending_keyframe_prompt",
                "width": width,
                "height": height,
                "image_count": candidate_count,
                "suffix": "native_h3_l2va_last",
            },
            depends_on=["native-story-prompt", "native-image-asset-check"],
            tags=["render", "image", "last-frame", "native-h3", "l2va"],
            tool_name="comfy.workflow.text_to_image",
            stage="render",
        ),
    ]
    frame_node = ending_source_node
    if require_human_review or stage_probe_auto_select:
        nodes.append(
            ExecutionNode(
                node_id="native-l2va-ending-review",
                skill_name="review.assets.select",
                inputs={
                    "limit": 1,
                    "review_all_candidates": True,
                    "review_scope": "last_frame",
                    "require_human_review": pre_video_requires_human,
                    "auto_select_for_probe": stage_probe_auto_select,
                    "review_notes": "Select the final conditioning frame. This frame is immutable after approval; reject identity drift, wrong pose, or unusable composition.",
                },
                depends_on=[ending_source_node],
                tags=["review", "last-frame", "native-h3", "l2va"],
                stage="review",
            )
        )
        frame_node = "native-l2va-ending-review"
    nodes.extend(
        [
            ExecutionNode(
                node_id="native-l2va-frame-source-check",
                skill_name="media.image.confirm_last_frame_source",
                inputs={
                    "frame_node": frame_node,
                    "character": str(goal.constraints.get("character") or ""),
                    "preserve_last_frame": bool(
                        require_human_review or stage_probe_auto_select
                    ),
                    "max_regenerations": 0,
                },
                depends_on=[frame_node],
                tags=["assets", "frame-source", "last-frame", "native-h3", "l2va"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="native-h3-render",
                skill_name="longvideo.render_native_h3_l2va",
                inputs={
                    "workflow_name": video_manifest.name,
                    "width": width,
                    "height": height,
                    "length": length,
                    "steps": steps,
                    "video_count": video_count,
                    "model_profile": model_profile,
                },
                depends_on=[
                    "native-story-prompt",
                    "native-video-asset-check",
                    "native-l2va-frame-source-check",
                ],
                tags=["render", "video", "native-h3", "l2va"],
                tool_name="comfy.render_h3",
                stage="render",
            ),
            *context.native_h3_finalize_nodes(
                goal,
                tags=["native-h3", "l2va"],
                include_keyframe_source_check=True,
                frame_source_check_node_id="native-l2va-frame-source-check",
                qa_inputs=(
                    {
                        "target_duration": length
                        / float(goal.constraints.get("native_h3_frame_rate") or 24),
                        "expected_width": width,
                        "expected_height": height,
                        "expected_fps": float(
                            goal.constraints.get("native_h3_frame_rate") or 24
                        ),
                    }
                ),
            ),
        ]
    )
    metadata = {
        "recipe": "native_h3_l2va_story",
        "storyboard_path": storyboard_path,
        "selected_workflow": video_manifest.name,
        "keyframe_workflow": image_manifest.name,
        "required_assets": [
            asset.to_dict()
            for asset in (
                *image_manifest.required_assets,
                *video_manifest.required_assets,
            )
        ],
        "native_h3": {
            "width": width,
            "height": height,
            "length": length,
            "steps": steps,
            "target_duration": int(goal.duration_seconds),
            "require_human_review": require_human_review,
            "stage_probe_auto_select": stage_probe_auto_select,
            "backend": "comfyui",
        },
        "render_mode": "last_frame_to_video",
        **(
            {"reference_video": context.reference_video_metadata(goal)}
            if reference_node
            else {}
        ),
        "graph_overview": [node.node_id for node in nodes],
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name=video_manifest.name,
        nodes=nodes,
        metadata=metadata,
        description=f"Native H3 L2VA story from storyboard '{storyboard_path}'",
    )


def build_native_h3_ref2va(
    context: PlanningContext,
    goal: GoalRequest,
    *,
    auto_reference_generation: bool | None = None,
) -> ExecutionPlan:
    """Build the multi-reference H3 route without audio refs.

    ``native_h3_ref2va`` uses a validated manifest directly when one is
    configured.  With an empty manifest it uses the same six-candidate
    T2I plus Discord selection gate as the composed
    ``text2image2native_h3_ref2va`` route.  The composed route keeps the
    candidate stage explicit in its strategy name.
    """
    if auto_reference_generation is None:
        auto_reference_generation = (
            goal.media_type == "text2image2native_h3_ref2va"
            or bool(goal.constraints.get("auto_reference_generation", False))
        )
    image_manifest = None
    if auto_reference_generation:
        image_manifest = context.manifest_from_goal_constraints(
            goal,
            *context.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=(
                "native_h3_reference_workflow_name",
                "keyframe_workflow_name",
                "image_workflow_name",
            ),
            allowed_media_types={"image"},
        )
    video_manifest = context.manifest_from_goal_constraints(
        goal,
        "minimax_h3_ref2va",
        constraint_keys=(
            "video_workflow_name",
            "native_h3_ref2va_workflow_name",
            "workflow_name",
        ),
        allowed_media_types={"native_h3_ref2va", "long_video"},
    )
    render_config = context.native_h3_render_config(
        goal, video_manifest, default_steps=20
    )
    width, height = render_config["width"], render_config["height"]
    length, steps, video_count = (
        render_config["length"],
        render_config["steps"],
        render_config["video_count"],
    )
    storyboard_path = context.native_h3_storyboard_path(goal, "native_h3_ref2va")
    reference_manifest = list(
        goal.constraints.get("native_h3_reference_manifest")
        or goal.constraints.get("reference_manifest")
        or []
    )
    reference_image_paths = list(
        goal.constraints.get("native_h3_reference_image_paths") or []
    )
    reference_video_paths = list(
        goal.constraints.get("native_h3_reference_video_paths") or []
    )
    reference_candidates = list(
        goal.constraints.get("native_h3_reference_candidates")
        or goal.constraints.get("media_paths")
        or []
    )
    raw_selection_limit = int(
        goal.constraints.get("native_h3_reference_selection_limit") or 4
    )
    reference_selection_limit = max(
        1,
        min(4 if auto_reference_generation else 12, raw_selection_limit),
    )
    raw_candidate_count = int(
        (
            goal.constraints.get("image_count")
            if auto_reference_generation
            else goal.constraints.get("native_h3_reference_candidate_count")
        )
        or goal.constraints.get("native_h3_reference_candidate_count")
        or 4
    )
    reference_candidate_count = max(
        4 if auto_reference_generation else 1,
        min(6 if auto_reference_generation else 6, raw_candidate_count),
    )
    stage_probe_auto_select = bool(
        goal.constraints.get("stage_probe_auto_select", False)
    )
    if "require_human_review" in goal.constraints:
        require_human_review = bool(goal.constraints.get("require_human_review"))
    else:
        # Generated references must never silently become Ref2VA inputs.
        # Only an explicit --no-review path may disable this gate.
        require_human_review = auto_reference_generation or bool(reference_candidates)
    if stage_probe_auto_select:
        require_human_review = False
    nodes = [
        ExecutionNode(
            node_id="native-story-prompt",
            skill_name="longvideo.prepare_native_h3_story",
            inputs={
                "storyboard_path": storyboard_path,
                "duration_seconds": int(goal.duration_seconds),
                "style": goal.style,
                "render_mode": "reference_to_video",
            },
            tags=["creative", "story", "native-h3", "ref2va"],
            stage="prompting",
        ),
    ]
    if auto_reference_generation:
        nodes.extend(
            [
                ExecutionNode(
                    node_id="native-image-asset-check",
                    skill_name="media.ensure_workflow",
                    inputs={
                        "workflow_name": image_manifest.name,
                    },
                    depends_on=["native-story-prompt"],
                    tags=["assets", "image", "native-h3", "ref2va"],
                    tool_name="asset.ensure_workflow_ready",
                    stage="assets",
                ),
                ExecutionNode(
                    node_id="native-video-asset-check",
                    skill_name="media.ensure_workflow",
                    inputs={
                        "workflow_name": video_manifest.name,
                    },
                    depends_on=["native-story-prompt"],
                    tags=["assets", "video", "native-h3", "ref2va"],
                    tool_name="asset.ensure_workflow_ready",
                    stage="assets",
                ),
                ExecutionNode(
                    node_id="native-ref2va-reference-candidates",
                    skill_name="media.image.generate_keyframe",
                    inputs={
                        "workflow_name": image_manifest.name,
                        "prompt_key": "opening_keyframe_prompt",
                        "width": width,
                        "height": height,
                        "image_count": reference_candidate_count,
                        "suffix": "native_h3_ref2va_reference",
                        "max_regenerations": 0,
                    },
                    depends_on=["native-story-prompt", "native-image-asset-check"],
                    tags=["render", "image", "references", "native-h3", "ref2va"],
                    tool_name="comfy.workflow.text_to_image",
                    stage="render",
                ),
            ]
        )
    else:
        nodes.append(
            ExecutionNode(
                node_id="native-video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": video_manifest.name,
                },
                depends_on=["native-story-prompt"],
                tags=["assets", "video", "native-h3", "ref2va"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            )
        )
    reference_check_dependencies = ["native-story-prompt"]
    if auto_reference_generation:
        candidate_node_id = "native-ref2va-reference-candidates"
        if require_human_review or stage_probe_auto_select:
            nodes.append(
                ExecutionNode(
                    node_id="native-ref2va-reference-review",
                    skill_name="review.assets.select",
                    inputs={
                        "limit": reference_selection_limit,
                        "review_all_candidates": True,
                        "review_scope": "reference",
                        "review_phase": "reference_selection",
                        "require_human_review": require_human_review,
                        "auto_select_for_probe": stage_probe_auto_select,
                        "review_notes": str(goal.constraints.get("review_notes") or ""),
                    },
                    depends_on=[candidate_node_id],
                    tags=["review", "references", "native-h3", "ref2va"],
                    stage="review",
                )
            )
            reference_check_dependencies.append("native-ref2va-reference-review")
        else:
            reference_check_dependencies.append(candidate_node_id)
    elif reference_candidates:
        candidate_node_id = "native-ref2va-reference-candidates"
        nodes.append(
            ExecutionNode(
                node_id=candidate_node_id,
                skill_name="publish.media.ingest",
                inputs={"media_paths": reference_candidates},
                depends_on=["native-story-prompt"],
                tags=["assets", "references", "native-h3", "ref2va"],
                stage="assets",
            )
        )
        if require_human_review or stage_probe_auto_select:
            nodes.append(
                ExecutionNode(
                    node_id="native-ref2va-reference-review",
                    skill_name="review.assets.select",
                    inputs={
                        "limit": reference_selection_limit,
                        "review_all_candidates": True,
                        "review_scope": "reference",
                        "review_phase": "reference_selection",
                        "require_human_review": require_human_review,
                        "auto_select_for_probe": stage_probe_auto_select,
                        "review_notes": str(goal.constraints.get("review_notes") or ""),
                    },
                    depends_on=[candidate_node_id],
                    tags=["review", "references", "native-h3", "ref2va"],
                    stage="review",
                )
            )
            reference_check_dependencies.append("native-ref2va-reference-review")
        else:
            reference_check_dependencies.append(candidate_node_id)
    nodes.extend(
        [
            ExecutionNode(
                node_id="native-ref2va-reference-check",
                skill_name="longvideo.validate_native_h3_references",
                inputs={
                    "reference_manifest": reference_manifest,
                    "reference_image_paths": reference_image_paths,
                    "reference_video_paths": reference_video_paths,
                    "max_images": int(
                        goal.constraints.get("native_h3_reference_max_images") or 9
                    ),
                    "max_videos": int(
                        goal.constraints.get("native_h3_reference_max_videos") or 3
                    ),
                    "selection_limit": reference_selection_limit,
                    "auto_reference_generation": auto_reference_generation,
                },
                depends_on=reference_check_dependencies,
                tags=["quality", "references", "native-h3", "ref2va"],
                stage="quality",
            ),
            ExecutionNode(
                node_id="native-h3-render",
                skill_name="longvideo.render_native_h3_ref2va",
                inputs={
                    "workflow_name": video_manifest.name,
                    "width": width,
                    "height": height,
                    "length": length,
                    "steps": steps,
                    "video_count": video_count,
                    "ref_image_size": str(
                        goal.constraints.get("native_h3_reference_image_size")
                        or "match"
                    ),
                    "model_profile": str(
                        goal.constraints.get("native_h3_model_profile") or "q4"
                    ),
                },
                depends_on=[
                    "native-story-prompt",
                    "native-video-asset-check",
                    "native-ref2va-reference-check",
                ],
                tags=["render", "video", "native-h3", "ref2va"],
                tool_name="comfy.render_h3",
                stage="render",
            ),
        ]
    )
    nodes.extend(
        context.native_h3_finalize_nodes(
            goal,
            tags=["native-h3", "ref2va"],
            include_keyframe_source_check=False,
            qa_inputs={
                "target_duration": length / 24,
                "expected_width": width,
                "expected_height": height,
                "expected_fps": 24,
                "frame_count": 12,
                "columns": 4,
            },
        )
    )
    required_assets = [asset.to_dict() for asset in video_manifest.required_assets]
    if image_manifest is not None:
        required_assets = [
            asset.to_dict() for asset in image_manifest.required_assets
        ] + required_assets
    metadata = {
        "recipe": "text2image2native_h3_ref2va"
        if auto_reference_generation
        else "native_h3_ref2va",
        "storyboard_path": storyboard_path,
        "selected_workflow": video_manifest.name,
        "selected_workflows": {
            "image": image_manifest.name if image_manifest is not None else "",
            "video": video_manifest.name,
        },
        "required_assets": required_assets,
        "image_required_assets": (
            [asset.to_dict() for asset in image_manifest.required_assets]
            if image_manifest is not None
            else []
        ),
        "video_required_assets": [
            asset.to_dict() for asset in video_manifest.required_assets
        ],
        "reference_manifest": reference_manifest,
        "reference_candidates": reference_candidates,
        "reference_candidate_count": reference_candidate_count
        if auto_reference_generation
        else 0,
        "reference_selection_limit": reference_selection_limit,
        "reference_audio_enabled": False,
        "native_h3": {
            "width": width,
            "height": height,
            "length": length,
            "steps": steps,
            "target_duration": int(goal.duration_seconds),
            "reference_image_size": str(
                goal.constraints.get("native_h3_reference_image_size") or "match"
            ),
            "stage_probe_auto_select": stage_probe_auto_select,
            "backend": "comfyui",
        },
        "render_mode": "reference_to_video",
        "graph_overview": [node.node_id for node in nodes],
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name=video_manifest.name,
        nodes=nodes,
        metadata=metadata,
        description=(
            f"Native H3 T2I-to-Ref2VA story from storyboard '{storyboard_path}' with reviewed generated references"
            if auto_reference_generation
            else f"Native H3 Ref2VA story from storyboard '{storyboard_path}' with image/video references only"
        ),
    )
