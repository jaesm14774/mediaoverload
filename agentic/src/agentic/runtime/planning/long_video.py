"""Long Video strategy graph assembly."""

from __future__ import annotations

import math

from agentic.runtime.contracts import (
    MIN_VIDEO_DURATION_SECONDS,
    ExecutionNode,
    ExecutionPlan,
    GoalRequest,
)
from agentic.runtime.editing import EDIT_PROFILES
from agentic.runtime.planning.common import PlanningContext


def build_long_video(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    minimum_h3_segment_count = max(
        1, math.ceil(float(goal.duration_seconds) / (362 / 24))
    )
    max_segment_count = max(
        minimum_h3_segment_count,
        int(goal.duration_seconds) // MIN_VIDEO_DURATION_SECONDS,
    )
    requested_segment_count = context.constraint_int(
        goal,
        "segment_count",
        max_segment_count,
    )
    segment_count = max(
        minimum_h3_segment_count,
        min(requested_segment_count, max_segment_count),
    )
    use_tts = bool(goal.constraints.get("use_tts", False))
    review_loop_enabled = context.review_loop_enabled(goal)
    pre_video_review = context.pre_video_review_enabled(goal)
    stage_probe_auto_select = bool(
        goal.constraints.get("stage_probe_auto_select", False)
    )
    review_notes = str(goal.constraints.get("review_notes", "") or "")
    selection_limit = context.review_selection_limit(
        goal, default=context.constraint_int(goal, "review_selection_limit", 3)
    )
    production_profile = (
        str(goal.constraints.get("longvideo_production_profile") or "text2longvideo")
        .strip()
        .lower()
    )
    if production_profile != "text2longvideo":
        raise ValueError("longvideo_production_profile must be 'text2longvideo'")
    review_policy = (
        str(
            goal.constraints.get("longvideo_review_policy")
            or goal.constraints.get("longvideo_frame_review_policy")
            or "opening_only"
        )
        .strip()
        .lower()
    )
    if review_policy not in {"opening_only", "anchors", "every_segment"}:
        raise ValueError(
            "longvideo_review_policy must be 'opening_only', 'anchors', or 'every_segment'"
        )
    continuity_mode = (
        str(goal.constraints.get("longvideo_continuity_mode") or "rendered_tail")
        .strip()
        .lower()
    )
    if continuity_mode != "rendered_tail":
        raise ValueError("longvideo_continuity_mode must be 'rendered_tail'")

    from agentic.runtime.video_conditioning import (
        ConditioningPlan,
        capabilities_from_manifests,
        production_recipe_sequence,
        recipe_candidates,
    )

    capabilities = capabilities_from_manifests(context.asset_registry.all_manifests())
    preferred_workflows = goal.constraints.get("longvideo_workflow_names") or []
    if isinstance(preferred_workflows, str):
        preferred_workflows = [preferred_workflows]
    candidates = recipe_candidates(
        capabilities, preferred_workflows=preferred_workflows
    )
    variant_seed = int(goal.constraints.get("edit_variant_seed") or 17)
    recipe_sequence = production_recipe_sequence(
        segment_count,
        candidates,
        use_reference_bundle=bool(
            goal.constraints.get("reference_manifest")
            or goal.constraints.get("native_h3_reference_manifest")
            or goal.constraints.get("reference_image_paths")
            or goal.constraints.get("native_h3_reference_image_paths")
            or goal.constraints.get("reference_video_paths")
            or goal.constraints.get("native_h3_reference_video_paths")
        ),
    )
    selected_capabilities = {
        recipe_name: candidates[recipe_name][0]
        for recipe_name in dict.fromkeys(recipe_sequence)
    }
    selected_workflows = {
        capability.workflow_name for capability in selected_capabilities.values()
    }
    recipe_contracts = {
        recipe_name: selected_capabilities[recipe_name].recipes[recipe_name]
        for recipe_name in selected_capabilities
    }
    requires_image_conditioning = any(
        recipe.requires_first or recipe.requires_last or recipe.requires_references
        for recipe in recipe_contracts.values()
    )
    requires_landing_refinement = any(
        recipe.requires_last for recipe in recipe_contracts.values()
    )
    image_manifest = (
        context.manifest_from_goal_constraints(
            goal,
            *context.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        if requires_image_conditioning
        else None
    )
    refine_manifest = (
        context.manifest_from_goal_constraints(
            goal,
            *context.DEFAULT_REFINE_WORKFLOWS,
            constraint_keys=("refine_workflow_name",),
            allowed_media_types={"image_refine"},
        )
        if requires_landing_refinement
        else None
    )

    image_defaults = image_manifest.recommended_defaults if image_manifest else {}
    frame_width = int(
        goal.constraints.get("canvas_width")
        or goal.constraints.get("longvideo_frame_width")
        or image_defaults.get("width", 1024)
    )
    frame_height = int(
        goal.constraints.get("canvas_height")
        or goal.constraints.get("longvideo_frame_height")
        or image_defaults.get("height", 1024)
    )
    frame_candidate_count = max(
        1,
        int(
            goal.constraints.get("longvideo_frame_candidate_count")
            or goal.constraints.get("pre_video_candidate_count")
            or (4 if review_policy != "opening_only" else 1)
        ),
    )
    reference_candidate_count = max(
        1,
        int(
            goal.constraints.get("longvideo_reference_candidate_count")
            or goal.constraints.get("native_h3_reference_candidate_count")
            or 4
        ),
    )
    require_human_review = bool(
        goal.constraints.get("require_human_review", False)
        or goal.constraints.get("enable_stage_review", False)
        or pre_video_review
    )
    if stage_probe_auto_select:
        require_human_review = False
    review_all_anchors = review_policy in {"anchors", "every_segment"}
    human_anchor_review = require_human_review and review_all_anchors
    opening_human_review = require_human_review and (
        pre_video_review
        or bool(goal.constraints.get("enable_stage_review", False))
        or review_policy == "opening_only"
    )

    configured_reference_manifest = list(
        goal.constraints.get("reference_manifest")
        or goal.constraints.get("native_h3_reference_manifest")
        or []
    )
    configured_reference_images = list(
        goal.constraints.get("reference_image_paths")
        or goal.constraints.get("native_h3_reference_image_paths")
        or []
    )
    configured_reference_videos = list(
        goal.constraints.get("reference_video_paths")
        or goal.constraints.get("native_h3_reference_video_paths")
        or []
    )
    has_configured_references = bool(
        configured_reference_manifest
        or configured_reference_images
        or configured_reference_videos
    )
    max_reference_images = int(
        goal.constraints.get("longvideo_reference_max_images")
        or goal.constraints.get("native_h3_reference_max_images")
        or 9
    )
    max_reference_videos = int(
        goal.constraints.get("longvideo_reference_max_videos")
        or goal.constraints.get("native_h3_reference_max_videos")
        or 3
    )
    reference_selection_limit = max(
        1,
        int(
            goal.constraints.get("longvideo_reference_selection_limit")
            or goal.constraints.get("native_h3_reference_selection_limit")
            or 4
        ),
    )
    segment_frame_rate = float(
        goal.constraints.get("longvideo_frame_rate")
        or goal.constraints.get("video_frame_rate")
        or 24
    )
    if segment_frame_rate <= 0:
        raise ValueError("longvideo_frame_rate must be greater than zero")
    if any(name.startswith("minimax_h3_") for name in selected_workflows) and segment_frame_rate != 24:
        raise ValueError("longvideo_frame_rate must be 24 when using MiniMax H3 workflows")
    segment_duration_seconds = float(goal.duration_seconds) / max(1, segment_count)
    segment_default_length = max(
        1,
        round(segment_duration_seconds * segment_frame_rate),
    )

    idea_variants = context.idea_director.generate_variations(goal)
    reference_node = context.reference_video_analysis_node(goal)
    planning_dependencies = [reference_node.node_id] if reference_node else []
    nodes: list[ExecutionNode] = [
        *([reference_node] if reference_node else []),
        ExecutionNode(
            node_id="idea-brief",
            skill_name="agent.goal.expand",
            inputs={
                "prompt": goal.prompt,
                "style": goal.style,
                "idea_variants": idea_variants,
            },
            depends_on=planning_dependencies,
            tags=["creative"],
        ),
        ExecutionNode(
            node_id="script-plan",
            skill_name="agent.story.segment",
            inputs={
                "segment_count": segment_count,
                "duration_seconds": goal.duration_seconds,
                "tone": "playful cinematic escalation",
                "production_profile": production_profile,
            },
            depends_on=["idea-brief"],
            tags=["story"],
        ),
    ]
    if requires_image_conditioning:
        assert image_manifest is not None
        nodes.append(
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": image_manifest.name,
                },
                depends_on=["script-plan"],
                tags=["assets", "image"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            )
        )
        if refine_manifest:
            nodes.append(
                ExecutionNode(
                    node_id="refine-asset-check",
                    skill_name="media.ensure_workflow",
                    inputs={
                        "workflow_name": refine_manifest.name,
                    },
                    depends_on=["script-plan"],
                    tags=["assets", "refine"],
                    tool_name="asset.ensure_workflow_ready",
                    stage="assets",
                )
            )

    def video_asset_check_for(workflow_name: str) -> str:
        node_id = f"video-asset-check-{workflow_name}"
        if not any(node.node_id == node_id for node in nodes):
            context.asset_registry.get_manifest(workflow_name)
            nodes.append(
                ExecutionNode(
                    node_id=node_id,
                    skill_name="media.ensure_workflow",
                    inputs={
                        "workflow_name": workflow_name,
                    },
                    depends_on=["script-plan"],
                    tags=["assets", "video", "longvideo"],
                    tool_name="asset.ensure_workflow_ready",
                    stage="assets",
                )
            )
        return node_id

    for workflow_name in sorted(selected_workflows):
        video_asset_check_for(workflow_name)

    segment_video_nodes: list[str] = []
    tts_nodes: list[str] = []

    def append_segment(
        prefix: str,
        index: int,
        previous_rendered_tail_node: str | None,
        *,
        retry: bool = False,
    ) -> tuple[str, str, str | None, str | None, str | None]:
        suffix = f"{index + 1:02d}"
        prompt_node = f"{prefix}-prompt-{suffix}"
        video_node = f"{prefix}-video-{suffix}"
        recipe_name = recipe_sequence[index]
        capability = selected_capabilities[recipe_name]
        recipe = recipe_contracts[recipe_name]
        prompt_dependencies = ["script-plan", "idea-brief"]
        if retry:
            prompt_dependencies.append("review-refine-prompt")
        nodes.append(
            ExecutionNode(
                node_id=prompt_node,
                skill_name="agent.segment.prepare",
                inputs={"segment_index": index, "recipe": recipe_name},
                depends_on=prompt_dependencies,
                tags=["story", "segment", "retry"] if retry else ["story", "segment"],
                stage="prompting",
            )
        )

        input_dependencies = [
            prompt_node,
            video_asset_check_for(capability.workflow_name),
        ]
        anchor_nodes: dict[str, str] = {}
        anchor_review_nodes: list[str] = []

        if recipe.requires_first:
            if previous_rendered_tail_node:
                anchor_nodes["first"] = previous_rendered_tail_node
                input_dependencies.append(previous_rendered_tail_node)
            else:
                # Keep the original segment-frame id stable for callers and
                # persisted plans while the payload now carries a generic
                # anchor contract.
                first_candidate = f"{prefix}-frame-{suffix}"
                first_dependencies = [
                    prompt_node,
                    "image-asset-check",
                ]
                first_count = (
                    frame_candidate_count
                    if ((index == 0 and pre_video_review) or human_anchor_review)
                    else 1
                )
                nodes.append(
                    ExecutionNode(
                        node_id=first_candidate,
                        skill_name="media.image.generate_keyframe",
                        inputs={
                            "workflow_name": image_manifest.name,
                            "segment_index": index,
                            "anchor_position": "first",
                            "width": frame_width,
                            "height": frame_height,
                            "image_count": first_count,
                            "suffix": f"{prefix}_anchor_first_{suffix}",
                        },
                        depends_on=first_dependencies,
                        tags=["render", "image", "anchor", "first", "segment"],
                        tool_name="comfy.workflow.text_to_image",
                        stage="render",
                    )
                )
                first_source = first_candidate
                first_review_enabled = (
                    first_count > 1
                    or (opening_human_review and index == 0)
                    or (require_human_review and review_policy == "every_segment")
                    or (require_human_review and review_policy == "anchors")
                )
                if first_review_enabled:
                    first_review = (
                        f"stage-review-{suffix}"
                        if prefix == "segment" and index == 0
                        else f"{prefix}-anchor-first-select-{suffix}"
                    )
                    nodes.append(
                        ExecutionNode(
                            node_id=first_review,
                            skill_name="review.assets.select",
                            inputs={
                                "limit": 1,
                                "review_all_candidates": True,
                                "review_scope": "first_frame",
                                "review_phase": "opening_frame",
                                "require_human_review": opening_human_review
                                or human_anchor_review,
                                "auto_select_for_probe": stage_probe_auto_select,
                                "review_notes": review_notes,
                            },
                            depends_on=[first_candidate],
                            tags=["review", "anchor", "first", "segment"],
                            stage="review",
                        )
                    )
                    first_source = first_review
                    anchor_review_nodes.append(first_review)
                anchor_nodes["first"] = first_source
                input_dependencies.append(first_source)

        if recipe.requires_last:
            last_candidate = f"{prefix}-anchor-last-candidates-{suffix}"
            assert image_manifest is not None and refine_manifest is not None
            prior_anchor_node = anchor_nodes.get("first")
            if not prior_anchor_node:
                raise ValueError("A landing-frame recipe requires its first-frame anchor")
            last_dependencies = [
                prompt_node,
                "image-asset-check",
                "refine-asset-check",
                prior_anchor_node,
            ]
            nodes.append(
                ExecutionNode(
                    node_id=last_candidate,
                    skill_name="media.image.generate_keyframe",
                    inputs={
                        "workflow_name": image_manifest.name,
                        "identity_refine_workflow_name": refine_manifest.name,
                        "use_prior_frame": True,
                        "segment_index": index,
                        "anchor_position": "last",
                        "width": frame_width,
                        "height": frame_height,
                        "image_count": frame_candidate_count
                        if review_all_anchors
                        else 1,
                        "suffix": f"{prefix}_anchor_last_{suffix}",
                    },
                    depends_on=last_dependencies,
                    tags=["render", "image", "anchor", "last", "segment"],
                    tool_name="comfy.workflow.image_to_image",
                    stage="render",
                )
            )
            last_source = last_candidate
            last_count = frame_candidate_count if review_all_anchors else 1
            if (
                last_count > 1
                or (human_anchor_review and recipe.requires_last)
                or (opening_human_review and index == 0)
            ):
                last_review = f"{prefix}-anchor-last-select-{suffix}"
                nodes.append(
                    ExecutionNode(
                        node_id=last_review,
                        skill_name="review.assets.select",
                        inputs={
                            "limit": 1,
                            "review_all_candidates": True,
                            "review_scope": "last_frame",
                            "review_phase": "ending_frame",
                            "require_human_review": human_anchor_review,
                            "auto_select_for_probe": stage_probe_auto_select,
                            "review_notes": review_notes,
                        },
                        depends_on=[last_candidate],
                        tags=["review", "anchor", "last", "segment"],
                        stage="review",
                    )
                )
                last_source = last_review
                anchor_review_nodes.append(last_review)
            anchor_nodes["last"] = last_source
            input_dependencies.append(last_source)

        reference_node = ""
        if recipe.requires_references:
            if has_configured_references:
                reference_check = f"{prefix}-reference-check-{suffix}"
                nodes.append(
                    ExecutionNode(
                        node_id=reference_check,
                        skill_name="longvideo.validate_references",
                        inputs={
                            "reference_manifest": configured_reference_manifest,
                            "reference_image_paths": configured_reference_images,
                            "reference_video_paths": configured_reference_videos,
                            "max_images": max_reference_images,
                            "max_videos": max_reference_videos,
                            "selection_limit": min(
                                reference_selection_limit,
                                recipe.reference_selection_limit
                                or reference_selection_limit,
                            ),
                        },
                        depends_on=[prompt_node],
                        tags=["quality", "reference", "segment"],
                        stage="quality",
                    )
                )
                reference_node = reference_check
            else:
                reference_candidate = f"{prefix}-reference-candidates-{suffix}"
                reference_dependencies = [
                    prompt_node,
                    "image-asset-check",
                ]
                nodes.append(
                    ExecutionNode(
                        node_id=reference_candidate,
                        skill_name="media.image.generate_keyframe",
                        inputs={
                            "workflow_name": image_manifest.name,
                            "segment_index": index,
                            "anchor_position": "reference",
                            "width": frame_width,
                            "height": frame_height,
                            "image_count": reference_candidate_count,
                            "suffix": f"{prefix}_references_{suffix}",
                        },
                        depends_on=reference_dependencies,
                        tags=["render", "image", "reference", "segment"],
                        tool_name="comfy.workflow.text_to_image",
                        stage="render",
                    )
                )
                reference_source = reference_candidate
                reference_review_enabled = (
                    require_human_review or stage_probe_auto_select
                ) and (review_policy == "every_segment" or index == 0)
                if reference_review_enabled:
                    reference_review = f"{prefix}-reference-select-{suffix}"
                    nodes.append(
                        ExecutionNode(
                            node_id=reference_review,
                            skill_name="review.assets.select",
                            inputs={
                                "limit": min(
                                    reference_selection_limit,
                                    recipe.reference_selection_limit
                                    or reference_selection_limit,
                                ),
                                "review_all_candidates": True,
                                "review_scope": "reference",
                                "review_phase": "reference_selection",
                                "require_human_review": require_human_review,
                                "auto_select_for_probe": stage_probe_auto_select,
                                "review_notes": review_notes,
                            },
                            depends_on=[reference_candidate],
                            tags=["review", "reference", "segment"],
                            stage="review",
                        )
                    )
                    reference_source = reference_review
                reference_check = f"{prefix}-reference-check-{suffix}"
                nodes.append(
                    ExecutionNode(
                        node_id=reference_check,
                        skill_name="longvideo.validate_references",
                        inputs={
                            "auto_reference_generation": True,
                            "selection_limit": min(
                                reference_selection_limit,
                                recipe.reference_selection_limit
                                or reference_selection_limit,
                            ),
                            "max_images": max_reference_images,
                            "max_videos": max_reference_videos,
                        },
                        depends_on=[prompt_node, reference_source],
                        tags=["quality", "reference", "segment"],
                        stage="quality",
                    )
                )
                reference_node = reference_check
            input_dependencies.append(reference_node)

        configured_render_length = context.longvideo_int(
            goal,
            "longvideo_length",
            int(goal.constraints.get("longvideo_h3_length") or segment_default_length),
            1000,
        )
        render_length = max(
            round(MIN_VIDEO_DURATION_SECONDS * max(segment_frame_rate, 1.0)),
            round(segment_duration_seconds * segment_frame_rate),
            configured_render_length,
        )
        is_h3_workflow = capability.workflow_name.startswith("minimax_h3_")
        if is_h3_workflow and render_length > 362:
            raise ValueError(
                "MiniMax H3 long-video segments cannot exceed 362 frames; increase the segment count or reduce longvideo_length"
            )
        rendered_duration_seconds = render_length / max(segment_frame_rate, 1.0)
        if is_h3_workflow:
            rendered_duration_seconds = render_length / 24
        h3_prompt_node = ""
        if capability.workflow_name.startswith("minimax_h3_") and (
            recipe.requires_first or recipe.requires_last or recipe.requires_references
        ):
            h3_prompt_node = f"{prefix}-h3-prompt-{suffix}"
            h3_prompt_dependencies = [prompt_node, *anchor_nodes.values()]
            if reference_node:
                h3_prompt_dependencies.append(reference_node)
            nodes.append(
                ExecutionNode(
                    node_id=h3_prompt_node,
                    skill_name="agent.segment.prepare_h3",
                    inputs={
                        "segment_index": index,
                        "scene_prompt_node": prompt_node,
                        "anchor_nodes": dict(anchor_nodes),
                        "reference_node": reference_node,
                        "recipe": recipe_name,
                        "duration_seconds": segment_duration_seconds,
                    },
                    depends_on=list(dict.fromkeys(h3_prompt_dependencies)),
                    tags=["story", "segment", "h3", "vision"],
                    stage="prompting",
                )
            )
            input_dependencies.append(h3_prompt_node)

        conditioning_plan = ConditioningPlan(
            recipe=recipe_name,
            workflow_name=capability.workflow_name,
            anchor_nodes=anchor_nodes,
            reference_node=reference_node,
            continuation_node=previous_rendered_tail_node or "",
        )
        render_inputs = {
            "recipe": recipe_name,
            "workflow_name": capability.workflow_name,
            "render_tool": recipe.render_tool,
            "segment_index": index,
            "video_count": context.longvideo_int(goal, "video_count", 1, 8),
            "conditioning_plan": conditioning_plan.to_dict(),
            "anchor_nodes": dict(anchor_nodes),
            "reference_node": reference_node,
            "h3_prompt_node": h3_prompt_node,
            "continuation": recipe.continuation,
            "continuity_mode": continuity_mode,
            "production_profile": production_profile,
            "width": int(
                goal.constraints.get("canvas_width")
                or context.longvideo_int(
                    goal,
                    "longvideo_width",
                    int(goal.constraints.get("longvideo_h3_width") or 512),
                    2048,
                )
            ),
            "height": int(
                goal.constraints.get("canvas_height")
                or context.longvideo_int(
                    goal,
                    "longvideo_height",
                    int(goal.constraints.get("longvideo_h3_height") or 288),
                    2048,
                )
            ),
            "length": render_length,
            "target_duration_seconds": segment_duration_seconds,
            "steps": context.longvideo_int(
                goal,
                "longvideo_steps",
                int(goal.constraints.get("longvideo_h3_steps") or 16),
                100,
            ),
            "model_profile": str(
                goal.constraints.get("longvideo_model_profile")
                or goal.constraints.get("native_h3_model_profile")
                or goal.constraints.get("longvideo_h3_model_profile")
                or "q2"
            ),
        }
        nodes.append(
            ExecutionNode(
                node_id=video_node,
                skill_name="longvideo.render_segment_video",
                inputs=render_inputs,
                depends_on=list(dict.fromkeys(input_dependencies)),
                tags=["render", "video", "segment", recipe_name]
                + (["retry"] if retry else []),
                tool_name=recipe.render_tool,
                stage="render",
            )
        )
        delivered_video_node = video_node
        if segment_duration_seconds < rendered_duration_seconds - 0.01:
            trim_node = f"{prefix}-trim-{suffix}"
            nodes.append(
                ExecutionNode(
                    node_id=trim_node,
                    skill_name="media.video.trim",
                    inputs={
                        "duration_seconds": segment_duration_seconds,
                        "output_name": f"longvideo_{prefix}_{suffix}_trimmed.mp4",
                    },
                    depends_on=[video_node],
                    tags=["package", "video", "segment", "duration"],
                    tool_name="media.trim_video",
                    stage="package",
                )
            )
            delivered_video_node = trim_node
        tail_node: str | None = None
        segment_qa_node: str | None = None
        if continuity_mode == "rendered_tail":
            tail_node = f"{prefix}-tail-{suffix}"
            nodes.append(
                ExecutionNode(
                    node_id=tail_node,
                    skill_name="media.video.extract_last_frame",
                    inputs={
                        "output_name": f"{prefix}_tail_{suffix}.png",
                        "source_video_node": delivered_video_node,
                    },
                    depends_on=[delivered_video_node],
                    tags=["quality", "continuity", "tail", "segment"]
                    + (["retry"] if retry else []),
                    tool_name="media.extract_last_frame",
                    stage="quality",
                )
            )
            segment_qa_node = f"{prefix}-qa-{suffix}"
            segment_duration = max(
                1.0, float(render_inputs["length"]) / max(segment_frame_rate, 1.0)
            )
            nodes.append(
                ExecutionNode(
                    node_id=segment_qa_node,
                    skill_name="media.video.inspect",
                    inputs={
                        "expected_width": render_inputs["width"],
                        "expected_height": render_inputs["height"],
                        "expected_fps": segment_frame_rate,
                        "target_duration": segment_duration,
                        "duration_tolerance": 0.75,
                        "require_audio": True,
                        "require_stereo_audio": True,
                        "analyze_audio": True,
                        "frame_count": 8,
                        "columns": 4,
                        "scale_width": 320,
                        "source_video_node": video_node,
                    },
                    depends_on=[video_node],
                    tags=["quality", "technical-qa", "segment"]
                    + (["retry"] if retry else []),
                    tool_name="media.video_qa",
                    stage="quality",
                )
            )
        tts_node: str | None = None
        if use_tts:
            tts_node = f"{prefix}-tts-audio-{suffix}"
            nodes.append(
                ExecutionNode(
                    node_id=tts_node,
                    skill_name="media.audio.narrate",
                    inputs={
                        "segment_index": index,
                        "voice": "en-US-AriaNeural",
                        "output_name": f"{prefix}_segment_{suffix}.mp3",
                    },
                    depends_on=[prompt_node],
                    tags=["audio", "segment"] + (["retry"] if retry else []),
                    tool_name="audio.generate_tts_real",
                    stage="audio",
                )
            )
        return delivered_video_node, prompt_node, tts_node, tail_node, segment_qa_node

    previous_rendered_tail_node: str | None = None
    segment_tail_nodes: list[str] = []
    segment_qa_nodes: list[str] = []
    for index in range(segment_count):
        video_node, _, tts_node, segment_tail_node, segment_qa_node = append_segment(
            "segment",
            index,
            previous_rendered_tail_node,
        )
        segment_video_nodes.append(video_node)
        if tts_node:
            tts_nodes.append(tts_node)
        if segment_tail_node:
            segment_tail_nodes.append(segment_tail_node)
            previous_rendered_tail_node = segment_tail_node
        if segment_qa_node:
            segment_qa_nodes.append(segment_qa_node)

    longvideo_edit_profile = str(
        goal.constraints.get("longvideo_edit_profile") or ""
    ).strip()
    if longvideo_edit_profile and longvideo_edit_profile not in EDIT_PROFILES:
        raise ValueError(
            f"Unsupported longvideo_edit_profile: {longvideo_edit_profile}"
        )
    if longvideo_edit_profile and use_tts:
        raise ValueError(
            "longvideo_edit_profile currently requires use_tts=false so the edit owns segment audio"
        )
    if longvideo_edit_profile:
        final_video_node = "edit-segment-timeline"
        nodes.append(
            ExecutionNode(
                node_id=final_video_node,
                skill_name="media.video.compose_timeline",
                inputs={
                    "profile": longvideo_edit_profile,
                    "output_width": int(
                        goal.constraints.get("canvas_width")
                        or goal.constraints.get("longvideo_width")
                        or goal.constraints.get("longvideo_h3_width")
                        or 512
                    ),
                    "output_height": int(
                        goal.constraints.get("canvas_height")
                        or goal.constraints.get("longvideo_height")
                        or goal.constraints.get("longvideo_h3_height")
                        or 288
                    ),
                    "fps": segment_frame_rate,
                    "target_duration_seconds": float(goal.duration_seconds),
                    "variant_seed": int(
                        goal.constraints.get("edit_variant_seed") or variant_seed
                    ),
                    "transition_duration_seconds": float(
                        goal.constraints.get("edit_transition_duration") or 0.1
                    ),
                },
                depends_on=segment_video_nodes,
                tags=["package", "video", "editing", "timeline"],
                tool_name="media.compose_edit",
                stage="package",
            )
        )
    else:
        final_video_node = "concat-final-video"
        nodes.append(
            ExecutionNode(
                node_id=final_video_node,
                skill_name="media.video.concat",
                inputs={"method": "demuxer"},
                depends_on=segment_video_nodes,
                tags=["package", "video"],
                tool_name="media.concat_videos",
                stage="package",
            )
        )
    preview_dependency = final_video_node
    if use_tts:
        nodes.extend(
            [
                ExecutionNode(
                    node_id="concat-final-audio",
                    skill_name="media.audio.concat",
                    inputs={},
                    depends_on=tts_nodes,
                    tags=["package", "audio"],
                    tool_name="audio.concat_tracks",
                    stage="audio",
                ),
                ExecutionNode(
                    node_id="mux-final-video",
                    skill_name="media.video.merge_audio",
                    inputs={},
                    depends_on=["concat-final-video", "concat-final-audio"],
                    tags=["package", "mux"],
                    tool_name="media.merge_audio_video",
                    stage="package",
                ),
            ]
        )
        preview_dependency = "mux-final-video"

    preview_dependency = context.append_video_speed_node(
        goal,
        nodes,
        source_node=preview_dependency,
        node_id="video-speed",
    )
    final_video_duration = float(goal.duration_seconds)
    if goal.duration_seconds > 0:
        scaled_duration = context.scaled_video_qa_inputs(
            goal,
            {"target_duration": float(goal.duration_seconds)},
        ).get("target_duration")
        if scaled_duration not in {None, ""}:
            final_video_duration = float(scaled_duration)
    if goal.duration_seconds > 0:
        nodes.append(
            ExecutionNode(
                node_id="video-trim",
                skill_name="media.video.trim",
                inputs={"duration_seconds": final_video_duration},
                depends_on=[preview_dependency],
                tags=["package", "video", "duration"],
                tool_name="media.trim_video",
                stage="package",
            )
        )
        preview_dependency = "video-trim"
    preview_dependency = context.append_nvidia_vfx_postprocess_node(
        goal,
        nodes,
        source_node=preview_dependency,
        node_id="video-nvidia-vfx",
    )
    final_video_node = preview_dependency

    longvideo_qa_inputs = context.scaled_video_qa_inputs(
        goal,
        {
            "expected_width": int(
                goal.constraints.get("canvas_width")
                or goal.constraints.get("longvideo_width")
                or goal.constraints.get("longvideo_h3_width")
                or 512
            ),
            "expected_height": int(
                goal.constraints.get("canvas_height")
                or goal.constraints.get("longvideo_height")
                or goal.constraints.get("longvideo_h3_height")
                or 288
            ),
            "expected_fps": segment_frame_rate,
            "target_duration": float(goal.duration_seconds)
            if goal.duration_seconds > 0
            else None,
            "duration_tolerance": 0.6,
            "require_audio": use_tts,
            "require_stereo_audio": False,
            "analyze_audio": use_tts,
            "frame_count": 8,
            "columns": 4,
            "scale_width": 480,
        },
    )
    nodes.append(
        ExecutionNode(
            node_id="longvideo-video-qa",
            skill_name="media.video.inspect",
            inputs=longvideo_qa_inputs,
            depends_on=[preview_dependency],
            tags=["quality", "technical-qa", "video", "longvideo"],
            tool_name="media.video_qa",
            stage="quality",
        )
    )
    nodes.append(
        ExecutionNode(
            node_id="preview-gif",
            skill_name="media.video.gif_preview",
            inputs={"fps": 12, "scale_width": 512},
            depends_on=[preview_dependency, "longvideo-video-qa"],
            tags=["preview", "gif"],
            tool_name="media.video_to_gif",
            stage="package",
        )
    )
    collect_node = "collect-longvideo-outputs"
    nodes.append(
        ExecutionNode(
            node_id=collect_node,
            skill_name="agent.output.collect",
            inputs={
                "keys": [
                    "saved_files",
                    "video_path",
                    "audio_path",
                    "gif_path",
                    "frame_path",
                ],
                "preferred_video_node": final_video_node,
            },
            depends_on=[
                preview_dependency,
                "longvideo-video-qa",
                "preview-gif",
                *segment_video_nodes,
                *segment_tail_nodes,
                *segment_qa_nodes,
                *tts_nodes,
            ],
            tags=["artifact", "summary"],
            stage="package",
        )
    )

    review_final_node = ""
    if review_loop_enabled:
        nodes.append(
            ExecutionNode(
                node_id="review-select",
                skill_name="review.assets.select",
                inputs={
                    "limit": selection_limit,
                    "review_scope": "final_video",
                    "review_notes": review_notes,
                },
                depends_on=[collect_node],
                tags=["review", "longvideo"],
                stage="review",
            )
        )
        nodes.append(
            ExecutionNode(
                node_id="review-refine-prompt",
                skill_name="agent.review.refine_prompt",
                inputs={"review_notes": review_notes, "retry_count": 1},
                depends_on=["idea-brief", "review-select"],
                tags=["review", "creative", "retry"],
                stage="review",
            )
        )
        retry_video_nodes: list[str] = []
        retry_tts_nodes: list[str] = []
        retry_tail_nodes: list[str] = []
        retry_qa_nodes: list[str] = []
        retry_rendered_tail: str | None = None
        for index in range(segment_count):
            retry_video, _, retry_tts, retry_tail, retry_qa = append_segment(
                "review-segment",
                index,
                retry_rendered_tail,
                retry=True,
            )
            retry_video_nodes.append(retry_video)
            if retry_tts:
                retry_tts_nodes.append(retry_tts)
            if retry_tail:
                retry_tail_nodes.append(retry_tail)
                retry_rendered_tail = retry_tail
            if retry_qa:
                retry_qa_nodes.append(retry_qa)
        if longvideo_edit_profile:
            retry_preview_dependency = "review-edit-segment-timeline"
            nodes.append(
                ExecutionNode(
                    node_id=retry_preview_dependency,
                    skill_name="media.video.compose_timeline",
                    inputs={
                        "profile": longvideo_edit_profile,
                        "output_width": int(
                            goal.constraints.get("canvas_width")
                            or goal.constraints.get("longvideo_width")
                            or goal.constraints.get("longvideo_h3_width")
                            or 512
                        ),
                        "output_height": int(
                            goal.constraints.get("canvas_height")
                            or goal.constraints.get("longvideo_height")
                            or goal.constraints.get("longvideo_h3_height")
                            or 288
                        ),
                        "fps": segment_frame_rate,
                        "target_duration_seconds": float(goal.duration_seconds),
                        "variant_seed": int(
                            goal.constraints.get("edit_variant_seed") or variant_seed
                        )
                        + 1,
                        "transition_duration_seconds": float(
                            goal.constraints.get("edit_transition_duration") or 0.1
                        ),
                    },
                    depends_on=retry_video_nodes,
                    tags=["package", "video", "editing", "timeline", "retry"],
                    tool_name="media.compose_edit",
                    stage="package",
                )
            )
        else:
            retry_preview_dependency = "review-concat-final-video"
            nodes.append(
                ExecutionNode(
                    node_id=retry_preview_dependency,
                    skill_name="media.video.concat",
                    inputs={"method": "demuxer"},
                    depends_on=retry_video_nodes,
                    tags=["package", "video", "retry"],
                    tool_name="media.concat_videos",
                    stage="package",
                )
            )
        if use_tts:
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="review-concat-final-audio",
                        skill_name="media.audio.concat",
                        inputs={},
                        depends_on=retry_tts_nodes,
                        tags=["package", "audio", "retry"],
                        tool_name="audio.concat_tracks",
                        stage="audio",
                    ),
                    ExecutionNode(
                        node_id="review-mux-final-video",
                        skill_name="media.video.merge_audio",
                        inputs={},
                        depends_on=[
                            "review-concat-final-video",
                            "review-concat-final-audio",
                        ],
                        tags=["package", "mux", "retry"],
                        tool_name="media.merge_audio_video",
                        stage="package",
                    ),
                ]
            )
            retry_preview_dependency = "review-mux-final-video"
        retry_preview_dependency = context.append_video_speed_node(
            goal,
            nodes,
            source_node=retry_preview_dependency,
            node_id="review-video-speed",
            retry=True,
        )
        if goal.duration_seconds > 0:
            nodes.append(
                ExecutionNode(
                    node_id="review-video-trim",
                    skill_name="media.video.trim",
                    inputs={"duration_seconds": final_video_duration},
                    depends_on=[retry_preview_dependency],
                    tags=["package", "video", "duration", "retry"],
                    tool_name="media.trim_video",
                    stage="package",
                )
            )
            retry_preview_dependency = "review-video-trim"
        retry_preview_dependency = context.append_nvidia_vfx_postprocess_node(
            goal,
            nodes,
            source_node=retry_preview_dependency,
            node_id="review-video-nvidia-vfx",
            retry=True,
        )
        nodes.append(
            ExecutionNode(
                node_id="review-longvideo-video-qa",
                skill_name="media.video.inspect",
                inputs=longvideo_qa_inputs,
                depends_on=[retry_preview_dependency],
                tags=["quality", "technical-qa", "video", "longvideo", "retry"],
                tool_name="media.video_qa",
                stage="quality",
            )
        )
        nodes.append(
            ExecutionNode(
                node_id="review-preview-gif",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=[retry_preview_dependency, "review-longvideo-video-qa"],
                tags=["preview", "gif", "retry"],
                tool_name="media.video_to_gif",
                stage="package",
            )
        )
        retry_collect = "review-collect-longvideo-outputs"
        nodes.append(
            ExecutionNode(
                node_id=retry_collect,
                skill_name="agent.output.collect",
                inputs={
                    "keys": [
                        "saved_files",
                        "video_path",
                        "audio_path",
                        "gif_path",
                        "frame_path",
                    ],
                    "preferred_video_node": retry_preview_dependency,
                },
                depends_on=[
                    retry_preview_dependency,
                    "review-longvideo-video-qa",
                    "review-preview-gif",
                    *retry_video_nodes,
                    *retry_tail_nodes,
                    *retry_qa_nodes,
                    *retry_tts_nodes,
                ],
                tags=["artifact", "summary", "retry"],
                stage="package",
            )
        )
        review_final_node = "review-final-select"
        final_video_node = retry_preview_dependency
        nodes.append(
            ExecutionNode(
                node_id=review_final_node,
                skill_name="review.assets.select",
                inputs={
                    "limit": selection_limit,
                    "review_scope": "final_video",
                    "review_notes": review_notes,
                },
                depends_on=[retry_collect],
                tags=["review", "longvideo", "retry"],
                stage="review",
            )
        )

    summary_dependencies = [review_final_node] if review_final_node else [collect_node]
    nodes.append(
        ExecutionNode(
            node_id="persist-longvideo-summary",
            skill_name="agent.summary.persist",
            inputs={
                "summary_name": "long_video_summary.json",
                "summary_scope": "long_video",
            },
            depends_on=summary_dependencies,
            tags=["artifact", "summary"],
            stage="package",
        )
    )
    publish_package_node = ""
    if production_profile == "text2longvideo":
        publish_package_node = "persist-publish-ready-longvideo"
        nodes.append(
            ExecutionNode(
                node_id=publish_package_node,
                skill_name="agent.summary.persist",
                inputs={
                    "summary_name": "publish_ready_longvideo.json",
                    "summary_scope": "publish_ready_longvideo",
                    "summary_metadata": {
                        "production_profile": production_profile,
                        "target_duration_seconds": float(goal.duration_seconds),
                        "segment_count": segment_count,
                        "recipe_sequence": list(recipe_sequence),
                        "continuity_mode": continuity_mode,
                        "segment_qa_nodes": list(segment_qa_nodes),
                        "tail_frame_nodes": list(segment_tail_nodes),
                        "final_video_node": final_video_node,
                        "publish_contract": {
                            "human_review_required": True,
                            "media_inspection_recorded": True,
                            "public_dispatch_requires_platform_receipt": True,
                        },
                    },
                },
                depends_on=["persist-longvideo-summary", *segment_qa_nodes],
                tags=["artifact", "summary", "publish", "longvideo"],
                stage="package",
            )
        )
    metadata = {
        "segment_count": segment_count,
        "continuity_mode": continuity_mode,
        "selected_workflow": image_manifest.name
        if image_manifest
        else selected_capabilities[recipe_sequence[0]].workflow_name,
        "keyframe_workflow": image_manifest.name if image_manifest else "",
        "landing_frame_workflow": refine_manifest.name if refine_manifest else "",
        "required_assets": [
            *(
                [asset.to_dict() for asset in image_manifest.required_assets]
                if image_manifest
                else []
            ),
            *(
                [asset.to_dict() for asset in refine_manifest.required_assets]
                if refine_manifest
                else []
            ),
        ],
        "graph_overview": [node.node_id for node in nodes],
        "idea_variants": idea_variants,
        "variant_seed": variant_seed,
        "recipe_sequence": recipe_sequence,
        "recipe_workflows": {
            name: selected_capabilities[name].workflow_name for name in recipe_contracts
        },
        "conditioning_contracts": {
            name: recipe_contracts[name].to_dict() for name in recipe_contracts
        },
        "review_policy": review_policy,
        "stage_probe_auto_select": stage_probe_auto_select,
        "use_tts": use_tts,
        "longvideo_edit_profile": longvideo_edit_profile,
        "production_profile": production_profile,
        "segment_qa_nodes": segment_qa_nodes,
        "segment_tail_nodes": segment_tail_nodes,
        "publish_package_node": publish_package_node,
        "video_speed": context.video_speed_config(goal),
        "final_video_node": final_video_node,
        "review_loop_enabled": review_loop_enabled,
        "review_notes": review_notes,
        **(
            {"reference_video": context.reference_video_metadata(goal)}
            if reference_node
            else {}
        ),
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="longvideo_real_v2",
        nodes=nodes,
        metadata=metadata,
        description=f"Capability-driven long-video workflow for goal '{goal.prompt}'",
    )
