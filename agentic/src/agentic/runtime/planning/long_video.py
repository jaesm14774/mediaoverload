"""Serial episode video graph assembly."""

from __future__ import annotations

import math

from agentic.runtime.contracts import (
    MIN_VIDEO_DURATION_SECONDS,
    ExecutionNode,
    ExecutionPlan,
    GoalRequest,
)
from agentic.runtime.planning.common import PlanningContext


def build_long_video(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    serial_mode = bool(goal.constraints.get("serial_mode", False))
    if not serial_mode:
        raise ValueError("long_video planning requires a serial series")
    serial_series_id = str(goal.constraints.get("series_id") or "").strip()
    if not serial_series_id:
        raise ValueError("Serial long-video planning requires an explicit series_id")
    if not str(goal.constraints.get("character") or "").strip():
        raise ValueError("Serial long-video planning requires an explicit fixed character")

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
    stage_probe_auto_select = bool(
        goal.constraints.get("stage_probe_auto_select", False)
    )
    review_policy = (
        str(
            goal.constraints.get("longvideo_review_policy")
            or "every_segment"
        )
        .strip()
        .lower()
    )
    if review_policy != "every_segment":
        raise ValueError("Serial long-video mode requires every-segment human review")
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
        recipe_candidates,
    )

    capabilities = capabilities_from_manifests(context.asset_registry.all_manifests())
    preferred_workflows = goal.constraints.get("longvideo_workflow_names") or []
    if isinstance(preferred_workflows, str):
        preferred_workflows = [preferred_workflows]
    candidates = recipe_candidates(
        capabilities, preferred_workflows=preferred_workflows
    )
    configured_references = bool(
        goal.constraints.get("reference_manifest")
        or goal.constraints.get("reference_image_paths")
        or goal.constraints.get("reference_video_paths")
    )
    if not configured_references:
        raise ValueError("Serial long-video mode requires a registered Ref2VA identity reference bundle")
    if not candidates.get("reference_bundle") or not candidates.get("anchor_first"):
        raise ValueError("Serial long-video mode requires configured H3 Ref2VA and I2VA workflows")
    if stage_probe_auto_select:
        raise ValueError("Serial long-video mode cannot auto-select a segment review")
    recipe_sequence = ["reference_bundle", *(["anchor_first"] * (segment_count - 1))]
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

    configured_reference_manifest = list(goal.constraints.get("reference_manifest") or [])
    configured_reference_images = list(goal.constraints.get("reference_image_paths") or [])
    configured_reference_videos = list(goal.constraints.get("reference_video_paths") or [])
    max_reference_images = int(
        goal.constraints.get("longvideo_reference_max_images") or 9
    )
    max_reference_videos = int(
        goal.constraints.get("longvideo_reference_max_videos") or 3
    )
    reference_selection_limit = max(
        1,
        int(
            goal.constraints.get("longvideo_reference_selection_limit") or 4
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
    render_width = int(
        goal.constraints.get("canvas_width")
        or context.longvideo_int(
            goal,
            "longvideo_width",
            int(goal.constraints.get("longvideo_h3_width") or 608),
            2048,
        )
    )
    render_height = int(
        goal.constraints.get("canvas_height")
        or context.longvideo_int(
            goal,
            "longvideo_height",
            int(goal.constraints.get("longvideo_h3_height") or 352),
            2048,
        )
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
            },
            depends_on=["idea-brief"],
            tags=["story"],
        ),
    ]
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
    segment_review_nodes_by_video: dict[str, str] = {}

    def append_segment(
        prefix: str,
        index: int,
        previous_rendered_tail_node: str | None,
    ) -> tuple[str, str | None, str | None]:
        suffix = f"{index + 1:02d}"
        prompt_node = f"{prefix}-prompt-{suffix}"
        video_node = f"{prefix}-video-{suffix}"
        recipe_name = recipe_sequence[index]
        capability = selected_capabilities[recipe_name]
        recipe = recipe_contracts[recipe_name]
        prompt_dependencies = ["script-plan", "idea-brief"]
        nodes.append(
            ExecutionNode(
                node_id=prompt_node,
                skill_name="agent.segment.prepare",
                inputs={"segment_index": index, "recipe": recipe_name},
                depends_on=prompt_dependencies,
                tags=["story", "segment"],
                stage="prompting",
            )
        )

        input_dependencies = [
            prompt_node,
            video_asset_check_for(capability.workflow_name),
        ]
        anchor_nodes: dict[str, str] = {}

        if recipe.requires_first:
            if not previous_rendered_tail_node:
                raise ValueError("Serial continuation requires the accepted prior segment tail")
            anchor_nodes["first"] = previous_rendered_tail_node
            input_dependencies.append(previous_rendered_tail_node)

        reference_node = ""
        if recipe.requires_references:
            reference_node = f"{prefix}-reference-check-{suffix}"
            nodes.append(
                ExecutionNode(
                    node_id=reference_node,
                    skill_name="longvideo.validate_references",
                    inputs={
                        "reference_manifest": configured_reference_manifest,
                        "reference_image_paths": configured_reference_images,
                        "reference_video_paths": configured_reference_videos,
                        "max_images": max_reference_images,
                        "max_videos": max_reference_videos,
                        "selection_limit": min(
                            reference_selection_limit,
                            recipe.reference_selection_limit or reference_selection_limit,
                        ),
                    },
                    depends_on=[prompt_node],
                    tags=["quality", "reference", "segment"],
                    stage="quality",
                )
            )
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
            "video_count": 1,
            "conditioning_plan": conditioning_plan.to_dict(),
            "anchor_nodes": dict(anchor_nodes),
            "reference_node": reference_node,
            "h3_prompt_node": h3_prompt_node,
            "continuation": recipe.continuation,
            "continuity_mode": continuity_mode,
            "width": render_width,
            "height": render_height,
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
                tags=["render", "video", "segment", recipe_name],
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
                        "segment_index": index,
                    },
                    depends_on=[video_node],
                    tags=["package", "video", "segment", "duration"],
                    tool_name="media.trim_video",
                    stage="package",
                )
            )
            delivered_video_node = trim_node
        continuation_source_node = delivered_video_node
        segment_qa_node = f"{prefix}-qa-{suffix}"
        segment_duration = segment_duration_seconds
        nodes.append(
            ExecutionNode(
                node_id=segment_qa_node,
                skill_name="media.video.inspect",
                inputs={
                    "expected_fps": segment_frame_rate,
                    "target_duration": segment_duration,
                    "duration_tolerance": 0.75,
                    "require_audio": True,
                    "require_stereo_audio": True,
                    "analyze_audio": True,
                    "frame_count": 8,
                    "columns": 4,
                    "scale_width": 320,
                    "source_video_node": delivered_video_node,
                    "segment_index": index,
                },
                depends_on=[delivered_video_node],
                tags=["quality", "technical-qa", "segment"],
                tool_name="media.video_qa",
                stage="quality",
            )
        )
        segment_review_node = f"{prefix}-review-{suffix}"
        nodes.append(
            ExecutionNode(
                node_id=segment_review_node,
                skill_name="review.assets.select",
                inputs={
                    "limit": 1,
                    "review_all_candidates": True,
                    "review_scope": "final_video",
                    "review_phase": "segment_acceptance",
                    "segment_index": index,
                    "require_human_review": True,
                    "review_notes": (
                        f"Segment {index + 1} of {segment_count} acceptance: inspect "
                        "the rendered action and continuity. "
                        "Accept only if this segment is suitable for the next segment to "
                        "continue from its observed final frame. Reject to stop the sequence "
                        "before that frame is used."
                    ),
                },
                depends_on=[delivered_video_node, segment_qa_node],
                tags=["review", "human-acceptance", "segment", "continuity"],
                stage="review",
            )
        )
        segment_review_nodes_by_video[delivered_video_node] = segment_review_node
        continuation_source_node = segment_review_node
        tail_node = f"{prefix}-tail-{suffix}"
        nodes.append(
            ExecutionNode(
                node_id=tail_node,
                skill_name="media.video.extract_last_frame",
                inputs={
                    "output_name": f"{prefix}_tail_{suffix}.png",
                    "source_video_node": continuation_source_node,
                    "segment_index": index,
                    "episode_end": index == segment_count - 1,
                },
                depends_on=[continuation_source_node],
                tags=["quality", "continuity", "tail", "segment"],
                tool_name="media.extract_last_frame",
                stage="quality",
            )
        )
        return delivered_video_node, tail_node, segment_qa_node

    previous_rendered_tail_node: str | None = None
    segment_tail_nodes: list[str] = []
    segment_qa_nodes: list[str] = []
    for index in range(segment_count):
        video_node, segment_tail_node, segment_qa_node = append_segment(
            "segment",
            index,
            previous_rendered_tail_node,
        )
        segment_video_nodes.append(video_node)
        segment_tail_nodes.append(segment_tail_node)
        previous_rendered_tail_node = segment_tail_node
        segment_qa_nodes.append(segment_qa_node)

    segment_acceptance_nodes = [
        segment_review_nodes_by_video[video_node] for video_node in segment_video_nodes
    ]

    final_video_node = "concat-final-video"
    nodes.append(
        ExecutionNode(
            node_id=final_video_node,
            skill_name="media.video.concat",
            inputs={"method": "demuxer"},
            depends_on=[*segment_video_nodes, *segment_acceptance_nodes],
            tags=["package", "video"],
            tool_name="media.concat_videos",
            stage="package",
        )
    )
    preview_dependency = final_video_node

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
            "expected_fps": segment_frame_rate,
            "target_duration": float(goal.duration_seconds)
            if goal.duration_seconds > 0
            else None,
            "duration_tolerance": 0.6,
            "require_audio": False,
            "require_stereo_audio": False,
            "analyze_audio": False,
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
            ],
            tags=["artifact", "summary"],
            stage="package",
        )
    )

    summary_dependencies = [collect_node]
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
    serial_episode_number = int(
        goal.constraints.get("serial_episode_number")
        or goal.constraints.get("episode_number")
        or 0
    )
    episode_topic = str(goal.constraints.get("episode_topic") or "").strip()
    platforms = [str(platform) for platform in (goal.constraints.get("platforms") or [])]
    serial_publish_node = "serial-publish-dispatch"
    nodes.extend(
        [
            ExecutionNode(
                node_id="serial-final-publish-review",
                skill_name="review.assets.select",
                inputs={
                    "limit": 1,
                    "review_scope": "final_video",
                    "review_phase": "serial_final_publish",
                    "preferred_video_node": final_video_node,
                    "platforms": platforms,
                    "require_human_review": True,
                },
                depends_on=[collect_node, final_video_node, "longvideo-video-qa"],
                tags=["review", "publish", "serial", "episode-final"],
                stage="review",
            ),
            ExecutionNode(
                node_id="serial-publish-process",
                skill_name="publish.media.process",
                inputs={
                    "output_dir": goal.constraints.get("output_dir"),
                    "platforms": platforms,
                },
                depends_on=["serial-final-publish-review"],
                tags=["publish", "media", "serial"],
                tool_name="publish.process_media",
                stage="package",
            ),
            ExecutionNode(
                node_id="serial-publish-caption",
                skill_name="publish.caption.prepare",
                inputs={
                    "hashtags": goal.constraints.get("hashtags", []),
                    "platforms": platforms,
                },
                depends_on=["serial-final-publish-review", "serial-publish-process"],
                tags=["publish", "caption", "serial"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id=serial_publish_node,
                skill_name="publish.social.dispatch",
                inputs={
                    "platforms": platforms,
                    "platform_configs": goal.constraints.get("platform_configs", {}),
                    "additional_params": goal.constraints.get("additional_params", {}),
                    "publish_mode": goal.constraints.get("publish_mode", "live"),
                    "dry_run": bool(goal.constraints.get("dry_run", False)),
                },
                depends_on=[
                    "serial-publish-process",
                    "serial-publish-caption",
                    "serial-final-publish-review",
                ],
                tags=["publish", "dispatch", "serial"],
                tool_name="publish.social",
                stage="publish",
            ),
            ExecutionNode(
                node_id="persist-serial-publication-summary",
                skill_name="agent.summary.persist",
                inputs={
                    "summary_name": "serial_publication_summary.json",
                    "summary_scope": "serial_publication",
                    "summary_metadata": {
                        "episode_number": serial_episode_number,
                        "episode_topic": episode_topic,
                        "platforms": platforms,
                        "publication_node": serial_publish_node,
                    },
                },
                depends_on=["serial-final-publish-review", serial_publish_node],
                tags=["artifact", "summary", "publish", "serial"],
                stage="package",
            ),
        ]
    )
    metadata = {
        "segment_count": segment_count,
        "continuity_mode": continuity_mode,
        "selected_workflow": selected_capabilities[recipe_sequence[0]].workflow_name,
        "graph_overview": [node.node_id for node in nodes],
        "idea_variants": idea_variants,
        "recipe_sequence": recipe_sequence,
        "recipe_workflows": {
            name: selected_capabilities[name].workflow_name for name in recipe_contracts
        },
        "conditioning_contracts": {
            name: recipe_contracts[name].to_dict() for name in recipe_contracts
        },
        "review_policy": review_policy,
        "stage_probe_auto_select": stage_probe_auto_select,
        "segment_qa_nodes": segment_qa_nodes,
        "segment_tail_nodes": segment_tail_nodes,
        "serial_publish_node": serial_publish_node,
        "video_speed": context.video_speed_config(goal),
        "final_video_node": final_video_node,
        "serial_mode": True,
        "series_id": serial_series_id,
        "serial_episode_number": int(goal.constraints.get("serial_episode_number") or 0),
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
