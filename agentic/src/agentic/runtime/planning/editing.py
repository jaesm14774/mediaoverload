"""Editing strategy graph assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest
from agentic.runtime.drama import DramaPlan, DramaPlanError, compile_drama_plan
from agentic.runtime.editing import EDIT_PROFILES, IMAGE_SUFFIXES, EditPlan
from agentic.runtime.planning.common import PlanningContext


def build_video_narrate(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    input_video_path = goal.constraints.get("input_video_path")
    narration_text = goal.constraints.get("text")
    if not input_video_path:
        raise ValueError("media_type 'video_narrate' requires --input-video")
    if not narration_text:
        raise ValueError("media_type 'video_narrate' requires --text")

    nodes = [
        ExecutionNode(
            node_id="tts-audio",
            skill_name="media.audio.narrate",
            inputs={"text": narration_text, "voice": "en-US-AriaNeural", "rate": "+0%"},
            tags=["audio"],
            tool_name="audio.generate_tts_real",
            stage="audio",
        ),
        ExecutionNode(
            node_id="mux-video",
            skill_name="media.video.merge_audio",
            inputs={"video_path": input_video_path},
            depends_on=["tts-audio"],
            tags=["video", "audio", "package"],
            tool_name="media.merge_audio_video",
            stage="package",
        ),
    ]
    editorial_node = context.append_editorial_pacing_node(
        nodes,
        source_node="mux-video",
        node_id="video-editorial-pacing",
    )
    nodes.append(
        ExecutionNode(
            node_id="gif-preview",
            skill_name="media.video.gif_preview",
            inputs={"fps": 12, "scale_width": 512},
            depends_on=[editorial_node],
            tags=["preview", "gif"],
            tool_name="media.video_to_gif",
            stage="package",
        ),
    )
    metadata = {
        "input_video_path": input_video_path,
        "graph_overview": [node.node_id for node in nodes],
        "use_case": "narrate_existing_video",
    }
    return ExecutionPlan(
        goal=goal,
        workflow_name="video_narrate_v1",
        nodes=nodes,
        metadata=metadata,
        description=f"Narrate existing video '{input_video_path}'",
    )


def build_image_sequence_edit(
    context: PlanningContext, goal: GoalRequest
) -> ExecutionPlan:
    raw_paths = (
        goal.constraints.get("edit_input_paths")
        or goal.constraints.get("media_paths")
        or []
    )
    if isinstance(raw_paths, str):
        input_paths = [raw_paths]
    else:
        input_paths = [str(path) for path in raw_paths if str(path).strip()]
    raw_drama_plan = goal.constraints.get("drama_plan")
    drama_plan_source = str(goal.constraints.get("drama_plan_source") or "").strip()
    if raw_drama_plan is None and drama_plan_source:
        source_path = Path(drama_plan_source).expanduser().resolve()
        if not source_path.is_file():
            raise DramaPlanError(f"DramaPlan file does not exist: {source_path}")
        raw_drama_plan = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        not input_paths
        and not isinstance(goal.constraints.get("edit_plan"), dict)
        and raw_drama_plan is None
    ):
        raise ValueError(
            "media_type 'image_sequence_edit' requires --edit-input, an edit_plan, or a drama_plan"
        )

    raw_edit_plan = goal.constraints.get("edit_plan")
    if raw_edit_plan is not None and raw_drama_plan is not None:
        raise ValueError(
            "image_sequence_edit accepts either edit_plan or drama_plan, not both"
        )
    explicit_plan = (
        EditPlan.from_dict(raw_edit_plan) if isinstance(raw_edit_plan, dict) else None
    )
    drama_plan = None
    compiled_drama_plan = None
    if raw_drama_plan is not None:
        if not isinstance(raw_drama_plan, dict):
            raise DramaPlanError("drama_plan must be an object or a JSON file path")
        drama_plan = DramaPlan.from_dict(raw_drama_plan).validate(require_assets=True)
        compiled_drama_plan = compile_drama_plan(drama_plan)
    if explicit_plan:
        effective_profile = explicit_plan.profile
    elif compiled_drama_plan:
        effective_profile = compiled_drama_plan.profile
    else:
        requested_profile = str(goal.constraints.get("edit_profile") or "").strip()
        if requested_profile:
            effective_profile = requested_profile
        elif input_paths and all(
            Path(path).suffix.lower() in IMAGE_SUFFIXES for path in input_paths
        ):
            effective_profile = "motion_cut_v1"
        else:
            effective_profile = "baseline_concat"
    if effective_profile not in EDIT_PROFILES:
        raise ValueError(f"Unsupported edit_profile: {effective_profile}")

    output_path = str(goal.constraints.get("edit_output_path") or "").strip()
    if explicit_plan:
        qa_target_duration = explicit_plan.target_duration_seconds
    elif compiled_drama_plan:
        natural_duration = sum(
            scene.duration_seconds for scene in drama_plan.scenes
        ) - sum(
            transition.duration_seconds
            for transition in compiled_drama_plan.transitions
        )
        qa_target_duration = compiled_drama_plan.target_duration_seconds or round(
            natural_duration, 3
        )
    else:
        qa_target_duration = (
            float(goal.duration_seconds) if goal.duration_seconds > 0 else None
        )
    compose_inputs: dict[str, Any] = {
        "profile": effective_profile,
        "output_width": explicit_plan.output_width
        if explicit_plan
        else compiled_drama_plan.output_width
        if compiled_drama_plan
        else int(
            goal.constraints.get("edit_width") or goal.constraints.get("width") or 576
        ),
        "output_height": explicit_plan.output_height
        if explicit_plan
        else compiled_drama_plan.output_height
        if compiled_drama_plan
        else int(
            goal.constraints.get("edit_height")
            or goal.constraints.get("height")
            or 1024
        ),
        "fps": explicit_plan.fps
        if explicit_plan
        else compiled_drama_plan.fps
        if compiled_drama_plan
        else float(
            goal.constraints.get("edit_fps")
            or goal.constraints.get("video_frame_rate")
            or 24
        ),
        "target_duration_seconds": qa_target_duration,
        "variant_seed": explicit_plan.variant_seed
        if explicit_plan
        else compiled_drama_plan.variant_seed
        if compiled_drama_plan
        else int(goal.constraints.get("edit_variant_seed") or 0),
        "transition_duration_seconds": float(
            goal.constraints.get("edit_transition_duration") or 0.10
        ),
        "require_audio": bool(goal.constraints.get("edit_require_audio", False))
        or effective_profile != "baseline_concat",
        "require_stereo_audio": bool(goal.constraints.get("edit_require_audio", False))
        or effective_profile != "baseline_concat",
        "analyze_audio": bool(goal.constraints.get("edit_analyze_audio", False)),
    }
    if input_paths:
        compose_inputs["input_paths"] = input_paths
    if explicit_plan:
        compose_inputs["edit_plan"] = explicit_plan.to_dict()
    if drama_plan:
        compose_inputs["drama_plan"] = drama_plan.to_dict()
    if output_path:
        compose_inputs["output_path"] = output_path

    nodes = [
        ExecutionNode(
            node_id="compose-edit",
            skill_name="media.video.compose_timeline",
            inputs=compose_inputs,
            tags=["render", "editing", "timeline"],
            tool_name="media.compose_edit",
            stage="package",
        ),
        ExecutionNode(
            node_id="edit-video-qa",
            skill_name="media.video.inspect",
            inputs={
                "target_duration": qa_target_duration,
                "duration_tolerance": 0.35,
                "expected_fps": compose_inputs["fps"],
                "require_audio": bool(goal.constraints.get("edit_require_audio", False))
                or effective_profile != "baseline_concat",
                "require_stereo_audio": bool(
                    goal.constraints.get("edit_require_audio", False)
                )
                or effective_profile != "baseline_concat",
                "analyze_audio": bool(
                    goal.constraints.get("edit_analyze_audio", False)
                ),
                "frame_count": 12,
                "columns": 4,
                "scale_width": 360,
            },
            depends_on=["compose-edit"],
            tags=["quality", "editing", "video"],
            tool_name="media.video_qa",
            stage="quality",
        ),
        ExecutionNode(
            node_id="edit-preview-gif",
            skill_name="media.video.gif_preview",
            inputs={"fps": 12, "scale_width": 512},
            depends_on=["compose-edit", "edit-video-qa"],
            tags=["preview", "editing"],
            tool_name="media.video_to_gif",
            stage="package",
        ),
        ExecutionNode(
            node_id="collect-edit-outputs",
            skill_name="agent.output.collect",
            inputs={
                "keys": [
                    "video_path",
                    "manifest_path",
                    "contact_sheet_path",
                    "gif_path",
                    "saved_files",
                ]
            },
            depends_on=["compose-edit", "edit-video-qa", "edit-preview-gif"],
            tags=["artifact", "editing"],
            stage="package",
        ),
        ExecutionNode(
            node_id="persist-edit-summary",
            skill_name="agent.summary.persist",
            inputs={
                "summary_name": "image_sequence_edit_summary.json",
                "summary_scope": "image_sequence_edit",
            },
            depends_on=["collect-edit-outputs"],
            tags=["artifact", "summary", "editing"],
            stage="package",
        ),
    ]
    editorial_node = context.append_editorial_pacing_node(
        nodes,
        source_node="compose-edit",
        node_id="edit-editorial-pacing",
    )
    qa_node = next(node for node in nodes if node.node_id == "edit-video-qa")
    qa_node.inputs = context.scaled_video_qa_inputs(goal, qa_node.inputs)
    qa_node.inputs["expected_fps"] = compose_inputs["fps"]
    if qa_target_duration is not None:
        qa_node.inputs["minimum_duration"] = min(
            float(qa_node.inputs["minimum_duration"]),
            float(qa_target_duration),
        )
    qa_node.depends_on = [editorial_node]
    preview_node = next(node for node in nodes if node.node_id == "edit-preview-gif")
    preview_node.depends_on = [editorial_node, "edit-video-qa"]
    collect_node = next(node for node in nodes if node.node_id == "collect-edit-outputs")
    collect_node.depends_on = [editorial_node, "edit-video-qa", "edit-preview-gif"]
    return ExecutionPlan(
        goal=goal,
        workflow_name="image_sequence_edit_v1",
        nodes=nodes,
        metadata={
            "graph_overview": [node.node_id for node in nodes],
            "input_paths": input_paths,
            "profile": compose_inputs.get("profile"),
            "variant_seed": compose_inputs.get("variant_seed"),
            "output_path": output_path,
            "qa_target_duration": qa_target_duration,
            "explicit_edit_plan": explicit_plan is not None,
            "explicit_drama_plan": drama_plan is not None,
        },
        description=(
            "Compile a short-drama plan into the deterministic timeline editor"
            if drama_plan
            else "Compose an agent-controlled timeline from generated images or video segments"
        ),
    )
