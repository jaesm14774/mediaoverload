"""News + character fusion story graph: N keyframes, N beat clips, N-1 seams of two bridge clips around a Klein frame, one video."""

from __future__ import annotations

from agentic.runtime.contracts import MIN_VIDEO_DURATION_SECONDS, ExecutionNode, ExecutionPlan, GoalRequest
from agentic.runtime.planning.common import PlanningContext
from agentic.runtime.prompt_tasks.news_fusion import MAX_BEATS, MIN_BEATS

DEFAULT_BEAT_COUNT = 3
CLIP_LENGTH = 124  # frames at 24 fps, about 5.2 s
BRIDGE_LENGTH = 107  # the shortest H3 render, about 4.5 s


def build_news_fusion_story(context: PlanningContext, goal: GoalRequest) -> ExecutionPlan:
    """One news item and one character become a fused story told as bridged clips.

    The beat count is fixed here because the graph is assembled before the LLM
    runs; the story writer must return exactly that many beats.
    """
    beats = context.constraint_int(goal, "news_fusion_beat_count", DEFAULT_BEAT_COUNT)
    if not MIN_BEATS <= beats <= MAX_BEATS:
        raise ValueError(f"news_fusion_beat_count must be between {MIN_BEATS} and {MAX_BEATS}")
    image = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_IMAGE_WORKFLOWS,
        constraint_keys=("keyframe_workflow_name", "image_workflow_name"),
        allowed_media_types={"image"},
    )
    refine = context.manifest_from_goal_constraints(
        goal,
        *context.DEFAULT_REFINE_WORKFLOWS,
        constraint_keys=("refine_workflow_name", "identity_refine_workflow_name"),
        allowed_media_types={"image_refine"},
    )
    clip_video = context.manifest_from_goal_constraints(
        goal,
        "minimax_h3_i2va",
        constraint_keys=("video_workflow_name",),
        allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
    )
    bridge_video = context.manifest_from_goal_constraints(
        goal,
        "minimax_h3_fl2va",
        constraint_keys=("bridge_video_workflow_name",),
        allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
    )
    render_config = context.native_h3_render_config(goal, clip_video)
    width, height = render_config["width"], render_config["height"]
    require_human_review = bool(goal.constraints.get("require_human_review", False))

    nodes = [
        ExecutionNode(
            node_id="fusion-story",
            skill_name="news_fusion.prepare_story",
            inputs={"beat_count": beats},
            tags=["creative", "story", "news-fusion"],
            stage="prompting",
        )
    ]
    for node_id, manifest in (
        ("fusion-image-asset-check", image),
        ("fusion-refine-asset-check", refine),
        ("fusion-i2va-asset-check", clip_video),
        ("fusion-fl2va-asset-check", bridge_video),
    ):
        nodes.append(
            ExecutionNode(
                node_id=node_id,
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": manifest.name},
                depends_on=["fusion-story"],
                tags=["assets", "news-fusion"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            )
        )

    # Beat 1 is drawn fresh; every later keyframe is a reference edit of the previous one.
    approved: list[str] = []  # the node whose output is each beat's approved keyframe
    for index in range(1, beats + 1):
        suffix = f"{index:02d}"
        keyframe = f"fusion-keyframe-{suffix}"
        fresh = index == 1
        nodes.append(
            ExecutionNode(
                node_id=keyframe,
                skill_name="media.image.generate_keyframe",
                inputs={
                    "workflow_name": image.name,
                    "identity_refine_workflow_name": refine.name,
                    "prompt_key": f"beat_{index}_keyframe_prompt",
                    "use_prior_frame": not fresh,
                    "width": width,
                    "height": height,
                    "image_count": 1,
                    "suffix": f"news_fusion_{suffix}",
                },
                depends_on=[
                    *([] if fresh else [approved[-1]]),
                    "fusion-story",
                    "fusion-image-asset-check",
                    *([] if fresh else ["fusion-refine-asset-check"]),
                ],
                tags=["render", "image", "continuity", "news-fusion"],
                tool_name="comfy.workflow.text_to_image" if fresh else "comfy.workflow.image_to_image",
                stage="render",
            )
        )
        source = keyframe
        if require_human_review:
            review = f"fusion-keyframe-review-{suffix}"
            nodes.append(
                ExecutionNode(
                    node_id=review,
                    skill_name="review.assets.select",
                    inputs={
                        "limit": 1,
                        "review_all_candidates": True,
                        "review_scope": "first_frame" if fresh else "last_frame",
                        "review_phase": f"fusion_keyframe_{suffix}",
                        "require_human_review": True,
                        "review_notes": f"News fusion keyframe {index} of {beats}: approve before any video renders.",
                    },
                    depends_on=[keyframe],
                    tags=["review", "news-fusion"],
                    stage="review",
                )
            )
            source = review
        approved.append(source)

    # Videos start only after every keyframe is approved, so no GPU time is spent on a rejected picture.
    ready = ["fusion-story", "fusion-i2va-asset-check", "fusion-fl2va-asset-check", *approved]
    video_order: list[str] = []
    tails: list[str] = []
    for index in range(1, beats + 1):
        suffix = f"{index:02d}"
        clip = f"fusion-clip-{suffix}"
        nodes.append(
            ExecutionNode(
                node_id=clip,
                skill_name="news_fusion.render_clip",
                inputs={
                    "kind": "clip",
                    "workflow_name": clip_video.name,
                    "index": index - 1,
                    "story_node": "fusion-story",
                    "anchor_nodes": {"first": approved[index - 1]},
                    "width": width,
                    "height": height,
                    "length": CLIP_LENGTH,
                },
                depends_on=list(ready),
                tags=["render", "video", "news-fusion"],
                tool_name="comfy.render_h3",
                stage="render",
            )
        )
        video_order.append(clip)
        if index == beats:
            break
        tail = f"fusion-tail-{suffix}"
        nodes.append(
            ExecutionNode(
                node_id=tail,
                skill_name="media.video.extract_last_frame",
                inputs={"output_name": f"fusion_tail_{suffix}.png", "source_video_node": clip},
                depends_on=[clip],
                tags=["quality", "continuity", "tail", "news-fusion"],
                tool_name="media.extract_last_frame",
                stage="quality",
            )
        )
        tails.append(tail)
        # The seam: Klein paints the character halfway between the clip's real last frame and the next
        # scene, so each H3 half crosses only half the gap. It needs the rendered tail, so it cannot be
        # approved before the videos start; the human judges it in the finished video.
        bridge_frame = f"fusion-bridge-frame-{suffix}"
        nodes.append(
            ExecutionNode(
                node_id=bridge_frame,
                skill_name="media.image.generate_keyframe",
                inputs={
                    "workflow_name": image.name,
                    "identity_refine_workflow_name": refine.name,
                    "prompt_key": f"bridge_{index}_frame_prompt",
                    "use_prior_frame": True,
                    "width": width,
                    "height": height,
                    "image_count": 1,
                    "suffix": f"news_fusion_bridge_{suffix}",
                },
                # The tail goes last: the keyframe skill reads the newest dependency first.
                depends_on=["fusion-story", "fusion-image-asset-check", "fusion-refine-asset-check", tail],
                tags=["render", "image", "continuity", "bridge", "news-fusion"],
                tool_name="comfy.workflow.image_to_image",
                stage="render",
            )
        )
        for half, first, last in (
            ("in", tail, bridge_frame),
            ("out", bridge_frame, approved[index]),
        ):
            bridge = f"fusion-bridge-{half}-{suffix}"
            nodes.append(
                ExecutionNode(
                    node_id=bridge,
                    skill_name="news_fusion.render_clip",
                    inputs={
                        "kind": f"bridge_{half}",
                        "workflow_name": bridge_video.name,
                        "index": index - 1,
                        "story_node": "fusion-story",
                        "anchor_nodes": {"first": first, "last": last},
                        "width": width,
                        "height": height,
                        "length": BRIDGE_LENGTH,
                    },
                    depends_on=[*ready, bridge_frame],
                    tags=["render", "video", "bridge", "news-fusion"],
                    tool_name="comfy.render_h3",
                    stage="render",
                )
            )
            video_order.append(bridge)

    nodes.append(
        ExecutionNode(
            node_id="concat-final-video",
            skill_name="media.video.concat",
            inputs={"method": "demuxer", "video_node_ids": list(video_order)},
            depends_on=list(video_order),
            tags=["package", "video", "news-fusion"],
            tool_name="media.concat_videos",
            stage="package",
        )
    )
    # No NVIDIA VFX pass runs here, so the QA frame rate stays the H3 native 24 fps.
    qa_inputs = {
        "expected_fps": 24,
        "minimum_duration": MIN_VIDEO_DURATION_SECONDS,
        "duration_tolerance": 0.6,
        "require_audio": True,
        "require_stereo_audio": True,
        "analyze_audio": True,
        "frame_count": 8,
        "columns": 4,
        "scale_width": 480,
    }
    nodes.extend(
        [
            ExecutionNode(
                node_id="longvideo-video-qa",
                skill_name="media.video.inspect",
                inputs=qa_inputs,
                depends_on=["concat-final-video"],
                tags=["quality", "technical-qa", "video", "news-fusion"],
                tool_name="media.video_qa",
                stage="quality",
            ),
            ExecutionNode(
                node_id="preview-gif",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=["concat-final-video", "longvideo-video-qa"],
                tags=["preview", "gif", "news-fusion"],
                tool_name="media.video_to_gif",
                stage="package",
            ),
            ExecutionNode(
                node_id="collect-longvideo-outputs",
                skill_name="agent.output.collect",
                inputs={
                    "keys": ["saved_files", "video_path", "audio_path", "gif_path", "frame_path"],
                    "preferred_video_node": "concat-final-video",
                },
                depends_on=["concat-final-video", "longvideo-video-qa", "preview-gif", *video_order],
                tags=["artifact", "summary", "news-fusion"],
                stage="package",
            ),
        ]
    )
    return ExecutionPlan(
        goal=goal,
        workflow_name=clip_video.name,
        nodes=nodes,
        metadata={
            "recipe": "news_fusion_story",
            "selected_workflow": clip_video.name,
            "keyframe_workflow": image.name,
            "bridge_workflow": bridge_video.name,
            "bridge_frame_workflow": refine.name,
            "required_assets": [
                asset.to_dict()
                for manifest in (image, refine, clip_video, bridge_video)
                for asset in manifest.required_assets
            ],
            "news_fusion": {
                "beat_count": beats,
                "width": width,
                "height": height,
                "clip_length": CLIP_LENGTH,
                "bridge_length": BRIDGE_LENGTH,
                "require_human_review": require_human_review,
                "backend": "comfyui",
            },
            "graph_overview": [node.node_id for node in nodes],
        },
        description=f"News fusion story: {beats} keyframe beats joined by {beats - 1} seams, each a Klein frame between two bridge clips",
    )
