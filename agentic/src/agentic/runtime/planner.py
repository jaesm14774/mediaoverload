from __future__ import annotations

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest
from agentic.assets.registry import AssetRegistry, WorkflowManifest
from agentic.runtime.creativity import IdeaDirector


class TaskPlanner:
    DEFAULT_IMAGE_WORKFLOWS = ("nova_model_plus_z_image_anime", "nova-anime-xl", "anima_anime")
    DEFAULT_REFINE_WORKFLOWS = ("image_to_image", "z_image_i2i_anime")
    DEFAULT_UPSCALE_WORKFLOWS = ("Tile Upscaler SDXL",)
    DEFAULT_I2V_WORKFLOWS = ("wan2.2_gguf_i2v", "wan2.2_gguf_i2v_audio")

    def __init__(self, asset_registry: AssetRegistry, idea_director: IdeaDirector | None = None) -> None:
        self.asset_registry = asset_registry
        self.idea_director = idea_director or IdeaDirector()

    def _preferred_manifest(self, *workflow_names: str) -> WorkflowManifest:
        for workflow_name in workflow_names:
            try:
                return self.asset_registry.get_manifest(workflow_name)
            except KeyError:
                continue
        requested = ", ".join(workflow_names)
        raise KeyError(f"None of the preferred workflow manifests are available: {requested}")

    def _manifest_from_goal_constraints(
        self,
        goal: GoalRequest,
        *workflow_names: str,
        constraint_keys: tuple[str, ...] = ("workflow_name",),
        allowed_media_types: set[str] | None = None,
    ) -> WorkflowManifest:
        for key in constraint_keys:
            selected_name = str(goal.constraints.get(key) or "").strip()
            if not selected_name:
                continue
            try:
                manifest = self.asset_registry.get_manifest(selected_name)
            except KeyError:
                continue
            if allowed_media_types and not set(manifest.media_types).intersection(allowed_media_types):
                continue
            return manifest
        return self._preferred_manifest(*workflow_names)

    def _pick_goal_workflow(self, goal: GoalRequest) -> WorkflowManifest:
        selected_name = str(goal.constraints.get("workflow_name") or "").strip()
        if selected_name:
            try:
                manifest = self.asset_registry.get_manifest(selected_name)
            except KeyError:
                manifest = None
            if manifest and goal.media_type in manifest.media_types:
                return manifest
        return self.asset_registry.pick_workflow(goal.media_type)

    @staticmethod
    def _constraint_int(goal: GoalRequest, key: str, default: int) -> int:
        value = goal.constraints.get(key)
        if value in {None, ""}:
            return default
        return max(1, int(value))

    def create_goal(
        self,
        prompt: str,
        media_type: str,
        duration_seconds: int,
        style: str,
        auto_download_assets: bool,
        constraints: dict | None = None,
    ) -> GoalRequest:
        return GoalRequest(
            prompt=prompt,
            media_type=media_type,
            duration_seconds=duration_seconds,
            style=style,
            auto_download_assets=auto_download_assets,
            constraints=constraints or {},
        )

    def build_plan(self, goal: GoalRequest) -> ExecutionPlan:
        if goal.media_type == "publish_review":
            return self._build_publish_review_plan(goal)
        if goal.media_type == "animated_sticker":
            return self._build_animated_sticker_plan(goal)
        if goal.media_type == "carousel":
            return self._build_carousel_plan(goal)
        if goal.media_type == "sticker_pack":
            return self._build_sticker_pack_plan(goal)
        if goal.media_type == "text2img2img":
            return self._build_text2img2img_plan(goal)
        if goal.media_type == "text2video":
            return self._build_text2video_plan(goal)
        if goal.media_type == "text2img2video":
            return self._build_text2img2video_plan(goal)
        if goal.media_type == "video_narrate":
            return self._build_video_narrate_plan(goal)
        workflow_manifest = self._pick_goal_workflow(goal)
        if goal.media_type == "storyboard":
            return self._build_storyboard_plan(goal, workflow_manifest)
        if goal.media_type == "image":
            return self._build_image_plan(goal, workflow_manifest)
        if goal.media_type in {"image_refine", "image_upscale", "image_to_video"}:
            return self._build_comfy_primitive_plan(goal, workflow_manifest)
        return self._build_long_video_plan(goal, workflow_manifest)

    @staticmethod
    def _review_loop_enabled(goal: GoalRequest) -> bool:
        if bool(goal.constraints.get("enable_review_loop", False)):
            return True
        review_notes = str(goal.constraints.get("review_notes", "") or "").strip()
        return bool(review_notes)

    @staticmethod
    def _review_selection_limit(goal: GoalRequest, default: int = 3) -> int:
        return int(goal.constraints.get("selection_limit") or goal.constraints.get("review_selection_limit") or default)

    @staticmethod
    def _stage_review_enabled(goal: GoalRequest) -> bool:
        return bool(goal.constraints.get("enable_stage_review", False))

    def _build_long_video_plan(self, goal: GoalRequest, workflow_manifest: WorkflowManifest) -> ExecutionPlan:
        segment_count = self._constraint_int(goal, "segment_count", max(2, goal.duration_seconds // 10))
        use_tts = bool(goal.constraints.get("use_tts", False))
        review_loop_enabled = self._review_loop_enabled(goal)
        stage_review_enabled = self._stage_review_enabled(goal)
        review_notes = str(goal.constraints.get("review_notes", "") or "")
        selection_limit = self._review_selection_limit(goal, default=self._constraint_int(goal, "review_selection_limit", 3))
        idea_variants = self.idea_director.generate_variations(goal)
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        transition_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_REFINE_WORKFLOWS,
            constraint_keys=("transition_workflow_name", "refine_workflow_name"),
            allowed_media_types={"image_refine"},
        )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_I2V_WORKFLOWS,
            constraint_keys=("video_workflow_name",),
            allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
        )
        image_count = self._constraint_int(goal, "image_count", 1)
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": idea_variants,
                },
                tags=["creative"],
            ),
            ExecutionNode(
                node_id="script-plan",
                skill_name="agent.story.segment",
                inputs={"segment_count": segment_count, "duration_seconds": goal.duration_seconds, "tone": "playful cinematic escalation"},
                depends_on=["idea-brief"],
                tags=["story"],
            ),
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": image_manifest.name,
                    "auto_download": goal.auto_download_assets,
                },
                depends_on=["script-plan"],
                tags=["assets", "image"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id="transition-asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": transition_manifest.name,
                    "auto_download": goal.auto_download_assets,
                },
                depends_on=["script-plan"],
                tags=["assets", "transition"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id="video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": video_manifest.name,
                    "auto_download": goal.auto_download_assets,
                },
                depends_on=["script-plan"],
                tags=["assets", "video"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
        ]

        segment_video_nodes: list[str] = []
        tts_nodes: list[str] = []
        previous_tail_node: str | None = None

        for index in range(segment_count):
            node_suffix = f"{index + 1:02d}"
            segment_prompt_node = f"segment-prompt-{node_suffix}"
            segment_frame_node = f"segment-frame-{node_suffix}"
            segment_video_node = f"segment-video-{node_suffix}"
            frame_extract_node = f"segment-tail-{node_suffix}"
            prompt_dependencies = ["script-plan", "idea-brief"]
            if previous_tail_node:
                prompt_dependencies.append(previous_tail_node)
            nodes.append(
                ExecutionNode(
                    node_id=segment_prompt_node,
                    skill_name="agent.segment.prepare",
                    inputs={"segment_index": index},
                    depends_on=prompt_dependencies,
                    tags=["story", "segment"],
                    stage="prompting",
                )
            )
            frame_dependencies = [segment_prompt_node, "image-asset-check", "transition-asset-check"]
            if previous_tail_node:
                frame_dependencies.append(previous_tail_node)
            nodes.append(
                ExecutionNode(
                    node_id=segment_frame_node,
                    skill_name="media.image.generate_keyframe",
                    inputs={
                        "workflow_name": image_manifest.name,
                        "segment_index": index,
                        "width": image_manifest.recommended_defaults.get("width", 1024),
                        "height": image_manifest.recommended_defaults.get("height", 1024),
                        "image_count": image_count,
                    },
                    depends_on=frame_dependencies,
                    tags=["render", "image", "segment"],
                    tool_name="comfy.workflow.text_to_image",
                    stage="render",
                )
            )
            segment_video_dependencies = [segment_prompt_node, segment_frame_node, "video-asset-check"]
            if stage_review_enabled and index == 0:
                stage_review_node = f"stage-review-{node_suffix}"
                nodes.append(
                    ExecutionNode(
                        node_id=stage_review_node,
                        skill_name="review.assets.select",
                        inputs={"limit": selection_limit, "review_notes": review_notes},
                        depends_on=[segment_frame_node],
                        tags=["review", "stage", "segment"],
                        stage="review",
                    )
                )
                segment_video_dependencies = [segment_prompt_node, stage_review_node, "video-asset-check"]
            nodes.append(
                ExecutionNode(
                    node_id=segment_video_node,
                    skill_name="media.image.animate",
                    inputs={
                        "workflow_name": video_manifest.name,
                        "segment_index": index,
                        "video_count": self._constraint_int(goal, "video_count", 1),
                    },
                    depends_on=segment_video_dependencies,
                    tags=["render", "video", "segment"],
                    tool_name="comfy.workflow.image_to_video",
                    stage="render",
                )
            )
            nodes.append(
                ExecutionNode(
                    node_id=frame_extract_node,
                    skill_name="media.video.extract_last_frame",
                    inputs={"segment_index": index},
                    depends_on=[segment_video_node],
                    tags=["frame", "segment"],
                    tool_name="media.extract_last_frame",
                    stage="package",
                )
            )
            segment_video_nodes.append(segment_video_node)
            previous_tail_node = frame_extract_node

            if use_tts:
                tts_node = f"tts-audio-{node_suffix}"
                nodes.append(
                    ExecutionNode(
                        node_id=tts_node,
                        skill_name="media.audio.narrate",
                        inputs={"segment_index": index, "voice": "en-US-AriaNeural", "output_name": f"segment_{node_suffix}.mp3"},
                        depends_on=[segment_prompt_node],
                        tags=["audio", "segment"],
                        tool_name="audio.generate_tts_real",
                        stage="audio",
                    )
                )
                tts_nodes.append(tts_node)

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
            nodes.append(
                ExecutionNode(
                    node_id="concat-final-audio",
                    skill_name="media.audio.concat",
                    inputs={},
                    depends_on=tts_nodes,
                    tags=["package", "audio"],
                    tool_name="audio.concat_tracks",
                    stage="audio",
                )
            )
            nodes.append(
                ExecutionNode(
                    node_id="mux-final-video",
                    skill_name="media.video.merge_audio",
                    inputs={},
                    depends_on=["concat-final-video", "concat-final-audio"],
                    tags=["package", "mux"],
                    tool_name="media.merge_audio_video",
                    stage="package",
                )
            )
            preview_dependency = "mux-final-video"

        preview_gif_node = "preview-gif"
        nodes.append(
            ExecutionNode(
                node_id=preview_gif_node,
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=[preview_dependency],
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
                inputs={"keys": ["saved_files", "video_path", "audio_path", "gif_path", "frame_path"]},
                depends_on=[preview_dependency, preview_gif_node, *segment_video_nodes, *tts_nodes],
                tags=["artifact", "summary"],
                stage="package",
            )
        )

        if review_loop_enabled:
            nodes.append(
                ExecutionNode(
                    node_id="review-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_notes": review_notes},
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

            review_segment_video_nodes: list[str] = []
            review_tts_nodes: list[str] = []
            review_previous_tail_node: str | None = None
            for index in range(segment_count):
                node_suffix = f"{index + 1:02d}"
                segment_prompt_node = f"review-segment-prompt-{node_suffix}"
                segment_frame_node = f"review-segment-frame-{node_suffix}"
                segment_video_node = f"review-segment-video-{node_suffix}"
                frame_extract_node = f"review-segment-tail-{node_suffix}"
                prompt_dependencies = ["script-plan", "idea-brief", "review-refine-prompt"]
                if review_previous_tail_node:
                    prompt_dependencies.append(review_previous_tail_node)
                nodes.append(
                    ExecutionNode(
                        node_id=segment_prompt_node,
                        skill_name="agent.segment.prepare",
                        inputs={"segment_index": index},
                        depends_on=prompt_dependencies,
                        tags=["story", "segment", "retry"],
                        stage="prompting",
                    )
                )
                frame_dependencies = [segment_prompt_node, "image-asset-check", "transition-asset-check"]
                if review_previous_tail_node:
                    frame_dependencies.append(review_previous_tail_node)
                nodes.append(
                    ExecutionNode(
                        node_id=segment_frame_node,
                        skill_name="media.image.generate_keyframe",
                        inputs={
                            "workflow_name": image_manifest.name,
                            "segment_index": index,
                            "width": image_manifest.recommended_defaults.get("width", 1024),
                            "height": image_manifest.recommended_defaults.get("height", 1024),
                            "image_count": image_count,
                        },
                        depends_on=frame_dependencies,
                        tags=["render", "image", "segment", "retry"],
                        tool_name="comfy.workflow.text_to_image",
                        stage="render",
                    )
                )
                nodes.append(
                    ExecutionNode(
                        node_id=segment_video_node,
                        skill_name="media.image.animate",
                        inputs={
                            "workflow_name": video_manifest.name,
                            "segment_index": index,
                            "video_count": self._constraint_int(goal, "video_count", 1),
                        },
                        depends_on=[segment_prompt_node, segment_frame_node, "video-asset-check"],
                        tags=["render", "video", "segment", "retry"],
                        tool_name="comfy.workflow.image_to_video",
                        stage="render",
                    )
                )
                nodes.append(
                    ExecutionNode(
                        node_id=frame_extract_node,
                        skill_name="media.video.extract_last_frame",
                        inputs={"segment_index": index},
                        depends_on=[segment_video_node],
                        tags=["frame", "segment", "retry"],
                        tool_name="media.extract_last_frame",
                        stage="package",
                    )
                )
                review_segment_video_nodes.append(segment_video_node)
                review_previous_tail_node = frame_extract_node

                if use_tts:
                    tts_node = f"review-tts-audio-{node_suffix}"
                    nodes.append(
                        ExecutionNode(
                            node_id=tts_node,
                            skill_name="media.audio.narrate",
                            inputs={"segment_index": index, "voice": "en-US-AriaNeural", "output_name": f"review_segment_{node_suffix}.mp3"},
                            depends_on=[segment_prompt_node],
                            tags=["audio", "segment", "retry"],
                            tool_name="audio.generate_tts_real",
                            stage="audio",
                        )
                    )
                    review_tts_nodes.append(tts_node)

            review_preview_dependency = "review-concat-final-video"
            nodes.append(
                ExecutionNode(
                    node_id="review-concat-final-video",
                    skill_name="media.video.concat",
                    inputs={"method": "demuxer"},
                    depends_on=review_segment_video_nodes,
                    tags=["package", "video", "retry"],
                    tool_name="media.concat_videos",
                    stage="package",
                )
            )
            if use_tts:
                nodes.append(
                    ExecutionNode(
                        node_id="review-concat-final-audio",
                        skill_name="media.audio.concat",
                        inputs={},
                        depends_on=review_tts_nodes,
                        tags=["package", "audio", "retry"],
                        tool_name="audio.concat_tracks",
                        stage="audio",
                    )
                )
                nodes.append(
                    ExecutionNode(
                        node_id="review-mux-final-video",
                        skill_name="media.video.merge_audio",
                        inputs={},
                        depends_on=["review-concat-final-video", "review-concat-final-audio"],
                        tags=["package", "mux", "retry"],
                        tool_name="media.merge_audio_video",
                        stage="package",
                    )
                )
                review_preview_dependency = "review-mux-final-video"
            nodes.append(
                ExecutionNode(
                    node_id="review-preview-gif",
                    skill_name="media.video.gif_preview",
                    inputs={"fps": 12, "scale_width": 512},
                    depends_on=[review_preview_dependency],
                    tags=["preview", "gif", "retry"],
                    tool_name="media.video_to_gif",
                    stage="package",
                )
            )
            nodes.append(
                ExecutionNode(
                    node_id="review-collect-longvideo-outputs",
                    skill_name="agent.output.collect",
                    inputs={"keys": ["saved_files", "video_path", "audio_path", "gif_path", "frame_path"]},
                    depends_on=[review_preview_dependency, "review-preview-gif", *review_segment_video_nodes, *review_tts_nodes],
                    tags=["artifact", "summary", "retry"],
                    stage="package",
                )
            )
            nodes.append(
                ExecutionNode(
                    node_id="review-final-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_notes": review_notes},
                    depends_on=["review-collect-longvideo-outputs"],
                    tags=["review", "longvideo", "retry"],
                    stage="review",
                )
            )
        summary_dependencies = [collect_node]
        if review_loop_enabled:
            summary_dependencies.append("review-final-select")
        nodes.append(
            ExecutionNode(
                node_id="persist-longvideo-summary",
                skill_name="agent.summary.persist",
                inputs={"summary_name": "long_video_summary.json", "summary_scope": "long_video"},
                depends_on=summary_dependencies,
                tags=["artifact", "summary"],
                stage="package",
            )
        )
        metadata = {
            "segment_count": segment_count,
            "selected_workflow": workflow_manifest.name,
            "required_assets": [
                *[asset.to_dict() for asset in image_manifest.required_assets],
                *[asset.to_dict() for asset in transition_manifest.required_assets],
                *[asset.to_dict() for asset in video_manifest.required_assets],
            ],
            "graph_overview": [node.node_id for node in nodes],
            "idea_variants": idea_variants,
            "image_workflow": image_manifest.name,
            "transition_workflow": transition_manifest.name,
            "video_workflow": video_manifest.name,
            "use_tts": use_tts,
            "review_loop_enabled": review_loop_enabled,
            "review_notes": review_notes,
        }
        description = f"Long-video workflow '{workflow_manifest.name}' for goal '{goal.prompt}'"
        return ExecutionPlan(
            goal=goal,
            workflow_name="longvideo_real_v1",
            nodes=nodes,
            metadata=metadata,
            description=description,
        )

    def _build_video_narrate_plan(self, goal: GoalRequest) -> ExecutionPlan:
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
            ExecutionNode(
                node_id="gif-preview",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=["mux-video"],
                tags=["preview", "gif"],
                tool_name="media.video_to_gif",
                stage="package",
            ),
        ]
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

    def _build_text2img2video_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        upscale_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_UPSCALE_WORKFLOWS,
            constraint_keys=("upscale_workflow_name",),
            allowed_media_types={"image_upscale"},
        )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_I2V_WORKFLOWS,
            constraint_keys=("video_workflow_name",),
            allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
        )
        review_loop_enabled = self._review_loop_enabled(goal)
        stage_review_enabled = self._stage_review_enabled(goal)
        review_notes = str(goal.constraints.get("review_notes", "") or "")
        selection_limit = self._review_selection_limit(goal, default=self._constraint_int(goal, "review_selection_limit", 2))
        image_count = self._constraint_int(goal, "image_count", 1)
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="image.ensure_workflow",
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
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
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
                    "image_count": image_count,
                },
                depends_on=["idea-brief", "image-asset-check"],
                tags=["render", "image"],
                tool_name="comfy.workflow.text_to_image",
                stage="render",
            ),
            ExecutionNode(
                node_id="upscale-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": upscale_manifest.name, "auto_download": goal.auto_download_assets},
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
            ExecutionNode(
                node_id="video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": video_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["upscale-image"],
                tags=["assets"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
        ]
        if stage_review_enabled:
            nodes.append(
                ExecutionNode(
                    node_id="stage-review-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_notes": review_notes},
                    depends_on=["upscale-image"],
                    tags=["review", "stage"],
                    stage="review",
                )
            )
        animate_dependencies = ["idea-brief", "upscale-image", "video-asset-check"]
        if stage_review_enabled:
            animate_dependencies = ["idea-brief", "stage-review-select", "video-asset-check"]
        nodes.extend(
            [
            ExecutionNode(
                node_id="animate-video",
                skill_name="image.animate",
                inputs={
                    "workflow_name": video_manifest.name,
                    "video_count": self._constraint_int(goal, "video_count", 1),
                },
                depends_on=animate_dependencies,
                tags=["render", "video"],
                tool_name="comfy.workflow.image_to_video",
                stage="render",
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
        if review_loop_enabled:
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="review-select",
                        skill_name="review.assets.select",
                        inputs={"limit": selection_limit, "review_notes": review_notes},
                        depends_on=["render-image", "upscale-image", "animate-video", "gif-preview"],
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
                            "width": image_manifest.recommended_defaults.get("width", 1024),
                            "height": image_manifest.recommended_defaults.get("height", 1024),
                            "image_count": image_count,
                        },
                        depends_on=["review-refine-prompt", "image-asset-check"],
                        tags=["render", "image", "retry"],
                        tool_name="comfy.workflow.text_to_image",
                        stage="render",
                    ),
                    ExecutionNode(
                        node_id="review-upscale-image",
                        skill_name="image.upscale",
                        inputs={},
                        depends_on=["review-render-image", "upscale-asset-check"],
                        tags=["render", "upscale", "retry"],
                        tool_name="comfy.workflow.image_upscale",
                        stage="render",
                    ),
                    ExecutionNode(
                        node_id="review-animate-video",
                        skill_name="image.animate",
                        inputs={
                            "workflow_name": video_manifest.name,
                            "video_count": self._constraint_int(goal, "video_count", 1),
                        },
                        depends_on=["review-refine-prompt", "review-upscale-image", "video-asset-check"],
                        tags=["render", "video", "retry"],
                        tool_name="comfy.workflow.image_to_video",
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
                        depends_on=["review-render-image", "review-upscale-image", "review-animate-video", "review-gif-preview"],
                        tags=["review", "retry"],
                        stage="review",
                    ),
                ]
            )
        summary_dependencies = ["render-image", "upscale-image", "animate-video", "gif-preview"]
        if review_loop_enabled:
            summary_dependencies.append("review-final-select")
        nodes.append(
            ExecutionNode(
                node_id="persist-text2img2video-summary",
                skill_name="agent.summary.persist",
                inputs={"summary_name": "text2img2video_summary.json", "summary_scope": "text2img2video"},
                depends_on=summary_dependencies,
                tags=["artifact", "summary"],
                stage="package",
            )
        )
        metadata = {
            "selected_workflows": [image_manifest.name, upscale_manifest.name, video_manifest.name],
            "required_assets": [
                *[asset.to_dict() for asset in image_manifest.required_assets],
                *[asset.to_dict() for asset in upscale_manifest.required_assets],
                *[asset.to_dict() for asset in video_manifest.required_assets],
            ],
            "graph_overview": [node.node_id for node in nodes],
            "review_loop_enabled": review_loop_enabled,
            "review_notes": review_notes,
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="text2img2video_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Migrated text2img2video chain for goal '{goal.prompt}'",
        )

    def _build_storyboard_plan(self, goal: GoalRequest, workflow_manifest: WorkflowManifest) -> ExecutionPlan:
        segment_count = self._constraint_int(
            goal,
            "segment_count",
            int(workflow_manifest.recommended_defaults.get("segment_count", 4)),
        )
        frame_width = workflow_manifest.recommended_defaults.get("frame_width", 1280)
        frame_height = workflow_manifest.recommended_defaults.get("frame_height", 720)
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative"],
            ),
            ExecutionNode(
                node_id="script-plan",
                skill_name="agent.story.segment",
                inputs={"segment_count": segment_count, "tone": "clear visual beats"},
                depends_on=["idea-brief"],
                tags=["story"],
            ),
            ExecutionNode(
                node_id="asset-check",
                skill_name="story.ensure_workflow",
                inputs={"workflow_name": workflow_manifest.name},
                depends_on=["script-plan"],
                tags=["assets"],
            ),
            ExecutionNode(
                node_id="storyboard-frames",
                skill_name="story.render_frames",
                inputs={"frame_width": frame_width, "frame_height": frame_height},
                depends_on=["asset-check"],
                tags=["render"],
            ),
            ExecutionNode(
                node_id="storyboard-package",
                skill_name="story.package_outputs",
                inputs={},
                depends_on=["storyboard-frames"],
                tags=["artifact"],
            ),
        ]
        metadata = {
            "segment_count": segment_count,
            "selected_workflow": workflow_manifest.name,
            "required_assets": [],
            "graph_overview": [node.node_id for node in nodes],
            "frame_width": frame_width,
            "frame_height": frame_height,
        }
        description = f"Storyboard workflow '{workflow_manifest.name}' for goal '{goal.prompt}'"
        return ExecutionPlan(
            goal=goal,
            workflow_name=workflow_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=description,
        )

    def _build_image_plan(self, goal: GoalRequest, workflow_manifest: WorkflowManifest) -> ExecutionPlan:
        width = workflow_manifest.recommended_defaults.get("width", 1024)
        height = workflow_manifest.recommended_defaults.get("height", 1024)
        image_count = self._constraint_int(goal, "image_count", 1)
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative"],
            ),
            ExecutionNode(
                node_id="asset-check",
                skill_name="image.ensure_workflow",
                inputs={
                    "workflow_name": workflow_manifest.name,
                    "auto_download": goal.auto_download_assets,
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
        metadata = {
            "selected_workflow": workflow_manifest.name,
            "required_assets": [asset.to_dict() for asset in workflow_manifest.required_assets],
            "graph_overview": [node.node_id for node in nodes],
            "width": width,
            "height": height,
        }
        description = f"Image workflow '{workflow_manifest.name}' for goal '{goal.prompt}'"
        return ExecutionPlan(
            goal=goal,
            workflow_name=workflow_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=description,
        )

    def _build_comfy_primitive_plan(self, goal: GoalRequest, workflow_manifest: WorkflowManifest) -> ExecutionPlan:
        constraints = goal.constraints
        input_image_path = constraints.get("input_image_path")
        if goal.media_type in {"image_refine", "image_upscale", "image_to_video"} and not input_image_path:
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
                    inputs={"workflow_name": workflow_manifest.name, "auto_download": goal.auto_download_assets},
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
                    inputs={"workflow_name": workflow_manifest.name, "auto_download": goal.auto_download_assets},
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
                    inputs={"workflow_name": workflow_manifest.name, "auto_download": goal.auto_download_assets},
                    depends_on=["prompt-prepare"],
                    tags=["assets"],
                ),
                ExecutionNode(
                    node_id="i2v-render",
                    skill_name="media.image.animate",
                    inputs={
                        "image_path": input_image_path,
                        "prompt": constraints.get("text", "") or goal.prompt,
                    },
                    depends_on=["prompt-prepare", "asset-check"],
                    tags=["render", "video"],
                ),
            ]

        metadata = {
            "selected_workflow": workflow_manifest.name,
            "required_assets": [asset.to_dict() for asset in workflow_manifest.required_assets],
            "graph_overview": [node.node_id for node in nodes],
            "input_image_path": input_image_path,
        }
        description = f"Legacy workflow '{workflow_manifest.name}' for media_type '{goal.media_type}'"
        return ExecutionPlan(
            goal=goal,
            workflow_name=workflow_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=description,
        )

    def _build_text2img2img_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        refine_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_REFINE_WORKFLOWS,
            constraint_keys=("refine_workflow_name", "transition_workflow_name"),
            allowed_media_types={"image_refine"},
        )
        image_count = self._constraint_int(goal, "image_count", 1)
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="image.ensure_workflow",
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["idea-brief"],
                tags=["assets"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-image",
                skill_name="image.render",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
                    "image_count": image_count,
                },
                depends_on=["idea-brief", "image-asset-check"],
                tags=["render", "image"],
                stage="render",
            ),
            ExecutionNode(
                node_id="refine-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": refine_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["render-image"],
                tags=["assets"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="refine-image",
                skill_name="media.image.refine",
                inputs={},
                depends_on=["idea-brief", "render-image", "refine-asset-check"],
                tags=["render", "image", "refine"],
                stage="render",
            ),
        ]
        metadata = {
            "selected_workflows": [image_manifest.name, refine_manifest.name],
            "required_assets": [
                *[asset.to_dict() for asset in image_manifest.required_assets],
                *[asset.to_dict() for asset in refine_manifest.required_assets],
            ],
            "graph_overview": [node.node_id for node in nodes],
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="text2img2img_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Agentic text2img2img chain for goal '{goal.prompt}'",
        )

    def _build_text2video_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_I2V_WORKFLOWS,
            constraint_keys=("video_workflow_name",),
            allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
        )
        review_loop_enabled = self._review_loop_enabled(goal)
        stage_review_enabled = self._stage_review_enabled(goal)
        review_notes = str(goal.constraints.get("review_notes", "") or "")
        selection_limit = self._review_selection_limit(goal, default=self._constraint_int(goal, "review_selection_limit", 2))
        image_count = self._constraint_int(goal, "image_count", 1)
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="image.ensure_workflow",
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["idea-brief"],
                tags=["assets"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-image",
                skill_name="image.render",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
                    "image_count": image_count,
                },
                depends_on=["idea-brief", "image-asset-check"],
                tags=["render", "image"],
                stage="render",
            ),
            ExecutionNode(
                node_id="video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": video_manifest.name, "auto_download": goal.auto_download_assets},
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
                    inputs={"limit": selection_limit, "review_notes": review_notes},
                    depends_on=["render-image"],
                    tags=["review", "stage"],
                    stage="review",
                )
            )
        animate_dependencies = ["idea-brief", "render-image", "video-asset-check"]
        if stage_review_enabled:
            animate_dependencies = ["idea-brief", "stage-review-select", "video-asset-check"]
        nodes.extend(
            [
            ExecutionNode(
                node_id="animate-video",
                skill_name="media.image.animate",
                inputs={
                    "workflow_name": video_manifest.name,
                    "video_count": self._constraint_int(goal, "video_count", 1),
                },
                depends_on=animate_dependencies,
                tags=["render", "video"],
                stage="render",
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
        if review_loop_enabled:
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="review-select",
                        skill_name="review.assets.select",
                        inputs={"limit": selection_limit, "review_notes": review_notes},
                        depends_on=["render-image", "animate-video", "gif-preview"],
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
                            "width": image_manifest.recommended_defaults.get("width", 1024),
                            "height": image_manifest.recommended_defaults.get("height", 1024),
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
                            "video_count": self._constraint_int(goal, "video_count", 1),
                        },
                        depends_on=["review-refine-prompt", "review-render-image", "video-asset-check"],
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
                        depends_on=["review-render-image", "review-animate-video", "review-gif-preview"],
                        tags=["review", "retry"],
                        stage="review",
                    ),
                ]
            )
        summary_dependencies = ["render-image", "animate-video", "gif-preview"]
        if review_loop_enabled:
            summary_dependencies.append("review-final-select")
        nodes.append(
            ExecutionNode(
                node_id="persist-text2video-summary",
                skill_name="agent.summary.persist",
                inputs={"summary_name": "text2video_summary.json", "summary_scope": "text2video"},
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
            "review_notes": review_notes,
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="text2video_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Agentic text2video chain for goal '{goal.prompt}'",
        )

    def _build_sticker_pack_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        character = goal.constraints.get("character", "")
        expression_count = self._constraint_int(goal, "sticker_expression_count", 8)
        images_per_prompt = self._constraint_int(goal, "images_per_prompt", 1)
        nodes = [
            ExecutionNode(
                node_id="sticker-expressions",
                skill_name="agent.sticker.expressions",
                inputs={"prompt": goal.prompt, "character": character, "expression_count": expression_count},
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
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["sticker-prompts"],
                tags=["assets", "sticker"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-stickers",
                skill_name="media.image.render_batch",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
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
            "required_assets": [asset.to_dict() for asset in image_manifest.required_assets],
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

    def _build_animated_sticker_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_I2V_WORKFLOWS,
            constraint_keys=("video_workflow_name",),
            allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
        )
        character = goal.constraints.get("character", "")
        review_loop_enabled = self._review_loop_enabled(goal)
        review_notes = str(goal.constraints.get("review_notes", "") or "")
        selection_limit = self._review_selection_limit(goal, default=self._constraint_int(goal, "review_selection_limit", 2))
        expression_count = self._constraint_int(goal, "sticker_expression_count", 8)
        images_per_prompt = self._constraint_int(goal, "images_per_prompt", 1)
        nodes = [
            ExecutionNode(
                node_id="sticker-expressions",
                skill_name="agent.sticker.expressions",
                inputs={"prompt": goal.prompt, "character": character, "expression_count": expression_count},
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
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
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
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
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
                inputs={"workflow_name": video_manifest.name, "auto_download": goal.auto_download_assets},
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
                    "video_count": self._constraint_int(goal, "video_count", 1),
                },
                depends_on=["motion-prompt", "video-asset-check"],
                tags=["render", "video", "sticker"],
                tool_name="comfy.workflow.image_to_video",
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
                        depends_on=["render-stickers", "animate-sticker", "gif-preview", "package-animated-sticker"],
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
                            "video_count": self._constraint_int(goal, "video_count", 1),
                        },
                        depends_on=["review-refine-prompt", "render-stickers", "video-asset-check"],
                        tags=["render", "video", "sticker", "retry"],
                        tool_name="comfy.workflow.image_to_video",
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

    def _build_carousel_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        slide_count = self._constraint_int(goal, "segment_count", max(3, goal.duration_seconds // 10))
        images_per_prompt = self._constraint_int(goal, "images_per_prompt", 1)
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
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
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["carousel-prompts"],
                tags=["assets", "carousel"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-carousel",
                skill_name="media.image.render_batch",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
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
            "required_assets": [asset.to_dict() for asset in image_manifest.required_assets],
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

    def _build_publish_review_plan(self, goal: GoalRequest) -> ExecutionPlan:
        platforms = goal.constraints.get("platforms") or []
        dry_run = bool(goal.constraints.get("dry_run", True))
        selection_limit = int(goal.constraints.get("selection_limit") or 4)
        nodes = [
            ExecutionNode(
                node_id="ingest-media",
                skill_name="publish.media.ingest",
                inputs={},
                tags=["publish", "review"],
                stage="ingest",
            ),
            ExecutionNode(
                node_id="review-select",
                skill_name="review.assets.select",
                inputs={"limit": selection_limit},
                depends_on=["ingest-media"],
                tags=["review", "publish"],
                stage="review",
            ),
            ExecutionNode(
                node_id="process-media",
                skill_name="publish.media.process",
                inputs={"output_dir": goal.constraints.get("output_dir")},
                depends_on=["review-select"],
                tags=["publish", "media"],
                tool_name="publish.process_media",
                stage="package",
            ),
            ExecutionNode(
                node_id="prepare-caption",
                skill_name="publish.caption.prepare",
                inputs={
                    "prefix": goal.constraints.get("caption_prefix", ""),
                    "hashtags": goal.constraints.get("hashtags", []),
                    "platforms": platforms,
                },
                depends_on=["review-select", "process-media"],
                tags=["publish", "caption"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="dispatch-publish",
                skill_name="publish.social.dispatch",
                inputs={
                    "platforms": platforms,
                    "platform_configs": goal.constraints.get("platform_configs", {}),
                    "additional_params": goal.constraints.get("additional_params", {}),
                    "dry_run": dry_run,
                },
                depends_on=["process-media", "prepare-caption"],
                tags=["publish", "dispatch"],
                tool_name="publish.social",
                stage="publish",
            ),
            ExecutionNode(
                node_id="persist-publish-review-summary",
                skill_name="agent.summary.persist",
                inputs={"summary_name": "publish_review_summary.json", "summary_scope": "publish_review"},
                depends_on=["review-select", "process-media", "prepare-caption", "dispatch-publish"],
                tags=["artifact", "summary", "publish"],
                stage="package",
            ),
        ]
        metadata = {
            "graph_overview": [node.node_id for node in nodes],
            "platforms": list(platforms),
            "dry_run": dry_run,
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="publish_review_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Agentic publish/review chain for goal '{goal.prompt}'",
        )

