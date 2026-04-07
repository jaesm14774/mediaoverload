from __future__ import annotations

from pathlib import Path

from agentic.assets.registry import AssetRegistry, WorkflowManifest
from agentic.runtime.registry import ToolRegistry


class BuiltinTools:
    def __init__(self, asset_registry: AssetRegistry) -> None:
        self.asset_registry = asset_registry

    def load_manifest(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        manifest = self.asset_registry.get_manifest(workflow_name)
        return {"manifest": manifest.to_dict()}

    def materialize_workflow(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        manifest = self.asset_registry.get_manifest(workflow_name)
        workflow_path = self.asset_registry.materialize_workflow(manifest)
        template = self.asset_registry.load_workflow_template(manifest)
        return {
            "workflow_name": manifest.name,
            "workflow_path": str(workflow_path),
            "template_preview": template.get("title") if template else None,
        }

    def ensure_workflow_ready(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        auto_download = bool(payload.get("auto_download", False))
        return self.asset_registry.ensure_workflow_ready(workflow_name, auto_download)

    def render_candidate_frames(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        manifest = self.asset_registry.get_manifest(workflow_name)
        default_batch = manifest.recommended_defaults.get("candidate_batch_size", 3)
        batch_size = int(payload.get("batch_size", default_batch))
        frames = [f"mock://{workflow_name}/candidate_frame_{index}.png" for index in range(batch_size)]
        return {"candidate_frames": frames, "selected_frame": frames[0] if frames else None}

    def render_long_video(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        segments = payload.get("segments", [])
        if not segments:
            segments = [
                {"segment_id": "segment-1", "video_path": f"mock://{workflow_name}/segment_1.mp4"},
                {"segment_id": "segment-2", "video_path": f"mock://{workflow_name}/segment_2.mp4"},
                {"segment_id": "segment-3", "video_path": f"mock://{workflow_name}/segment_3.mp4"},
            ]
        return {"rendered_segments": segments}

    def generate_transition_anchors(self, payload: dict[str, object]) -> dict[str, object]:
        segment_count = int(payload.get("segment_count", 0))
        anchor_per_segment = int(payload.get("anchor_per_segment", 2))
        anchors: list[str] = []
        for segment_index in range(segment_count):
            for anchor_index in range(anchor_per_segment):
                anchors.append(f"mock://anchors/segment_{segment_index+1}_anchor_{anchor_index+1}.png")
        return {"anchors": anchors, "anchor_per_segment": anchor_per_segment}

    def blend_segments(self, payload: dict[str, object]) -> dict[str, object]:
        segments = payload.get("segments", [])
        anchors = payload.get("anchors", [])
        plan = [
            {
                "from_segment": seg["segment_id"],
                "to_segment": segments[idx + 1]["segment_id"] if idx + 1 < len(segments) else None,
                "anchor_frame": anchors[idx] if idx < len(anchors) else None,
            }
            for idx, seg in enumerate(segments[:-1])
        ]
        return {"stitched_segments": segments, "transition_plan": plan}

    def generate_tts(self, payload: dict[str, object]) -> dict[str, object]:
        voice = str(payload.get("voice", "en-US-AriaNeural"))
        segment_count = int(payload.get("segment_count", 1))
        audio_tracks = [f"mock://tts/{voice}/segment_{index + 1}.mp3" for index in range(segment_count)]
        return {"voice": voice, "audio_tracks": audio_tracks}

    def concat_video(self, payload: dict[str, object]) -> dict[str, object]:
        segment_count = int(payload.get("segment_count", 0))
        stitched = payload.get("stitched_segments", [])
        return {
            "final_video": "mock://final/long_video.mp4",
            "segment_count": segment_count,
            "stitched_segments": stitched,
        }

    def score_output(self, payload: dict[str, object]) -> dict[str, object]:
        segment_count = int(payload.get("segment_count", 0))
        score = min(0.99, 0.72 + segment_count * 0.05)
        return {"quality_score": round(score, 2), "status": "accepted" if score >= 0.8 else "retry"}


def register_builtin_tools(tool_registry: ToolRegistry, asset_registry: AssetRegistry) -> None:
    tools = BuiltinTools(asset_registry)
    tool_registry.register("workflow.load_manifest", tools.load_manifest, "Load workflow manifest")
    tool_registry.register("workflow.materialize", tools.materialize_workflow, "Materialize workflow template path")
    tool_registry.register("asset.ensure_workflow_ready", tools.ensure_workflow_ready, "Prepare workflow assets")
    tool_registry.register("comfy.render_candidate_frames", tools.render_candidate_frames, "Render candidate first frames")
    tool_registry.register("comfy.render_long_video", tools.render_long_video, "Render long video segments")
    tool_registry.register("transition.generate_anchors", tools.generate_transition_anchors, "Generate transition anchors")
    tool_registry.register("transition.blend_segments", tools.blend_segments, "Blend transitions between segments")
    tool_registry.register("audio.generate_tts", tools.generate_tts, "Generate narration tracks")
    tool_registry.register("media.concat_video", tools.concat_video, "Concatenate video segments")
    tool_registry.register("evaluation.score_output", tools.score_output, "Score final output")

