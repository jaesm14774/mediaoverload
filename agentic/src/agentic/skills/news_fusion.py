"""Skills for the news + character fusion story: write it, then render clips and bridges."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agentic.minimax_prompting import format_h3_video_prompt
from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.news_grounding import is_usable_news_selection
from agentic.runtime.prompt_tasks.news_fusion import generate_news_fusion_story
from agentic.runtime.prompting import include_role_description, selected_role_description
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.skills.shared import build_run_dir
from agentic.tools.context_services import NewsContextService

FRAME_KEYS = ("frame_path", "selected_frame_path", "selected_assets", "saved_files")
# Each seam is two fl2va halves: clip end -> Klein connecting frame -> next keyframe.
BRIDGE_HALVES = {"bridge_in": "first_scene", "bridge_out": "second_scene"}
LOCKED_CAMERA ="Cute 2D cartoon look; the camera never moves; constant lively body motion until the last frame."


def _first_path(outputs: Any) -> str:
    if not isinstance(outputs, dict):
        return ""
    for key in FRAME_KEYS:
        value = outputs.get(key)
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
        if isinstance(value, str) and value:
            return value
    return ""


class NewsFusionSkills:
    def __init__(
        self,
        tools: ToolRegistry,
        output_root: Path,
        llm_engine: LLMPromptEngine | None = None,
        news_service: NewsContextService | None = None,
    ) -> None:
        self.tools = tools
        self.output_root = Path(output_root)
        self._llm_engine = llm_engine
        self.news_service = news_service or NewsContextService()

    @property
    def llm_engine(self) -> LLMPromptEngine:
        if self._llm_engine is None:
            self._llm_engine = LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
        return self._llm_engine

    def prepare_story(self, context: SkillContext) -> SkillResult:
        goal = context.plan.goal
        constraints = goal.constraints
        beat_count = int(context.node.inputs["beat_count"])
        news_context = dict(constraints.get("news_context") or {})
        if news_context.get("error"):
            raise RuntimeError(f"News fusion news context failed: {news_context['error']}")
        if not is_usable_news_selection(str(news_context.get("title") or ""), str(news_context.get("keyword") or "")):
            selected = self.news_service.select_news(
                random_inspiration=True, exclude_categories=list(NewsContextService.TONE_FIT_EXCLUDE_CATEGORIES)
            )
            if selected is None:
                raise RuntimeError("News fusion requires a selectable news item; none was available")
            news_context = selected.to_dict()
        story = generate_news_fusion_story(
            self.llm_engine.client,
            character=str(constraints.get("character") or "the protagonist"),
            character_profile=selected_role_description(goal),
            news_context=news_context,
            beat_count=beat_count,
            style=goal.style,
        )
        outputs: dict[str, Any] = {
            **story,
            "news_context": news_context,
            "beat_count": beat_count,
        }
        for index, beat in enumerate(story["beats"], start=1):
            outputs[f"beat_{index}_keyframe_prompt"] = beat["keyframe_prompt"]
        for index, bridge in enumerate(story["bridges"], start=1):
            outputs[f"bridge_{index}_frame_prompt"] = bridge["frame_prompt"]
        return SkillResult(
            status="success",
            outputs=outputs,
            metrics={"beat_count": beat_count},
            logs=[f"Wrote a {beat_count}-beat fusion story from news: {news_context.get('title', '')}"],
        )

    def render_clip(self, context: SkillContext) -> SkillResult:
        inputs = context.node.inputs
        kind = str(inputs["kind"])
        index = int(inputs["index"])
        story = context.state.node_outputs.get(str(inputs["story_node"]))
        if not isinstance(story, dict):
            raise RuntimeError("Fusion clip rendering needs the written story")
        if kind == "clip":
            source = story["beats"][index]
            scene, soundscape, music = source["clip_scene"], source["soundscape"], source["music"]
        elif kind in BRIDGE_HALVES:
            source = story["bridges"][index]
            scene, soundscape, music = source[BRIDGE_HALVES[kind]], source["soundscape"], source["music"]
        else:
            raise ValueError(f"Unknown fusion clip kind: {kind}")
        anchors = dict(inputs.get("anchor_nodes") or {})
        first = _first_path(context.state.node_outputs.get(str(anchors.get("first") or "")))
        last = _first_path(context.state.node_outputs.get(str(anchors.get("last") or "")))
        if not first:
            raise RuntimeError(f"Fusion {kind} {index + 1} did not resolve its first frame")
        if anchors.get("last") and not last:
            raise RuntimeError(f"Fusion {kind} {index + 1} did not resolve its last frame")
        mode = "fl2va" if kind in BRIDGE_HALVES else "i2va"
        length = int(inputs["length"])
        # The video model only knows the picture and this text, so the character's anatomy rides in every clip.
        prompt = format_h3_video_prompt(
            include_role_description(f"{LOCKED_CAMERA} {scene}", context.plan.goal),
            mode=mode,
            duration_seconds=length / 24,
            overall_soundscape=soundscape,
            non_diegetic_music=music,
        )
        payload: dict[str, Any] = {
            "workflow_name": str(inputs["workflow_name"]),
            "run_dir": str(build_run_dir(self.output_root, scene, f"fusion_{kind}_{index + 1}", default_slug="news_fusion")),
            "prompt": prompt,
            "h3_mode": mode,
            "width": int(inputs["width"]),
            "height": int(inputs["height"]),
            "length": length,
            "video_count": 1,
            "character": str(context.plan.goal.constraints.get("character") or ""),
            "image_path": first,
        }
        if last:
            payload["last_image_path"] = last
        result = self.tools.call("comfy.render_h3", payload)
        return SkillResult(
            status="success",
            outputs={**result, "h3_mode": mode, "first_frame_path": first, "last_frame_path": last},
            metrics={"video_count": len(result.get("saved_files", []))},
            logs=[f"Rendered fusion {kind} {index + 1} with {mode}."],
        )


def register_news_fusion_skills(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    output_root: Path,
    llm_engine: LLMPromptEngine | None = None,
) -> None:
    skills = NewsFusionSkills(tool_registry, output_root, llm_engine=llm_engine)
    skill_registry.register("news_fusion.prepare_story", skills.prepare_story, "Write a news + character fusion story in N beats")
    skill_registry.register("news_fusion.render_clip", skills.render_clip, "Render one fusion beat clip or bridge clip with H3")
