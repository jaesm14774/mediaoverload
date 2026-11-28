"""Skills for the real-world intruder: write the scene and gag, draw the photo and the sketch, render the clip, tape the sketch in."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agentic.minimax_prompting import format_h3_cut_time, format_h3_video_prompt
from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.news_grounding import is_usable_news_selection
from agentic.runtime.prompt_tasks.real_world_intruder import (
    BEAT_FIELDS,
    BEAT_STARTS,
    generate_real_world_intruder_story,
    keyframe_prompt,
    sketch_prompt,
)
from agentic.runtime.prompting import include_role_description, selected_role_description
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.skills.shared import build_run_dir

FRAME_KEYS = ("selected_frame_path", "selected_assets", "saved_files")
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}
# The reference clips lose their charm when the camera drifts or the intruder swells to fill the frame.
LOCKED_SCENE = (
    "Real photograph look with one flat 2D cartoon character; locked-off static tripod shot, the camera never moves and "
    "never zooms in the first two shots, and nothing in the real photo moves except the character and what it touches; "
    "the character stays small, about one fifth of the frame height, until the third shot; "
    "constant lively body motion until the last frame, ending in the middle of a motion and never in a calm pose."
)


# Live run: with the reaction buried in a long sentence, H3 animated the event and left the character standing.
PAYOFF_TAIL = (
    "The character's own big reaction fills the whole shot: its body leaves its pose, arms flung wide, "
    "and it is still mid-reaction on the last frame."
)


def timed_shots(story: dict[str, Any]) -> str:
    """The three beats as H3 shots of one unbroken take, each cut at its start time."""
    shots = []
    for number, (field, start) in enumerate(zip(BEAT_FIELDS, BEAT_STARTS), start=1):
        cut = "" if number == 1 else f"At {format_h3_cut_time(str(start))}, in the same unbroken shot, "
        tail = f" {PAYOFF_TAIL}" if number == len(BEAT_FIELDS) else ""
        shots.append(f"[Shot {number}] {cut}{story[field]}{tail}")
    return " ".join(shots)


def _first_file(outputs: Any, keys: tuple[str, ...] = FRAME_KEYS, suffixes: set[str] | None = None) -> str:
    if not isinstance(outputs, dict):
        return ""
    for key in keys:
        value = outputs.get(key)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and item and (suffixes is None or Path(item).suffix.lower() in suffixes):
                return item
    return ""


class RealWorldIntruderSkills:
    def __init__(
        self,
        tools: ToolRegistry,
        output_root: Path,
        llm_engine: LLMPromptEngine | None = None,
    ) -> None:
        self.tools = tools
        self.output_root = Path(output_root)
        self._llm_engine = llm_engine

    @property
    def llm_engine(self) -> LLMPromptEngine:
        if self._llm_engine is None:
            self._llm_engine = LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
        return self._llm_engine

    def prepare_story(self, context: SkillContext) -> SkillResult:
        goal = context.plan.goal
        constraints = goal.constraints
        news_context = dict(constraints.get("news_context") or {})
        if news_context.get("error"):
            raise RuntimeError(f"Real-world intruder news context failed: {news_context['error']}")
        if not is_usable_news_selection(str(news_context.get("title") or ""), str(news_context.get("keyword") or "")):
            raise RuntimeError("Real-world intruder requires a routed news item; none was provided")
        character = str(constraints.get("character") or "the protagonist")
        profile = selected_role_description(goal)
        story = generate_real_world_intruder_story(
            self.llm_engine.client, character=character, character_profile=profile, news_context=news_context
        )
        outputs = {
            **story,
            "news_context": news_context,
            "keyframe_prompt": keyframe_prompt(story, character=character, character_profile=profile),
            "sketch_prompt": sketch_prompt(character=character, character_profile=profile),
        }
        return SkillResult(
            status="success",
            outputs=outputs,
            logs=[f"Wrote a real-world intrusion from news: {news_context.get('title', '')}"],
        )

    def render_image(self, context: SkillContext) -> SkillResult:
        """Draw one picture from a prompt the story already finished; no run-wide style is added on top."""
        inputs = context.node.inputs
        story = context.state.node_outputs.get(str(inputs["story_node"]))
        prompt = str((story or {}).get(str(inputs["prompt_key"])) or "").strip()
        if not prompt:
            raise RuntimeError(f"The story has no {inputs['prompt_key']} to draw")
        result = self.tools.call(
            "comfy.workflow.text_to_image",
            {
                "workflow_name": str(inputs["workflow_name"]),
                "run_dir": str(build_run_dir(self.output_root, prompt, str(inputs["suffix"]), default_slug="real_world_intruder")),
                "prompt": prompt,
                "width": int(inputs["width"]),
                "height": int(inputs["height"]),
                "image_count": 1,
            },
        )
        saved = [str(path) for path in result.get("saved_files", [])]
        return SkillResult(
            status="success",
            outputs={**result, "generated_files": saved, "rejected_files": []},
            metrics={"image_count": len(saved)},
            logs=[f"Drew the {inputs['suffix']} picture."],
        )

    def render_clip(self, context: SkillContext) -> SkillResult:
        inputs = context.node.inputs
        story = context.state.node_outputs.get(str(inputs["story_node"]))
        if not isinstance(story, dict):
            raise RuntimeError("Intruder clip rendering needs the written story")
        first = _first_file(context.state.node_outputs.get(str(inputs["anchor_node"])))
        if not first:
            raise RuntimeError("Intruder clip did not resolve its keyframe")
        length = int(inputs["length"])
        # The video model only knows the picture and this text, so the character's anatomy rides along.
        prompt = format_h3_video_prompt(
            include_role_description(f"{LOCKED_SCENE} {timed_shots(story)}", context.plan.goal),
            mode="i2va",
            duration_seconds=length / 24,
            overall_soundscape=story["soundscape"],
            non_diegetic_music=story["music"],
        )
        result = self.tools.call(
            "comfy.render_h3",
            {
                "workflow_name": str(inputs["workflow_name"]),
                "run_dir": str(build_run_dir(self.output_root, story["setup_beat"], "intruder_clip", default_slug="real_world_intruder")),
                "prompt": prompt,
                "h3_mode": "i2va",
                "width": int(inputs["width"]),
                "height": int(inputs["height"]),
                "length": length,
                "video_count": 1,
                "character": str(context.plan.goal.constraints.get("character") or ""),
                "image_path": first,
            },
        )
        return SkillResult(
            status="success",
            outputs={**result, "h3_mode": "i2va", "first_frame_path": first, "news_context": story.get("news_context") or {}},
            metrics={"video_count": len(result.get("saved_files", []))},
            logs=["Rendered the intruder clip with i2va."],
        )

    def compose(self, context: SkillContext) -> SkillResult:
        inputs = context.node.inputs
        clip = _first_file(context.state.node_outputs.get(str(inputs["clip_node"])), ("saved_files", "video_path"), VIDEO_SUFFIXES)
        sketch = _first_file(context.state.node_outputs.get(str(inputs["sketch_node"])))
        if not clip:
            raise RuntimeError("Sketch inset needs the rendered clip")
        if not sketch:
            raise RuntimeError("Sketch inset needs the drawn sketch")
        output_path = Path(clip).with_name(f"{Path(clip).stem}_sketch_inset.mp4")
        result = self.tools.call(
            "media.sketch_inset", {"video_path": clip, "sketch_path": sketch, "output_path": str(output_path)}
        )
        video_path = str(result["video_path"])
        return SkillResult(
            status="success",
            outputs={**result, "saved_files": [video_path], "sketch_path": sketch},
            logs=["Taped the kid's sketch into the clip's top-left corner."],
        )


def register_real_world_intruder_skills(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    output_root: Path,
    llm_engine: LLMPromptEngine | None = None,
) -> None:
    skills = RealWorldIntruderSkills(tool_registry, output_root, llm_engine=llm_engine)
    skill_registry.register("real_world_intruder.prepare_story", skills.prepare_story, "Write a real scene and a gag for a drawn character from the news")
    skill_registry.register("real_world_intruder.render_image", skills.render_image, "Draw the real-photo keyframe or the kid's sketch")
    skill_registry.register("real_world_intruder.render_clip", skills.render_clip, "Render the intruder clip with H3 image-to-video")
    skill_registry.register("real_world_intruder.compose", skills.compose, "Tape the kid's sketch into the corner of the clip")
