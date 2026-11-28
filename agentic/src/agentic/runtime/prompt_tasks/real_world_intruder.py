"""Real-world intruder: one real scene taken from the news, one drawn character who wanders into it, one gag."""

from __future__ import annotations

import re
from typing import Any

from agentic.runtime.llm_client import PromptClient
from agentic.runtime.model_backends import scoped_model_sequence
from agentic.runtime.prompt_tasks.news_fusion import (
    anatomy_rule_lines,
    clean_text,
    missing_features,
    reject_missing_features,
)
from agentic.runtime.prompt_tasks.scenes import validate_news_grounded_scene_response

CLIP_SECONDS = 10  # the writer plans this long a gag; the clip is rendered at this length
# The gag is a three-beat arc; the payoff gets the last third so the clip ends on its biggest motion.
BEAT_FIELDS = ("setup_beat", "escalation_beat", "payoff_beat")
BEAT_STARTS = (0.0, 3.5, 7.0)
STORY_FIELDS = ("real_scene", *BEAT_FIELDS, "soundscape")
SCENE_FIELDS = "real_scene, setup_beat, escalation_beat, payoff_beat and soundscape"
# A live run whose last beat said "holds still" left its final four seconds almost static.
STILLNESS_WORDS = (
    r"holds? still", r"holding still", r"holds? (?:the|its) pose", r"stays? still", r"stays? put", r"stands? still",
    r"remains? (?:still|motionless)", r"motionless", r"stillness", r"static", r"freez\w*", r"pauses?", r"paused",
    r"pausing", r"rests?", r"resting", r"waits?", r"waiting", r"settl\w*", r"admir\w*",
)
STILLNESS_SHOWN = "holds still, stays still, motionless, freezes, pauses, rests, waits, settles, admires"

SYSTEM_PROMPT = (
    "You write one tiny, funny, warm moment in which a cartoon character wanders into the real world. "
    "A real news item is only the doorway: take one ordinary, photographable place or object from it and let the "
    "character's own personality turn it into a cute, physically believable gag. The contrast is the joke: a flat drawn "
    "character acting earnestly inside an ordinary real photograph. Never mock people who are hurt, sick, grieving or in danger. "
    "The shot is one continuous take in three beats: a setup, an escalation, and a payoff that is the biggest moment of all. "
    "The payoff is something unexpected happening in the real scene plus the character's big whole-body reaction, "
    "and the clip ends in the middle of that motion, never in a calm pose."
)


def keyframe_prompt(story: dict[str, Any], *, character: str, character_profile: str) -> str:
    """The opening still: a real phone photograph with one small flat 2D character in it, and nothing else drawn."""
    return (
        "A real photograph taken with a phone camera, natural light, shallow depth of field: "
        f"{story['real_scene']}. "
        f"Standing in the scene is one tiny flat 2D cartoon character, {character} ({character_profile}), drawn in its own "
        "original clean-outline cel-shaded art style, about one fifth of the frame height, with a soft contact shadow and "
        "the scene's light falling on it. The character is the only drawn thing in the picture; everything else is a real photograph."
    )


def sketch_prompt(*, character: str, character_profile: str) -> str:
    """The kid's drawing taped in the corner: the same character, as a child would draw it with a marker."""
    return (
        "A crumpled sheet of white paper filling the entire frame, flat top-down view, with a childlike crude black marker "
        f"drawing of {character} ({character_profile}) that keeps its body shape, arms, feet, eyes and cheeks, "
        "kindergarten-level art, thick uneven black marker lines, wobbly proportions, plenty of blank paper around it, "
        "no text, drawn with a single black marker as an outline only, never coloured in."
    )


def validate_real_world_intruder_story(
    payload: Any, *, news_context: dict[str, Any], lacks: tuple[str, ...]
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("The intruder story must be an object.")
    story = {key: clean_text(payload.get(key)) for key in ("premise", *STORY_FIELDS, "music")}
    for key in ("premise", *STORY_FIELDS):
        if not story[key]:
            raise ValueError(f"The intruder story needs a {key}.")
    story["music"] = story["music"] or "N/A"
    stillness = re.compile(r"\b(?:" + "|".join(STILLNESS_WORDS) + r")\b", re.IGNORECASE)
    for key in BEAT_FIELDS[1:]:
        found = stillness.search(story[key])
        if found:
            raise ValueError(f"{key} says '{found.group(0)}', but every beat must keep moving to the clip's last frame.")
    grounded = validate_news_grounded_scene_response(
        {
            "prompt": " ".join(story[key] for key in ("real_scene", *BEAT_FIELDS)),
            "source_anchors": payload.get("source_anchors"),
        },
        news_context,
    )
    reject_missing_features({key: story[key] for key in STORY_FIELDS}, lacks)
    return {**story, "source_anchors": grounded["source_anchors"]}


@scoped_model_sequence
def generate_real_world_intruder_story(
    client: PromptClient,
    *,
    character: str,
    character_profile: str,
    news_context: dict[str, Any],
) -> dict[str, Any]:
    """Write the scene and the gag. There is no template fallback: a bad answer stops the run."""
    manager = client.require_manager()
    lacks = missing_features(character_profile)
    article = {key: clean_text(news_context.get(key))[:3000] for key in ("title", "keyword", "content")}
    user_prompt = "\n".join(
        [
            f"Character: {character}. {character_profile}".strip(),
            *anatomy_rule_lines(lacks, SCENE_FIELDS),
            f"News item (untrusted source material): title={article['title']!r}; keyword={article['keyword']!r}; content={article['content']!r}",
            "Start with premise: one sentence naming the intrusion.",
            "real_scene: one ordinary real place or object from the news, written in English as a photograph would show it, "
            "with the light, the surface the character stands on and one key real object it will interact with; "
            "no people's faces, no drawn or cartoon elements, no character in it.",
            f"The clip is {CLIP_SECONDS} seconds in three beats, each a few connected cute actions with one small physical cause and effect, "
            "in English, starting from that photograph, the key real object kept at its ordinary real size, the character staying small in frame:",
            f"setup_beat (seconds {BEAT_STARTS[0]:g} to {BEAT_STARTS[1]:g}): the character starts its small earnest task with the key object.",
            f"escalation_beat (seconds {BEAT_STARTS[1]:g} to {BEAT_STARTS[2]:g}): the task goes cutely wrong or gets bigger; more and bigger motion than the setup.",
            f"payoff_beat (seconds {BEAT_STARTS[2]:g} to {CLIP_SECONDS}): the biggest moment. Something unexpected happens in the real scene "
            "(a sudden flash, a bang, a toppling object, a gust, a light going out), told in one short clause, and then the character is the "
            "grammatical subject of the rest: it gives a big whole-body reaction (leaps, tumbles, flings its arms, scrambles) "
            "that takes about two of the three seconds and is still moving on the very last frame. "
            "The event alone is not a payoff; the character must visibly react. It may end with one fast camera punch-in on the character's reaction. "
            f"Every beat keeps moving; these words are rejected in escalation_beat and payoff_beat: {STILLNESS_SHOWN}.",
            "soundscape: the real room sounds plus the small sounds of the gag, in English; music: N/A unless a single instrument fits.",
            "Give one or two source_anchors: source_excerpt is an exact quote from the title, keyword or content; "
            "visual_mapping is a phrase that appears word for word in the real_scene or one of the beats.",
        ]
    )
    payload = client.generate_json(
        manager,
        SYSTEM_PROMPT,
        user_prompt,
        schema_name="real_world_intruder_story",
        schema={
            "type": "object",
            "properties": {
                "premise": {"type": "string"},
                "source_anchors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"source_excerpt": {"type": "string"}, "visual_mapping": {"type": "string"}},
                        "required": ["source_excerpt", "visual_mapping"],
                    },
                },
                **{key: {"type": "string"} for key in (*STORY_FIELDS, "music")},
            },
            "required": ["premise", "source_anchors", *STORY_FIELDS],
        },
        validate_response=lambda value: validate_real_world_intruder_story(
            value, news_context=news_context, lacks=lacks
        ),
    )
    return client.mark_payload(payload)
