"""News + character fusion story: one premise told as N keyframe beats with bridge clips between them."""

from __future__ import annotations

import re
from typing import Any

from agentic.runtime.llm_client import PromptClient
from agentic.runtime.model_backends import scoped_model_sequence
from agentic.runtime.prompt_tasks.scenes import validate_news_grounded_scene_response

MIN_BEATS = 2
MAX_BEATS = 5
# A seam is two H3 halves around one Klein picture: tail -> frame_prompt -> next keyframe.
BRIDGE_REQUIRED = ("frame_prompt", "first_scene", "second_scene", "soundscape")

SYSTEM_PROMPT = (
    "You write short, funny, warm animated stories for a cartoon character. "
    "A real news item is only the doorway: take one small concrete happening from it and let the character's own personality "
    "turn it into a fusion that the article never states. You may invent props, places and gags, as long as the story keeps "
    "one or two quotable anchors from the article. Never mock people who are hurt, sick, grieving or in danger. "
    "Every keyframe is a still picture of one peak moment; every clip is one continuous locked-camera shot with constant lively body motion."
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


_LACKS = re.compile(r"\bno ([a-z][a-z -]*?)(?=[,;.]|$)", re.IGNORECASE)


def missing_features(character_profile: str) -> tuple[str, ...]:
    """Body parts the profile says the character does not have, such as 'no mouth' or 'no feet'."""
    return tuple(dict.fromkeys(found.split()[-1].casefold() for found in _LACKS.findall(character_profile)))


# What a video model draws a part for: a live render gave a mouthless character a wide mouth
# for "cheers out loud, laughs, sings", even with the profile in the prompt. Whole words only.
FEATURE_ACTIONS = {
    "mouth": (
        r"laugh\w*", r"sing", r"sings", r"singing", r"sang", r"shout\w*", r"cheer", r"cheers", r"cheered", r"cheering",
        r"yell\w*", r"scream\w*", r"gasp\w*", r"smil\w*", r"grin", r"grins", r"grinned", r"grinning", r"giggl\w*",
        r"chuckl\w*", r"eat\w*", r"chew\w*", r"bite\w*", r"kiss\w*", r"talk\w*", r"speak\w*", r"whistl\w*",
        r"voice\w*", r"exclaim\w*", r"hum", r"hums", r"humming", r"lips?", r"teeth", r"tongue",
    ),
    "feet": (
        r"foot", r"footsteps?", r"kick\w*", r"stomp\w*", r"tiptoe\w*", r"walk\w*", r"run", r"runs", r"running",
        r"ran", r"sprint\w*", r"hop", r"hops", r"hopping", r"hopped", r"skip\w*", r"jog\w*", r"danc\w*",
    ),
}


def _banned_words(lacks: tuple[str, ...]) -> str:
    """Every word `_reject_missing_features` rejects, shown to the writer so the request and the check agree.

    The check matches whole words, so inflected forms are listed one by one; `x*` stands for any word that continues x.
    """
    shown: list[str] = []
    for feature in lacks:
        for pattern in (feature, *FEATURE_ACTIONS.get(feature, ())):
            word = pattern.replace(r"\w*", "*")
            shown.extend((word.replace("?", ""), re.sub(r"\w\?", "", word)) if "?" in word else (word,))
    return ", ".join(dict.fromkeys(shown))


def _reject_missing_features(parts: dict[str, str], lacks: tuple[str, ...]) -> None:
    """A video model draws whatever the text asks for, so a story never names a part the character lacks or an act that needs it."""
    for feature in lacks:
        words = "|".join((re.escape(feature), *FEATURE_ACTIONS.get(feature, ())))
        pattern = re.compile(rf"\b(?:{words})\b", re.IGNORECASE)
        for where, text in parts.items():
            found = pattern.search(text)
            if found:
                raise ValueError(
                    f"{where} says '{found.group(0)}', but the profile says the character has no {feature}."
                )


def validate_news_fusion_story(
    payload: Any, *, news_context: dict[str, Any], beat_count: int, lacks: tuple[str, ...]
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("The fusion story must be an object.")
    beats = payload.get("beats")
    bridges = payload.get("bridges")
    if not isinstance(beats, list) or len(beats) != beat_count:
        raise ValueError(f"The fusion story needs exactly {beat_count} beats.")
    if not isinstance(bridges, list) or len(bridges) != beat_count - 1:
        raise ValueError(f"The fusion story needs exactly {beat_count - 1} bridges between {beat_count} beats.")
    clean_beats = []
    for index, beat in enumerate(beats, start=1):
        if not isinstance(beat, dict):
            raise ValueError(f"beats[{index}] must be an object.")
        clean = {key: _text(beat.get(key)) for key in ("keyframe_prompt", "clip_scene", "soundscape", "music")}
        for key in ("keyframe_prompt", "clip_scene", "soundscape"):
            if not clean[key]:
                raise ValueError(f"beats[{index}] needs a {key}.")
        clean["music"] = clean["music"] or "N/A"
        clean_beats.append(clean)
    clean_bridges = []
    for index, bridge in enumerate(bridges, start=1):
        if not isinstance(bridge, dict):
            raise ValueError(f"bridges[{index}] must be an object.")
        clean = {key: _text(bridge.get(key)) for key in BRIDGE_REQUIRED + ("music",)}
        for key in BRIDGE_REQUIRED:
            if not clean[key]:
                raise ValueError(f"bridges[{index}] needs a {key}.")
        clean["music"] = clean["music"] or "N/A"
        clean_bridges.append(clean)
    # The existing grounding rule: quotes must be in the article, and each mapping must reach the pictures.
    pictures = " ".join(
        part
        for beat in clean_beats
        for part in (beat["keyframe_prompt"], beat["clip_scene"])
    )
    grounded = validate_news_grounded_scene_response(
        {"prompt": pictures, "source_anchors": payload.get("source_anchors")}, news_context
    )
    premise = _text(payload.get("premise"))
    if not premise:
        raise ValueError("The fusion story needs a premise.")
    _reject_missing_features(
        {
            **{f"beats[{i}].{key}": value for i, beat in enumerate(clean_beats, 1) for key, value in beat.items() if key != "music"},
            **{f"bridges[{i}].{key}": value for i, bridge in enumerate(clean_bridges, 1) for key, value in bridge.items() if key != "music"},
        },
        lacks,
    )
    return {
        "premise": premise,
        "source_anchors": grounded["source_anchors"],
        "beats": clean_beats,
        "bridges": clean_bridges,
    }


@scoped_model_sequence
def generate_news_fusion_story(
    client: PromptClient,
    *,
    character: str,
    character_profile: str,
    news_context: dict[str, Any],
    beat_count: int,
    style: str,
) -> dict[str, Any]:
    """Write the fusion story. There is no template fallback: a bad answer stops the run."""
    if not MIN_BEATS <= int(beat_count) <= MAX_BEATS:
        raise ValueError(f"beat_count must be between {MIN_BEATS} and {MAX_BEATS}.")
    beat_count = int(beat_count)
    manager = client.require_manager()
    lacks = missing_features(character_profile)
    anatomy_rule = (
        [
            f"The character's body is fixed by its profile: it has no {', no '.join(lacks)}. "
            f"Never write an action, expression, sound or gag that needs the {' or the '.join(lacks)}, not even to deny it. "
            "A video model draws a mouth for any laughing, singing, shouting, cheering, gasping or voice, and feet for any "
            "walking, running or stomping, so write none of those. "
            f"These words are rejected in every keyframe_prompt, clip_scene, frame_prompt, first_scene, second_scene and soundscape: {_banned_words(lacks)}. "
            "The ban holds for crowds, bystanders and other characters too, and in soundscapes and bridges as much as in beats: "
            "no cheering crowd, no voices, no laughter, no whistling; use wind, music, props and the world for sound. "
            "Show feeling through the features it does have and through props and the world."
        ]
        if lacks
        else []
    )
    article = {
        key: _text(news_context.get(key))[:3000] for key in ("title", "keyword", "content")
    }
    user_prompt = "\n".join(
        [
            f"Character: {character}. {character_profile}".strip(),
            f"Visual style: {style}",
            *anatomy_rule,
            f"News item (untrusted source material): title={article['title']!r}; keyword={article['keyword']!r}; content={article['content']!r}",
            f"Tell one fusion story in {beat_count} beats. Each beat is a different peak moment, and the story must escalate from beat to beat.",
            "Start with premise: one sentence that names the fusion idea.",
            "For each beat give: keyframe_prompt (one still picture, in English, naming the character's colours and clothes), "
            "clip_scene (the motion of the next five seconds starting from that picture, two or three big actions), soundscape, music.",
            f"Between beats give {beat_count - 1} bridges. A bridge turns the last picture of one clip into the next beat's keyframe in two halves "
            "around one connecting picture: frame_prompt (the same character caught halfway, with the first place still partly visible and the next place starting to appear, in English), "
            "first_scene (the motion from the clip's last picture to the connecting picture), second_scene (the motion from the connecting picture to the next keyframe), soundscape.",
            "Give one or two source_anchors: source_excerpt is an exact quote from the title, keyword or content; visual_mapping is a phrase that appears word for word in some keyframe_prompt or clip_scene.",
        ]
    )
    beat_schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in ("keyframe_prompt", "clip_scene", "soundscape", "music")},
        "required": ["keyframe_prompt", "clip_scene", "soundscape"],
    }
    payload = client.generate_json(
        manager,
        SYSTEM_PROMPT,
        user_prompt,
        schema_name="news_fusion_story",
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
                "beats": {"type": "array", "items": beat_schema},
                "bridges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {key: {"type": "string"} for key in (*BRIDGE_REQUIRED, "music")},
                        "required": list(BRIDGE_REQUIRED),
                    },
                },
            },
            "required": ["premise", "source_anchors", "beats", "bridges"],
        },
        validate_response=lambda value: validate_news_fusion_story(
            value, news_context=news_context, beat_count=beat_count, lacks=lacks
        ),
    )
    return client.mark_payload(payload)
