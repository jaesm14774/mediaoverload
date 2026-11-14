"""News-grounded scene prompts: anchors must be traceable, not typographically identical.

The model writes the excerpt and mapping; the grounding guarantee is that both are
really in the article and the prompt, regardless of punctuation or character width.
"""

import pytest

from agentic.runtime.llm_client import PromptGenerationError
from agentic.runtime.llm_engine import LLMPromptEngine
from provider_fakes import ScriptedLLMManager

ARTICLE = {
    "title": "鋒面通過各地轉雨",
    "keyword": "鋒面",
    "content": "今天鋒面通過、各地轉雨，最低氣溫１５度，外出記得攜帶雨具。",
    "event_title": "SECRET-EVENT-TITLE",
    "selection_evidence": {"note": "SECRET-EVIDENCE-SENTENCE"},
}
PROMPT = (
    "Kracko floats through the drizzle and holds an umbrella above its head, "
    "shivering a little."
)


def _scene(*anchors, prompt=PROMPT):
    return {
        "suitable": True,
        "prompt": prompt,
        "creative_seed": "rain",
        "source": "autonomous_llm",
        "source_anchors": list(anchors),
    }


def _generate(payload):
    manager = ScriptedLLMManager([payload], repeat_last=True)
    engine = LLMPromptEngine(mode="llm", manager=manager)
    result = engine.generate_autonomous_scene_prompt(
        character="Kracko",
        style="anime",
        media_type="image",
        news_context=ARTICLE,
        news_grounding_required=True,
    )
    return result, manager


def test_user_given_an_excerpt_that_differs_only_in_punctuation_and_character_width_when_the_scene_is_grounded_then_it_is_accepted():
    """User Given a Chinese article with full-width punctuation and digits When the writer quotes it with half-width punctuation and digits Then the scene is accepted with its anchors."""
    result, _ = _generate(
        _scene(
            {
                "source_excerpt": "今天鋒面通過,各地轉雨",
                "visual_mapping": "Kracko floats through the drizzle and holds an umbrella above its head.",
            },
            {
                "source_excerpt": "最低氣溫15度",
                "visual_mapping": "Kracko floats through the drizzle",
            },
        )
    )
    assert len(result["source_anchors"]) == 2
    assert result["prompt"]


def test_user_given_an_excerpt_that_is_not_in_the_article_when_the_scene_is_grounded_then_it_is_rejected_and_the_error_names_it():
    """User Given a writer that invents an excerpt When the scene is grounded Then generation fails and the failure shows the invented excerpt."""
    with pytest.raises(PromptGenerationError) as failure:
        _generate(
            _scene(
                {
                    "source_excerpt": "颱風即將登陸",
                    "visual_mapping": "Kracko floats through the drizzle",
                }
            )
        )
    assert "颱風即將登陸" in str(failure.value)


def test_user_given_a_mapping_that_is_not_in_the_prompt_when_the_scene_is_grounded_then_it_is_rejected_and_the_error_names_it():
    """User Given a mapping the prompt never states When the scene is grounded Then generation fails and the failure shows the missing mapping."""
    with pytest.raises(PromptGenerationError) as failure:
        _generate(
            _scene(
                {
                    "source_excerpt": "各地轉雨",
                    "visual_mapping": "Kracko dances on a rainbow",
                }
            )
        )
    assert "Kracko dances on a rainbow" in str(failure.value)


def test_user_given_an_article_with_extra_selection_fields_when_the_writer_is_briefed_then_it_only_sees_text_it_is_allowed_to_quote():
    """User Given an article carrying selection evidence beyond its title, keyword and content When the writer is briefed Then the brief shows the quotable article and none of the extra fields."""
    _, manager = _generate(
        _scene(
            {
                "source_excerpt": "各地轉雨",
                "visual_mapping": "Kracko floats through the drizzle",
            }
        )
    )
    brief = " ".join(
        message["content"] for message in manager.text_model.requests[0]["messages"]
    )
    assert "各地轉雨" in brief and "鋒面通過各地轉雨" in brief
    assert "SECRET-EVIDENCE-SENTENCE" not in brief
    assert "SECRET-EVENT-TITLE" not in brief


def test_user_given_an_excerpt_stitched_across_two_article_fields_when_the_scene_is_grounded_then_it_is_rejected():
    """User Given a writer that joins the end of the title to the start of the content When the scene is grounded Then it is rejected, because no single passage of the article says that."""
    with pytest.raises(PromptGenerationError):
        _generate(
            _scene(
                {
                    "source_excerpt": "各地轉雨鋒面",
                    "visual_mapping": "Kracko floats through the drizzle",
                }
            )
        )
