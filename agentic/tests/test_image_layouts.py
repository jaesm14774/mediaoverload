from __future__ import annotations

from agentic.runtime.prompt_tasks import scenes
from agentic.runtime.prompting import IMAGE_PROMPT_CONTRACT

CAMERA_SHOT_TERMS = ("low-angle", "over-the-shoulder", "eye-level", "bokeh", "close-up shot")


def test_user_given_an_illustration_set_when_variants_are_planned_then_each_has_its_own_layout_not_a_camera_shot():
    """User Given a text2img set in an illustrated style When the variants are planned Then every variant is offered a different layout idea and none is a photographic camera shot."""
    presentations = [presentation for _, presentation, _ in scenes.IMAGE_FALLBACK_PRESENTATIONS]
    assert len(set(presentations)) == len(presentations) >= 4
    for presentation in presentations:
        assert all(term not in presentation.casefold() for term in CAMERA_SHOT_TERMS)
    for presentation in presentations:
        assert presentation in scenes.IMAGE_VARIANT_RESPONSE_CONTRACT


def test_user_given_a_news_brief_when_the_scene_prompt_is_requested_then_the_writer_gets_the_illustration_contract():
    """User Given a news-grounded image request When the first scene prompt is written Then the writer is told to keep characters small and to leave out signs, price tags, and captions or draw them without writing."""
    sent = []

    class Manager:
        pass

    class Client:
        def require_manager(self):
            return Manager()

        def generate_json(self, manager, system, user_prompt, **kwargs):
            sent.append(user_prompt)
            return {"prompt": "A small Kirby under an umbrella.", "creative_seed": "1", "source": "n/a",
                    "source_anchors": [{"source_excerpt": "rain", "visual_mapping": "A small Kirby under an umbrella."}]}

        def mark_payload(self, payload):
            return payload

        def template_fallback(self, fallback, exc):
            raise exc

    scenes.generate_autonomous_scene_prompt(
        Client(), character="Kirby", style="storybook_watercolor", media_type="image",
        news_context={"title": "Heavy rain hits the north", "keyword": "rain", "content": "Rain all weekend."},
        news_grounding_required=True,
    )
    assert sent and IMAGE_PROMPT_CONTRACT in sent[0]
    assert "no writing" in sent[0]
