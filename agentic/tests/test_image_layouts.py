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


class _SetWriter:
    """Fake prompt client; its JSON contract is the one `LLMClient.generate_json` serves."""

    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.sent = payload, error, []

    def require_manager(self):
        if self.error:
            raise self.error
        return object()

    def generate_json(self, manager, system, user_prompt, **kwargs):
        self.sent.append(user_prompt + "\n" + str(kwargs.get("output_contract") or ""))
        return kwargs["validate_response"](self.payload)

    def mark_payload(self, payload):
        return payload

    def template_fallback(self, fallback, exc):
        return fallback

    def generation_error(self, operation, exc=None):
        return RuntimeError(f"{operation}: {exc}")


MOMENT_ACTING = (
    ("hopeful", "Kirby stretches tall on tiptoe peering over the suitcase"),
    ("worried", "Kirby shrinks behind the suitcase with both arms hugging its handle"),
    ("determined", "Kirby leans forward dragging the heavy suitcase by one stubby arm"),
    ("tired", "Kirby flops face down across the top of the suitcase"),
    ("startled", "Kirby leaps backward with arms flung up as the suitcase tips over"),
    ("content", "Kirby sits cross-legged on the suitcase with eyes closed in a smile"),
)


def _six_moments(acting=MOMENT_ACTING):
    return {
        "creative_brief": "Kirby and a suitcase",
        "prompt_sets": [
            {
                "label": f"m{i}",
                "presentation": f"{feeling} beat {i}",
                "feeling": feeling,
                "pose": pose,
                "prompt": f"{pose}, soft pastel drawing of Kirby with a suitcase.",
            }
            for i, (feeling, pose) in enumerate(acting, start=1)
        ],
    }


def _image_goal(**constraints):
    from agentic.runtime.contracts import GoalRequest

    return GoalRequest(
        prompt="Kirby holds a ticket beside an airport sign",
        media_type="image",
        style="pastel",
        constraints={"character": "Kirby", **constraints},
    )


NEWS = {"title": "Airport reopens after typhoon", "keyword": "airport", "content": "Travellers queued with suitcases."}


def test_user_given_a_news_driven_image_run_when_candidates_are_planned_then_the_six_are_different_moments_around_one_source_object():
    """User Given a bare news-driven image run When the six candidates are planned Then the writer is asked for six different moments of the character with one object from the source, not six views of one scene, and all six reach review."""
    writer = _SetWriter(_six_moments())

    result = scenes.expand_goal(
        writer, _image_goal(news_grounding_required=True, news_context=NEWS), "pastel", []
    )

    asked = writer.sent[0]
    assert scenes.IMAGE_MOMENT_SET_CONTRACT in asked
    assert "Preserve the requested scene" not in asked
    assert "Do not create alternate stories" not in asked
    assert len({item["prompt"] for item in result["prompt_sets"]}) == 6
    assert all(item["self_contained"] for item in result["prompt_sets"])


def test_user_given_a_news_driven_image_run_when_the_writer_is_down_then_the_run_stops_instead_of_showing_one_scene_six_times():
    """User Given a bare news-driven image run When the prompt writer is unavailable Then the run fails loudly rather than sending six copies of one scene to review."""
    import pytest

    writer = _SetWriter(error=ValueError("no model"))

    with pytest.raises(RuntimeError):
        scenes.expand_goal(
            writer, _image_goal(news_grounding_required=True, news_context=NEWS), "pastel", []
        )


def _plan_moments(payload):
    return scenes.expand_goal(
        _SetWriter(payload), _image_goal(news_grounding_required=True, news_context=NEWS), "pastel", []
    )


def test_user_given_six_beats_that_repeat_a_feeling_when_candidates_are_planned_then_the_set_is_refused():
    """User Given a news-driven run whose writer gives two beats the same feeling When the six candidates are planned Then the set is refused instead of reaching review as six moods of one."""
    import pytest

    repeated = list(MOMENT_ACTING)
    repeated[3] = ("Hopeful", repeated[3][1])
    with pytest.raises(RuntimeError):
        _plan_moments(_six_moments(repeated))


def test_user_given_six_beats_that_stand_in_the_same_pose_when_candidates_are_planned_then_the_set_is_refused():
    """User Given a writer that describes two beats with almost the same body When the six candidates are planned Then the set is refused, because the characters would look identical in review."""
    import pytest

    same_pose = list(MOMENT_ACTING)
    same_pose[1] = ("worried", "Kirby stretches tall on tiptoe peering over the suitcase now")
    with pytest.raises(RuntimeError):
        _plan_moments(_six_moments(same_pose))


def test_user_given_a_prompt_that_draws_its_pose_with_different_punctuation_when_candidates_are_planned_then_the_set_is_accepted():
    """User Given a writer whose prompt carries each pose with only punctuation or quote differences When the six candidates are planned Then the set is accepted, because the acting is the same."""
    payload = _six_moments()
    payload["prompt_sets"][0]["pose"] += "."
    payload["prompt_sets"][1]["prompt"] = payload["prompt_sets"][1]["prompt"].replace(",", ";", 1)

    assert len(_plan_moments(payload)["prompt_sets"]) == 6


def test_user_given_a_beat_whose_prompt_leaves_out_its_own_pose_when_candidates_are_planned_then_the_set_is_refused():
    """User Given a writer whose prompt for a beat does not draw the pose it announced When the six candidates are planned Then the set is refused, so the picture cannot ignore the acting."""
    import pytest

    payload = _six_moments()
    payload["prompt_sets"][2]["prompt"] = "Kirby stands next to a suitcase, soft pastel drawing."
    with pytest.raises(RuntimeError):
        _plan_moments(payload)


def test_user_given_a_news_driven_image_run_when_the_writer_is_briefed_then_it_must_give_every_beat_its_own_feeling_and_pose_and_keep_the_character_small():
    """User Given a news-driven image run When the writer is briefed Then each beat must carry its own feeling and pose, and the character stays small in a larger world with open space."""
    writer = _SetWriter(_six_moments())

    scenes.expand_goal(
        writer, _image_goal(news_grounding_required=True, news_context=NEWS), "pastel", []
    )

    asked = writer.sent[0]
    assert "feeling" in asked and "pose" in asked
    assert "larger world" in asked
    assert "quiet plain background" not in asked


def test_user_given_their_own_scene_when_candidates_are_planned_then_all_six_keep_that_scene():
    """User Given a scene they wrote themselves When the six candidates are planned Then the writer is told to keep that scene and vary only the layout."""
    writer = _SetWriter(_six_moments())

    scenes.expand_goal(writer, _image_goal(), "pastel", [])

    asked = writer.sent[0]
    assert "Preserve the requested scene" in asked
    assert scenes.IMAGE_MOMENT_SET_CONTRACT not in asked


def test_user_given_a_news_item_when_the_scene_is_written_then_the_character_lives_one_reported_happening_and_no_reported_person_is_drawn():
    """User Given a news-driven run When the scene is written from the article Then the writer is told to let the character live one everyday happening the article reports, keep it traceable to an excerpt, and leave reported people, flags, slogans and writing out of the picture."""
    sent = []

    class Client:
        def require_manager(self):
            return object()

        def generate_json(self, manager, system, user_prompt, **kwargs):
            sent.append(user_prompt)
            return {"prompt": "Kirby hugs an empty pot.", "creative_seed": "1", "source": "n/a",
                    "source_anchors": [{"source_excerpt": "湯圓", "visual_mapping": "Kirby hugs an empty pot."}]}

        def mark_payload(self, payload):
            return payload

    result = scenes.generate_autonomous_scene_prompt(
        Client(), character="Kirby", style="pastel", media_type="sticker_pack",
        news_context={"title": "夜市老攤重新開張 湯圓半小時賣完", "keyword": "湯圓", "content": "第一鍋湯圓半小時賣完。"},
        news_grounding_required=True,
    )

    assert scenes.NEWS_PREMISE_CONTRACT in sent[0]
    assert result["source_anchors"][0]["source_excerpt"] == "湯圓"


def test_user_given_an_article_with_no_gentle_happening_when_the_scene_is_written_then_no_scene_is_made_from_it():
    """User Given an article the writer judges to hold no gentle everyday happening When the scene is written Then no scene is produced from it and the run is told the article is unsuitable."""
    import pytest
    from agentic.runtime.news_grounding import UnsuitableNewsError

    class Client:
        def require_manager(self):
            return object()

        def generate_json(self, manager, system, user_prompt, **kwargs):
            return kwargs["validate_response"](
                {"suitable": False, "prompt": "", "creative_seed": "", "source": "", "source_anchors": []}
            )

    with pytest.raises(UnsuitableNewsError):
        scenes.generate_autonomous_scene_prompt(
            Client(), character="Kirby", style="pastel", media_type="image",
            news_context={"title": "工廠意外", "keyword": "意外", "content": "事故調查中。"},
            news_grounding_required=True,
        )


def test_user_given_the_first_picked_article_is_unsuitable_when_a_bare_run_starts_then_another_article_is_picked():
    """User Given a bare run whose first picked article is unsuitable When the run resolves its scene Then it picks another article and builds the scene from that one."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from agentic.app.generation_routing import _resolve_autonomous_prompt
    from agentic.runtime.news_grounding import UnsuitableNewsError

    articles = iter(
        SimpleNamespace(to_dict=lambda title=title: {"title": title, "keyword": "k", "content": title})
        for title in ("工廠停工調查", "夜市老攤重新開張 湯圓半小時賣完")
    )

    def write_scene(self, *, news_context, **kwargs):
        if "湯圓" not in news_context["title"]:
            raise UnsuitableNewsError(news_context["title"])
        return {"prompt": "Kirby hugs an empty pot.", "source": "autonomous_llm", "source_anchors": []}

    with patch(
        "agentic.tools.context_services.NewsContextService.select_news",
        side_effect=lambda: next(articles),
    ), patch(
        "agentic.runtime.llm_engine.LLMPromptEngine.generate_autonomous_scene_prompt", write_scene
    ):
        bundle = _resolve_autonomous_prompt(
            prompt="", character_name="Kirby", style="pastel", media_type="image",
            generation={}, generation_type="text2img",
        )

    assert "湯圓" in bundle["news_context"]["title"]
    assert bundle["prompt"] == "Kirby hugs an empty pot."


def test_user_given_a_post_with_media_when_the_caption_is_written_then_the_words_follow_the_media_and_tie_back_to_the_source():
    """User Given a post whose media came from a news item When the caption is written Then the writer is told to write what the media shows, tie it to the one source happening it comes from, and tag only what the caption mentions."""
    from agentic.runtime.prompt_tasks.publishing import SOCIAL_CAPTION_SYSTEM_PROMPT as rules

    assert "the post and the media must tell the same thing" in rules
    assert "Every claim about the scene must be visible in the media" in rules
    assert "tie the scene to the one thing in the article it" in rules
    assert "make the post about the article's central event" not in rules
    assert "Tag only" in rules
