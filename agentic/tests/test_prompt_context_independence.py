"""Prompt-writing boundary using the HTTP-contract-verified scripted provider."""
import json
import re

from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.storyboard import format_native_h3_keyframe_prompt
from provider_fakes import ScriptedLLMManager


def test_user_given_a_source_specific_scene_when_native_story_requests_are_written_then_generic_guidance_does_not_prescribe_other_props():
    """User Given a timetable change at a train station When the production Native H3 writer constructs its requests Then it carries the source scene without prescribing a card, door, chain-link analogy, or another character's expression."""
    news = {"title": "A train station combines two timetables", "keyword": "train;station;timetable", "content": "The station combines two service timetables. A train follows the updated timetable.", "category": "transport"}
    trace = {"source_category": "product_or_service", "source_title": news["title"], "source_fact": news["content"], "news_mechanism": "combining two service timetables", "news_consequence": "a train follows the updated timetable", "source_prop": "train"}
    spine = {"premise": "Magolor watches a waiting train.", "stakes": "The train waits for its scheduled departure.", "climax": "The train departs on the updated timetable.", "resolution": "Magolor watches the departing train."}
    shots = [{"time": f"{i * 5}-{(i + 1) * 5}s", "title": f"Timetable beat {i + 1}", "action": action, "physical_cause": "The updated timetable determines the departure.", "state_change": action, "camera": "A medium view keeps the selected character and the train visible."} for i, action in enumerate((spine["premise"], spine["climax"], spine["resolution"]))]
    story = {"title": "The updated departure", "world": {"setting": "the train station", "visual_language": "hand-painted illustration"}, "story_spine": spine, "news_trace": trace, "native_shots": shots}
    plan = {"news_trace": trace, "recommended_scene": {"payoff_kind": "curiosity", "source_prop": "train"}, "story_spine": spine, "shots": shots}
    manager = ScriptedLLMManager([plan, story, story], repeat_last=True)
    result = LLMPromptEngine(mode="llm", manager=manager).generate_native_h3_storyboard(
        character="Magolor", style="hand-painted illustration", duration_seconds=15,
        base_storyboard={"character": "Magolor"}, news_context=news,
    )
    assert result["prompt_mode"] == "llm"
    assert "train" in json.dumps(result["story"]).lower()
    requests = "\n".join(message["content"] for request in manager.text_model.requests for message in request["messages"])
    assert news["title"] in requests
    assert news["content"] in requests
    for unrelated_direction in (
        "Choose visibly interlocking links or loops",
        "Keep a small analogy card on a low surface",
        "reported outcome is a door opening",
        "Google Gems are saved software instructions",
        "give Kirby bright wide eyes",
        "give Kirby the visibly open curious mouth",
    ):
        assert unrelated_direction not in requests


def test_user_given_a_recurring_visit_story_when_keyframes_are_formatted_then_the_selected_scene_is_preserved_instead_of_a_preset_village():
    """User Given recurring visits visualized in a library or an explicitly supplied rural doorway When opening and ending keyframes are formatted Then their own setting and action survive without an injected village, door, or wall."""
    for setting, object_name in (("A Helsinki library reading room", "bookshelf"), ("A Changhua home beside a wooden doorway", "wooden doorway"), ("A Helsinki library reading room with a long row of bookshelves around the reading area, a clearly separated entrance and a visible source-defined meeting space", "bookshelf")):
        opening = f"MetaKnight stands beside a painted footprint route and the {object_name}"
        ending = f"MetaKnight holds a thoughtful pose beside the {object_name} and painted footprint route"
        story = {
            "character": "MetaKnight", "world": {"setting": setting, "visual_language": "ink illustration"},
            "news_trace": {"source_category": "human_interest", "source_fact": "A helper returns over time to build trust", "source_action_sequence": ["The helper returns on repeated visits to build trust"], "visual_translation": "an uncounted painted footprint route", "visual_anchors": ["painted footprint route"]},
            "native_shots": [{"action": opening, "state_change": opening, "camera": "a wide view of the selected setting"}, {"action": ending, "state_change": ending, "camera": "the same wide view"}],
        }
        for frame, pose in (("opening", opening), ("ending", ending)):
            prompt = format_native_h3_keyframe_prompt(story, style="ink illustration", frame=frame)
            assert setting in prompt
            assert pose in prompt
            assert object_name in prompt
            if "Helsinki" in setting:
                assert "Changhua" not in prompt
                assert "wooden doorway" not in prompt
                assert "stone wall" not in prompt


def test_user_given_selected_task_objects_when_a_linked_result_is_formatted_then_no_other_analogy_is_injected():
    """User Given linked timetables or explicitly selected wooden rings When the ending prompt is formatted Then the chosen objects remain without a forced ring-shaped or food-shaped substitute."""
    for object_name in ("two printed timetables", "three wooden rings"):
        pose = f"Magolor holds {object_name} joined together on the source-defined work surface"
        story = {"character": "Magolor", "news_trace": {"source_category": "product_or_service", "source_fact": "The service combines its timetables", "visual_anchors": [object_name]}, "native_shots": [{"action": pose, "state_change": pose}]}
        prompt = format_native_h3_keyframe_prompt(story, frame="ending")
        assert pose in prompt
        assert object_name in prompt
        if "timetables" in object_name:
            assert not re.search(r"\b(?:rings?|donut|bagel)\b", prompt, re.IGNORECASE)
