import dataclasses
from pathlib import Path

from agentic.app.character_payload import build_goal_payload_from_character_config
from agentic.app.serial_look import serial_series_look
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.serial_character import build_serial_character_reference_prompt
from character_workflow_helpers import make_character_workflow_request
from provider_fakes import ScriptedLLMManager

REPO_ROOT = Path(__file__).resolve().parents[2]
KIRBY_CONFIG = REPO_ROOT / "configs" / "characters" / "kirby.yaml"
CARD = "角色名稱：柯比\n外觀辨識特徵：粉紅色球狀身體、紅色小腳、藍色橢圓眼睛\n\n角色參考圖提示：\nA pink round creature."


def test_user_given_a_series_style_when_the_identity_prompt_is_built_then_it_carries_that_style_and_keeps_one_whole_centered_character_with_room_around_it():
    """User Given a series style When the identity reference prompt is built Then it states that style and asks for one whole, centred character on a plain ground with room around it, so a round body never fills the frame edge to edge."""
    prompt = build_serial_character_reference_prompt(
        character_name="柯比",
        character_card=CARD,
        style_direction="soft watercolor storybook illustration, pale paper grain",
    )

    assert "soft watercolor storybook illustration" in prompt
    assert "置中" in prompt
    assert "完整入鏡" in prompt
    assert "四周留白" in prompt
    assert "三分之二" not in prompt
    assert "沿用角色設定檔的既有美術風格" not in prompt


DEMEANOR_CARD = (
    "角色名稱：Kirby\n外觀辨識特徵：粉紅色球狀身體、紅色小腳\n\n"
    "角色參考圖提示：\nKirby 戴著用舊郵差帽改的小帽子，雙眼圓亮帶高光，視線越過畫面望向遠方的信箱，"
    "眉毛上揚、嘴角抿著一股不服輸的勁，身體微微前傾，像隨時要衝出去。"
)


def test_user_given_a_story_specific_character_card_when_the_identity_prompt_is_built_then_the_reference_keeps_the_story_acting_and_never_demands_a_blank_face():
    """User Given a card that designs this story's protagonist with a signature look and a determined gaze When the identity prompt is built Then the image model is asked to draw that acting and look, and is never told to stay neutral, expressionless or free of distinguishing detail."""
    prompt = build_serial_character_reference_prompt(
        character_name="Kirby",
        character_card=DEMEANOR_CARD,
        style_direction="soft watercolor storybook illustration",
    )

    assert "舊郵差帽" in prompt
    assert "不服輸" in prompt
    assert "中性" not in prompt


def test_user_given_a_card_without_acting_when_the_identity_prompt_is_built_then_the_builder_still_asks_for_a_focused_gaze_off_camera():
    """User Given a card that only lists the look When the identity prompt is built Then the image model is still told to give the eyes a focus and not stare blankly at the camera."""
    prompt = build_serial_character_reference_prompt(
        character_name="柯比", character_card=CARD, style_direction="watercolor"
    )

    assert "呆望鏡頭" in prompt and "高光" in prompt


def test_user_given_a_reference_prompt_that_covers_look_and_acting_in_one_paragraph_when_the_identity_prompt_is_built_then_nothing_of_it_is_lost():
    """User Given a card whose single reference paragraph holds both the signature look and the acting When the identity prompt is built Then both reach the image model."""
    prompt = build_serial_character_reference_prompt(
        character_name="Kirby", character_card=DEMEANOR_CARD, style_direction="watercolor"
    )

    assert "舊郵差帽" in prompt and "身體微微前傾" in prompt


def test_user_given_a_story_when_the_character_card_prompt_is_built_then_the_designer_is_asked_for_a_story_specific_look_and_acting_instead_of_the_stock_character():
    """User Given an approved story and a pinned franchise character When the card prompt is built Then the designer must keep the species but design this story's look, gaze and acting into the reference paragraph, rather than hand back the stock model sheet."""
    from agentic.runtime.serial_character import build_serial_character_prompts

    system_prompt, user_prompt = build_serial_character_prompts(
        approved_story="Kirby 想把一封信送到山頂，卻一直被風吹回來。",
        character_name="Kirby",
    )

    assert "Kirby" in user_prompt
    for required in ("造型", "眼神", "表情", "段落"):
        assert required in system_prompt
    assert "只寫外觀" not in system_prompt


LLM_CARD = (
    "**角色卡**\n1. 角色名稱：Kirby (卡比)\n2. 外觀辨識特徵：粉紅色圓球體、紅色小腳\n"
    "3. 性格或習慣：憧憬飛行、好奇地觀察星空變化\n4. 一直想要但還沒得到的東西：飛上天\n\n"
    "**圖片生成器參考圖提示**  \n提示：Kirby，粉紅色圓球體，紅色小腳，全身正面視角。"
)


def test_user_given_a_character_card_when_the_identity_prompt_is_built_then_only_the_visual_prompt_reaches_the_image_model_so_no_card_text_gets_painted():
    """User Given a character card with personality and wishes When the identity prompt is built Then only the visual reference prompt is sent, so the image model has no card text to paint onto the picture."""
    prompt = build_serial_character_reference_prompt(
        character_name="Kirby",
        character_card=LLM_CARD,
        style_direction="soft watercolor storybook illustration",
    )

    assert "粉紅色圓球體，紅色小腳" in prompt
    assert prompt.count("粉紅色圓球體") == 1
    for non_visual in ("性格或習慣", "憧憬飛行", "飛上天", "角色卡", "**"):
        assert non_visual not in prompt


def test_user_given_a_card_with_notes_after_the_visual_prompt_when_the_identity_prompt_is_built_then_the_trailing_notes_never_reach_the_image_model():
    """User Given a card whose visual prompt is followed by notes about personality When the identity prompt is built Then only the visual prompt is sent."""
    card = LLM_CARD + "\n\n審核備註：性格急躁、想要飛上天，不可改名。"

    prompt = build_serial_character_reference_prompt(
        character_name="Kirby", character_card=card, style_direction="watercolor"
    )

    assert "紅色小腳，全身正面視角" in prompt
    assert "審核備註" not in prompt and "性格急躁" not in prompt


def test_user_given_a_card_without_a_reference_prompt_section_when_the_card_is_validated_then_it_is_refused_before_it_is_saved():
    """User Given a freshly written card with no reference prompt section When it is validated Then it is refused, so no review is ever opened for a card that cannot produce reference images."""
    import pytest

    from agentic.runtime.serial_character import extract_reference_prompt

    with pytest.raises(ValueError):
        extract_reference_prompt("角色名稱：Kirby\n外觀辨識特徵：粉紅色圓球體")
    assert extract_reference_prompt(LLM_CARD).startswith("Kirby，粉紅色圓球體")


def test_user_given_a_character_card_without_a_reference_prompt_section_when_the_identity_prompt_is_built_then_it_is_refused_instead_of_pasting_the_whole_card():
    """User Given a card that has no reference prompt section When the identity prompt is built Then it is refused, rather than pasting the whole card into the image prompt."""
    import pytest

    with pytest.raises(ValueError):
        build_serial_character_reference_prompt(
            character_name="Kirby",
            character_card="角色名稱：Kirby\n外觀辨識特徵：粉紅色圓球體",
            style_direction="watercolor",
        )


def test_user_given_an_identity_reference_request_when_prompts_are_expanded_then_every_candidate_is_the_same_self_contained_prompt_without_asking_a_model():
    """User Given an identity reference request When its candidate prompts are expanded Then all candidates share one self-contained prompt and no model is asked to invent compositions, so the candidates differ only by noise."""
    manager = ScriptedLLMManager([])
    goal = GoalRequest(
        prompt="centred pink round character, watercolor storybook look",
        media_type="image",
        style="watercolor",
        constraints={"identity_reference": True, "character": "Kirby"},
    )

    brief = LLMPromptEngine(mode="llm", manager=manager).expand_goal(
        goal, "watercolor", [], prompt_variant_count=6
    )

    prompts = {item["prompt"] for item in brief["prompt_sets"]}
    assert prompts == {goal.prompt}
    assert len(brief["prompt_sets"]) == 6
    assert all(item["self_contained"] for item in brief["prompt_sets"])
    assert manager.text_model.requests == []


def test_user_given_a_series_look_when_identity_references_are_requested_then_the_run_uses_that_style_and_canvas(tmp_path):
    """User Given a series with a locked look When its identity references are requested Then the run's style and canvas are exactly the series look, not a fresh random draw."""
    base = make_character_workflow_request(
        REPO_ROOT,
        KIRBY_CONFIG,
        prompt="reference prompt",
        preferred_generation_type="text2img",
    )
    import yaml

    generation = yaml.safe_load(KIRBY_CONFIG.read_text(encoding="utf-8"))["generation"]
    look = serial_series_look(generation, "auto-kirby-series-1", output_root=tmp_path)
    request = dataclasses.replace(
        base,
        generation=dataclasses.replace(
            base.generation, creative_profile=look, identity_reference=True
        ),
    )

    constraints = build_goal_payload_from_character_config(request)["constraints"]

    assert constraints["visual_style_profile"]["id"] == look["style"]["id"]
    assert constraints["canvas_width"] == look["canvas"]["width"]
    assert constraints["canvas_height"] == look["canvas"]["height"]
    assert constraints["identity_reference"] is True
