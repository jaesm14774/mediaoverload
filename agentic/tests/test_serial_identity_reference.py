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
CARD = "角色名稱：柯比\n外觀辨識特徵：粉紅色球狀身體、紅色小腳、藍色橢圓眼睛\n\n角色參考圖提示\nA pink round creature."


def test_user_given_a_series_style_when_the_identity_prompt_is_built_then_it_carries_that_style_and_asks_for_a_large_centered_character():
    """User Given a series style When the identity reference prompt is built Then it states that style and asks for one large, centred, full-body character on a plain ground, and leaves layout to the reference rather than a random composition."""
    prompt = build_serial_character_reference_prompt(
        character_name="柯比",
        character_card=CARD,
        style_direction="soft watercolor storybook illustration, pale paper grain",
    )

    assert "soft watercolor storybook illustration" in prompt
    assert "置中" in prompt
    assert "約佔畫面高度的三分之二" in prompt
    assert "沿用角色設定檔的既有美術風格" not in prompt


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


def test_user_given_a_series_look_when_identity_references_are_requested_then_the_run_uses_that_style_and_canvas():
    """User Given a series with a locked look When its identity references are requested Then the run's style and canvas are exactly the series look, not a fresh random draw."""
    base = make_character_workflow_request(
        REPO_ROOT,
        KIRBY_CONFIG,
        prompt="reference prompt",
        preferred_generation_type="text2img",
    )
    import yaml

    generation = yaml.safe_load(KIRBY_CONFIG.read_text(encoding="utf-8"))["generation"]
    look = serial_series_look(generation, "auto-kirby-series-1")
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
