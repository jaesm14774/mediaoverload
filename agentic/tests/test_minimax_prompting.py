from __future__ import annotations

import json
import unittest
from pathlib import Path


from agentic.assets.registry import AssetRegistry
from agentic.minimax_prompting import compose_minimax_h3_prompt, subject_reference
from agentic.runtime.contracts import GoalRequest


class ImageConditioningWorkflowTests(unittest.TestCase):


    def test_krea2_keyframe_and_identity_workflows_are_registered(self) -> None:
        """User Given image-assisted H3 When workflows are resolved Then opening
        and continuity recipes validate and continuity preserves source content.
        """
        repo_root = Path(__file__).resolve().parents[2]
        registry = AssetRegistry(repo_root / "agentic", asset_root=repo_root)
        for workflow_name in ("krea2_turbo", "krea2_turbo_img2img"):
            validation = registry.validate_workflow(workflow_name)
            self.assertTrue(validation["valid"], validation)
        continuity = json.loads((repo_root / "configs" / "workflow" / "krea2_turbo_img2img.json").read_text(encoding="utf-8"))
        self.assertGreater(continuity["9"]["inputs"]["denoise"], 0)
        self.assertLess(continuity["9"]["inputs"]["denoise"], 1)


class MiniMaxH3PromptTests(unittest.TestCase):
    def test_user_given_approved_first_frame_when_h3_prompt_is_composed_then_official_i2va_fields_anchor_the_action(self) -> None:
        """User Given an approved first frame and one scene action, When the H3 I2VA prompt is composed, Then it starts with the official frame alignment and contains the three audiovisual fields."""
        prompt = compose_minimax_h3_prompt(
            duration_seconds=5,
            character="MetaKnight",
            style="bold inked 2D comic",
            shots=[
                {
                    "action": "MetaKnight lowers his hand from the mask and turns toward the warning monitor.",
                    "camera": "The camera pushes in slowly while keeping the monitor beside him.",
                    "state_change": "MetaKnight faces the warning with his hand lowered.",
                }
            ],
            render_mode="i2va",
            prior_frame=True,
            overall_soundscape="A soft electronic pulse sounds once under quiet clinic room tone.",
        )

        self.assertTrue(
            prompt.startswith(
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            )
        )
        self.assertIn("integrated_multimodal_description: [Shot 1]", prompt)
        self.assertIn("overall_soundscape: A soft electronic pulse", prompt)
        self.assertIn("non_diegetic_music: N/A", prompt)
        self.assertIn("MetaKnight lowers his hand", prompt)

    def test_user_given_character_profile_when_prompt_is_composed_then_it_is_passed_as_reference(self) -> None:
        reference = subject_reference(
            "Waddle Dee",
            {
                "character_profile": {
                    "role_description": "Waddle Dee has a tan pear-shaped face and no mouth.",
                    "keywords": "Waddle Dee, tan, no mouth",
                }
            },
        )
        self.assertIn("Waddle Dee", reference)
        self.assertIn("tan pear-shaped face and no mouth", reference)
        self.assertNotIn("must", reference.lower())

    def test_local_h3_prompt_uses_context_ir_order_and_i2v_input_relation(self) -> None:
        prompt = compose_minimax_h3_prompt(
            duration_seconds=6,
            character="Kirby",
            style="polished 2D anime",
            story_spine={"objective": "save the seed", "obstacle": "a strong gust"},
            shots=[
                {
                    "time": "0-6s",
                    "title": "Rescue",
                    "action": "Kirby runs and catches the seed",
                    "camera": "tracking shot follows the run",
                    "state_change": "the seed is safe",
                }
            ],
            prior_frame=True,
        )
        self.assertLess(len(prompt), 7000)
        self.assertNotIn("Video action and camera direction:", prompt)
        self.assertIn("integrated_multimodal_description: [Shot 1]", prompt)
        self.assertIn("overall_soundscape:", prompt)
        self.assertLess(prompt.index("aims to save the seed"), prompt.index("Kirby runs and catches the seed"))
        self.assertLess(prompt.index("Kirby runs and catches the seed"), prompt.index("tracking shot follows the run"))
        self.assertNotIn("Input relation:", prompt)

    def test_first_last_frame_prompt_carries_ending_condition(self) -> None:
        prompt = compose_minimax_h3_prompt(
            duration_seconds=15,
            character="Kirby",
            style="polished 2D anime",
            shots=[{"time": "0-15s", "action": "Kirby reaches the closing gate", "state_change": "the gate opens"}],
            render_mode="first_last_frame_to_video",
        )
        self.assertIn("Kirby reaches the closing gate", prompt)
        self.assertIn("the gate opens", prompt)
        self.assertNotIn("Input relation:", prompt)

    def test_pair_prompt_lists_configured_subjects_without_interaction_rules(self) -> None:
        prompt = compose_minimax_h3_prompt(
            duration_seconds=6,
            character="Kirby",
            style="polished 2D anime",
            story_spine={"objective": "push the glowing mechanism together"},
            shots=[
                {
                    "time": "0-6s",
                    "action": "the two Kirby subject slots push the mechanism together",
                    "state_change": "the mechanism opens",
                }
            ],
            subject_context={
                "subjects": [
                    {"role": "primary", "name": "Kirby"},
                    {"role": "secondary", "name": "Kirby"},
                ],
                "interaction_contract": {"required": True, "same_frame": True},
            },
        )

        self.assertIn("Character references", prompt)
        self.assertIn("Kirby", prompt)
        self.assertNotIn("unrequested third subject", prompt)
        self.assertNotIn("must remain", prompt)

    def test_visual_prompt_has_stable_subject_action_camera_order(self) -> None:
        """User Given an H3 first/last-frame story When its prompt is composed Then action and payoff guidance precede the camera rendering details."""
        prompt = compose_minimax_h3_prompt(
            duration_seconds=15,
            character="Kirby",
            style="2D anime",
            story_spine={
                "premise": "a seed falls through a flower garden",
                "objective": "Kirby tries to catch it",
            },
            shots=[
                {
                    "cause": "a flower releases a seed",
                    "action": "Kirby runs toward the falling seed",
                    "camera": "tracking shot",
                    "state_change": "the seed begins to glow",
                    "effect": "the glowing seed surprises Kirby",
                }
            ],
            render_mode="fl2va",
        )
        self.assertLess(
            prompt.index("Kirby runs toward the falling seed"),
            prompt.index("tracking shot"),
        )
        self.assertIn("a flower releases a seed", prompt)
        self.assertIn("the glowing seed surprises Kirby", prompt)
        self.assertNotIn("Input relation:", prompt)

    def test_user_given_subject_scene_when_composed_then_h3_keeps_description_without_default_audio_direction(self) -> None:
        """User Given a subject and scene, When composed for H3, Then the description remains without generic writing guidance or an invented soundtrack."""
        goal = GoalRequest(
            prompt="Kirby races through a neon night market and catches a falling star",
            media_type="long_video",
            duration_seconds=5,
            style="cinematic anime",
            constraints={"character": "Kirby"},
        )
        prompt = compose_minimax_h3_prompt(
            duration_seconds=goal.duration_seconds,
            character=goal.constraints["character"],
            style=goal.style,
            shots=[{"action": "Kirby runs through a glowing market."}],
        )
        self.assertIn("Character: Kirby", prompt)
        self.assertIn("Kirby runs through a glowing market", prompt)
        self.assertNotIn("Audio direction:", prompt)
        self.assertNotIn("Video action and camera direction:", prompt)


if __name__ == "__main__":
    unittest.main()
