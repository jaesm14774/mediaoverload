from __future__ import annotations

import unittest

from agentic.runtime.illustration_style import apply_paper_storybook_art_direction


class IllustrationStyleContractTests(unittest.TestCase):
    def test_user_scene_keeps_its_palette_without_default_foliage_cues(self) -> None:
        """User: Given a foliage-free space workshop, When adding the shared storybook style, Then its scene stays foliage-free."""
        scene = "A cobalt and silver orbital workshop with a compact radiator; no plants or leaves."

        styled_prompt = apply_paper_storybook_art_direction(scene)

        self.assertIn(scene, styled_prompt)
        self.assertNotIn("leaf green", styled_prompt.casefold())

    def test_user_given_expressive_comedy_when_shared_style_is_added_then_emotion_is_not_flattened(self) -> None:
        """User Given a playful character-comedy scene When shared storybook art direction is added Then the palette and poses remain vivid and emotionally legible."""
        scene = "Kirby and Bandana Waddle Dee share a comic surprise in a storybook watercolor scene."

        styled_prompt = apply_paper_storybook_art_direction(scene)

        self.assertIn("specific scene environment", styled_prompt)
        self.assertIn("readable foreground, middle distance, and background instead of blank paper", styled_prompt)
        self.assertIn("vivid contrast", styled_prompt)
        self.assertIn("strong diagonal silhouette", styled_prompt)
        self.assertIn("open laughing or startled expressions", styled_prompt)

    def test_user_given_oil_painted_video_frame_when_shared_art_direction_is_added_then_selected_medium_stays_coherent(self) -> None:
        """User: Given a video frame explicitly uses oil paint, When shared illustration direction is added, Then it keeps oil paint instead of imposing watercolor."""
        scene = "Kirby leaps through an oil-painted airfield with visible brush marks and warm light."

        styled_prompt = apply_paper_storybook_art_direction(scene)

        self.assertIn(scene, styled_prompt)
        self.assertIn("selected medium", styled_prompt)
        self.assertNotIn("watercolor and gouache", styled_prompt.casefold())


if __name__ == "__main__":
    unittest.main()
