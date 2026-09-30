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


if __name__ == "__main__":
    unittest.main()
