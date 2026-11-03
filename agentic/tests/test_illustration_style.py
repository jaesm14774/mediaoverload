from __future__ import annotations

import unittest

from agentic.runtime.illustration_style import apply_isolated_subject_art_direction


class IllustrationStyleContractTests(unittest.TestCase):
    def test_user_given_isolated_sticker_when_art_direction_is_added_then_background_contract_is_preserved(self) -> None:
        """User Given a sticker requires an isolated chroma-key background When its subject finish is composed Then the silhouette and background remain isolated without adding scenery."""
        scene = "Kirby on a transparent chroma-key background, full body, centered."

        styled_prompt = apply_isolated_subject_art_direction(scene)

        self.assertIn(scene, styled_prompt)
        self.assertIn("Preserve the requested silhouette", styled_prompt)
        self.assertIn("do not add a floor or environment", styled_prompt)
        self.assertNotIn("foreground, middle distance, and background", styled_prompt)


if __name__ == "__main__":
    unittest.main()
