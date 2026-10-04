from __future__ import annotations

import random
import unittest

from agentic.runtime.route_selection import select_weighted_route


class RouteSelectionTests(unittest.TestCase):
    def test_user_given_weighted_random_route_selection_when_draw_repeats_then_repeat_is_allowed(self) -> None:
        """User Given independent weighted draws When the RNG repeats a route Then selection accepts that repeat."""
        rng = random.Random(4)
        candidates = ["image", "short_video", "long_video"]
        routes = [
            select_weighted_route(
                {"image": 1, "short_video": 1, "long_video": 1},
                candidates,
                rng=rng,
            )
            for _ in range(2)
        ]

        self.assertEqual(routes, ["image", "image"])


if __name__ == "__main__":
    unittest.main()
