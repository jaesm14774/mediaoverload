from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from agentic.runtime.route_selection import select_weighted_route


class RouteSelectionTests(unittest.TestCase):
    def test_persisted_shuffle_bag_spreads_weighted_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "kirby.json"
            routes = [
                select_weighted_route(
                    {"image": 1, "short_video": 1, "long_video": 1},
                    ["image", "short_video", "long_video"],
                    rng=random.Random(4),
                    state_path=state_path,
                    diversity_config={"enabled": True},
                )
                for _ in range(3)
            ]

            self.assertEqual(set(routes), {"image", "short_video", "long_video"})
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(len(state["history"]), 3)

    def test_missing_or_unwritable_state_keeps_weighted_selection_available(self) -> None:
        selected = select_weighted_route(
            {"image": 1},
            ["image"],
            rng=random.Random(4),
            state_path=Path(tempfile.gettempdir()) / "missing-parent" / "route.json",
            diversity_config={"enabled": False},
        )
        self.assertEqual(selected, "image")


if __name__ == "__main__":
    unittest.main()
