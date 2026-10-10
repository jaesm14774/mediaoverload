from __future__ import annotations

import unittest
import os
from pathlib import Path
from unittest.mock import patch

from agentic.app.main import build_runtime


class RuntimeRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.planner, cls.runner, cls.run_memory = build_runtime(cls.project_root)


    def test_runtime_uses_comfyui_root_environment_for_asset_checks(self) -> None:
        configured_root = self.project_root / ".tmp-tests" / "configured-comfy-root"
        with patch.dict(os.environ, {"COMFYUI_ROOT": str(configured_root)}, clear=False):
            planner, _runner, _memory = build_runtime(self.project_root)

        self.assertEqual(planner.asset_registry.asset_root, configured_root.resolve())


if __name__ == "__main__":
    unittest.main()
