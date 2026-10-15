"""Run the local selection UI without installing a separate web framework."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentic" / "src"))

from agentic.tools.style_picker import main  # noqa: E402


if __name__ == "__main__":
    main()
