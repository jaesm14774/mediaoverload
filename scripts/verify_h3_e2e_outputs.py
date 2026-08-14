"""Re-run strict technical QA against already-generated H3 E2E artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.runtime.registry import ToolRegistry
from agentic.tools.media_services import register_media_service_tools


MODE_OUTPUTS = {
    "t2va": ("t2va/videos", "Kirby_H3_t2v", 362 / 24),
    "i2va": ("i2va/videos", "Kirby_H3_draft", 124 / 24),
    "fl2va": ("fl2va/videos", "Kirby_H3_native15", 362 / 24),
    "l2va": ("l2va/videos", "Kirby_H3_native15", 362 / 24),
    "ref2va": ("ref2va/videos", "H3_Ref2VA", 124 / 24),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated H3 E2E outputs with strict video/audio QA")
    parser.add_argument("--output-root", default=r"D:\ComfyUI_windows_portable\ComfyUI\output\mediaoverload_h3_p2_e2e")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    tools = ToolRegistry()
    register_media_service_tools(tools, output_root)
    report: dict[str, object] = {"output_root": str(output_root), "reference_audio_enabled": False, "modes": {}}
    for mode, (relative_dir, prefix, duration) in MODE_OUTPUTS.items():
        candidates = sorted((output_root / relative_dir).glob(f"{prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            report["modes"][mode] = {"passed": False, "error": "generated MP4 not found"}
            continue
        video_path = candidates[-1]
        qa = tools.call(
            "media.video_qa",
            {
                "video_path": str(video_path),
                "target_duration": duration,
                "duration_tolerance": 0.9,
                "expected_width": 608,
                "expected_height": 352,
                "expected_fps": 24,
                "require_audio": True,
                "require_stereo_audio": True,
                "analyze_audio": True,
                "warn_if_no_audio": True,
                "contact_sheet_path": str(output_root / mode / "qa" / "contact_sheet_final_verification.jpg"),
                "frame_count": 8,
                "columns": 4,
                "scale_width": 320,
            },
        )
        report["modes"][mode] = {"passed": bool(qa.get("passed")), "video_path": str(video_path), "qa": qa}
        print(f"{'PASS' if qa.get('passed') else 'FAIL'} {mode}: {video_path}")
    report_path = output_root / "h3_modes_p2_final_verification.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT {report_path}")
    return 0 if all(bool(item.get("passed")) for item in report["modes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
