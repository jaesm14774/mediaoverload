from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic.runtime.creative_reflection import inspect_recent_runs, render_markdown, write_report


class CreativeReflectionTests(unittest.TestCase):
    def _write_run(self, root: Path, run_id: str, updated_at: str, *, prompt: str, storyboard_path: str, segments: list[dict[str, str]], review_text: str = "", review_status: str = "reject", qa: dict | None = None, strategy: str = "text2longvideo") -> None:
        run_dir = root / "agentic" / "logs" / "runs" / run_id
        run_dir.mkdir(parents=True)
        review_path = root / "output" / "kirby" / "review_sessions" / f"{run_id}.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps({"status": review_status, "text": review_text, "selected_paths": []}), encoding="utf-8")
        records = [
            {"node_id": "idea-brief", "outputs": {"prompt": prompt, "creative_brief": prompt}},
            {"node_id": "script-plan", "outputs": {"segments": segments}},
            {"node_id": "review-select", "outputs": {"review_session_path": f"/app/output/kirby/review_sessions/{run_id}.json"}},
        ]
        if qa is not None:
            records.append({"node_id": "native-h3-qa", "outputs": {"semantic_qa": qa, "technical_qa": {"passed": True}}})
        manifest = {
            "run_id": run_id,
            "updated_at": updated_at,
            "status": "failed",
            "failure_node": "review-select",
            "failure_reason": "Human reviewer rejected all candidates.",
            "source_generation_type": strategy,
            "routing_summary": {"strategy": strategy, "prompt": prompt},
            "plan": {"goal": {"prompt": prompt, "constraints": {"storyboard_path": storyboard_path}}},
            "generation": {"result": {"records": records}},
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_detects_static_storyboard_drift_and_missing_review_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_run(
                root,
                "run-a",
                "2026-08-19T10:00:00+00:00",
                prompt="Kirby escapes a financial storm in Tokyo using a backpack umbrella",
                storyboard_path="configs/storyboards/retired_static_meadow.yaml",
                segments=[
                    {"segment_id": "1", "visual": "Kirby stands in a meadow beside a star seed", "action": "", "start_state": "meadow", "end_state": "star seed"},
                    {"segment_id": "2", "visual": "Kirby carries the star seed", "action": "walks", "start_state": "living room", "end_state": "sky"},
                ],
                review_text="未提供故事摘要",
            )
            report = inspect_recent_runs(root, count=1)
            run = report["runs"][0]
            categories = {cause["category"] for cause in run["root_causes"]}
            self.assertIn("storyboard_drift", categories)
            self.assertIn("segment_contract_gap", categories)
            self.assertIn("continuity_break", categories)
            self.assertIn("review_context_missing", categories)

    def test_separates_publish_or_qa_boundary_from_creative_causes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_run(
                root,
                "run-b",
                "2026-08-19T11:00:00+00:00",
                prompt="Kirby catches one balloon",
                storyboard_path="",
                segments=[{"segment_id": "1", "visual": "Kirby catches a balloon", "action": "catches", "start_state": "air", "end_state": "balloon held"}],
                review_text="A clear balloon gag",
                review_status="accept",
                qa={"enabled": False, "status": "disabled", "passed": None, "score": 0},
            )
            manifest_path = root / "agentic" / "logs" / "runs" / "run-b" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["failure_node"] = "dispatch-publish"
            manifest["failure_reason"] = "YouTube OAuth failed"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = inspect_recent_runs(root, count=1)
            categories = {cause["category"] for cause in report["runs"][0]["root_causes"]}
            self.assertIn("publish_boundary", categories)
            self.assertNotIn("human_review_rejection_without_reason", categories)

    def test_detects_native_news_mechanism_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_run(
                root,
                "run-news-loop",
                "2026-08-19T12:00:00+00:00",
                prompt="Kirby reacts to a citywide lantern blackout",
                storyboard_path="",
                segments=[{"segment_id": "1", "visual": "Kirby watches a dark street", "action": "looks", "start_state": "street lit", "end_state": "street dark"}],
                review_text="A generic floating orb gag",
                qa={
                    "enabled": True,
                    "status": "failed",
                    "passed": False,
                    "score": 20,
                    "checks": {
                        "news_mechanism_present": True,
                        "news_consequence_present": True,
                        "news_anchor_roles_complete": False,
                        "news_anchor_diversity": False,
                        "news_anchor_not_default_object_loop": False,
                        "news_mechanism_reaches_story": False,
                        "news_consequence_reaches_payoff": False,
                    },
                },
                strategy="native_h3_ref2va",
            )
            report = inspect_recent_runs(root, count=1)
            categories = {cause["category"] for cause in report["runs"][0]["root_causes"]}
            self.assertIn("news_mechanism_collapse", categories)
            self.assertIn("news_mechanism_collapse", report["batch_counts"])
            self.assertIn("v2 mechanism contract", report["recommended_next_experiment"])

    def test_native_report_prefers_story_source_over_numeric_creative_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "agentic" / "logs" / "runs" / "native-source"
            run_dir.mkdir(parents=True)
            records = [
                {
                    "node_id": "native-story-prompt",
                    "outputs": {
                        "story_source": "headline about a secure corridor",
                        "news_context": {"title": "headline about a secure corridor"},
                        "creative_seed": 9847120394,
                        "story_spine": {"premise": "Kirby follows the corridor"},
                    },
                },
            ]
            manifest = {
                "run_id": "native-source",
                "updated_at": "2026-08-19T13:00:00+00:00",
                "status": "success",
                "source_generation_type": "native_h3_story",
                "routing_summary": {"strategy": "native_h3_story", "prompt": "9847120394"},
                "plan": {"goal": {"prompt": "9847120394", "constraints": {}}},
                "generation": {"result": {"records": records}},
            }
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = inspect_recent_runs(root, count=1)

        self.assertEqual(
            report["runs"][0]["story"]["source_prompt"],
            "headline about a secure corridor",
        )

    def test_writes_json_markdown_and_reflection_memory(self) -> None:
        report = {"generated_at": "now", "run_count_inspected": 1, "run_count_requested": 1, "batch_counts": {}, "recommended_next_experiment": "one", "loop": [], "runs": []}
        with tempfile.TemporaryDirectory() as temp:
            paths = write_report(report, Path(temp))
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertIn("creative self-reflection", render_markdown(report))
