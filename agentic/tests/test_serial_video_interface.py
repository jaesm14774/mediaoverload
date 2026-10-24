"""Acceptance tests for serial-video selection in the existing MediaOverload CLI."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import run_media_interface
from agentic.app.main import _serial_publish_configuration


def _write_config(path: Path, *, episodes_per_run: int = 2) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "character": {"name": "Kirby"},
                "generation": {
                    "output_dir": "output",
                    "serial_video": {
                        "enabled": True,
                        "default_duration_seconds": 40,
                        "episodes_per_run": episodes_per_run,
                        "style": "watercolor storybook animation",
                        "discord_review_timeout_seconds": 720,
                        "identity_references": ["assets/kirby_identity.png"],
                        "hashtags": ["#Kirby", "#連載"],
                    },
                },
                "social_media": {
                    "platforms": {
                        "instagram_graph": {"enabled": True},
                        "facebook": {"enabled": True},
                        "youtube": {"enabled": True},
                    },
                    "default_hashtags": ["#MediaOverload"],
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


class SerialRunnerBoundary:
    """Captures the serial CLI contract without invoking external media providers."""

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def __call__(self, argv: list[str]) -> None:
        self.calls.append(argv)
        if self.fail_on_call == len(self.calls):
            raise SystemExit(3)


def test_user_given_serial_video_is_selected_when_interface_runs_then_yaml_defaults_reach_the_serial_runner(tmp_path, capsys):
    """User Given the creator selects serial_video in the existing interface When YAML enables the strategy Then YAML defaults route to the serial runner and do not invoke the ordinary workflow."""
    config_path = _write_config(tmp_path / "kirby.yaml")
    serial = SerialRunnerBoundary()

    run_media_interface.main(
        ["--config", str(config_path), "--generation-type", "serial_video"],
        serial_runner=serial,
    )

    assert len(serial.calls) == 2
    first = serial.calls[0]
    assert "--serial-auto-next" in first
    assert "--serial-mode" in first
    assert "--execute" in first
    assert "--character" in first and first[first.index("--character") + 1] == "Kirby"
    assert "--duration-seconds" in first and first[first.index("--duration-seconds") + 1] == "40"
    assert "--style" in first and first[first.index("--style") + 1] == "watercolor storybook animation"
    assert "--discord-review-timeout-seconds" in first
    assert "--serial-character-reference" in first
    assert first[first.index("--character-config") + 1] == str(config_path.resolve())
    assert "--platform" not in first
    assert "--hashtag" in first
    assert capsys.readouterr().out == ""


def test_user_given_the_selected_character_yaml_enables_three_platforms_when_serial_publish_config_is_loaded_then_all_three_are_targets(tmp_path):
    """User Given the selected character YAML enables Instagram, Facebook, and YouTube When serial publishing configuration is loaded Then all enabled destinations are selected from that same file."""
    config_path = _write_config(tmp_path / "kirby.yaml")

    platforms, platform_configs = _serial_publish_configuration(
        tmp_path,
        "Kirby",
        None,
        config_path,
    )

    assert set(platforms) == {"instagram_graph", "facebook", "youtube"}
    assert set(platform_configs) == set(platforms)


def test_user_given_cli_overrides_and_an_episode_limit_when_serial_video_runs_then_only_the_first_episode_uses_the_supplied_story(tmp_path):
    """User Given explicit duration, story, topic, output, and episode-count choices When serial_video runs Then overrides are honored and later episodes return to automatic idea generation."""
    config_path = _write_config(tmp_path / "kirby.yaml", episodes_per_run=1)
    serial = SerialRunnerBoundary()
    output_dir = tmp_path / "renders"

    run_media_interface.main(
        [
            "--config", str(config_path),
            "--generation-type", "serial_video",
            "--duration-seconds", "60",
            "--serial-episodes", "3",
            "--prompt", "A new firefly adventure",
            "--episode-topic", "火螢的新冒險",
            "--publish-platform", "youtube",
            "--output-dir", str(output_dir),
            "--dry-run-publish",
        ],
        serial_runner=serial,
    )

    assert len(serial.calls) == 3
    first, second, third = serial.calls
    assert first[first.index("--duration-seconds") + 1] == "60"
    assert first[first.index("--goal") + 1] == "A new firefly adventure"
    assert first[first.index("--episode-topic") + 1] == "火螢的新冒險"
    assert first[first.index("--output-dir") + 1] == str(output_dir)
    assert first.count("--platform") == 1
    assert first[first.index("--platform") + 1] == "youtube"
    assert "--dry-run-publish" in first
    for later in (second, third):
        assert "--goal" not in later
        assert "--episode-topic" not in later


def test_user_given_a_discord_failure_during_a_multi_episode_run_when_the_interface_invokes_serial_generation_then_it_stops_before_another_episode(tmp_path):
    """User Given a multi-episode serial request When a Discord-gated serial run fails Then the existing interface propagates failure and starts no later episode."""
    config_path = _write_config(tmp_path / "kirby.yaml", episodes_per_run=4)
    serial = SerialRunnerBoundary(fail_on_call=1)

    with pytest.raises(SystemExit) as failure:
        run_media_interface.main(
            ["--config", str(config_path), "--generation-type", "serial_video"],
            serial_runner=serial,
        )

    assert failure.value.code == 3
    assert len(serial.calls) == 1
