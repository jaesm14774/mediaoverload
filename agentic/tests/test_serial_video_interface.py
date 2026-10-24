"""Acceptance tests for serial-video selection in the existing MediaOverload CLI."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import run_media_interface
from agentic.app.main import _serial_publish_configuration, parse_args
from agentic.app.character_requests import (
    CharacterGenerationOptions,
    CharacterReviewOptions,
    CharacterRuntimeOptions,
    CharacterWorkflowRequest,
)
from agentic.app.serial_video_dispatch import dispatch_serial_video_request


def _write_config(
    path: Path,
    *,
    episodes_per_run: int = 2,
    weighted_serial_only: bool = False,
) -> Path:
    generation = {
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
    }
    if weighted_serial_only:
        generation["generation_type_weights"] = {"serial_video": 1}
        generation["routing_overrides"] = {"strategy_candidates": ["serial_video"]}
    path.write_text(
        yaml.safe_dump(
            {
                "character": {"name": "Kirby"},
                "generation": generation,
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


def test_user_given_a_serial_topic_option_when_main_help_is_requested_then_it_is_described_as_story_planning_input(capsys):
    """User Given a creator reads serial CLI help When --help is requested Then the episode topic is described as planning input rather than public copy."""
    with pytest.raises(SystemExit) as help_exit:
        parse_args(["--help"])

    assert help_exit.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--episode-topic" in help_text
    assert "not added to public captions" in help_text
    assert "shown in serial episode publication copy" not in help_text


def test_user_given_serial_video_has_positive_weight_and_is_selected_by_normal_routing_then_serial_runner_starts(tmp_path, capsys):
    """User Given serial_video is enabled, weighted, and selected by ordinary routing When the interface runs without a generation override Then it starts the existing serial runner."""
    config_path = _write_config(tmp_path / "kirby.yaml", weighted_serial_only=True)
    serial = SerialRunnerBoundary()

    run_media_interface.main(
        ["--config", str(config_path)],
        serial_runner=serial,
    )

    assert len(serial.calls) == 2
    assert all("--serial-mode" in call and "--execute" in call for call in serial.calls)
    assert "--serial-auto-next" in serial.calls[0]
    assert "--media-type" in serial.calls[0]
    assert serial.calls[0][serial.calls[0].index("--media-type") + 1] == "long_video"
    assert '"generation_type": "serial_video"' in capsys.readouterr().out


def test_user_given_a_character_request_selects_serial_video_when_the_shared_dispatcher_runs_then_auto_next_uses_its_idea_generator(tmp_path):
    """User Given a character request selects serial_video When the shared dispatcher runs Then it invokes serial auto-next and preserves request options."""
    config_path = _write_config(tmp_path / "kirby.yaml", episodes_per_run=1)
    serial = SerialRunnerBoundary()
    request = CharacterWorkflowRequest(
        repo_root=tmp_path,
        config_path=config_path,
        generation=CharacterGenerationOptions(
            prompt="A firefly finds a way home.",
            duration_seconds=60,
            output_dir=str(tmp_path / "renders"),
        ),
        review=CharacterReviewOptions(
            dry_run_publish=True,
            publish_platforms=("youtube",),
        ),
        runtime=CharacterRuntimeOptions(comfy_host="127.0.0.1", comfy_port=8188),
    )

    result = dispatch_serial_video_request(request, serial_runner=serial)

    assert result == {
        "status": "success",
        "generation_type": "serial_video",
        "episodes_started": 1,
    }
    command = serial.calls[0]
    assert "--serial-auto-next" in command
    assert "--goal" in command and command[command.index("--goal") + 1] == "A firefly finds a way home."
    assert "--duration-seconds" in command and command[command.index("--duration-seconds") + 1] == "60"
    assert "--platform" in command and command[command.index("--platform") + 1] == "youtube"
    assert "--dry-run-publish" in command
    assert "--comfy-host" in command and command[command.index("--comfy-host") + 1] == "127.0.0.1"


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
