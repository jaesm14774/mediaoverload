import json
import random

import pytest
import yaml

from agentic.app.serial_look import draw_serial_look, serial_series_look
from agentic.app.style_catalog import StyleLibrary

CANVAS_PROFILES = {
    "landscape_16_9": {"aspect_ratio": "16:9", "width": 640, "height": 360},
    "portrait_4_5": {"aspect_ratio": "4:5", "width": 512, "height": 640},
    "square_1_1": {"aspect_ratio": "1:1", "width": 512, "height": 512},
}
ENABLED = ["paper", "glass", "ink", "clay"]


def _library(tmp_path, enabled=ENABLED) -> StyleLibrary:
    (tmp_path / "catalog.json").write_text(json.dumps({"styles": [
        {"id": name, "name": name.title(), "descriptor": f"{name} visual texture"}
        for name in ("paper", "glass", "ink", "clay", "neon")
    ]}), encoding="utf-8")
    (tmp_path / "krea2.yaml").write_text(yaml.safe_dump({"enabled": list(enabled)}), encoding="utf-8")
    return StyleLibrary("krea2", tmp_path)


def _generation() -> dict:
    return {
        "creative_profile": {
            "canvas": {
                "mode": "random",
                "random_weights": {"landscape_16_9": 2, "portrait_4_5": 3, "square_1_1": 1},
                "profiles": CANVAS_PROFILES,
            },
            "style": {"mode": "random", "model": "krea2"},
        },
    }


def test_user_given_a_series_when_any_episode_is_made_then_the_look_is_the_same_every_time(tmp_path):
    """User Given one series When its look is asked for again for a later episode, even after unrelated random draws Then the illustration style and canvas are exactly the same."""
    (tmp_path / "styles").mkdir()
    library = _library(tmp_path / "styles")
    first = serial_series_look(_generation(), "auto-kirby-abc123", output_root=tmp_path, style_library=library)
    random.seed(99)
    random.random()
    second = serial_series_look(_generation(), "auto-kirby-abc123", output_root=tmp_path, style_library=library)

    assert first == second


def test_user_given_a_series_with_a_style_when_the_enabled_styles_change_then_later_episodes_keep_that_style(tmp_path):
    """User Given a series that already drew its style When the creator unchecks that style, adds others and reweights the canvas Then every later episode still gets the same style and canvas."""
    styles = tmp_path / "styles"
    styles.mkdir()
    library = _library(styles)
    born = serial_series_look(_generation(), "series-keep", output_root=tmp_path, style_library=library)
    later = _generation()
    later["creative_profile"]["canvas"]["random_weights"] = {"square_1_1": 1}
    library.model_path.write_text(
        yaml.safe_dump({"enabled": [name for name in ("paper", "glass", "ink", "clay", "neon") if name != born["style"]["id"]]}),
        encoding="utf-8",
    )

    assert serial_series_look(later, "series-keep", output_root=tmp_path, style_library=library) == born


def test_user_given_many_series_when_their_looks_are_drawn_then_each_uses_an_enabled_style_and_the_series_differ(tmp_path):
    """User Given many series When each gets its look Then every style is one of the model's enabled styles, never one outside them, and different series do not all share one style."""
    styles = tmp_path / "styles"
    styles.mkdir()
    library = _library(styles)
    looks = [
        serial_series_look(_generation(), f"series-{index}", output_root=tmp_path, style_library=library)
        for index in range(60)
    ]

    drawn = {look["style"]["id"] for look in looks}
    assert drawn <= set(ENABLED)
    assert len(drawn) >= 2
    assert {look["canvas"]["id"] for look in looks} <= set(CANVAS_PROFILES)


def test_user_given_a_character_naming_a_model_without_a_style_file_when_the_config_is_checked_then_it_fails_naming_the_model():
    """User Given a serial character whose model has no style file When the config is checked at startup Then it stops naming the model, before any series is created."""
    generation = _generation()
    generation["creative_profile"]["style"]["model"] = "no-such-model"
    with pytest.raises(ValueError, match="no-such-model"):
        draw_serial_look(generation, "config-check")


def test_user_given_the_shipped_krea2_file_when_a_series_look_is_drawn_then_it_resolves_from_the_enabled_styles(tmp_path):
    """User Given the shipped Krea 2 style file and a character that only names its model When a series look is drawn Then it resolves from the enabled styles without any character style list."""
    look = serial_series_look(_generation(), "series-1", output_root=tmp_path)
    assert look["style"]["id"] in StyleLibrary().profiles()


def test_user_given_a_series_look_when_its_style_is_read_then_it_describes_the_medium_the_references_and_clips_must_share(tmp_path):
    """User Given a resolved series look When its style is read Then it carries a label and a prompt describing the medium, which the references and the clips both use."""
    styles = tmp_path / "styles"
    styles.mkdir()
    look = serial_series_look(_generation(), "series-1", output_root=tmp_path, style_library=_library(styles))

    assert look["style"]["label"].strip()
    assert look["style"]["prompt"].strip()
