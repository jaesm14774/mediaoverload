import random

import pytest

from agentic.app.serial_look import serial_series_look

CANVAS_PROFILES = {
    "landscape_16_9": {"aspect_ratio": "16:9", "width": 640, "height": 360},
    "portrait_4_5": {"aspect_ratio": "4:5", "width": 512, "height": 640},
    "square_1_1": {"aspect_ratio": "1:1", "width": 512, "height": 512},
}
POOL = ["storybook_watercolor", "soft_pastel_characters", "ghibli-style", "children-s-book-drawing"]


def _generation(style_ids=POOL) -> dict:
    return {
        "creative_profile": {
            "canvas": {
                "mode": "random",
                "random_weights": {"landscape_16_9": 2, "portrait_4_5": 3, "square_1_1": 1},
                "profiles": CANVAS_PROFILES,
            }
        },
        "serial_video": {"style_ids": list(style_ids)},
    }


def test_user_given_a_series_when_any_episode_is_made_then_the_look_is_the_same_every_time():
    """User Given one series When its look is resolved again for a later episode, even after unrelated random draws Then the illustration style and canvas are exactly the same."""
    first = serial_series_look(_generation(), "auto-kirby-abc123")
    random.seed(99)
    random.random()
    second = serial_series_look(_generation(), "auto-kirby-abc123")

    assert first == second


def test_user_given_many_series_when_their_looks_are_drawn_then_each_uses_the_illustration_pool_and_the_series_differ():
    """User Given many series When each gets its look Then every style comes from the configured illustration pool, never outside it, and different series do not all share one style."""
    looks = [serial_series_look(_generation(), f"series-{index}") for index in range(60)]

    styles = {look["style"]["id"] for look in looks}
    assert styles <= set(POOL)
    assert len(styles) >= 2
    assert {look["canvas"]["id"] for look in looks} <= set(CANVAS_PROFILES)


def test_user_given_a_pool_with_an_unknown_style_when_a_look_is_requested_then_it_fails_naming_the_style():
    """User Given a serial style pool containing a style that is not in the catalog When a look is requested Then it stops and names the unknown style instead of silently picking another."""
    with pytest.raises(ValueError, match="not-a-real-style"):
        serial_series_look(_generation(["storybook_watercolor", "not-a-real-style"]), "series-1")


def test_user_given_a_series_look_when_its_style_is_read_then_it_describes_the_medium_the_references_and_clips_must_share():
    """User Given a resolved series look When its style is read Then it carries a label and a prompt describing the medium, which the references and the clips both use."""
    look = serial_series_look(_generation(), "series-1")

    assert look["style"]["label"].strip()
    assert look["style"]["prompt"].strip()
