"""Story-card body copy layout; the card font is a real local dependency."""

import re

from PIL import Image, ImageDraw
import pytest

from agentic.runtime.story_cards import _wrap_text, load_story_card_font, resolve_story_card_font


BODY = "他每天看著日曆，直到2034年才發現日子早已用完，還有AI2000次提醒沒有人回覆。"


@pytest.mark.parametrize("max_width", range(200, 420, 17))
def test_user_given_a_page_with_years_and_latin_names_when_the_text_wraps_then_each_number_stays_on_one_line(max_width):
    """User Given page text holding a year and a Latin-letter name When the body wraps at any card width Then every such run is read whole on a single line and no text is lost."""
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    font = load_story_card_font(resolve_story_card_font(), 40)

    lines = _wrap_text(draw, BODY, font, max_width)

    assert "".join(lines) == BODY
    for run in re.findall(r"[A-Za-z0-9]+", BODY):
        assert any(run in line for line in lines), run


QUOTED = "他說「ABCDEFGHIJKLMNOPQRSTUVWXYZ0123」然後走開。"


@pytest.mark.parametrize("max_width", range(200, 420, 7))
def test_user_given_a_quoted_latin_name_when_the_text_wraps_then_no_line_ends_with_an_opening_bracket(max_width):
    """User Given a page quoting a long Latin name inside brackets When the body wraps at any card width Then an opening bracket never ends a line and no text is lost."""
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    font = load_story_card_font(resolve_story_card_font(), 40)

    lines = _wrap_text(draw, QUOTED, font, max_width)

    assert "".join(lines) == QUOTED
    assert not any(line and line[-1] in "（「『【《〈" for line in lines)
