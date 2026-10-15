from __future__ import annotations

import json
import random
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import yaml

from agentic.app.creative_profiles import resolve_creative_profile
from agentic.app.style_catalog import StyleLibrary
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.prompting import build_goal_brief
from agentic.tools.style_picker import create_server


CHARACTER_STYLE_IDS = (
    "storybook_watercolor", "soft_pastel_characters", "bold_cartoon_comic",
    "cinematic_painterly", "stylized_3d_game", "photographic_character_fusion",
)


@pytest.fixture
def library(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "source": {"url": "https://example.com/styles"},
        "preview_base_url": "https://example.com/",
        "scenes": [{"id": "animal-landscape", "label": "Animal and Landscape"}],
        "styles": [
            {"id": name, "name": name.title(), "descriptor": f"{name} visual texture",
             "categories": ["Illustration"], "order": index,
             "previews": {"animal-landscape": f"images/{name}.webp"}}
            for index, name in enumerate(("paper", "glass", "ink"))
        ],
    }), encoding="utf-8")
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(yaml.safe_dump({"enabled": ["paper", "glass", "ink"]}), encoding="utf-8")
    return StyleLibrary(catalog_path, selection_path)


@pytest.fixture
def picker(library):
    server = create_server(library=library, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", library
    server.shutdown()
    server.server_close()
    thread.join()


def read_state(base):
    with urlopen(base + "/api/styles") as response:
        return json.load(response)


def save(base, state, ids, **headers):
    request = Request(base + "/api/selection", method="PUT", data=json.dumps({
        "enabled_ids": ids, "revision": state["revision"], "token": state["token"],
    }).encode(), headers={"Content-Type": "application/json", **headers})
    with urlopen(request) as response:
        return json.load(response)


def creative_config():
    return {
        "canvas": {"mode": "fixed", "fixed": "wide", "profiles": {
            "wide": {"aspect_ratio": "16:9", "width": 640, "height": 360},
        }},
        "style": {"mode": "random", "catalog": "krea2"},
    }


def test_user_given_imported_and_character_styles_when_reading_the_config_then_both_are_available():
    """User Given 288 imported styles and the six original character styles When reading the library Then both collections retain their definitions and the imported previews."""
    state = StyleLibrary().snapshot()
    ids = {style["id"] for style in state["styles"]}
    character_ids = set(CHARACTER_STYLE_IDS)
    imported = [style for style in state["styles"] if style["id"] not in character_ids]
    assert character_ids <= ids
    assert len(imported) == 288
    assert set(state["enabled_ids"]) <= ids
    assert all(style["descriptor"] for style in state["styles"])
    assert all(style["previews"] for style in imported)


@pytest.mark.parametrize("style_id", CHARACTER_STYLE_IDS)
def test_user_given_an_original_character_style_when_selected_from_krea_then_its_full_guidance_reaches_generation(tmp_path, style_id):
    """User Given an original character style in the Krea library When selecting it for image and video generation Then its prompt, composition, palette, motion, and avoidance guidance are preserved."""
    catalog = StyleLibrary().catalog
    original = next(style for style in catalog["styles"] if style["id"] == style_id)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(yaml.safe_dump({"enabled": [style_id]}), encoding="utf-8")
    library = StyleLibrary(catalog_path, selection_path)
    style = resolve_creative_profile(creative_config(), rng=random.Random(42), style_library=library)["style"]
    assert style["id"] == style_id
    assert style["prompt"] == original["descriptor"]
    for field in ("composition", "palette", "motion", "avoid"):
        assert style[field] == original[field]
        assert style[field]
    for media_type in ("image", "text2img2video"):
        goal = GoalRequest(
            prompt="Magolor watches a train depart.", media_type=media_type,
            style=style["label"], constraints={"visual_style_profile": style},
        )
        brief = build_goal_brief(goal, style["label"], [])
        for field in ("prompt", "composition", "palette"):
            assert style[field] in brief["prompt"]
        assert style["avoid"] in brief["negative_prompt"]
        if media_type == "image":
            assert style["motion"] not in brief["prompt"]
        else:
            assert style["motion"] in brief["prompt"]


def test_user_given_all_styles_when_opening_the_page_then_the_picker_and_current_selection_are_available(picker):
    """User Given all styles enabled When opening the local page Then the selection controls and saved state are available."""
    base, _ = picker
    with urlopen(base + "/") as response:
        page = response.read().decode()
    assert "儲存並套用" in page
    assert "全選" in page
    state = read_state(base)
    assert set(state["enabled_ids"]) == {"paper", "glass", "ink"}


def test_user_given_unwanted_styles_when_saving_then_reload_and_random_use_only_the_retained_styles(picker):
    """User Given unwanted styles When saving only paper and ink Then reloading preserves those choices and random never selects glass."""
    base, library = picker
    state = read_state(base)
    saved = save(base, state, ["ink", "paper"])
    reloaded = read_state(base)
    assert saved["revision"] == reloaded["revision"]
    assert set(reloaded["enabled_ids"]) == {"paper", "ink"}
    assert yaml.safe_load(library.selection_path.read_text(encoding="utf-8"))["enabled"] == ["paper", "ink"]
    profiles = [resolve_creative_profile(creative_config(), rng=random.Random(seed), style_library=library)
                for seed in range(40)]
    assert {profile["style"]["id"] for profile in profiles} == {"paper", "ink"}
    for profile in profiles:
        assert profile["style"]["prompt"] == f'{profile["style"]["id"]} visual texture'


@pytest.mark.parametrize("ids", [[], ["missing"], ["paper", "paper"], [True], "paper"])
def test_user_given_an_invalid_selection_when_saving_then_the_saved_pool_is_unchanged(picker, ids):
    """User Given an empty, unknown, duplicate, or malformed selection When saving Then a clear error preserves the last valid pool."""
    base, library = picker
    before = library.selection_path.read_bytes()
    with pytest.raises(HTTPError) as error:
        save(base, read_state(base), ids)
    assert error.value.code == 400
    assert library.selection_path.read_bytes() == before


def test_user_given_an_older_page_when_saving_then_newer_config_changes_are_preserved(picker):
    """User Given two pages with the same starting state When one saves first Then the stale page cannot overwrite the newer choices."""
    base, library = picker
    original = read_state(base)
    save(base, original, ["ink"])
    with pytest.raises(HTTPError) as error:
        save(base, original, ["glass"])
    assert error.value.code == 409
    assert library.snapshot()["enabled_ids"] == ["ink"]


def test_user_given_an_unrelated_webpage_when_it_submits_a_selection_then_local_config_is_protected(picker):
    """User Given an unrelated website When it attempts to save a selection Then the local config remains unchanged."""
    base, library = picker
    with pytest.raises(HTTPError) as error:
        save(base, read_state(base), ["ink"], Origin="https://example.com")
    assert error.value.code == 403
    assert len(library.snapshot()["enabled_ids"]) == 3


def test_user_given_the_same_seed_and_saved_pool_when_resolving_then_style_selection_is_reproducible(library):
    """User Given the same seed and saved style pool When resolving creative profiles Then the selected style is reproducible."""
    first = resolve_creative_profile(creative_config(), rng=random.Random(42), style_library=library)
    second = resolve_creative_profile(creative_config(), rng=random.Random(42), style_library=library)
    assert first == second


def test_user_given_a_fixed_catalog_style_when_it_is_excluded_then_the_configuration_fails_clearly(library):
    """User Given a fixed style that was excluded When resolving a run Then the excluded style cannot bypass the chosen pool."""
    state = library.snapshot()
    library.save(["ink"], revision=state["revision"])
    config = creative_config()
    config["style"].update(mode="fixed", fixed="glass")
    with pytest.raises(ValueError, match="glass"):
        resolve_creative_profile(config, style_library=library)


def test_user_given_a_manually_broken_config_when_loading_then_the_page_reports_the_problem(picker):
    """User Given a manually edited config containing an unknown style When loading the picker Then it reports the invalid config instead of silently changing the pool."""
    base, library = picker
    library.selection_path.write_text("enabled: [missing]\n", encoding="utf-8")
    with pytest.raises(HTTPError) as error:
        read_state(base)
    assert error.value.code == 500
    assert "missing" in json.load(error.value)["error"]


@pytest.mark.parametrize("body,content_type", [(b"not-json", "application/json"), (b"[]", "application/json"), (b"{}", "text/plain"), (b"", "application/json")])
def test_user_given_a_malformed_save_when_submitting_then_the_saved_pool_remains_valid(picker, body, content_type):
    """User Given a malformed save request When it reaches the picker Then the saved style pool remains intact."""
    base, library = picker
    request = Request(base + "/api/selection", method="PUT", data=body, headers={"Content-Type": content_type})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 400
    assert len(library.profiles()) == 3


def test_user_given_an_expired_page_token_when_saving_then_the_selection_is_not_changed(picker):
    """User Given an expired page token When saving Then the picker asks for a reload and preserves the selection."""
    base, library = picker
    state = read_state(base)
    state["token"] = "expired"
    with pytest.raises(HTTPError) as error:
        save(base, state, ["ink"])
    assert error.value.code == 403
    assert len(library.profiles()) == 3


@pytest.mark.parametrize("path", ["/app.css", "/app.js"])
def test_user_given_the_picker_when_loading_its_assets_then_styles_and_interactions_are_available(picker, path):
    """User Given the picker When loading its assets Then the page's presentation and interaction files are available."""
    base, _ = picker
    with urlopen(base + path) as response:
        assert response.status == 200
        assert response.read()


def test_user_given_an_unknown_page_when_opening_it_then_the_picker_returns_not_found(picker):
    """User Given an unknown page When opening it Then the picker returns a clear not-found response."""
    base, _ = picker
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/missing")
    assert error.value.code == 404


@pytest.mark.parametrize("catalog", [{"styles": []}, {"styles": [None]}, {"styles": [{"id": "ink", "name": "Ink"}]}, {"styles": [
    {"id": "ink", "name": "Ink", "descriptor": "ink texture"},
    {"id": "ink", "name": "Ink duplicate", "descriptor": "other texture"},
]}])
def test_user_given_an_incomplete_catalog_when_loading_it_then_invalid_style_definitions_are_rejected(library, catalog):
    """User Given an incomplete or duplicate style catalog When loading it Then invalid definitions cannot enter the random pool."""
    library.catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError):
        library.profiles()


@pytest.mark.parametrize("style", [{"catalog": "unknown", "mode": "random"}, {"catalog": "krea2", "mode": "random", "profiles": {}}])
def test_user_given_an_ambiguous_style_source_when_starting_a_run_then_configuration_is_rejected(library, style):
    """User Given an unknown or ambiguous style source When starting a run Then configuration is rejected instead of guessing a source."""
    config = creative_config()
    config["style"] = style
    with pytest.raises(ValueError):
        resolve_creative_profile(config, style_library=library)
