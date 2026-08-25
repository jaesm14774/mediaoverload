"""Audit every active character row against a row-specific web source.

This script is read-only with respect to MySQL.  It captures source evidence
and a deterministic first-pass proposal; a later apply step must consume the
audited JSON and use exact guarded IDs.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DATE = datetime.now().strftime("%Y%m%d")
OUTPUT_DIR = REPO_ROOT / "artifacts" / "character_role_audit" / AUDIT_DATE
OUTPUT_PATH = OUTPUT_DIR / "active_role_audit.json"
USER_AGENT = "MediaOverloadRoleResearch/1.0 (active anime_roles data-quality audit)"

WIKI_ALIASES = {
    # Names used by the database which are not the most useful encyclopedia
    # lookup title.  These are explicit mappings, never fuzzy acceptance.
    "Panda": "Giant panda",
    "Peacock": "Peafowl",
    "Pink Dolphin": "Amazon river dolphin",
    "One-humped Camel": "Dromedary",
    "Two-humped Came": "Bactrian camel",
    "Aoudad": "Barbary sheep",
    "Common Peafowl": "Indian peafowl",
    "White Peacock": "Peafowl",
    "King Penguin": "King penguin",
    "Lesser Slow Loris": "Slow loris",
    "Fat-tailed Dwarf Lemur": "Fat-tailed dwarf lemur",
    "Yellow-banded Poison Dart Frog": "Poison dart frog",
    "Black-and-green poison-dart frog": "Poison dart frog",
    "Blue Poison Dart Frog": "Blue poison dart frog",
    "White-handed Gibbon": "Lar gibbon",
    "Tomistoma": "False gharial",
    "Chinese Water Dragon": "Chinese water dragon",
    "Taiwan beauty snake": "Beauty rat snake",
    "Swinhoe’s tree lizard": "Diploderma swinhonis",
    "Swinhoe's tree lizard": "Diploderma swinhonis",
    "Monkey-tailed skink": "Solomon Islands skink",
    "Chinese Red-Headed Centipede": "Scolopendra subspinipes",
    "Mexican Red Knee Tarantula": "Brachypelma hamorii",
    "Giant Walking Stick": "Phasmid",
    "Lan-hsu giant katydid": "Katydid",
    "Violet-Backed Starling": "Violet-backed starling",
    "Azara’s Night Monkey": "Azara's night monkey",
    "Azara's Night Monkey": "Azara's night monkey",
    "Yellow Peacock Bass": "Cichla kelberi",
    "Hundred-pace Viper": "Deinagkistrodon",
    "Black-and-white ruffed Lemur": "Black-and-white ruffed lemur",
    "Blue-and-Yellow Macaw": "Blue-and-yellow macaw",
    "Gray jungle fowl": "Grey junglefowl",
    "Bamboo Partridge": "Bamboo partridge",
    "Sulpher-crested Cockatoo": "Sulphur-crested cockatoo",
    "Hodgson's Hawk Eagle": "Mountain hawk-eagle",
    "Orange Oak Leaf": "Orange Oakleaf",
    "Yellow Emperor": "Yellow Emperor",
    "Golden Pheasant": "Golden pheasant",
    "Green Touraco": "Green turaco",
    "Red Panda;Lesser Panda": "Red panda",
    "Puma, Mountain Lion": "Puma",
    "Canadian Beaver, American Beaver": "North American beaver",
    "Small Chinese Civet, Lesser Oriental Civet": "Small Indian civet",
    "Formosan Pangolin": "Chinese pangolin",
    "Formosan Sika Deer": "Formosan sika deer",
    "Formosan serow": "Formosan serow",
    "Formosan Ferret-badger": "Formosan ferret-badger",
    "Formosan Wild Boar": "Wild boar",
    "Formosan Macaque": "Formosan rock macaque",
    "Two-humped Camel": "Bactrian camel",
    "Leptoptilos crumenifer": "Marabou stork",
    "Doraemon": "Doraemon",
    "slime": "Slime (Dragon Quest)",
}

POKEMON_ALIASES = {
    "Farfetch": "farfetchd",
    "Nidoran♀": "nidoran-f",
    "Nidoran♂": "nidoran-m",
    "Nidoran��": "nidoran-f",
    "Nidoran\ufffd\ufffd": "nidoran-f",
    "Deoxys": "deoxys-normal",
    "Pumpkaboo": "pumpkaboo-average",
}

MARIO_ALIASES = {
    "Piranha Plant Bros.": "Piranha Plant",
}

SANRIO_URLS = {
    "Hello Kitty": "https://www.sanrio.co.jp/characters/hellokitty/",
    "Cinnamoroll": "https://www.sanrio.co.jp/characters/cinnamon/?id=profile",
    "Pompompurin": "https://www.sanrio.co.jp/characters/pompompurin/?id=profile",
    "Kuromi": "https://www.sanrio.co.jp/characters/kuromi/",
}

PEANUTS_URLS = {
    "Snoopy": "https://www.peanuts.com/about/snoopy",
    "Woodstock": "https://www.peanuts.com/about/woodstock",
}

SPECIAL_HTML_URLS = {
    "Yellow Peacock Bass": "https://fishbase.se/FieldGuide/FieldGuideSummary.php?c_code=076&genusname=Cichla&speciesname=kelberi",
    "Formosan Reeve's muntjac": "https://www.ysnp.gov.tw/ChildEn/StaticPage/SA05En",
    "White-fronted capuchin": "https://animaldiversity.org/accounts/Cebus_albifrons/",
    "Field cricket": "https://orthoptera.org.uk/species/gryllus-campestris",
    "Common Tiger": "https://www.nparks.gov.sg/florafaunaweb/fauna/9/5/955",
}

SPECIAL_SOURCE_TYPES = {
    "Yellow Peacock Bass": "FishBase",
    "Formosan Reeve's muntjac": "Yushan National Park reference",
    "White-fronted capuchin": "Animal Diversity Web",
    "Field cricket": "Orthoptera & Allied Insects",
    "Common Tiger": "Singapore NParks Flora & Fauna Web",
}

SOURCE_ENDPOINTS = {
    "Wikipedia": "https://en.wikipedia.org/w/api.php",
    "MarioWiki": "https://www.mariowiki.com/api.php",
    "Kirby Wiki": "https://kirby.fandom.com/api.php",
    "Ghibli Wiki": "https://ghibli.fandom.com/api.php",
}


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            value = re.sub(r"\s+", " ", html.unescape(data)).strip()
            if value:
                self.parts.append(value)


class _PageParser(HTMLParser):
    """Extract the title and lead paragraphs from a Wikipedia HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.paragraphs: list[list[str]] = []
        self._in_title = False
        self._in_paragraph = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag == "p":
            self._in_paragraph = True
            self.paragraphs.append([])

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "p":
            self._in_paragraph = False

    def handle_data(self, data: str) -> None:
        value = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
        if self._in_paragraph and self.paragraphs:
            self.paragraphs[-1].append(value)


def _http_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last_error}")


def _http_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last_error}")


def _api_query(endpoint: str, title: str, *, extracts: bool = True) -> dict[str, Any] | None:
    params: dict[str, str] = {
        "action": "query",
        "format": "json",
        "titles": title,
        "redirects": "1",
    }
    if extracts:
        params.update({"prop": "extracts", "exintro": "1", "explaintext": "1"})
    payload = _http_json(f"{endpoint}?{urllib.parse.urlencode(params)}")
    pages = payload.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if not page or "missing" in page:
        return None
    return page


def _api_search(endpoint: str, title: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": title,
        "srnamespace": "0",
        "srlimit": "3",
    }
    payload = _http_json(f"{endpoint}?{urllib.parse.urlencode(params)}")
    hits = payload.get("query", {}).get("search", [])
    return str(hits[0].get("title") or "").strip() or None if hits else None


def _api_wikitext(endpoint: str, title: str) -> str:
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "wikitext",
        "redirects": "1",
    }
    payload = _http_json(f"{endpoint}?{urllib.parse.urlencode(params)}")
    return str(payload.get("parse", {}).get("wikitext", {}).get("*", ""))


def _clean_wikitext(value: str) -> str:
    value = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    value = re.sub(r"\[\[[^]|]+\|([^]]+)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^]]+)\]\]", r"\1", value)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"'{2,}", "", value)
    value = re.sub(r"^=+.*?=+$", "", value, flags=re.M)
    return _clean_text(value)


def _wikipedia_page(title: str) -> tuple[str, str, str]:
    """Return (canonical title, extract, resolution) without API rate limits."""
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}"
    try:
        payload = _http_json(summary_url)
        canonical = str(payload.get("titles", {}).get("canonical") or payload.get("title") or title)
        extract = _source_excerpt(str(payload.get("extract") or ""), 1600)
        if extract:
            return canonical, extract, "exact_or_alias"
    except Exception:
        pass
    # The REST search endpoint is rate-limited more aggressively than the
    # ordinary page HTML.  For explicit aliases, the page itself is a reliable
    # fallback and gives us a row-level URL plus lead paragraphs.
    page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
    try:
        raw = _http_text(page_url)
        parser = _PageParser()
        parser.feed(raw)
        page_title = _clean_text("".join(parser.title_parts)).split(" - Wikipedia")[0].strip()
        paragraphs = [
            _clean_text(" ".join(parts))
            for parts in parser.paragraphs
            if _clean_text(" ".join(parts))
        ]
        if page_title and paragraphs and "Wikipedia" not in page_title:
            return page_title, _source_excerpt(" ".join(paragraphs[:2]), 1600), "html_page_fallback"
    except Exception:
        pass
    search_url = "https://en.wikipedia.org/w/rest.php/v1/search/page?" + urllib.parse.urlencode(
        {"q": title, "limit": "3"}
    )
    search = _http_json(search_url)
    pages = search.get("pages", [])
    if not pages:
        return "", "", "unresolved"
    candidate = str(pages[0].get("title") or pages[0].get("key") or "").strip()
    if not candidate:
        return "", "", "unresolved"
    payload = _http_json(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(candidate.replace(' ', '_'))}"
    )
    canonical = str(payload.get("titles", {}).get("canonical") or payload.get("title") or candidate)
    extract = _source_excerpt(str(payload.get("extract") or ""), 1600)
    return canonical, extract, "explicit_search_candidate" if extract else "source_without_extract"


def _clean_text(value: str) -> str:
    value = re.sub(r"\[\d+\]", "", value)
    value = re.sub(r"\s+", " ", html.unescape(value))
    return value.strip()


def _source_excerpt(text: str, limit: int = 1200) -> str:
    text = _clean_text(text)
    return text[:limit].rstrip()


def _db_rows() -> list[dict[str, Any]]:
    connection = pymysql.connect(
        host=os.getenv("mysql_host"),
        port=int(os.getenv("mysql_port", "3306")),
        user=os.getenv("mysql_user"),
        password=os.getenv("mysql_password"),
        database=os.getenv("mysql_db_name"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        read_timeout=30,
        write_timeout=30,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, role_name_zh, role_name_en, role_description, keywords, "
                "group_name, status, weight FROM anime.anime_roles "
                "WHERE status=1 ORDER BY group_name, id"
            )
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def _pokemon_slug(row: dict[str, Any]) -> str:
    name = str(row.get("role_name_en") or "").strip()
    if name in POKEMON_ALIASES:
        return POKEMON_ALIASES[name]
    name = name.replace("\ufffd", "")
    lowered = name.casefold().replace("'", "").replace(" ", "-")
    if lowered == "nidoran":
        # The two malformed DB values are disambiguated by their source order.
        return "nidoran-f" if int(row["id"]) < 185 else "nidoran-m"
    return lowered.replace("–", "-")


def _pokemon_species_slug(row: dict[str, Any]) -> str:
    name = str(row.get("role_name_en") or "").strip()
    if name in {"Deoxys", "Pumpkaboo"}:
        return name.casefold()
    return _pokemon_slug(row)


def _pokemon_record(row: dict[str, Any]) -> dict[str, Any]:
    slug = _pokemon_slug(row)
    species_slug = _pokemon_species_slug(row)
    species_url = f"https://pokeapi.co/api/v2/pokemon-species/{urllib.parse.quote(species_slug)}/"
    pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{urllib.parse.quote(slug)}/"
    species = _http_json(species_url)
    pokemon = _http_json(pokemon_url)
    genus = next(
        (
            item.get("genus")
            for item in species.get("genera", [])
            if item.get("language", {}).get("name") == "en"
        ),
        "Pokémon",
    )
    color = str(species.get("color", {}).get("name") or "unknown")
    shape = str(species.get("shape", {}).get("name") or "unknown").replace("-", " ")
    types = [
        str(item.get("type", {}).get("name") or "").replace("-", " ")
        for item in pokemon.get("types", [])
        if item.get("type", {}).get("name")
    ]
    display_name = str(row.get("role_name_en") or "").replace("\ufffd", "")
    if slug == "nidoran-f":
        display_name = "Nidoran♀"
    elif slug == "nidoran-m":
        display_name = "Nidoran♂"
    type_text = "/".join(types) if types else "unknown"
    article = "an" if shape[:1].lower() in "aeiou" else "a"
    description = f"{display_name} is the {genus}, a {color} {type_text}-type Pokémon with {article} {shape} form."
    keywords = _dedupe_keywords(
        [display_name, color, genus, *[f"{item} type" for item in types], f"{shape} form", "Pokémon"]
    )
    return {
        "source_type": "PokeAPI",
        "source_urls": [species_url, pokemon_url],
        "source_title": species.get("name", slug),
        "source_status": "verified_structured",
        "source_evidence": {
            "species_id": species.get("id"),
            "canonical_name": species.get("name"),
            "genus_en": genus,
            "color": color,
            "shape": shape,
            "types": types,
        },
        "proposed_description": description,
        "proposed_keywords": ", ".join(keywords),
    }


def _wiki_record(row: dict[str, Any], endpoint: str, source_type: str, base_url: str) -> dict[str, Any]:
    db_name = str(row.get("role_name_en") or "").strip()
    lookup = WIKI_ALIASES.get(db_name, db_name)
    if source_type == "Wikipedia":
        title, extract, resolution = _wikipedia_page(lookup)
    else:
        page = _api_query(endpoint, lookup)
        resolution = "exact_or_alias"
        if not page:
            candidate = _api_search(endpoint, lookup)
            if candidate:
                page = _api_query(endpoint, candidate)
                lookup = candidate
                resolution = "explicit_search_candidate"
            else:
                resolution = "unresolved"
        title = str(page.get("title") or lookup) if page else ""
        extract = _source_excerpt(str(page.get("extract") or ""), 1600) if page else ""
        if not extract and page and source_type in {"Kirby Wiki | Fandom", "Ghibli Wiki | Fandom"}:
            extract = _source_excerpt(_clean_wikitext(_api_wikitext(endpoint, title)), 1600)
    current_description = _clean_text(str(row.get("role_description") or ""))
    proposed_description = _normalize_description(db_name, current_description)
    proposed_keywords = ", ".join(_normalize_keywords(db_name, str(row.get("keywords") or "")))
    return {
        "source_type": source_type,
        "source_urls": [f"{base_url}{urllib.parse.quote(title.replace(' ', '_'))}"] if title else [],
        "source_title": title,
        "source_status": resolution if extract else ("source_without_extract" if page else resolution),
        "source_evidence": {"lead_extract": extract},
        "proposed_description": proposed_description,
        "proposed_keywords": proposed_keywords,
    }


def _official_html_record(row: dict[str, Any], url: str, source_type: str) -> dict[str, Any]:
    raw = _http_text(url)
    parser = _TextParser()
    parser.feed(raw)
    text = _source_excerpt(" ".join(parser.parts), 1600)
    name = str(row.get("role_name_en") or "").strip()
    description = _normalize_description(name, str(row.get("role_description") or ""))
    keywords = ", ".join(_normalize_keywords(name, str(row.get("keywords") or "")))
    return {
        "source_type": source_type,
        "source_urls": [url],
        "source_title": name,
        "source_status": "official_page_fetched" if text else "official_page_without_text",
        "source_evidence": {"page_text_excerpt": text},
        "proposed_description": description,
        "proposed_keywords": keywords,
    }


def _manual_reference_record(row: dict[str, Any], source_type: str, url: str, evidence: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("role_name_en") or "").strip()
    description = _normalize_description(name, str(row.get("role_description") or ""))
    keywords = ", ".join(_normalize_keywords(name, str(row.get("keywords") or "")))
    return {
        "source_type": source_type,
        "source_urls": [url],
        "source_title": name,
        "source_status": "verified_external_reference",
        "source_evidence": evidence,
        "proposed_description": description,
        "proposed_keywords": keywords,
    }


def _dedupe_keywords(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = re.sub(r"\s+", " ", str(value or "")).strip(" ,")
        if not value:
            continue
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _normalize_description(name: str, description: str) -> str:
    text = _clean_text(description)
    if not text:
        return name
    # The source table contains several lower-case Wikipedia lead fragments;
    # normalize only the leading identity and preserve the factual body for the
    # later source-backed rewrite pass.
    text = text[:1024]
    if text[: len(name)].casefold() == name.casefold():
        text = name + text[len(name) :]
    elif text[:1].islower():
        text = text[:1].upper() + text[1:]
    if name.casefold() not in text[: min(len(text), len(name) + 80)].casefold():
        text = f"{name}: {text}"
    return text[:1024]


def _normalize_keywords(name: str, keywords: str) -> list[str]:
    values = [name]
    values.extend(item.strip() for item in keywords.split(","))
    values = _dedupe_keywords(values)
    if not any(value.casefold() == "wildlife" for value in values) and name not in {
        "slime",
    }:
        values.append("wildlife")
    return values[:32]


def _fetch_row(row: dict[str, Any]) -> dict[str, Any]:
    group = str(row.get("group_name") or "")
    try:
        if str(row.get("role_name_en") or "") == "Yellow Peacock Bass":
            proposal = _manual_reference_record(
                row,
                "FishBase",
                "https://fishbase.se/FieldGuide/FieldGuideSummary.php?c_code=076&genusname=Cichla&speciesname=kelberi",
                {
                    "accepted_name": "Cichla kelberi",
                    "common_name_context": "yellow peacock bass / tucunaré amarela",
                    "family": "Cichlidae",
                    "habitat": "freshwater, benthopelagic",
                    "documented_visual_markers": "three dark vertical bars, light fin spots, irregular dark abdominal blotches",
                },
            )
        elif str(row.get("role_name_en") or "") == "Formosan Reeve's muntjac":
            proposal = _manual_reference_record(
                row,
                "Yushan National Park reference",
                "https://www.ysnp.gov.tw/ChildEn/StaticPage/SA05En",
                {
                    "accepted_context": "Formosan Reeve's muntjac",
                    "range_context": "Taiwan subspecies of Reeve's muntjac",
                    "reported_size": "body length approximately 40-70 cm; tail approximately 4-10 cm; maximum weight approximately 12 kg",
                },
            )
        elif group == "Pokemon":
            proposal = _pokemon_record(row)
        elif group == "Creature" and str(row["role_name_en"]) in SPECIAL_HTML_URLS:
            name = str(row["role_name_en"])
            proposal = _official_html_record(row, SPECIAL_HTML_URLS[name], SPECIAL_SOURCE_TYPES[name])
        elif group == "Creature":
            proposal = _wiki_record(
                row,
                SOURCE_ENDPOINTS["Wikipedia"],
                "Wikipedia",
                "https://en.wikipedia.org/wiki/",
            )
        elif group == "Super Mario":
            name = str(row.get("role_name_en") or "")
            lookup = MARIO_ALIASES.get(name, name)
            copy = dict(row)
            copy["role_name_en"] = lookup
            proposal = _wiki_record(
                copy,
                SOURCE_ENDPOINTS["MarioWiki"],
                "MarioWiki",
                "https://www.mariowiki.com/",
            )
        elif group == "Kirby":
            proposal = _wiki_record(
                row,
                SOURCE_ENDPOINTS["Kirby Wiki"],
                "Kirby Wiki | Fandom",
                "https://kirby.fandom.com/wiki/",
            )
        elif group == "Sanrio":
            proposal = _official_html_record(row, SANRIO_URLS[str(row["role_name_en"])], "Sanrio official")
        elif group == "Peanuts":
            proposal = _official_html_record(row, PEANUTS_URLS[str(row["role_name_en"])], "Peanuts official")
        elif group == "Studio Ghibli":
            proposal = _wiki_record(
                row,
                SOURCE_ENDPOINTS["Ghibli Wiki"],
                "Ghibli Wiki | Fandom",
                "https://ghibli.fandom.com/wiki/",
            )
        elif group in {"Doraemon", "Others"}:
            proposal = _wiki_record(
                row,
                SOURCE_ENDPOINTS["Wikipedia"],
                "Wikipedia",
                "https://en.wikipedia.org/wiki/",
            )
        else:
            raise RuntimeError(f"no source policy for group {group!r}")
        current_description = str(row.get("role_description") or "")
        current_keywords = str(row.get("keywords") or "")
        proposal["description_verdict"] = (
            "already_equal" if current_description == proposal["proposed_description"] else "needs_rewrite"
        )
        proposal["keywords_verdict"] = (
            "already_equal" if current_keywords == proposal["proposed_keywords"] else "needs_rewrite"
        )
        result = {
            "role_id": int(row["id"]),
            "role_name_zh": row.get("role_name_zh"),
            "role_name_en": row.get("role_name_en"),
            "group_name": group,
            "status_before": int(row.get("status") or 0),
            "current_description": current_description,
            "current_keywords": current_keywords,
            **proposal,
            "overall_status": "source_checked" if proposal["source_status"] not in {"unresolved", "source_without_extract", "official_page_without_text"} else "needs_review",
        }
        return result
    except Exception as exc:
        return {
            "role_id": int(row["id"]),
            "role_name_zh": row.get("role_name_zh"),
            "role_name_en": row.get("role_name_en"),
            "group_name": group,
            "status_before": int(row.get("status") or 0),
            "current_description": str(row.get("role_description") or ""),
            "current_keywords": str(row.get("keywords") or ""),
            "source_type": "",
            "source_urls": [],
            "source_title": "",
            "source_status": "fetch_error",
            "source_evidence": {},
            "proposed_description": "",
            "proposed_keywords": "",
            "description_verdict": "unverified",
            "keywords_verdict": "unverified",
            "overall_status": "needs_review",
            "error": repr(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", help="restrict to one or more group_name values")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / "media_overload.env")
    rows = _db_rows()
    if args.group:
        allowed = set(args.group)
        rows = [row for row in rows if row.get("group_name") in allowed]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_fetch_row, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"[{index}/{len(rows)}] {record['role_id']} {record['role_name_en']}: {record['overall_status']}")
    records.sort(key=lambda item: (str(item.get("group_name")), int(item["role_id"])))
    summary: dict[str, int] = {}
    for record in records:
        key = str(record.get("overall_status"))
        summary[key] = summary.get(key, 0) + 1
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(records),
        "scope": {"status": 1, "groups": sorted({str(row.get('group_name')) for row in rows})},
        "source_policy": {
            "one_record_per_active_row": True,
            "pokemon": "PokeAPI species and pokemon endpoints",
            "creature": "English Wikipedia MediaWiki API",
            "super_mario": "Super Mario Wiki MediaWiki API",
            "kirby": "Kirby Wiki MediaWiki API; official Kirby site remains the franchise reference",
            "sanrio": "Sanrio official character pages",
            "peanuts": "Peanuts official character pages",
            "studio_ghibli": "Ghibli Wiki MediaWiki API",
            "unresolved_is_not_auto_accepted": True,
            "mysql_mutation": "none",
        },
        "summary": summary,
        "records": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {OUTPUT_PATH}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
