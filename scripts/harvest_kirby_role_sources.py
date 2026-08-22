"""Harvest source-backed Kirby role material without changing MySQL.

The database contains a mixture of canonical characters, variants, enemies,
and several names for which no reliable Kirby character page could be found.
This script creates an auditable source snapshot first. A later apply step must
only write fields that are supported by this snapshot.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "kirby_role_research" / datetime.now().strftime("%Y%m%d")
WIKI_API = "https://kirby.fandom.com/api.php"
WIKI_BASE = "https://kirby.fandom.com/wiki/"
USER_AGENT = "MediaOverloadRoleResearch/1.0 (local data quality audit)"
ROLE_GROUP = "Kirby"

# These are spelling/spacing variants present in the DB, not guesses about a
# different character. Anything outside this explicit set is left unresolved.
CANONICAL_ALIASES = {
    "MetaKnight": "Meta Knight",
    "ChuChu": "Chuchu",
    "KingDedede": "King Dedede",
}


def _api(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{WIKI_API}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


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
                "WHERE group_name=%s ORDER BY id",
                (ROLE_GROUP,),
            )
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def _page_for_title(title: str) -> dict[str, Any] | None:
    payload = _api(
        {
            "action": "query",
            "format": "json",
            "titles": title,
            "redirects": "1",
        }
    )
    pages = payload.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if not page or "missing" in page:
        return None
    return page


def _search_title(title: str) -> str | None:
    payload = _api(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": title,
            "srnamespace": "0",
            "srlimit": "5",
        }
    )
    hits = payload.get("query", {}).get("search", [])
    if not hits:
        return None
    return str(hits[0].get("title") or "").strip() or None


def _wikitext(title: str) -> str:
    payload = _api(
        {
            "action": "parse",
            "format": "json",
            "prop": "wikitext",
            "redirects": "1",
            "page": title,
        }
    )
    return str(payload.get("parse", {}).get("wikitext", {}).get("*", ""))


def _clean_markup(value: str) -> str:
    text = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[([^]|]+)\|([^]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^]]+)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sections(source: str) -> list[tuple[int, str, str]]:
    matches = list(re.finditer(r"^(={2,6})\s*([^=\n]+?)\s*\1\s*$", source, flags=re.M))
    result: list[tuple[int, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        result.append((len(match.group(1)), match.group(2).strip(), source[match.end() : end]))
    return result


def _extract_material(source: str) -> dict[str, str]:
    before_first_section = source[: source.find("==") if "==" in source else len(source)]
    # The infobox is metadata, not visual prose. Keep it separately only as a
    # source hint; the description writer must not copy story relationships.
    infobox = "\n".join(
        line for line in before_first_section.splitlines() if not line.lstrip().startswith("{{")
    )
    appearance = ""
    for level, title, body in _sections(source):
        normalized = re.sub(r"\s+", " ", title).casefold()
        if normalized in {"physical appearance", "appearance"}:
            appearance = _clean_markup(body)
            break
    intro = _clean_markup(before_first_section)
    return {
        "intro": intro[:5000],
        "physical_appearance": appearance[:7000],
        "infobox_hint": _clean_markup(infobox)[:2000],
    }


def main() -> None:
    load_dotenv(REPO_ROOT / "media_overload.env")
    rows = _db_rows()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        db_name = str(row.get("role_name_en") or "").strip()
        lookup = CANONICAL_ALIASES.get(db_name, db_name)
        page = _page_for_title(lookup)
        source_title = str(page.get("title") or "") if page else ""
        resolution = "exact_or_alias" if page else "unresolved"
        if not page and db_name not in CANONICAL_ALIASES:
            # Search is evidence only. We do not auto-accept a fuzzy result.
            search_hit = _search_title(db_name)
            source_title = search_hit or ""
            resolution = "search_candidate" if search_hit else "unresolved"

        record: dict[str, Any] = {
            "role_id": int(row["id"]),
            "role_name_zh": row.get("role_name_zh"),
            "role_name_en": db_name,
            "status_before": int(row.get("status") or 0),
            "source_title": source_title,
            "resolution": resolution,
            "source_url": f"{WIKI_BASE}{urllib.parse.quote(source_title.replace(' ', '_'))}" if source_title else "",
            "source": "Kirby Wiki | Fandom" if source_title else "",
            "material": {},
        }
        if source_title and resolution == "exact_or_alias":
            try:
                material = _extract_material(_wikitext(source_title))
                record["material"] = material
                if not material["intro"] and not material["physical_appearance"]:
                    record["resolution"] = "source_without_extractable_text"
            except Exception as exc:  # retain row-level evidence and continue
                record["resolution"] = "source_fetch_error"
                record["error"] = repr(exc)
        records.append(record)
        print(f"[{index}/{len(rows)}] {db_name}: {record['resolution']}")
        time.sleep(0.2)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "group_name": ROLE_GROUP,
        "row_count": len(rows),
        "source_policy": {
            "primary_reference": "https://kirby.nintendo.com/about/",
            "character_reference": "https://kirby.fandom.com/wiki/Category:Characters",
            "enemy_reference": "https://kirby.fandom.com/wiki/Category:Enemies",
            "fuzzy_search_results_are_not_auto_accepted": True,
            "descriptions_must_be_appearance_only": True,
        },
        "records": records,
    }
    output_path = OUTPUT_ROOT / "source_harvest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
