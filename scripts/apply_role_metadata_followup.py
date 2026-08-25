"""Apply the small post-audit correction set with exact current-value guards."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_PATH = REPO_ROOT / "artifacts" / "character_role_audit" / "20260824" / "metadata_update_proposals.json"
REPORT_PATH = REPO_ROOT / "artifacts" / "character_role_audit" / "20260824" / "metadata_followup_report.json"
TABLE_NAME = "anime.anime_roles"

EXPECTED_CURRENT: dict[int, dict[str, str]] = {
    1373: {
        "role_name_en": "White-fronted capuchin",
        "group_name": "Creature",
        "role_description": "White-fronted capuchin: White-fronted capuchin can refer to any of a number of species of gracile capuchin monkey which used to be considered as the single species Cebus albifrons. White-fronted capuchins are found in seven different countries in South America: Bolivia, Brazil, Colombia, Venezuela, Ecuador, Peru, and Trinidad and Tobago.",
        "keywords": "White-fronted capuchin, white, monkey, wildlife",
    },
    1481: {
        "role_name_en": "Field cricket",
        "group_name": "Creature",
        "role_description": "Field cricket: Field cricket may refer to:",
        "keywords": "Field cricket, insect, wildlife",
    },
    1495: {
        "role_name_en": "Common Tiger",
        "group_name": "Creature",
        "role_description": "Common Tiger: Common tiger may refer to:Ictinogomphus ferox, a dragonfly of Africa Danaus genutia, a butterfly of India, also called the striped tiger Danaus melanippus, a butterfly of tropical Asia, also called the black veined tiger Danaus plexippus, a butterfly of North America, also called the monarch",
        "keywords": "Common Tiger, black, butterfly, tiger, wildlife",
    },
    1591: {
        "role_name_en": "Puma, Mountain Lion",
        "group_name": "Creature",
        "role_description": "Puma, Mountain Lion: Puma or PUMA may refer to:",
        "keywords": "Puma, Mountain Lion, cougar, puma, tawny feline, powerful limbs, long tail, wildlife",
    },
    1592: {
        "role_name_en": "Canadian Beaver, American Beaver",
        "group_name": "Creature",
        "role_description": "Canadian Beaver, American Beaver: The North American beaver is one of two extant beaver species, along with the Eurasian beaver. It is native to North America and has been introduced in South America (Patagonia) and Europe.",
        "keywords": "Canadian Beaver, American Beaver, North American beaver, semiaquatic rodent, brown fur, webbed feet, flat tail, wildlife",
    },
    1613: {
        "role_name_en": "Small Chinese Civet, Lesser Oriental Civet",
        "group_name": "Creature",
        "role_description": "Small Chinese Civet, Lesser Oriental Civet: The small Indian civet is a civet native to South and Southeast Asia. It is listed as least concern on the IUCN Red List because of its widespread distribution, widespread habitat use and healthy populations living in agricultural and secondary landscapes of many range states.",
        "keywords": "Small Chinese Civet, Lesser Oriental Civet, Small Chinese Civet, Lesser Oriental Civet, small, red, wildlife",
    },
}


def _connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=os.getenv("mysql_host"),
        port=int(os.getenv("mysql_port", "3306")),
        user=os.getenv("mysql_user"),
        password=os.getenv("mysql_password"),
        database=os.getenv("mysql_db_name"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        read_timeout=60,
        write_timeout=60,
        autocommit=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", required=True)
    args = parser.parse_args()
    del args
    load_dotenv(REPO_ROOT / "media_overload.env")
    payload = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    proposals = {int(row["role_id"]): row for row in payload["records"]}
    target_ids = sorted(EXPECTED_CURRENT)
    if any(role_id not in proposals for role_id in target_ids):
        raise RuntimeError("follow-up target is missing from the proposal artifact")

    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "proposal_path": str(PROPOSAL_PATH),
        "target_ids": target_ids,
        "committed": False,
        "affected_rows": 0,
        "changed_description_count": 0,
        "changed_keywords_count": 0,
        "updated": [],
    }
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            placeholders = ",".join(["%s"] * len(target_ids))
            cursor.execute(
                f"SELECT id, role_name_en, group_name, status, role_description, keywords "
                f"FROM {TABLE_NAME} WHERE status=1 AND id IN ({placeholders}) ORDER BY id FOR UPDATE",
                target_ids,
            )
            rows = [dict(row) for row in cursor.fetchall()]
            if [int(row["id"]) for row in rows] != target_ids:
                raise RuntimeError("follow-up target IDs are not the exact active set")

            for row in rows:
                role_id = int(row["id"])
                expected = EXPECTED_CURRENT[role_id]
                for field in ("role_name_en", "group_name", "role_description", "keywords"):
                    if str(row[field] or "") != expected[field]:
                        raise RuntimeError(f"current-value guard failed for id={role_id}, field={field}")
                if int(row["status"] or 0) != 1:
                    raise RuntimeError(f"status guard failed for id={role_id}")

            for row in rows:
                role_id = int(row["id"])
                proposal = proposals[role_id]
                cursor.execute(
                    f"UPDATE {TABLE_NAME} SET role_description=%s, keywords=%s "
                    f"WHERE id=%s AND role_name_en=%s AND group_name=%s AND status=1 "
                    f"AND COALESCE(role_description,'')=%s AND COALESCE(keywords,'')=%s",
                    (
                        proposal["proposed_description"],
                        proposal["proposed_keywords"],
                        role_id,
                        EXPECTED_CURRENT[role_id]["role_name_en"],
                        EXPECTED_CURRENT[role_id]["group_name"],
                        EXPECTED_CURRENT[role_id]["role_description"],
                        EXPECTED_CURRENT[role_id]["keywords"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"guarded update affected {cursor.rowcount} rows for id={role_id}")
                report["affected_rows"] += 1
                if EXPECTED_CURRENT[role_id]["role_description"] != proposal["proposed_description"]:
                    report["changed_description_count"] += 1
                if EXPECTED_CURRENT[role_id]["keywords"] != proposal["proposed_keywords"]:
                    report["changed_keywords_count"] += 1
                report["updated"].append(
                    {
                        "id": role_id,
                        "role_name_en": proposal["role_name_en"],
                        "group_name": proposal["group_name"],
                        "source_type": proposal["source_type"],
                        "source_urls": proposal["source_urls"],
                    }
                )
        connection.commit()
        report["committed"] = True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id, role_name_en, group_name, status, role_description, keywords "
                f"FROM {TABLE_NAME} WHERE status=1 AND id IN ({placeholders}) ORDER BY id",
                target_ids,
            )
            readback = [dict(row) for row in cursor.fetchall()]
        for row in readback:
            proposal = proposals[int(row["id"])]
            if str(row["role_description"] or "") != str(proposal["proposed_description"] or ""):
                raise RuntimeError(f"description read-back mismatch for id={row['id']}")
            if str(row["keywords"] or "") != str(proposal["proposed_keywords"] or ""):
                raise RuntimeError(f"keywords read-back mismatch for id={row['id']}")
    finally:
        connection.close()

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {REPORT_PATH}")
    print(json.dumps({key: report[key] for key in ("committed", "affected_rows", "changed_description_count", "changed_keywords_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
