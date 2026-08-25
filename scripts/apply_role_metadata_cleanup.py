"""Apply the final source-normalization cleanup set with exact guards."""

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
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "character_role_audit" / "20260824"
PROPOSAL_PATH = ARTIFACT_DIR / "metadata_update_proposals.json"
REPORT_PATH = ARTIFACT_DIR / "metadata_cleanup_report.json"
TABLE_NAME = "anime.anime_roles"

TARGET_IDS = [
    1448, 1457, 1458, 1459, 1460, 1461, 1462, 1469, 1492, 1497, 1527,
    1529, 1530, 1532, 1533, 1567, 1568, 1570, 1587, 1611, 1612,
]


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
    if any(role_id not in proposals for role_id in TARGET_IDS):
        raise RuntimeError("cleanup target is missing from the proposal artifact")

    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "proposal_path": str(PROPOSAL_PATH),
        "target_ids": TARGET_IDS,
        "committed": False,
        "affected_rows": 0,
        "changed_description_count": 0,
        "changed_keywords_count": 0,
        "updated": [],
    }
    placeholders = ",".join(["%s"] * len(TARGET_IDS))
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id, role_name_en, group_name, status, role_description, keywords "
                f"FROM {TABLE_NAME} WHERE status=1 AND id IN ({placeholders}) ORDER BY id FOR UPDATE",
                TARGET_IDS,
            )
            rows = [dict(row) for row in cursor.fetchall()]
            if [int(row["id"]) for row in rows] != TARGET_IDS:
                raise RuntimeError("cleanup target IDs are not the exact active set")

            for row in rows:
                role_id = int(row["id"])
                proposal = proposals[role_id]
                if int(row["status"] or 0) != 1:
                    raise RuntimeError(f"status guard failed for id={role_id}")
                for field, proposal_field in (
                    ("role_name_en", "role_name_en"),
                    ("group_name", "group_name"),
                    ("role_description", "current_description"),
                    ("keywords", "current_keywords"),
                ):
                    if str(row[field] or "") != str(proposal[proposal_field] or ""):
                        raise RuntimeError(f"current-value guard failed for id={role_id}, field={field}")

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
                        proposal["role_name_en"],
                        proposal["group_name"],
                        proposal["current_description"] or "",
                        proposal["current_keywords"] or "",
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"guarded update affected {cursor.rowcount} rows for id={role_id}")
                report["affected_rows"] += 1
                if proposal["current_description"] != proposal["proposed_description"]:
                    report["changed_description_count"] += 1
                if proposal["current_keywords"] != proposal["proposed_keywords"]:
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
                f"SELECT id, role_description, keywords FROM {TABLE_NAME} "
                f"WHERE status=1 AND id IN ({placeholders}) ORDER BY id",
                TARGET_IDS,
            )
            for row in cursor.fetchall():
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
