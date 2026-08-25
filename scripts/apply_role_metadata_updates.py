"""Apply the completed active-role metadata proposals with exact guards."""

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
REPORT_PATH = REPO_ROOT / "artifacts" / "character_role_audit" / "20260824" / "metadata_update_report.json"
TABLE_NAME = "anime.anime_roles"


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
    proposal_payload = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    proposals = {int(row["role_id"]): row for row in proposal_payload["records"]}
    if len(proposals) != 556 or proposal_payload.get("row_count") != 556:
        raise RuntimeError("proposal artifact must contain exactly 556 rows")

    connection = _connect()
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "proposal_path": str(PROPOSAL_PATH),
        "scope": {"status": 1, "row_count": 556},
        "committed": False,
        "affected_rows": 0,
        "changed_description_count": 0,
        "changed_keywords_count": 0,
        "updated": [],
    }
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id, role_name_en, group_name, status, role_description, keywords "
                f"FROM {TABLE_NAME} WHERE status=1 ORDER BY id FOR UPDATE"
            )
            rows = [dict(row) for row in cursor.fetchall()]
            if len(rows) != 556:
                raise RuntimeError(f"active row count changed before apply: {len(rows)}")
            if {int(row["id"]) for row in rows} != set(proposals):
                raise RuntimeError("active ID set differs from proposal artifact")

            for row in rows:
                role_id = int(row["id"])
                proposal = proposals[role_id]
                if str(row["role_name_en"] or "") != str(proposal["role_name_en"]):
                    raise RuntimeError(f"role_name_en changed concurrently for id={role_id}")
                if str(row["group_name"] or "") != str(proposal["group_name"]):
                    raise RuntimeError(f"group_name changed concurrently for id={role_id}")
                if int(row["status"] or 0) != 1:
                    raise RuntimeError(f"status changed concurrently for id={role_id}")
                if str(row["role_description"] or "") != str(proposal["current_description"] or ""):
                    raise RuntimeError(f"role_description old value differs for id={role_id}")
                if str(row["keywords"] or "") != str(proposal["current_keywords"] or ""):
                    raise RuntimeError(f"keywords old value differs for id={role_id}")

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
                report["affected_rows"] += cursor.rowcount
                if str(row["role_description"] or "") != str(proposal["proposed_description"] or ""):
                    report["changed_description_count"] += 1
                if str(row["keywords"] or "") != str(proposal["proposed_keywords"] or ""):
                    report["changed_keywords_count"] += 1
                report["updated"].append(
                    {
                        "id": role_id,
                        "role_name_en": proposal["role_name_en"],
                        "group_name": proposal["group_name"],
                        "source_type": proposal["source_type"],
                        "source_urls": proposal["source_urls"],
                        "rewrite_basis": proposal["rewrite_basis"],
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
                f"FROM {TABLE_NAME} WHERE status=1 ORDER BY id"
            )
            readback = [dict(row) for row in cursor.fetchall()]
        if len(readback) != 556:
            raise RuntimeError(f"read-back active row count mismatch: {len(readback)}")
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
