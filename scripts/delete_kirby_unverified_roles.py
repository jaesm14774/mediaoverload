"""Delete the explicitly approved unverified Kirby role rows."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_PATH = REPO_ROOT / "artifacts" / "kirby_role_research" / "20260822" / "deleted_roles_backup.json"
ROLE_GROUP = "Kirby"
TARGET_IDS = [
    72, 87, 88, 92, 95, 100, 101, 103, 104, 105, 106, 107, 112, 113, 114,
    115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149, 150, 151, 152, 153, 1032, 1039, 1040, 1687,
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
        read_timeout=30,
        write_timeout=30,
        autocommit=False,
    )


def main() -> None:
    load_dotenv(REPO_ROOT / "media_overload.env")
    if len(TARGET_IDS) != 56 or len(set(TARGET_IDS)) != 56:
        raise RuntimeError("Target ID list must contain 56 unique IDs")

    connection = _connect()
    rows: list[dict[str, Any]] = []
    try:
        with connection.cursor() as cursor:
            placeholders = ",".join(["%s"] * len(TARGET_IDS))
            cursor.execute(
                f"SELECT id, role_name_zh, role_name_en, role_description, keywords, "
                f"group_name, status, weight FROM anime.anime_roles "
                f"WHERE id IN ({placeholders}) ORDER BY id FOR UPDATE",
                TARGET_IDS,
            )
            rows = [dict(row) for row in cursor.fetchall()]
            if len(rows) != len(TARGET_IDS):
                raise RuntimeError(f"Expected 56 target rows, found {len(rows)}")
            if any(row.get("group_name") != ROLE_GROUP for row in rows):
                raise RuntimeError("Target list contains a non-Kirby row")
            if any(int(row.get("status")) != -1 for row in rows):
                raise RuntimeError("Target list contains a row whose status is not -1")

        backup = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": "User-approved deletion of 56 source-insufficient Kirby roles",
            "group_name": ROLE_GROUP,
            "rows": rows,
        }
        BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
        BACKUP_PATH.write_text(json.dumps(backup, ensure_ascii=False, indent=2), encoding="utf-8")

        with connection.cursor() as cursor:
            placeholders = ",".join(["%s"] * len(TARGET_IDS))
            cursor.execute(
                f"DELETE FROM anime.anime_roles WHERE group_name=%s AND id IN ({placeholders})",
                [ROLE_GROUP, *TARGET_IDS],
            )
            if cursor.rowcount != len(TARGET_IDS):
                raise RuntimeError(f"Expected to delete 56 rows, deleted {cursor.rowcount}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    verify_connection = _connect()
    try:
        with verify_connection.cursor() as cursor:
            placeholders = ",".join(["%s"] * len(TARGET_IDS))
            cursor.execute(
                f"SELECT COUNT(*) AS remaining FROM anime.anime_roles WHERE id IN ({placeholders})",
                TARGET_IDS,
            )
            remaining = int(cursor.fetchone()["remaining"])
            cursor.execute(
                "SELECT COUNT(*) AS kirby_count FROM anime.anime_roles WHERE group_name=%s",
                (ROLE_GROUP,),
            )
            kirby_count = int(cursor.fetchone()["kirby_count"])
    finally:
        verify_connection.close()

    if remaining != 0:
        raise RuntimeError(f"Verification failed: {remaining} deleted IDs remain")
    print(json.dumps({
        "deleted": len(TARGET_IDS),
        "remaining_target_rows": remaining,
        "kirby_group_count_after": kirby_count,
        "backup": str(BACKUP_PATH),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
