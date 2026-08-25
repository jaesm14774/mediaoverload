"""Read-only final gate for the active anime_roles metadata update."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import pymysql
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "character_role_audit" / "20260824"
AUDIT_PATH = ARTIFACT_DIR / "active_role_audit.json"
PROPOSAL_PATH = ARTIFACT_DIR / "metadata_update_proposals.json"
REPORT_PATH = ARTIFACT_DIR / "metadata_final_validation.json"


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
    )


def main() -> None:
    load_dotenv(REPO_ROOT / "media_overload.env")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    proposal_payload = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    audit_records = {int(row["role_id"]): row for row in audit["records"]}
    proposals = {int(row["role_id"]): row for row in proposal_payload["records"]}
    issues: list[str] = []

    if audit.get("row_count") != 556 or len(audit_records) != 556:
        issues.append(f"audit row count is {audit.get('row_count')} / {len(audit_records)}")
    if audit.get("summary") != {"source_checked": 556}:
        issues.append(f"audit summary is {audit.get('summary')}")
    if proposal_payload.get("row_count") != 556 or len(proposals) != 556:
        issues.append("proposal artifact is not exactly 556 rows")
    if set(audit_records) != set(proposals):
        issues.append("audit and proposal ID sets differ")

    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, role_name_en, group_name, status, role_description, keywords "
                "FROM anime.anime_roles WHERE status=1 ORDER BY id"
            )
            rows = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                "SELECT role_name_en, group_name, COUNT(*) AS duplicate_count "
                "FROM anime.anime_roles WHERE status=1 "
                "GROUP BY role_name_en, group_name HAVING COUNT(*) > 1"
            )
            duplicates = [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()

    if len(rows) != 556:
        issues.append(f"active DB row count is {len(rows)}")
    if {int(row["id"]) for row in rows} != set(proposals):
        issues.append("active DB ID set differs from proposal ID set")
    if duplicates:
        issues.append(f"duplicate active business keys: {duplicates}")

    suspicious = re.compile(r"may refer to|disambiguation|can refer to|source without extract", re.I)
    for row in rows:
        role_id = int(row["id"])
        name = str(row["role_name_en"] or "")
        description = str(row["role_description"] or "")
        keywords = str(row["keywords"] or "")
        proposal = proposals.get(role_id)
        if proposal is None:
            continue
        if description != str(proposal["proposed_description"] or ""):
            issues.append(f"id={role_id}: description differs from final proposal")
        if keywords != str(proposal["proposed_keywords"] or ""):
            issues.append(f"id={role_id}: keywords differ from final proposal")
        if not description or len(description) > 1024:
            issues.append(f"id={role_id}: empty or oversized description")
        if not keywords or len(keywords) > 1024:
            issues.append(f"id={role_id}: empty or oversized keywords")
        if not description.casefold().startswith(name.casefold()):
            issues.append(f"id={role_id}: description does not start with role name")
        if not keywords.casefold().startswith((name + ",").casefold()):
            issues.append(f"id={role_id}: keywords do not start with role name")
        if suspicious.search(description + " " + keywords):
            issues.append(f"id={role_id}: disambiguation/source residue remains")
        terms = [term.strip().casefold() for term in keywords.split(",") if term.strip()]
        if len(terms) != len(set(terms)):
            issues.append(f"id={role_id}: duplicate keyword term")

    required_features = {
        64: "no mouth",
        90: "no mouth",
        1029: "no feet",
        1136: "no visible mouth",
        1683: "no mouth",
    }
    by_id = {int(row["id"]): row for row in rows}
    for role_id, feature in required_features.items():
        combined = f"{by_id.get(role_id, {}).get('role_description', '')} {by_id.get(role_id, {}).get('keywords', '')}".casefold()
        if feature not in combined:
            issues.append(f"id={role_id}: required feature missing: {feature}")

    report = {
        "active_count": len(rows),
        "source_checked_count": audit.get("summary", {}).get("source_checked", 0),
        "proposal_count": len(proposals),
        "duplicate_business_keys": duplicates,
        "group_counts": dict(Counter(str(row["group_name"]) for row in rows)),
        "source_type_counts": dict(Counter(str(row.get("source_type")) for row in proposals.values())),
        "required_feature_checks": required_features,
        "issue_count": len(issues),
        "issues": issues,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {REPORT_PATH}")
    print(json.dumps({key: report[key] for key in ("active_count", "source_checked_count", "proposal_count", "issue_count")}, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
