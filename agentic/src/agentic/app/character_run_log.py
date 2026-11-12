"""Turn a finished character workflow result into the run log entry."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import PurePath, PureWindowsPath
from typing import Any

from agentic.runtime.run_log import (
    PublicationEntry,
    ReviewEntry,
    RunEntry,
    derive_review_outcome,
)

REVIEW_SKILL = "review.assets.select"
DISPATCH_NODE = "dispatch-publish"


def serial_run_entry(
    *,
    run_id: str,
    status: str,
    dry_run: bool,
) -> RunEntry:
    """Serial series keep their reviews and publications in the serial ledger.

    The prompt stays as the request recorded it when the run began.
    """
    return RunEntry(
        run_id=run_id,
        status=status,
        strategy="serial_video",
        review_outcome=None,
        dry_run=dry_run,
    )


def project_relative_path(raw: str, root: PurePath) -> str:
    """Record a media location the same way on Windows and in Docker.

    `root` is where this process sees the project (a Windows folder, or `/app`
    in the container), so stripping it leaves the identical `output/...` path
    on both. A file outside the project keeps its absolute path.
    """
    path = type(root)(raw)
    if not path.is_absolute():
        return PureWindowsPath(raw).as_posix()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def run_entry_from_result(
    result: dict[str, Any], *, dry_run: bool, repo_root: PurePath
) -> RunEntry:
    reviews = _reviews(
        "generation", result.get("plan"), _records(result.get("generation"))
    ) + _reviews("publish", *_publish_plan_and_records(result.get("publish")))
    reviews = [replace(review, ordinal=index) for index, review in enumerate(reviews, start=1)]
    news = result.get("news_context") or {}
    return RunEntry(
        run_id=str(result["run_id"]),
        status=str(result["status"]),
        strategy=_text(result.get("source_generation_type")),
        media_type=_text(result.get("agentic_media_type")),
        workflow_name=_text((result.get("plan") or {}).get("workflow_name")),
        character_name=_text(result.get("character")),
        prompt=_text(result.get("prompt")),
        news_title=_text(news.get("title")),
        news_url=_text(news.get("url")),
        review_outcome=derive_review_outcome(review.decision for review in reviews),
        dry_run=dry_run,
        failure_reason=_text(result.get("failure_reason")),
        output_paths=_relative_paths(
            (result.get("artifacts") or {}).get("media_paths"), repo_root
        ),
        reviews=tuple(reviews),
        publications=tuple(_publications(result.get("publish"), repo_root)),
    )


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _records(stage: Any) -> list[dict[str, Any]]:
    result = (stage or {}).get("result") or {}
    return list(result.get("records") or [])


def _publish_plan_and_records(publish: Any) -> tuple[Any, list[dict[str, Any]]]:
    publish = publish or {}
    return publish.get("plan"), _records(publish)


def _reviews(
    plan_name: str, plan: Any, records: list[dict[str, Any]]
) -> list[ReviewEntry]:
    node_inputs = {
        node["node_id"]: node.get("inputs") or {}
        for node in (plan or {}).get("nodes", [])
    }
    reviews: list[ReviewEntry] = []
    for record in records:
        outputs = record.get("outputs") or {}
        decision = _text(outputs.get("review_decision"))
        if record.get("skill_name") != REVIEW_SKILL or decision is None:
            continue
        inputs = node_inputs.get(record["node_id"], {})
        reviews.append(
            ReviewEntry(
                ordinal=0,
                plan=plan_name,
                stage=str(
                    inputs.get("review_phase")
                    or inputs.get("review_scope")
                    or record["node_id"]
                ),
                node_id=str(record["node_id"]),
                decision=decision,
                reviewer=str(outputs.get("reviewer") or ""),
                review_session_id=str(outputs.get("review_session_id") or ""),
                selected_count=_count(outputs.get("selected_count")),
                candidate_count=len(outputs.get("ranked_candidates") or []),
            )
        )
    return reviews


def _relative_paths(paths: Any, repo_root: PurePath) -> tuple[str, ...]:
    return tuple(project_relative_path(str(path), repo_root) for path in paths or ())


def _publications(publish: Any, repo_root: PurePath) -> list[PublicationEntry]:
    result = (publish or {}).get("result") or {}
    node_outputs = (result.get("state") or {}).get("node_outputs") or {}
    dispatch = node_outputs.get(DISPATCH_NODE) or {}
    results = dispatch.get("results") or {}
    skipped = dispatch.get("skipped_platforms") or {}
    errors = dispatch.get("errors") or {}
    receipts = {
        **(dispatch.get("publish_receipts") or {}),
        **(dispatch.get("handoff_receipts") or {}),
    }
    handoffs = dispatch.get("handoffs") or {}
    dispatch_plan = dispatch.get("dispatch_plan") or {}
    dispatched_at = _naive_utc(dispatch.get("published_at"))
    entries: list[PublicationEntry] = []
    for platform in dict.fromkeys([*results, *skipped]):
        receipt = receipts.get(platform) or {}
        platform_plan = dispatch_plan.get(platform) or {}
        if platform in skipped:
            outcome, error = "skipped", str(skipped[platform])
        elif platform in handoffs:
            outcome, error = "awaiting_user_action", ""
        elif results.get(platform):
            outcome, error = "published", ""
        else:
            outcome, error = "failed", str(errors.get(platform) or "")
        verified = receipt.get("verified")
        entries.append(
            PublicationEntry(
                platform=str(platform),
                outcome=outcome,
                external_id=str(receipt.get("external_id") or ""),
                receipt_status=str(receipt.get("status") or ""),
                visibility=str(receipt.get("visibility") or ""),
                verified=None if verified is None else bool(verified),
                error=error,
                caption=str(platform_plan.get("caption") or dispatch.get("caption") or ""),
                media_paths=_relative_paths(
                    platform_plan.get("media_paths") or dispatch.get("media_paths"),
                    repo_root,
                ),
                receipt_details=dict(receipt.get("details") or {}),
                dispatched_at=dispatched_at,
            )
        )
    return entries


def _count(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _naive_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed
