"""Character Workflow responsibilities."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from agentic.app.character_requests import CharacterWorkflowRequest
from agentic.app.main import _build_prompt_summary, build_runtime
from agentic.runtime.observability import RunRecorder
from agentic.runtime.strategies import NEWS_GROUNDED_GENERATION_TYPES
from agentic.runtime.step_logger import create_run_logger
from agentic.tools.context_services import DiscordRunNotificationService
from agentic.tools.social_services import record_facebook_profile_handoff_delivery
from agentic.app.character_selection import (
    _fixed_character_selection,
    _selection_subject_mode,
    _subject_mode,
    resolve_character_selection,
)
from agentic.app.character_requests import (
    SUBJECT_MODE_INTERACTION,
    SUBJECT_MODE_RANDOM,
    SUBJECT_MODE_SINGLE,
    SUPPORTED_SUBJECT_MODES,
)
from agentic.app.character_config import VIDEO_EXTENSIONS, load_character_config
from agentic.app.character_publishing import (
    _effective_publish_after_generate,
    _extract_publish_visual_grounding,
    _publish_review_scope,
    _publish_selection_limit,
    _publish_stage_status,
    _resolve_publish_prompt,
    _review_stage_status,
    collect_media_paths_from_run_result,
)
from agentic.app.generation_routing import _resolve_routing_asset_root
from agentic.app.character_payload import build_goal_payload_from_character_config


def _extract_failure_details(result: dict[str, Any] | None) -> dict[str, str]:
    """Promote the first failed node's diagnostic into the run manifest."""
    if not isinstance(result, dict):
        return {}
    records = result.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("status") or "").lower() == "success":
                continue
            logs = record.get("logs")
            reason = ""
            if isinstance(logs, list):
                reason = next(
                    (str(item).strip() for item in logs if str(item).strip()), ""
                )
            if not reason:
                reason = (
                    f"{record.get('skill_name') or record.get('node_id') or 'workflow node'} "
                    f"finished with status {record.get('status') or 'failed'}"
                )
            details = {
                "failure_reason": reason,
                "failure_node": str(record.get("node_id") or ""),
                "failure_skill": str(record.get("skill_name") or ""),
            }
            return {key: value for key, value in details.items() if value}
    if str(result.get("status") or "").lower() not in {"", "success"}:
        return {
            "failure_reason": f"Workflow finished with status {result.get('status')}"
        }
    return {}


def _regeneration_requested(result: dict[str, Any]) -> bool:
    return any(
        record.get("skill_name") == "review.assets.select"
        and record.get("outputs", {}).get("regeneration_requested") is True
        for record in result.get("records", [])
    )


def run_character_workflow(
    request: CharacterWorkflowRequest,
    *,
    runtime_builder: Callable[..., Any] | None = None,
    notification_service: DiscordRunNotificationService | None = None,
) -> dict[str, Any]:
    repo_root = request.repo_root
    config_path = request.config_path
    runtime = request.runtime
    review = request.review
    dry_run_publish = review.dry_run_publish
    review_notes = review.review_notes
    effective_publish_after_generate = _effective_publish_after_generate(
        requested=review.publish_after_generate,
        stage_probe=review.stage_probe,
    )
    run_id = uuid.uuid4().hex[:12]
    recorder = RunRecorder(repo_root / "logs" / "runs", run_id)
    logger, log_path = create_run_logger(
        repo_root / "logs" / "runs", run_id, recorder=recorder
    )
    os.environ["AGENTIC_RUN_LOGGER_NAME"] = logger.name
    logger.info(
        "workflow.start | run_id=%s | config=%s", run_id, Path(config_path).resolve()
    )
    logger.info("character.config.load | path=%s", Path(config_path).resolve())
    selection_group_name = ""
    selection_subject_mode = "single"
    if Path(config_path).exists():
        try:
            selection_config = load_character_config(config_path)
            selection_character = dict(selection_config.get("character", {}) or {})
            selection_subject_mode = (
                str(request.generation.subject_mode or "").strip().lower()
            )
            if selection_subject_mode not in SUPPORTED_SUBJECT_MODES:
                selection_subject_mode = _subject_mode(selection_config)
            selection_generation = dict(selection_config.get("generation", {}) or {})
            selection_interaction = dict(
                selection_generation.get("two_character_interaction", {}) or {}
            )
            selection_group_name = str(
                selection_character.get("group_name") or ""
            ).strip()
            if selection_subject_mode == SUBJECT_MODE_INTERACTION:
                selection_group_name = str(
                    selection_interaction.get("group_name") or selection_group_name
                ).strip()
            character_selection = resolve_character_selection(
                request, config=selection_config
            )
            selection_subject_mode = _selection_subject_mode(
                character_selection,
                configured_subject_mode=selection_subject_mode,
            )
        except Exception as exc:
            record_event = getattr(recorder, "record_event", None)
            if callable(record_event):
                record_event(
                    "character.subject.selection_failed"
                    if selection_subject_mode
                    in {SUBJECT_MODE_INTERACTION, SUBJECT_MODE_RANDOM}
                    else "character.group.selection_failed",
                    group_name=selection_group_name,
                    subject_mode=selection_subject_mode,
                    error=f"{type(exc).__name__}: {exc}",
                )
            logger.exception(
                "character.subject.selection_failed | mode=%s | group=%s | error=%s",
                selection_subject_mode,
                selection_group_name,
                exc,
            )
            raise
    else:
        character_selection = _fixed_character_selection(Path(config_path).stem)
        selection_subject_mode = SUBJECT_MODE_SINGLE
    selected_character_name = str(
        character_selection.get("selected_character") or ""
    ).strip()
    if selection_subject_mode == SUBJECT_MODE_INTERACTION:
        subjects = character_selection.get("subjects")
        if not isinstance(subjects, list) or len(subjects) != 2:
            raise ValueError(
                "two_character_interaction selection did not return two subject slots"
            )
        selected_character_name = str(subjects[0].get("name") or "").strip()
    record_event = getattr(recorder, "record_event", None)
    if selection_subject_mode == SUBJECT_MODE_INTERACTION:
        if callable(record_event):
            record_event("character.subjects.selected", **character_selection)
        logger.info(
            "character.subjects.selected | mode=%s | group=%s | subjects=%s | candidate_count=%s",
            selection_subject_mode,
            character_selection.get("group_name", ""),
            [
                subject.get("name", "")
                for subject in character_selection.get("subjects", [])
            ],
            character_selection.get("candidate_count", 0),
        )
    elif str(character_selection.get("mode") or "") == "group":
        if callable(record_event):
            record_event("character.group.selected", **character_selection)
        logger.info(
            "character.group.selected | group=%s | selected=%s | candidate_count=%s",
            character_selection.get("group_name", ""),
            selected_character_name,
            character_selection.get("candidate_count", 0),
        )
    else:
        if callable(record_event):
            record_event("character.config.selected", **character_selection)
    payload = build_goal_payload_from_character_config(
        replace(
            request,
            generation=replace(
                request.generation,
                selected_character_name=selected_character_name,
                character_selection=character_selection,
            ),
            review=replace(
                review, publish_after_generate=effective_publish_after_generate
            ),
            runtime=replace(
                runtime,
                asset_root=runtime.asset_root
                or _resolve_routing_asset_root(repo_root, runtime.comfy_root),
            ),
        ),
        recorder=recorder,
    )
    logger.info(
        "routing.selected | strategy=%s | media_type=%s | workflow=%s | prompt_source=%s",
        payload["source_generation_type"],
        payload["media_type"],
        payload["selected_workflow_name"],
        payload["constraints"].get("prompt_source", ""),
    )
    logger.info(
        "character.config.summary | payload=%s",
        json.dumps(payload.get("character_config_summary", {}), ensure_ascii=False),
    )
    if payload["constraints"].get("news_context"):
        logger.info(
            "news.context | payload=%s",
            json.dumps(payload["constraints"]["news_context"], ensure_ascii=False),
        )
    planner, runner, run_memory = (runtime_builder or build_runtime)(
        repo_root / "agentic",
        output_root=Path(payload["resolved_output_dir"]),
        comfy_host=runtime.comfy_host,
        comfy_port=runtime.comfy_port,
        comfy_root=Path(runtime.comfy_root).resolve() if runtime.comfy_root else None,
        run_id=run_id,
        logger=logger,
        recorder=recorder,
    )
    goal = planner.create_goal(
        prompt=payload["prompt"],
        media_type=payload["media_type"],
        duration_seconds=int(payload["duration_seconds"]),
        style=str(payload["style"]),
        constraints=dict(payload["constraints"]),
    )
    plan = planner.build_plan(goal)
    logger.info(
        "plan.built | workflow=%s | node_count=%s", plan.workflow_name, len(plan.nodes)
    )
    generation_result = runner.run(plan)
    generation_dict = generation_result.to_dict()
    generation_media_paths = collect_media_paths_from_run_result(generation_dict)
    logger.info("generation.finished | status=%s", generation_result.status)

    publish_dict: dict[str, Any] | None = None
    publish_gate_failure: str = ""
    regeneration_requested = _regeneration_requested(generation_dict)
    if effective_publish_after_generate and generation_result.status == "success":
        media_paths = generation_media_paths
        platform_configs = payload["constraints"].get("platform_configs", {})
        if not media_paths:
            publish_gate_failure = "Final publish review was required, but generation produced no publishable media."
        elif not isinstance(platform_configs, dict) or not platform_configs:
            publish_gate_failure = "Final publish review was required, but no compatible publishing platform is configured."
        else:
            logger.info(
                "publish.start | media_count=%s | platforms=%s",
                len(media_paths),
                list(platform_configs.keys()),
            )
            publish_prompt, publish_prompt_source = _resolve_publish_prompt(
                generation_dict,
                fallback_prompt=str(payload["prompt"]),
            )
            publish_goal = planner.create_goal(
                prompt=publish_prompt,
                media_type="publish_review",
                duration_seconds=0,
                style=str(payload["style"]),
                constraints={
                    **{
                        key: payload["constraints"][key]
                        for key in (
                            "canvas_width",
                            "canvas_height",
                            "canvas_aspect_ratio",
                            "story_card_width",
                            "story_card_height",
                            "source_generation_type",
                            "discord_review_timeout_seconds",
                        )
                        if key in payload["constraints"]
                    },
                    "media_paths": media_paths,
                    "platforms": list(platform_configs.keys()),
                    "platform_configs": dict(platform_configs),
                    "hashtags": list(payload["constraints"].get("hashtags", [])),
                    "character": payload["character_name"],
                    "dry_run": dry_run_publish,
                    "publish_mode": str(
                        payload["constraints"].get("publish_mode") or ""
                    ),
                    "additional_params": {
                        "facebook_profile_share_url": str(
                            payload["constraints"].get("facebook_profile_share_url")
                            or ""
                        )
                    },
                    "output_dir": str(
                        Path(payload["resolved_output_dir"]) / "publish_ready"
                    ),
                    "selection_limit": _publish_selection_limit(
                        media_paths,
                        payload["constraints"].get("review_selection_limit"),
                        source_generation_type=str(
                            payload.get("source_generation_type") or ""
                        ),
                    ),
                    "review_notes": review_notes,
                    # Every publishable media artifact must be explicitly
                    # approved in Discord before any platform dispatch.
                    "require_human_review": True,
                    "review_scope": _publish_review_scope(media_paths),
                    "review_all_candidates": _publish_review_scope(media_paths)
                    == "final_media",
                    "allow_regenerate_review": bool(review.allow_regenerate),
                    "publish_prompt_source": publish_prompt_source,
                    # Keep editorial rotation different across successful
                    # workflow runs while preserving a traceable key.
                    "post_strategy_seed": run_id,
                    "visual_grounding": _extract_publish_visual_grounding(
                        generation_dict
                    ),
                    "news_context": dict(
                        payload["constraints"].get("news_context") or {}
                    ),
                    "news_grounding_required": bool(
                        payload["constraints"].get("news_driven")
                        or payload["constraints"].get("news_grounding_required")
                        or payload["source_generation_type"]
                        in NEWS_GROUNDED_GENERATION_TYPES
                    ),
                    "news_trace_contract": (
                        "source context -> active mechanism -> visible consequence"
                    ),
                },
            )
            publish_plan = planner.build_plan(publish_goal)
            publish_result = runner.run(publish_plan)
            publish_result_dict = publish_result.to_dict()
            publish_state_data = publish_result_dict.get("state") or {}
            publish_node_outputs = (
                publish_state_data.get("node_outputs", {})
                if isinstance(publish_state_data, dict)
                else {}
            )
            publish_outputs = (
                publish_node_outputs.get("dispatch-publish", {})
                if isinstance(publish_node_outputs, dict)
                else {}
            )
            regeneration_requested = _regeneration_requested(publish_result_dict)
            publish_state = str(publish_outputs.get("publication_state") or "unknown")
            publicly_visible = bool(publish_outputs.get("publicly_visible", False))
            logger.info(
                "publish.finished | status=%s | publication_state=%s | publicly_visible=%s",
                publish_result.status,
                publish_state,
                publicly_visible,
            )
            publish_dict = {
                "plan": publish_plan.to_dict(),
                "result": publish_result_dict,
                "prompt_summary": _build_prompt_summary(
                    publish_result_dict.get("state", {})
                ),
            }
    failure_details = {} if regeneration_requested else _extract_failure_details(generation_dict)
    overall_status = (
        "regeneration_requested"
        if regeneration_requested
        else ("success" if generation_result.status == "success" else "failed")
    )
    if publish_gate_failure:
        logger.error("publish.blocked | reason=%s", publish_gate_failure)
        overall_status = "failed"
        failure_details = {
            "failure_reason": publish_gate_failure,
            "failure_node": "final-publish-gate",
            "failure_skill": "publish.review",
        }
    if publish_dict:
        publish_result_status = str(
            (publish_dict.get("result") or {}).get("status") or ""
        ).lower()
        if publish_result_status not in {"success"} and not regeneration_requested:
            overall_status = "failed"
            publish_failure_details = _extract_failure_details(
                publish_dict.get("result")
            )
            if not failure_details:
                failure_details = publish_failure_details
            if not failure_details:
                publish_result_data = publish_dict.get("result")
                publish_state_data = (
                    publish_result_data.get("state", {})
                    if isinstance(publish_result_data, dict)
                    else {}
                )
                publish_node_outputs = (
                    publish_state_data.get("node_outputs", {})
                    if isinstance(publish_state_data, dict)
                    else {}
                )
                dispatch_outputs = (
                    publish_node_outputs.get("dispatch-publish", {})
                    if isinstance(publish_node_outputs, dict)
                    else {}
                )
                dispatch_errors = (
                    dispatch_outputs.get("errors", {})
                    if isinstance(dispatch_outputs, dict)
                    else {}
                )
                if dispatch_errors:
                    failure_details = {
                        "failure_reason": f"Publish dispatch errors: {dispatch_errors}"
                    }
        publish_state_data = (
            (publish_dict.get("result") or {}).get("state", {})
            if isinstance(publish_dict.get("result"), dict)
            else {}
        )
        publish_node_outputs = (
            publish_state_data.get("node_outputs", {})
            if isinstance(publish_state_data, dict)
            else {}
        )
        dispatch_outputs = (
            publish_node_outputs.get("dispatch-publish", {})
            if isinstance(publish_node_outputs, dict)
            else {}
        )
        if (
            isinstance(dispatch_outputs, dict)
            and str(dispatch_outputs.get("status") or "").strip().lower()
            == "awaiting_user_action"
        ):
            overall_status = "awaiting_user_action"
    logger.info(
        "workflow.end | run_id=%s | status=%s | log_path=%s | failure_reason=%s",
        run_id,
        overall_status,
        log_path,
        failure_details.get("failure_reason", ""),
    )

    discord_notification: dict[str, Any] = {}
    if publish_dict:
        publish_result = publish_dict.get("result") or {}
        publish_state_data = (
            publish_result.get("state") if isinstance(publish_result, dict) else {}
        )
        publish_node_outputs = (
            publish_state_data.get("node_outputs", {})
            if isinstance(publish_state_data, dict)
            else {}
        )
        dispatch_outputs = (
            publish_node_outputs.get("dispatch-publish", {})
            if isinstance(publish_node_outputs, dict)
            else {}
        )
        platform_results = (
            dispatch_outputs.get("results", {})
            if isinstance(dispatch_outputs, dict)
            else {}
        )
        errors = (
            dispatch_outputs.get("errors", {})
            if isinstance(dispatch_outputs, dict)
            else {}
        )
        publish_summary = (
            "Regeneration requested; starting a new end-to-end run"
            if regeneration_requested
            else (
                f"{str(dispatch_outputs.get('status') or publish_result.get('status') or overall_status)} | "
                f"state={str(dispatch_outputs.get('publication_state') or 'unknown')} | "
                f"public={bool(dispatch_outputs.get('publicly_visible', False))}"
            )
        )
        discord_lines = [
            f"MediaOverload | {payload['character_name']} | run {run_id}",
            f"Publish: {publish_summary}",
            f"Platforms: {json.dumps(platform_results, ensure_ascii=False)}",
        ]
        if errors:
            discord_lines.append(
                f"Errors: {json.dumps(errors, ensure_ascii=False)[:700]}"
            )
        handoff_attachment_paths = [
            str(path)
            for path in (
                dispatch_outputs.get("handoff_attachment_paths", [])
                if isinstance(dispatch_outputs, dict)
                else []
            )
            if path
        ]
        handoff_messages = [
            str(handoff.get("notification_text") or "")
            for handoff in (
                dispatch_outputs.get("handoffs", {}).values()
                if isinstance(dispatch_outputs.get("handoffs", {}), dict)
                else []
            )
            if isinstance(handoff, dict)
            and str(handoff.get("notification_text") or "").strip()
        ]
        discord_lines.extend(handoff_messages)
        discord_lines.append(f"Log: {log_path}")
        notifier = notification_service or DiscordRunNotificationService()
        if handoff_attachment_paths:
            discord_notification = notifier.notify(
                "\n".join(discord_lines),
                media_paths=handoff_attachment_paths,
                channel_env="discord_handoff_channel_id",
            )
        else:
            discord_notification = notifier.notify(
                "\n".join(discord_lines)
            )
        handoff_delivery_results: list[dict[str, object]] = []
        for handoff_platform, handoff in (
            dispatch_outputs.get("handoffs", {}).items()
            if isinstance(dispatch_outputs, dict)
            and isinstance(dispatch_outputs.get("handoffs", {}), dict)
            else []
        ):
            if not isinstance(handoff, dict):
                continue
            handoff_path = str(handoff.get("handoff_path") or "").strip()
            if not handoff_path:
                continue
            try:
                delivery_result = record_facebook_profile_handoff_delivery(
                    handoff_path,
                    discord_notification,
                    expected_attachment_count=len(handoff_attachment_paths),
                )
                handoff_delivery_results.append(delivery_result)
                handoff["delivery"] = delivery_result.get("delivery", {})
                receipt = delivery_result.get("handoff", {})
                if isinstance(receipt, dict):
                    handoff["receipt"] = receipt.get(
                        "receipt", handoff.get("receipt", {})
                    )
                    handoff_receipts = dispatch_outputs.get("handoff_receipts")
                    if isinstance(handoff_receipts, dict):
                        handoff_receipts[str(handoff_platform)] = handoff["receipt"]
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.exception(
                    "facebook.profile.handoff.delivery_record.error | error=%s", exc
                )
                handoff_delivery_results.append(
                    {"handoff_path": handoff_path, "error": str(exc)}
                )
        if isinstance(dispatch_outputs, dict) and handoff_delivery_results:
            dispatch_outputs["handoff_delivery"] = handoff_delivery_results
        logger.info(
            "discord.notification.end | status=%s | message_id=%s",
            str(discord_notification.get("status") or "unknown"),
            str(discord_notification.get("message_id") or ""),
        )

    review_status = _review_stage_status(generation_dict)
    publish_status = _publish_stage_status(
        publish_dict,
        requested=effective_publish_after_generate,
        generation_status=str(generation_result.status),
        media_paths=generation_media_paths,
        platform_configs=payload["constraints"].get("platform_configs", {}),
    )
    if regeneration_requested:
        review_status = "regeneration_requested"
        publish_status = "regeneration_requested" if publish_dict else "not_run_regeneration_requested"
    result_payload = {
        "status": overall_status,
        "regeneration_requested": regeneration_requested,
        "failure_reason": failure_details.get("failure_reason"),
        "failure_node": failure_details.get("failure_node"),
        "failure_skill": failure_details.get("failure_skill"),
        "run_id": run_id,
        "log_path": str(log_path),
        "config_path": str(Path(config_path).resolve()),
        "source_generation_type": payload["source_generation_type"],
        "agentic_media_type": payload["media_type"],
        "character": payload["character_name"],
        "character_selection": payload.get("character_selection", {}),
        "routing_summary": payload.get("routing_summary", {}),
        "plan": plan.to_dict(),
        "generation": {
            "result": generation_dict,
            "prompt_summary": _build_prompt_summary(generation_dict.get("state", {})),
        },
        "routing": payload.get("routing", {}),
        "character_config_summary": payload.get("character_config_summary", {}),
        "publish": publish_dict,
        "stage_status": {
            "render": {
                "status": str(generation_result.status),
                "artifact_count": len(generation_media_paths),
            },
            "review": {"status": review_status},
            "publish": {"status": publish_status},
        },
        "artifacts": {
            "media_paths": generation_media_paths,
            "video_paths": [
                path
                for path in generation_media_paths
                if Path(path).suffix.lower() in VIDEO_EXTENSIONS
            ],
            "image_paths": [
                path
                for path in generation_media_paths
                if Path(path).suffix.lower() not in VIDEO_EXTENSIONS
            ],
        },
        "discord_notification": discord_notification,
        "memory": run_memory.as_serializable(),
    }
    recorder.finalize(result_payload)
    return result_payload


def dumps_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
