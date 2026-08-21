from __future__ import annotations

import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import schedule
except ImportError:  # pragma: no cover - exercised in environments without optional runtime deps
    schedule = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
for candidate in (REPO_ROOT, AGENTIC_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from agentic.app.character_workflow import run_character_workflow


LOGGER = logging.getLogger("mediaoverload.scheduler")
QUIET_HOURS_END = 6


@dataclass(slots=True)
class SchedulerConfig:
    enabled: bool
    mode: str
    interval_hours: int
    daily_time: str
    character: str | None
    config_path: Path | None
    prompt: str
    news_driven: bool
    news_history_path: str | None
    run_immediately: bool
    dry_run_publish: bool
    publish_mode: str
    publish_platforms: list[str] | None
    publish_after_generate: bool
    enable_review_loop: bool
    no_review: bool
    review_notes: str
    output_dir: str | None
    comfy_host: str | None
    comfy_port: int | None
    comfy_root: str | None
    auto_download_assets: bool
    preferred_generation_type: str | None
    duration_seconds: int | None
    sleep_seconds: int
    random_seed: int | None
    routing_history_path: str | None = None


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_csv(value: Any) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    return values or None


def _load_env() -> None:
    for env_path in (REPO_ROOT / "media_overload.env", REPO_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


def _resolve_character_config(character: str | None, config_path: str | None) -> Path | None:
    if config_path:
        return Path(config_path).resolve()
    if character:
        candidate = REPO_ROOT / "configs" / "characters" / f"{character.lower()}.yaml"
        if candidate.exists():
            return candidate
        raise ValueError(f"Character config not found: {candidate}")
    return None


def load_scheduler_config() -> SchedulerConfig:
    _load_env()
    character = str(os.getenv("SCHEDULER_CHARACTER", "")).strip() or None
    raw_config_path = str(os.getenv("SCHEDULER_CONFIG", "")).strip() or None
    mode = str(os.getenv("SCHEDULER_MODE", "interval")).strip().lower() or "interval"
    interval_hours = int(os.getenv("SCHEDULER_INTERVAL_HOURS", "24"))
    daily_time = str(os.getenv("SCHEDULER_DAILY_TIME", "09:00")).strip() or "09:00"
    comfy_port_raw = str(os.getenv("SCHEDULER_COMFY_PORT", "")).strip()
    duration_raw = str(os.getenv("SCHEDULER_DURATION_SECONDS", "")).strip()
    random_seed_raw = str(os.getenv("SCHEDULER_RANDOM_SEED", "")).strip()
    routing_history_path = str(os.getenv("SCHEDULER_ROUTING_HISTORY_PATH", "")).strip() or None
    return SchedulerConfig(
        enabled=_parse_bool(os.getenv("SCHEDULER_ENABLED"), default=True),
        mode=mode,
        interval_hours=max(1, interval_hours),
        daily_time=daily_time,
        character=character,
        config_path=_resolve_character_config(character, raw_config_path),
        prompt=str(os.getenv("SCHEDULER_PROMPT", "")).strip(),
        news_driven=_parse_bool(os.getenv("SCHEDULER_NEWS_DRIVEN"), default=True),
        news_history_path=str(os.getenv("SCHEDULER_NEWS_HISTORY_PATH", "")).strip() or None,
        run_immediately=_parse_bool(os.getenv("SCHEDULER_RUN_IMMEDIATELY"), default=False),
        dry_run_publish=_parse_bool(os.getenv("SCHEDULER_DRY_RUN_PUBLISH"), default=False),
        publish_mode=str(os.getenv("SCHEDULER_PUBLISH_MODE", "")).strip().lower(),
        publish_platforms=_parse_csv(os.getenv("SCHEDULER_PUBLISH_PLATFORMS")),
        publish_after_generate=_parse_bool(os.getenv("SCHEDULER_PUBLISH_AFTER_GENERATE"), default=True),
        enable_review_loop=_parse_bool(os.getenv("SCHEDULER_ENABLE_REVIEW_LOOP"), default=True),
        no_review=_parse_bool(os.getenv("SCHEDULER_NO_REVIEW"), default=False),
        review_notes=str(os.getenv("SCHEDULER_REVIEW_NOTES", "")).strip(),
        output_dir=str(os.getenv("SCHEDULER_OUTPUT_DIR", "")).strip() or None,
        comfy_host=str(os.getenv("SCHEDULER_COMFY_HOST", "")).strip() or None,
        comfy_port=int(comfy_port_raw) if comfy_port_raw else None,
        comfy_root=str(os.getenv("SCHEDULER_COMFY_ROOT", "")).strip() or None,
        auto_download_assets=_parse_bool(os.getenv("SCHEDULER_AUTO_DOWNLOAD_ASSETS"), default=False),
        preferred_generation_type=str(os.getenv("SCHEDULER_PREFERRED_GENERATION_TYPE", "")).strip() or None,
        duration_seconds=int(duration_raw) if duration_raw else None,
        sleep_seconds=max(5, int(os.getenv("SCHEDULER_SLEEP_SECONDS", "30"))),
        random_seed=int(random_seed_raw) if random_seed_raw else None,
        routing_history_path=routing_history_path,
    )


def run_scheduled_job(config: SchedulerConfig, *, rng: random.Random | None = None) -> dict[str, Any]:
    if config.config_path is None:
        raise ValueError("Scheduler requires SCHEDULER_CHARACTER or SCHEDULER_CONFIG")
    if rng is None:
        rng = random.Random()
    LOGGER.info(
        "scheduler.run | config=%s | prompt=%s | news_driven=%s | strategy_selection=%s | preferred_generation_type=%s",
        config.config_path,
        config.prompt or "<autonomous>",
        config.news_driven,
        "explicit_override" if config.preferred_generation_type else "weighted_random",
        config.preferred_generation_type or "<weighted>",
    )
    result = run_character_workflow(
        REPO_ROOT,
        config.config_path,
        prompt=config.prompt,
        preferred_generation_type=config.preferred_generation_type,
        duration_seconds=config.duration_seconds,
        dry_run_publish=config.dry_run_publish,
        publish_mode=config.publish_mode,
        publish_platforms=config.publish_platforms,
        publish_after_generate=config.publish_after_generate,
        output_dir=config.output_dir,
        enable_review_loop=config.enable_review_loop,
        review_notes=config.review_notes,
        no_review=config.no_review,
        news_driven=config.news_driven,
        news_history_path=config.news_history_path,
        routing_history_path=(
            config.routing_history_path
            or str(REPO_ROOT / "agentic" / "state" / "routing_selection" / f"{config.config_path.stem}.json")
        ),
        comfy_host=config.comfy_host,
        comfy_port=config.comfy_port,
        comfy_root=config.comfy_root,
        auto_download_assets=config.auto_download_assets,
        rng=rng,
    )
    LOGGER.info(
        "scheduler.result | status=%s | strategy=%s | publish=%s",
        result.get("status"),
        result.get("source_generation_type"),
        ((result.get("publish") or {}).get("result") or {}).get("status", ""),
    )
    return result


def _run_scheduled_job_safe(config: SchedulerConfig, *, rng: random.Random | None = None) -> dict[str, Any]:
    local_hour = time.localtime().tm_hour
    if local_hour < QUIET_HOURS_END:
        LOGGER.info("scheduler.skip | reason=quiet_hours | local_hour=%s", local_hour)
        return {"status": "skipped", "reason": "quiet_hours"}
    try:
        return run_scheduled_job(config, rng=rng)
    except Exception:
        LOGGER.exception("scheduler.run.failed | config=%s", config.config_path)
        return {"status": "failed", "source_generation_type": "", "error": "scheduled job failed"}


def register_jobs(config: SchedulerConfig, *, rng: random.Random | None = None) -> None:
    if schedule is None:
        raise RuntimeError("Missing scheduler dependency; install the 'schedule' package")
    if config.mode == "daily":
        schedule.every().day.at(config.daily_time).do(_run_scheduled_job_safe, config=config, rng=rng)
        LOGGER.info("scheduler.registered | mode=daily | at=%s", config.daily_time)
        return
    schedule.every(config.interval_hours).hours.do(_run_scheduled_job_safe, config=config, rng=rng)
    LOGGER.info("scheduler.registered | mode=interval | every_hours=%s", config.interval_hours)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("SCHEDULER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_scheduler_config()
    if not config.enabled:
        LOGGER.info("scheduler.disabled")
        return
    if schedule is None:
        raise RuntimeError("Missing scheduler dependency; install the 'schedule' package")
    if config.config_path is None:
        raise ValueError("Scheduler requires SCHEDULER_CHARACTER or SCHEDULER_CONFIG")

    rng = random.Random(config.random_seed) if config.random_seed is not None else random.Random()
    register_jobs(config, rng=rng)
    if config.run_immediately:
        _run_scheduled_job_safe(config, rng=rng)

    LOGGER.info("scheduler.loop.start | sleep_seconds=%s", config.sleep_seconds)
    while True:
        schedule.run_pending()
        time.sleep(config.sleep_seconds)


if __name__ == "__main__":
    main()
