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


@dataclass(slots=True)
class SchedulerConfig:
    enabled: bool
    mode: str
    interval_hours: int
    daily_time: str
    character: str | None
    config_path: Path | None
    prompt: str
    run_immediately: bool
    dry_run_publish: bool
    publish_after_generate: bool
    enable_review_loop: bool
    review_notes: str
    output_dir: str | None
    comfy_host: str | None
    comfy_port: int | None
    comfy_root: str | None
    auto_download_assets: bool
    preferred_generation_type: str | None
    sleep_seconds: int
    random_seed: int | None


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
    random_seed_raw = str(os.getenv("SCHEDULER_RANDOM_SEED", "")).strip()
    return SchedulerConfig(
        enabled=_parse_bool(os.getenv("SCHEDULER_ENABLED"), default=True),
        mode=mode,
        interval_hours=max(1, interval_hours),
        daily_time=daily_time,
        character=character,
        config_path=_resolve_character_config(character, raw_config_path),
        prompt=str(os.getenv("SCHEDULER_PROMPT", "")).strip(),
        run_immediately=_parse_bool(os.getenv("SCHEDULER_RUN_IMMEDIATELY"), default=False),
        dry_run_publish=_parse_bool(os.getenv("SCHEDULER_DRY_RUN_PUBLISH"), default=False),
        publish_after_generate=_parse_bool(os.getenv("SCHEDULER_PUBLISH_AFTER_GENERATE"), default=True),
        enable_review_loop=_parse_bool(os.getenv("SCHEDULER_ENABLE_REVIEW_LOOP"), default=True),
        review_notes=str(os.getenv("SCHEDULER_REVIEW_NOTES", "")).strip(),
        output_dir=str(os.getenv("SCHEDULER_OUTPUT_DIR", "")).strip() or None,
        comfy_host=str(os.getenv("SCHEDULER_COMFY_HOST", "")).strip() or None,
        comfy_port=int(comfy_port_raw) if comfy_port_raw else None,
        comfy_root=str(os.getenv("SCHEDULER_COMFY_ROOT", "")).strip() or None,
        auto_download_assets=_parse_bool(os.getenv("SCHEDULER_AUTO_DOWNLOAD_ASSETS"), default=False),
        preferred_generation_type=str(os.getenv("SCHEDULER_PREFERRED_GENERATION_TYPE", "")).strip() or None,
        sleep_seconds=max(5, int(os.getenv("SCHEDULER_SLEEP_SECONDS", "30"))),
        random_seed=int(random_seed_raw) if random_seed_raw else None,
    )


def run_scheduled_job(config: SchedulerConfig, *, rng: random.Random | None = None) -> dict[str, Any]:
    if config.config_path is None:
        raise ValueError("Scheduler requires SCHEDULER_CHARACTER or SCHEDULER_CONFIG")
    LOGGER.info(
        "scheduler.run | config=%s | prompt=%s | preferred_generation_type=%s",
        config.config_path,
        config.prompt or "<autonomous>",
        config.preferred_generation_type or "<weighted>",
    )
    result = run_character_workflow(
        REPO_ROOT,
        config.config_path,
        prompt=config.prompt,
        preferred_generation_type=config.preferred_generation_type,
        dry_run_publish=config.dry_run_publish,
        publish_after_generate=config.publish_after_generate,
        output_dir=config.output_dir,
        enable_review_loop=config.enable_review_loop,
        review_notes=config.review_notes,
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


def register_jobs(config: SchedulerConfig, *, rng: random.Random | None = None) -> None:
    if schedule is None:
        raise RuntimeError("Missing scheduler dependency; install the 'schedule' package")
    if config.mode == "daily":
        schedule.every().day.at(config.daily_time).do(run_scheduled_job, config=config, rng=rng)
        LOGGER.info("scheduler.registered | mode=daily | at=%s", config.daily_time)
        return
    schedule.every(config.interval_hours).hours.do(run_scheduled_job, config=config, rng=rng)
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

    rng = random.Random(config.random_seed) if config.random_seed is not None else random.Random()
    register_jobs(config, rng=rng)
    if config.run_immediately:
        run_scheduled_job(config, rng=rng)

    LOGGER.info("scheduler.loop.start | sleep_seconds=%s", config.sleep_seconds)
    while True:
        schedule.run_pending()
        time.sleep(config.sleep_seconds)


if __name__ == "__main__":
    main()
