"""Run log acceptance: every run, its Discord review verdict and its publications land in MySQL.

The scenarios drive `run_character_workflow` end to end with the real runner,
the real review service and the real Discord review view. Only ComfyUI
generation and the social platforms are replaced by fakes at their boundary;
MySQL is real and every assertion reads the tables directly.
"""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import replace
from datetime import timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath

import pymysql
import pytest
import yaml
from dotenv import dotenv_values
from PIL import Image
from pymysql.cursors import DictCursor

from agentic.app.character_requests import CharacterWorkflowRequest
from agentic.app.character_run_log import project_relative_path
from agentic.app.character_workflow import run_character_workflow
from agentic.assets.registry import AssetRegistry
from agentic.memory import RunMemory
from agentic.runtime.contracts import (
    ExecutionNode,
    ExecutionPlan,
    GoalRequest,
    RunState,
    SkillContext,
    SkillResult,
)
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.run_log import RunEntry, RunLog, derive_review_outcome, utc_now
from agentic.runtime.runner import WorkflowRunner
from agentic.skills.agent_social import AgentSocialSkills
from agentic.tools.context_services import DiscordHumanReviewService
from agentic.tools.publishing_adapter import PublishingAdapter
from agentic.tools.social_services import SocialServiceTools

from test_character_regeneration import (
    CaptionBoundary,
    NotificationBoundary,
    ReviewTransportBoundary,
)

ADMIN_ENV_KEYS = ("mysql_host", "mysql_port", "mysql_user", "mysql_password")
TEST_DATABASE_PREFIX = "mediaoverload_runs_test_"


# --------------------------------------------------------------------------- #
# Real MySQL
# --------------------------------------------------------------------------- #


@pytest.fixture
def mysql_options():
    values = dotenv_values(Path(__file__).resolve().parents[2] / "media_overload.env")
    options = {
        "host": str(values.get("mysql_host") or ""),
        "port": int(values.get("mysql_port") or 3306),
        "user": str(values.get("mysql_user") or ""),
        "password": str(values.get("mysql_password") or ""),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "connect_timeout": 5,
    }
    if not all(options[key] for key in ("host", "user", "password")):
        pytest.fail("Run log behavior tests require the configured real MySQL service.")
    return options


@pytest.fixture
def database(mysql_options):
    name = f"{TEST_DATABASE_PREFIX}{uuid.uuid4().hex}"
    connection = pymysql.connect(**mysql_options, autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        connection.close()
    yield name
    assert name.startswith(TEST_DATABASE_PREFIX), "Refusing to drop a database outside the test prefix."
    connection = pymysql.connect(**mysql_options, autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{name}`")
    finally:
        connection.close()


@pytest.fixture
def run_log(mysql_options, database):
    return RunLog(database, mysql_options=mysql_options)


class Ledger:
    """Read the run log tables the way an operator would, straight from MySQL."""

    def __init__(self, mysql_options, database):
        self.mysql_options = mysql_options
        self.database = database

    def rows(self, query, params=()):
        connection = pymysql.connect(**self.mysql_options, database=self.database)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return list(cursor.fetchall())
        finally:
            connection.close()

    def run(self, run_id):
        rows = self.rows("SELECT * FROM runs WHERE run_id = %s", (run_id,))
        assert len(rows) == 1, f"expected exactly one runs row for {run_id}, got {len(rows)}"
        return rows[0]

    def reviews(self, run_id):
        return self.rows(
            "SELECT * FROM run_reviews WHERE run_id = %s ORDER BY ordinal", (run_id,)
        )

    def publications(self, run_id):
        return self.rows(
            "SELECT * FROM run_publications WHERE run_id = %s ORDER BY platform", (run_id,)
        )


@pytest.fixture
def ledger(mysql_options, database, run_log):
    return Ledger(mysql_options, database)


# --------------------------------------------------------------------------- #
# Workflow boundary: real runner / review / publish skills, fake generation + platforms
# --------------------------------------------------------------------------- #


class PlatformBoundary(PublishingAdapter):
    """Real media processing; only the provider calls are replaced."""

    def __init__(self, *, accept=True, on_publish=None):
        super().__init__()
        # `accept` is one verdict for every platform, or a verdict per platform.
        self.accept = accept
        self.on_publish = on_publish
        self.published = []

    def register_platform(self, platform_name, platform_config):
        del platform_name, platform_config

    def publish(self, post, platforms=None):
        platform = (platforms or ["twitter"])[0]
        self.published.append((platform, post))
        if self.on_publish:
            self.on_publish()
        accepted = self.accept.get(platform, True) if isinstance(self.accept, dict) else self.accept
        return {platform: accepted}

    def get_publish_receipts(self):
        return {
            platform: {
                "platform": platform,
                "external_id": f"{platform}-4242",
                "status": "published",
                "verified": True,
                "visibility": "public",
                "details": {"permalink": f"https://example.test/{platform}-4242"},
            }
            for platform in ("twitter", "threads")
        }


class PlanBoundary:
    """Generation plan with an optional stage review; the real planner builds the publish plan."""

    create_goal = TaskPlanner.create_goal

    def __init__(self, root, *, stage_review):
        self.stage_review = stage_review
        self.publish_planner = TaskPlanner(AssetRegistry(root))

    def build_plan(self, goal):
        if goal.media_type == "publish_review":
            return self.publish_planner.build_plan(goal)
        nodes = [ExecutionNode(node_id="generate-image", skill_name="fixture.generate")]
        if self.stage_review:
            nodes += [
                ExecutionNode(
                    node_id="opening-frame-review",
                    skill_name="review.assets.select",
                    depends_on=["generate-image"],
                    inputs={
                        "review_scope": "first_frame",
                        "require_human_review": True,
                        "limit": 1,
                    },
                ),
                ExecutionNode(
                    node_id="render-video",
                    skill_name="fixture.render",
                    depends_on=["opening-frame-review"],
                ),
            ]
        return ExecutionPlan(goal=goal, workflow_name="run-log-fixture", nodes=nodes)


class RuntimeBoundary:
    def __init__(self, transport, *, stage_review=False, platforms=None):
        self.transport = transport
        self.stage_review = stage_review
        self.platforms = platforms or PlatformBoundary()

    def __call__(self, root, **values):
        output_root = values["output_root"]
        run_id = values["run_id"]
        service = DiscordHumanReviewService(output_root, feedback_process=self.transport)
        tools = ToolRegistry()
        social_tools = SocialServiceTools(output_root)
        social_tools._publishing = self.platforms
        tools.register("publish.process_media", social_tools.process_media, "Prepare media")
        tools.register("publish.social", social_tools.publish_social, "Publish media")
        social = AgentSocialSkills(
            tools, output_root, prompt_engine=CaptionBoundary(), discord_review=service
        )
        registry = SkillRegistry()

        def generate(context):
            image = output_root / f"{run_id}.png"
            Image.new("RGB", (64, 40), "blue").save(image)
            return SkillResult(status="success", outputs={"media_paths": [str(image)]})

        def render(context):
            previous = context.state["opening-frame-review"]["media_paths"]
            return SkillResult(status="success", outputs={"media_paths": previous})

        registry.register("fixture.generate", generate, "Generation boundary")
        registry.register("fixture.render", render, "Render boundary")
        registry.register("review.assets.select", social.select_best_assets, "Real review")
        registry.register("publish.media.ingest", social.ingest_media, "Real ingest")
        registry.register("publish.media.process", social.process_media, "Real process")
        registry.register("publish.caption.prepare", social.prepare_caption, "Real caption")
        registry.register("publish.social.dispatch", social.publish_social, "Real dispatch")
        registry.register(
            "agent.summary.persist",
            lambda context: SkillResult(status="success", outputs={}),
            "Summary boundary",
        )
        memory = RunMemory()
        runner = WorkflowRunner(
            registry, run_memory=memory, logger=values["logger"], recorder=values["recorder"]
        )
        return PlanBoundary(root, stage_review=self.stage_review), runner, memory


@pytest.fixture
def review_environment(monkeypatch):
    monkeypatch.setenv("discord_review_allowed_user_ids", "42")
    monkeypatch.setenv("discord_review_bot_token", "contract-fake")
    monkeypatch.setenv("discord_review_channel_id", "123")


@pytest.fixture
def project(tmp_path):
    shutil.copytree(Path(__file__).resolve().parents[2] / "configs", tmp_path / "configs")
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "character": {"name": "Fixture"},
                "generation": {"subject_mode": "single", "output_dir": str(tmp_path / "output")},
                "social_media": {"platforms": {"twitter": {"enabled": True}}},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, config_path


@pytest.fixture
def two_platform_project(project):
    _, config_path = project
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["social_media"]["platforms"]["threads"] = {"enabled": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return project


def start_run(project, runtime, run_log, *, prompt="Fixture scene", dry_run=False):
    root, config_path = project
    request = CharacterWorkflowRequest(root, config_path)
    request = replace(
        request,
        generation=replace(
            request.generation,
            prompt=prompt,
            preferred_generation_type="text2img",
            news_driven=False,
        ),
        review=replace(request.review, dry_run_publish=dry_run, allow_regenerate=True),
    )
    return run_character_workflow(
        request,
        runtime_builder=runtime,
        notification_service=NotificationBoundary(),
        run_log=run_log,
    )


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def test_user_given_a_run_published_after_review_when_it_finishes_then_the_run_log_shows_the_accepted_run_and_where_it_was_published(
    project, run_log, ledger, review_environment
):
    """User Given a run whose opening frame and final media are both accepted in Discord When it publishes to Twitter Then the run log records the strategy, prompt, an accepted verdict backed by both reviews, and the platform post with its id and time."""
    started = utc_now() - timedelta(seconds=1)
    runtime = RuntimeBoundary(
        ReviewTransportBoundary(["Accept", "Accept"]), stage_review=True
    )
    result = start_run(project, runtime, run_log, prompt="A quiet morning at the pond")
    run = ledger.run(result["run_id"])
    assert run["status"] == "success", run["failure_reason"]
    assert run["strategy"] == "text2img"
    assert run["workflow_name"] == "run-log-fixture"
    assert run["prompt"] == "A quiet morning at the pond"
    assert run["character_name"] == "Fixture"
    assert run["review_outcome"] == "accepted"
    assert not run["dry_run"]
    assert run["started_at"] >= started
    assert run["finished_at"] >= run["started_at"]
    stages = [(review["stage"], review["decision"]) for review in ledger.reviews(run["run_id"])]
    assert stages == [("first_frame", "approved"), ("final_media", "approved")]
    publications = ledger.publications(run["run_id"])
    assert [row["platform"] for row in publications] == ["twitter"]
    assert publications[0]["outcome"] == "published"
    assert publications[0]["external_id"] == "twitter-4242"
    assert publications[0]["dispatched_at"] >= run["started_at"]
    assert publications[0]["dispatched_at"] <= run["finished_at"]
    assert publications[0]["caption"]
    assert publications[0]["media_paths"]


def test_user_given_a_final_review_when_the_reviewer_rejects_then_the_run_is_rejected_and_nothing_is_published(
    project, run_log, ledger, review_environment
):
    """User Given a run with only a final review When the reviewer rejects Then the run log marks the run rejected and lists no publication."""
    platforms = PlatformBoundary()
    runtime = RuntimeBoundary(ReviewTransportBoundary(["Reject"]), platforms=platforms)
    result = start_run(project, runtime, run_log)
    run = ledger.run(result["run_id"])
    assert run["review_outcome"] == "rejected"
    assert run["status"] == "failed"
    assert [row["decision"] for row in ledger.reviews(run["run_id"])] == ["rejected"]
    assert ledger.publications(run["run_id"]) == []
    assert platforms.published == []


def test_user_given_a_multi_stage_run_when_an_early_stage_is_accepted_and_the_final_is_rejected_then_the_run_is_partial(
    project, run_log, ledger, review_environment
):
    """User Given a run with an opening-frame review and a final review When the opening frame is accepted but the final media is rejected Then the run log marks the run partial and keeps both verdicts in order."""
    runtime = RuntimeBoundary(
        ReviewTransportBoundary(["Accept", "Reject"]), stage_review=True
    )
    result = start_run(project, runtime, run_log)
    run = ledger.run(result["run_id"])
    assert run["review_outcome"] == "partial"
    stages = [(review["stage"], review["decision"]) for review in ledger.reviews(run["run_id"])]
    assert stages == [("first_frame", "approved"), ("final_media", "rejected")]
    assert ledger.publications(run["run_id"]) == []


def test_user_given_a_multi_stage_run_when_the_first_stage_is_rejected_then_the_run_is_rejected(
    project, run_log, ledger, review_environment
):
    """User Given a run with an opening-frame review When the opening frame is rejected Then the run log marks the run rejected and no later stage is recorded."""
    runtime = RuntimeBoundary(ReviewTransportBoundary(["Reject"]), stage_review=True)
    result = start_run(project, runtime, run_log)
    run = ledger.run(result["run_id"])
    assert run["review_outcome"] == "rejected"
    assert [review["stage"] for review in ledger.reviews(run["run_id"])] == ["first_frame"]


def test_user_given_a_final_review_when_the_reviewer_asks_to_regenerate_then_the_run_is_regenerated(
    project, run_log, ledger, review_environment
):
    """User Given a run with a final review When the reviewer asks to regenerate Then the run log marks the run regenerated, not rejected, with no publication."""
    runtime = RuntimeBoundary(ReviewTransportBoundary(["重新生成"]))
    result = start_run(project, runtime, run_log)
    run = ledger.run(result["run_id"])
    assert run["status"] == "regeneration_requested"
    assert run["review_outcome"] == "regenerated"
    assert [row["decision"] for row in ledger.reviews(run["run_id"])] == ["regenerate"]
    assert ledger.publications(run["run_id"]) == []


def test_user_given_a_review_nobody_answered_when_discord_times_out_then_the_run_is_unreviewed(
    project, run_log, ledger, review_environment
):
    """User Given a final review When Discord times out before anyone decides Then the run log records the timeout and does not claim a human accepted or rejected the run."""

    async def nobody_answers(**values):
        return ("timeout", None, None, None, {"status": "timeout"})

    runtime = RuntimeBoundary(nobody_answers)
    result = start_run(project, runtime, run_log)
    run = ledger.run(result["run_id"])
    assert run["review_outcome"] == "unreviewed"
    assert [row["decision"] for row in ledger.reviews(run["run_id"])] == ["timed_out"]
    assert ledger.publications(run["run_id"]) == []


def test_user_given_a_dry_run_publish_when_the_final_media_is_accepted_then_the_run_log_flags_it_as_a_dry_run_without_publications(
    project, run_log, ledger, review_environment
):
    """User Given a dry-run publish When the reviewer accepts the final media Then the run is accepted, flagged as a dry run, and no platform post is recorded."""
    platforms = PlatformBoundary()
    runtime = RuntimeBoundary(ReviewTransportBoundary(["Accept"]), platforms=platforms)
    result = start_run(project, runtime, run_log, dry_run=True)
    run = ledger.run(result["run_id"])
    assert run["review_outcome"] == "accepted"
    assert run["dry_run"]
    assert ledger.publications(run["run_id"]) == []
    assert platforms.published == []


def test_user_given_a_platform_that_refuses_the_post_when_publishing_then_the_run_log_shows_the_failed_platform_and_why(
    project, run_log, ledger, review_environment
):
    """User Given an accepted final review When the platform refuses the post Then the run is failed and the publication row shows the platform as failed with its error."""
    platforms = PlatformBoundary(accept=False)
    runtime = RuntimeBoundary(ReviewTransportBoundary(["Accept"]), platforms=platforms)
    result = start_run(project, runtime, run_log)
    run = ledger.run(result["run_id"])
    assert run["status"] == "failed"
    assert run["review_outcome"] == "accepted"
    publication = ledger.publications(run["run_id"])[0]
    assert publication["platform"] == "twitter"
    assert publication["outcome"] == "failed"
    assert "Publishing returned false for twitter" in publication["error"]


def test_user_given_a_run_that_crashes_when_generation_raises_then_the_run_log_shows_it_failed_and_the_error_still_surfaces(
    project, run_log, ledger, review_environment
):
    """User Given a run whose generation backend raises When the run aborts Then the caller still sees the error and the run log shows the run failed with the reason."""

    def broken_runtime(root, **values):
        raise RuntimeError("ComfyUI is unreachable")

    with pytest.raises(RuntimeError, match="ComfyUI is unreachable"):
        start_run(project, broken_runtime, run_log)
    (run,) = ledger.rows("SELECT * FROM runs")
    assert run["status"] == "failed"
    assert "ComfyUI is unreachable" in run["failure_reason"]
    assert run["finished_at"] is not None
    assert run["review_outcome"] == "unreviewed"


def test_user_given_a_published_run_when_the_run_log_database_disappears_then_the_published_run_still_reports_success(
    project, run_log, ledger, mysql_options, database, review_environment
):
    """User Given a run that is about to publish When the run log database becomes unavailable mid-run Then the publication still completes and the caller receives the successful result instead of an error that could trigger a double post."""

    def lose_database():
        connection = pymysql.connect(**mysql_options, autocommit=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP DATABASE `{database}`")
        finally:
            connection.close()

    platforms = PlatformBoundary(on_publish=lose_database)
    runtime = RuntimeBoundary(ReviewTransportBoundary(["Accept"]), platforms=platforms)
    result = start_run(project, runtime, run_log)
    assert result["status"] == "success"
    assert len(platforms.published) == 1


def test_user_given_a_misconfigured_run_log_when_a_run_starts_then_it_fails_before_any_generation(
    project, mysql_options, review_environment
):
    """User Given a run log pointing at an unreachable MySQL When a run is requested Then it fails immediately, before spending any generation time."""
    unreachable = {**mysql_options, "port": 1}
    with pytest.raises(pymysql.err.OperationalError):
        RunLog(f"{TEST_DATABASE_PREFIX}never", mysql_options=unreachable)


def test_user_given_no_run_log_configuration_when_the_application_builds_its_run_log_then_runs_proceed_unrecorded(
    monkeypatch,
):
    """User Given no run log database is configured When an entry point builds its run log Then there is none and runs proceed unrecorded."""
    monkeypatch.delenv("run_log_mysql_database", raising=False)
    assert RunLog.from_environment() is None


def test_user_given_run_log_settings_in_the_environment_when_an_entry_point_builds_its_run_log_then_runs_are_recorded_in_that_database(
    monkeypatch, mysql_options, database
):
    """User Given the run log database and account in media_overload.env When an entry point builds its run log Then runs are recorded in that database and reopening it keeps what is already there."""
    monkeypatch.setenv("mysql_host", mysql_options["host"])
    monkeypatch.setenv("mysql_port", str(mysql_options["port"]))
    monkeypatch.setenv("run_log_mysql_database", database)
    monkeypatch.setenv("run_log_mysql_user", mysql_options["user"])
    monkeypatch.setenv("run_log_mysql_password", mysql_options["password"])
    first = RunLog.from_environment()
    first.begin("run-1", prompt="kept across reopen", dry_run=False)
    reopened = RunLog.from_environment()
    assert reopened.database_name == database
    rows = Ledger(mysql_options, database).rows("SELECT prompt FROM runs")
    assert [row["prompt"] for row in rows] == ["kept across reopen"]


@pytest.fixture
def unreachable_run_log(monkeypatch):
    """media_overload.env settings that enable the run log against a MySQL nobody listens on."""
    monkeypatch.setenv("mysql_host", "127.0.0.1")
    monkeypatch.setenv("mysql_port", "1")
    monkeypatch.setenv("run_log_mysql_database", f"{TEST_DATABASE_PREFIX}unreachable")
    monkeypatch.setenv("run_log_mysql_user", "nobody")
    monkeypatch.setenv("run_log_mysql_password", "nothing")


def test_user_given_an_unreachable_run_log_when_the_cli_starts_a_run_then_it_fails_before_any_generation(
    unreachable_run_log, capsys
):
    """User Given run log settings that point at an unreachable MySQL When the manual command starts a run Then it fails at once instead of generating media nobody can log."""
    import run_media_interface

    with pytest.raises(pymysql.err.OperationalError):
        run_media_interface.main(["--character", "kirby", "--prompt", "x"])
    assert capsys.readouterr().out == ""


def test_user_given_an_unreachable_run_log_when_a_scheduled_job_starts_then_that_job_fails_before_any_generation_and_the_scheduler_keeps_running(
    unreachable_run_log, monkeypatch
):
    """User Given run log settings that point at an unreachable MySQL When a scheduled job starts Then the job fails before any generation, and the scheduler's guarded runner reports it instead of crashing."""
    from scheduler.scheduler import (
        _run_scheduled_job_safe,
        load_scheduler_config,
        run_scheduled_job,
    )

    monkeypatch.setenv("SCHEDULER_CHARACTER", "kirby")
    config = load_scheduler_config()
    with pytest.raises(pymysql.err.OperationalError):
        run_scheduled_job(config)
    outcome = _run_scheduled_job_safe(config, current_hour=12)
    assert outcome["status"] == "failed"


def test_user_given_a_run_that_already_finished_when_a_late_failure_is_reported_then_the_finished_run_is_not_overwritten(
    run_log, ledger
):
    """User Given a run that finished and published When an interrupt is reported afterwards Then the run log keeps the finished result instead of marking it failed."""
    run_log.begin("run-done", prompt="p", dry_run=False)
    run_log.finish(RunEntry(run_id="run-done", status="success", review_outcome="accepted"))
    run_log.fail("run-done", reason="KeyboardInterrupt: ")
    run = ledger.run("run-done")
    assert run["status"] == "success"
    assert run["failure_reason"] is None


def test_user_given_two_platforms_when_one_refuses_the_post_then_the_run_log_shows_each_platform_outcome(
    two_platform_project, run_log, ledger, review_environment
):
    """User Given an accepted final review and two target platforms When one platform publishes and the other refuses Then the run log lists both with their own outcome, id and error."""
    platforms = PlatformBoundary(accept={"twitter": True, "threads": False})
    runtime = RuntimeBoundary(ReviewTransportBoundary(["Accept"]), platforms=platforms)
    result = start_run(two_platform_project, runtime, run_log)
    run = ledger.run(result["run_id"])
    assert run["status"] == "failed"
    by_platform = {row["platform"]: row for row in ledger.publications(run["run_id"])}
    assert set(by_platform) == {"twitter", "threads"}
    assert by_platform["twitter"]["outcome"] == "published"
    assert by_platform["twitter"]["external_id"] == "twitter-4242"
    assert by_platform["threads"]["outcome"] == "failed"
    assert by_platform["threads"]["error"]


def test_user_given_a_serial_series_run_when_it_completes_then_the_run_log_lists_the_run_and_leaves_review_to_the_serial_ledger(
    project, run_log, ledger, review_environment
):
    """User Given a run routed to the serial video strategy When the serial dispatcher finishes Then the run log lists the run with the serial strategy and does not invent a review verdict."""
    root, config_path = project
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["generation"]["serial_video"] = {"enabled": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    request = CharacterWorkflowRequest(root, config_path)
    request = replace(
        request,
        generation=replace(
            request.generation,
            prompt="Series premiere",
            preferred_generation_type="serial_video",
            news_driven=False,
        ),
    )
    run_character_workflow(
        request,
        serial_dispatcher=lambda: {"status": "success", "episode_id": "ep-1"},
        run_log=run_log,
    )
    (run,) = ledger.rows("SELECT * FROM runs")
    assert run["strategy"] == "serial_video"
    assert run["prompt"] == "Series premiere"
    assert run["status"] == "success"
    assert run["review_outcome"] is None
    assert ledger.reviews(run["run_id"]) == []


# --------------------------------------------------------------------------- #
# Review verdict contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("decisions", "outcome"),
    [
        ([], "unreviewed"),
        (["automatic"], "unreviewed"),
        (["approved"], "accepted"),
        (["approved", "approved"], "accepted"),
        (["rejected"], "rejected"),
        (["approved", "rejected"], "partial"),
        (["approved", "timed_out"], "partial"),
        (["approved", "invalid_selection"], "partial"),
        (["approved", "unavailable"], "partial"),
        (["timed_out"], "unreviewed"),
        (["unavailable"], "unreviewed"),
        (["regenerate"], "regenerated"),
        (["approved", "regenerate"], "regenerated"),
    ],
)
def test_user_given_review_decisions_when_the_verdict_is_derived_then_it_reflects_what_a_human_accepted(decisions, outcome):
    """User Given the decisions taken across a run's review stages When the run verdict is derived Then only human decisions count and an accepted stage followed by a non-acceptance is partial."""
    assert derive_review_outcome(decisions) == outcome


class DiscordBoundary:
    def __init__(self, status, selected=None):
        self.status = status
        self.selected = selected

    async def __call__(self, **values):
        return (self.status, "reviewer", None, self.selected, {"status": "received"})


def review_node_outputs(
    tmp_path,
    *,
    discord,
    scope="final_media",
    policy=None,
    human=True,
    media_type="publish_review",
    media_count=1,
):
    media_paths = []
    for index in range(media_count):
        media = tmp_path / f"candidate-{index}.png"
        Image.new("RGB", (64, 40), "blue").save(media)
        media_paths.append(str(media))
    service = DiscordHumanReviewService(tmp_path, feedback_process=discord)
    goal = GoalRequest(
        prompt="Candidate",
        media_type=media_type,
        constraints={
            "review_scope": scope,
            "allow_regenerate_review": True,
            "platforms": [],
            "require_human_review": human,
            **(policy or {}),
        },
    )
    plan = ExecutionPlan(goal=goal, workflow_name="review-contract", nodes=[])
    node = ExecutionNode(
        node_id="review",
        skill_name="review.assets.select",
        depends_on=["source"],
        inputs={"review_scope": scope, "require_human_review": human},
    )
    state = RunState(goal={}, metadata={}, node_outputs={"source": {"media_paths": media_paths}})
    tools = ToolRegistry()
    skills = AgentSocialSkills(tools, tmp_path, prompt_engine=CaptionBoundary(), discord_review=service)
    return skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))


@pytest.mark.parametrize(
    ("status", "scope", "policy", "media_count", "decision"),
    [
        ("accept", "final_media", None, 1, "approved"),
        ("reject", "final_media", None, 1, "rejected"),
        ("regenerate", "final_media", None, 1, "regenerate"),
        ("timeout", "final_media", None, 1, "timed_out"),
        ("timeout", "final_media", {"pre_video_review_failure_policy": "fallback_to_top"}, 1, "timed_out"),
        # Accepting two opening-frame candidates without picking one is not a usable verdict.
        ("accept", "first_frame", None, 2, "invalid_selection"),
        # An anchor set is exactly two paired assets; accepting three is not a usable verdict.
        ("accept", "anchor_set", None, 3, "invalid_selection"),
    ],
)
def test_user_given_any_review_exit_when_the_review_stage_finishes_then_it_states_the_decision(
    status, scope, policy, media_count, decision, tmp_path, review_environment
):
    """User Given a Discord review that ends in any way When the review stage returns Then its outputs state which decision was taken so the run log never has to guess."""
    result = review_node_outputs(
        tmp_path,
        discord=DiscordBoundary(status),
        scope=scope,
        policy=policy,
        media_count=media_count,
    )
    assert result.outputs["review_decision"] == decision


def test_user_given_required_review_when_discord_is_not_configured_then_the_stage_states_it_was_unavailable(
    tmp_path, monkeypatch
):
    """User Given a required human review When Discord is not configured Then the stage is blocked and states that review was unavailable."""
    for key in ("discord_review_bot_token", "discord_review_channel_id"):
        monkeypatch.delenv(key, raising=False)
    result = review_node_outputs(tmp_path, discord=DiscordBoundary("accept"))
    assert result.status == "blocked"
    assert result.outputs["review_decision"] == "unavailable"


def test_user_given_review_that_is_not_required_when_no_human_is_asked_then_the_stage_states_it_was_automatic(
    tmp_path, monkeypatch
):
    """User Given a review that is not required and no Discord configured When assets are selected Then the stage states the selection was automatic, not human."""
    for key in ("discord_review_bot_token", "discord_review_channel_id"):
        monkeypatch.delenv(key, raising=False)
    result = review_node_outputs(
        tmp_path,
        discord=DiscordBoundary("accept"),
        human=False,
        scope="assets",
        media_type="text2img",
    )
    assert result.status == "success"
    assert result.outputs["review_decision"] == "automatic"


# --------------------------------------------------------------------------- #
# Where the produced media lives on disk
# --------------------------------------------------------------------------- #


def stored_paths(value):
    return json.loads(value) if isinstance(value, str) else list(value)


def assert_project_relative(root, paths):
    assert paths, "expected at least one recorded path"
    for path in paths:
        assert not Path(path).is_absolute() and "\\" not in path, path
        assert (root / path).is_file(), f"{path} is not a file under the project root"


def test_user_given_a_published_run_when_it_finishes_then_the_run_log_points_to_the_media_inside_the_project(
    project, run_log, ledger, review_environment
):
    """User Given an accepted run that was published When I read the run log Then I can find the produced media from the project folder, with the same relative path on Windows and in Docker."""
    runtime = RuntimeBoundary(ReviewTransportBoundary(["Accept"]))
    result = start_run(project, runtime, run_log)
    root, _ = project
    run = ledger.run(result["run_id"])
    assert_project_relative(root, stored_paths(run["output_paths"]))
    (publication,) = ledger.publications(run["run_id"])
    assert_project_relative(root, stored_paths(publication["media_paths"]))


def test_user_given_a_rejected_run_when_it_finishes_then_the_run_log_still_points_to_the_rejected_media(
    project, run_log, ledger, review_environment
):
    """User Given a run whose final media the reviewer rejects When I read the run log Then I can still open the rejected media from the project folder."""
    runtime = RuntimeBoundary(ReviewTransportBoundary(["Reject"]))
    result = start_run(project, runtime, run_log)
    root, _ = project
    run = ledger.run(result["run_id"])
    assert run["review_outcome"] == "rejected"
    assert_project_relative(root, stored_paths(run["output_paths"]))


def test_user_given_a_run_that_crashes_before_producing_media_when_it_fails_then_the_run_log_lists_no_media(
    project, run_log, ledger, review_environment
):
    """User Given a run that fails before any media exists When I read the run log Then it lists no media instead of a missing path."""

    def broken_runtime(root, **values):
        raise RuntimeError("ComfyUI is unreachable")

    with pytest.raises(RuntimeError):
        start_run(project, broken_runtime, run_log)
    (run,) = ledger.rows("SELECT * FROM runs")
    assert stored_paths(run["output_paths"]) == []


@pytest.mark.parametrize(
    ("root", "raw", "expected"),
    [
        (PureWindowsPath("C:/work/mediaoverload"), "C:\\work\\mediaoverload\\output\\kirby\\a.png", "output/kirby/a.png"),
        (PurePosixPath("/app"), "/app/output/kirby/a.png", "output/kirby/a.png"),
        (PurePosixPath("/app"), "output/kirby/a.png", "output/kirby/a.png"),
        (PurePosixPath("/app"), "output\\kirby\\a.png", "output/kirby/a.png"),
        (PurePosixPath("/app"), "/data/elsewhere/a.png", "/data/elsewhere/a.png"),
    ],
    ids=["windows-host", "docker-container", "already-relative", "windows-style-relative", "outside-project-kept"],
)
def test_user_given_the_same_output_seen_from_windows_or_docker_when_it_is_recorded_then_both_store_the_same_project_relative_path(
    root, raw, expected
):
    """User Given the project mounted at a Windows folder or at /app in Docker When a media path is recorded Then both runtimes store the same project-relative path, and a file outside the project keeps its absolute path."""
    assert project_relative_path(raw, root) == expected
