"""重新生成 acceptance at the user entry points and the real Discord view boundary."""
from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
import subprocess
import shutil

import pytest
import yaml
from types import SimpleNamespace

import discord
from PIL import Image

import run_media_interface
from agentic.app.character_requests import CharacterWorkflowRequest
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext, SkillResult
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.runtime.planner import TaskPlanner
from agentic.memory import RunMemory
from agentic.app.character_workflow import run_character_workflow
from agentic.app.main import build_runtime
from agentic.tools.media_services import register_media_service_tools
from agentic.skills.agent_social import AgentSocialSkills
from agentic.tools.context_services import DiscordHumanReviewService
from agentic.tools.context_services import DiscordRunNotificationService
from agentic.assets.registry import AssetRegistry
from scheduler.scheduler import load_scheduler_config, run_scheduled_job


class WorkflowBoundary:
    """A workflow boundary fake matching the public 重新生成/final result contract."""
    def __init__(self, decisions=("regeneration_requested", "success")):
        self.requests = []
        self.completed_runs = []
        self.decisions = iter(decisions)

    def __call__(self, request):
        assert isinstance(request, CharacterWorkflowRequest)
        self.requests.append(request)
        result = {"run_id": f"run-{len(self.requests)}", "source_generation_type": "game_sprite"}
        status = next(self.decisions)
        result.update(status=status, regeneration_requested=status == "regeneration_requested")
        self.completed_runs.append(result)
        return result


def test_user_given_manual_final_review_when_regenerate_is_selected_then_a_new_full_run_finishes(capsys):
    """User Given a manually started final review When 重新生成 is selected Then the current result is abandoned and a fresh full run finishes without waiting for the scheduler."""
    workflow = WorkflowBoundary()
    run_media_interface.main(["--character", "kirby"], workflow_runner=workflow)
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "success"
    assert result["run_id"] != workflow.completed_runs[0]["run_id"]
    assert all(request.review.allow_regenerate for request in workflow.requests)
    assert all(request.config_path.name == "kirby.yaml" for request in workflow.requests)
    assert workflow.completed_runs[0]["status"] == "regeneration_requested"


def test_user_given_scheduled_final_review_when_regenerate_is_selected_then_a_fresh_run_finishes(monkeypatch):
    """User Given a scheduled final review When 重新生成 is selected Then the same full rerun contract works immediately and stops after a final decision."""
    monkeypatch.setenv("SCHEDULER_CHARACTER", "kirby")
    workflow = WorkflowBoundary()
    result = run_scheduled_job(load_scheduler_config(), workflow_runner=workflow)
    assert result["status"] == "success"
    assert result["run_id"] != workflow.completed_runs[0]["run_id"]
    assert all(request.review.allow_regenerate for request in workflow.requests)


class InteractionResponse:
    """An interaction acknowledgement fake with discord.py's public async methods."""
    def __init__(self):
        self.acknowledged = False
        self.messages = []

    async def defer(self):
        self.acknowledged = True

    async def send_message(self, content, *, ephemeral=False):
        self.messages.append((content, ephemeral))


class ReviewMessage:
    async def edit(self, **values):
        self.values = values


class CaptionBoundary:
    def prepare_publish_caption(self, goal, prefix, hashtags, platforms, media_paths=None, review_notes="", visual_paths=None):
        return {"caption": "Final caption", "hashtags": "", "platform_bundle": {}, "dispatch_ready": True}


def test_user_given_final_review_when_regenerate_is_clicked_then_the_decision_reaches_the_review_service(tmp_path, monkeypatch):
    """User Given a final review with Edit retained When an authorized reviewer clicks 重新生成 Then Discord acknowledges it and the review service emits the full-rerun decision with no selected media."""
    from agentic.tools.discord_review_view import create_discord_review_view

    monkeypatch.setenv("discord_review_allowed_user_ids", "42")
    monkeypatch.setenv("discord_review_bot_token", "contract-fake")
    monkeypatch.setenv("discord_review_channel_id", "123")
    media = tmp_path / "final.png"
    Image.new("RGB", (64, 40), "blue").save(media)
    observed = {}

    async def feedback(**values):
        # Verify the fake's input vocabulary against the real transport boundary.
        from agentic.tools.context_services import _run_discord_file_feedback_process
        inspect.signature(_run_discord_file_feedback_process).bind(**values)
        files = [discord.File(path) for path in values["filepaths"]]
        view = create_discord_review_view(
            files=files, content=values["text"], timeout=values["timeout"],
            allow_asset_selection=values["allow_asset_selection"],
            allow_text_edit=values["allow_text_edit"], selection_mode=values["selection_mode"],
            selection_required=values["selection_required"], selection_limit=values["selection_limit"],
            allow_regenerate=values["allow_regenerate"],
        )
        observed["labels"] = {button.label for button in view.children if isinstance(button, discord.ui.Button)}
        message = ReviewMessage()
        view.message = message
        response = InteractionResponse()
        interaction = SimpleNamespace(user=SimpleNamespace(id=42), response=response)
        button = next(button for button in view.children if getattr(button, "label", None) == "重新生成")
        await button.callback(interaction)
        observed["acknowledged"] = response.acknowledged
        observed["message"] = message.values
        for file in files:
            file.close()
        return view.result, view.user_name, view.content, view.selected_files, {"status": "decision_received"}

    service = DiscordHumanReviewService(tmp_path, feedback_process=feedback)
    decision = service.review_candidates(
        text="Final caption", media_paths=[str(media)], review_scope="final_video", allow_regenerate=True,
    )
    assert observed["labels"] == {"Accept", "Reject", "Edit", "重新生成"}
    assert observed["acknowledged"]
    assert observed["message"]["view"] is None
    assert decision.status == "regeneration_requested"
    assert decision.selected_paths == []

    goal = GoalRequest(prompt="Final caption", media_type="publish_review", constraints={
        "review_scope": "final_media", "allow_regenerate_review": True, "platforms": [],
    })
    plan = ExecutionPlan(goal=goal, workflow_name="publish_review_v1", nodes=[])
    node = ExecutionNode(node_id="review-select", skill_name="review.assets.select", depends_on=["ingest-media"])
    state = RunState(goal={"prompt": goal.prompt}, metadata={}, node_outputs={"ingest-media": {"media_paths": [str(media)]}})
    tools = ToolRegistry()
    register_media_service_tools(tools, tmp_path)
    skills = AgentSocialSkills(tools, tmp_path, prompt_engine=CaptionBoundary(), discord_review=service)
    result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))
    assert result.status == "blocked"
    assert result.outputs["regeneration_requested"] is True
    assert result.outputs["media_paths"] == []


def test_user_given_disabled_regeneration_when_buttons_are_created_then_only_standard_controls_are_available():
    """User Given regeneration is disabled When review controls are created Then only Accept, Reject and Edit remain available."""
    from agentic.tools.discord_review_view import create_discord_review_view

    async def inspect_view():
        view = create_discord_review_view(files=[], content="Intermediate asset", timeout=30, allow_asset_selection=False)
        labels = {button.label for button in view.children}
        view.stop()
        return labels

    assert asyncio.run(inspect_view()) == {"Accept", "Reject", "Edit"}


def test_user_given_an_unauthorized_reviewer_when_regenerate_is_clicked_then_no_rerun_is_requested(monkeypatch):
    """User Given 重新生成 is available When an unauthorized user clicks it Then the review refuses the interaction and does not request another run."""
    from agentic.tools.discord_review_view import create_discord_review_view

    monkeypatch.setenv("discord_review_allowed_user_ids", "42")

    async def click():
        view = create_discord_review_view(files=[], content="Final caption", timeout=30, allow_asset_selection=False, allow_regenerate=True)
        response = InteractionResponse()
        interaction = SimpleNamespace(user=SimpleNamespace(id=43), response=response)
        button = next(button for button in view.children if button.label == "重新生成")
        await button.callback(interaction)
        result = view.result
        view.stop()
        return result, response

    result, response = asyncio.run(click())
    assert result is None
    assert not response.acknowledged
    assert response.messages == [("You are not authorized for this review.", True)]


def test_user_given_a_final_review_when_regenerate_is_clicked_twice_then_only_the_first_decision_is_accepted(monkeypatch):
    """User Given a final review When 重新生成 interactions overlap Then the first decision is acknowledged and later interactions report that the review is complete."""
    from agentic.tools.discord_review_view import create_discord_review_view

    monkeypatch.setenv("discord_review_allowed_user_ids", "42")

    async def click():
        view = create_discord_review_view(files=[], content="Final caption", timeout=30, allow_asset_selection=False, allow_regenerate=True)
        button = next(button for button in view.children if button.label == "重新生成")
        first, second = InteractionResponse(), InteractionResponse()
        await button.callback(SimpleNamespace(user=SimpleNamespace(id=42), response=first))
        await button.callback(SimpleNamespace(user=SimpleNamespace(id=42), response=second))
        return view.result, first, second

    result, first, second = asyncio.run(click())
    assert result == "regenerate"
    assert first.acknowledged
    assert not second.acknowledged
    assert second.messages == [("This review is already complete.", True)]


class ReviewTransportBoundary:
    """Script user decisions through the same discord.py View as the real transport."""

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.reviewed = []

    async def __call__(self, **values):
        from agentic.tools.context_services import _run_discord_file_feedback_process
        from agentic.tools.discord_review_view import create_discord_review_view

        inspect.signature(_run_discord_file_feedback_process).bind(**values)
        files = [discord.File(path) for path in values["filepaths"]]
        try:
            view = create_discord_review_view(
                files=files, content=values["text"], timeout=values["timeout"],
                allow_asset_selection=values["allow_asset_selection"],
                allow_text_edit=values["allow_text_edit"],
                selection_mode=values["selection_mode"],
                selection_required=values["selection_required"],
                selection_limit=values["selection_limit"],
                allow_regenerate=values["allow_regenerate"],
            )
            label = next(self.decisions)
            self.reviewed.append({
                "labels": {item.label for item in view.children if isinstance(item, discord.ui.Button)},
                "media_paths": values["filepaths"],
            })
            if label == "Accept" and values["selection_required"]:
                count = 1 if values["selection_mode"] == "single" else min(
                    len(files), values["selection_limit"] or len(files),
                )
                view.selected_files = list(range(count))
            response = InteractionResponse()
            button = next(item for item in view.children if getattr(item, "label", None) == label)
            await button.callback(SimpleNamespace(user=SimpleNamespace(id=42), response=response))
            assert response.acknowledged
            return view.result, view.user_name, view.content, view.selected_files, {"status": "decision_received"}
        finally:
            for file in files:
                file.close()


@pytest.fixture
def review_environment(monkeypatch):
    monkeypatch.setenv("discord_review_allowed_user_ids", "42")
    monkeypatch.setenv("discord_review_bot_token", "contract-fake")
    monkeypatch.setenv("discord_review_channel_id", "123")


@pytest.mark.parametrize("scope", ["first_frame", "last_frame", "anchor_set", "anchors", "reference", "assets", "final_media", "final_video"])
def test_user_given_any_review_stage_when_regenerate_is_clicked_then_current_media_are_abandoned(scope, tmp_path, review_environment):
    """User Given any image, reference, anchor or final review When 重新生成 is clicked without selecting an asset Then the review requests a fresh full run and releases no media downstream."""
    media = tmp_path / "candidate.png"
    Image.new("RGB", (64, 40), "blue").save(media)
    if scope == "final_video":
        video = tmp_path / "candidate.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-loop", "1", "-i", str(media),
            "-t", "0.1", "-pix_fmt", "yuv420p", str(video),
        ], check=True)
        media = video
    transport = ReviewTransportBoundary(["重新生成"])
    service = DiscordHumanReviewService(tmp_path, feedback_process=transport)
    goal = GoalRequest(prompt="Candidate", media_type="publish_review", constraints={
        "review_scope": scope, "allow_regenerate_review": True, "platforms": [],
    })
    plan = ExecutionPlan(goal=goal, workflow_name="review-contract", nodes=[])
    node = ExecutionNode(node_id="review", skill_name="review.assets.select", depends_on=["source"])
    state = RunState(goal={}, metadata={}, node_outputs={"source": {"media_paths": [str(media)]}})
    tools = ToolRegistry()
    register_media_service_tools(tools, tmp_path)
    skills = AgentSocialSkills(tools, tmp_path, prompt_engine=CaptionBoundary(), discord_review=service)
    result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))
    assert "重新生成" in transport.reviewed[0]["labels"]
    assert result.status == "blocked"
    assert result.outputs["regeneration_requested"] is True
    assert result.outputs["selected_assets"] == []
    assert result.outputs["media_paths"] == []
    saved = json.loads(Path(result.outputs["review_session_path"]).read_text(encoding="utf-8"))
    assert saved["normalized_status"] == "regeneration_requested"


class StagePlanBoundary:
    """A small plan fixture using the real goal and graph contracts."""

    create_goal = TaskPlanner.create_goal

    def __init__(self, scope):
        self.scope = scope

    def build_plan(self, goal):
        return ExecutionPlan(goal=goal, workflow_name="stage-contract", nodes=[
            ExecutionNode(node_id="generate-image", skill_name="fixture.generate"),
            ExecutionNode(node_id="user-review", skill_name="review.assets.select", depends_on=["generate-image"], inputs={
                "review_scope": self.scope, "require_human_review": True,
                "limit": 2 if self.scope == "anchor_set" else 1,
            }),
            ExecutionNode(node_id="render-video", skill_name="fixture.render", depends_on=["user-review"]),
        ])


class FinalPlanBoundary(StagePlanBoundary):
    def __init__(self, scope, root):
        super().__init__(scope)
        self.publish_planner = TaskPlanner(AssetRegistry(root))

    def build_plan(self, goal):
        if goal.media_type == "publish_review":
            return self.publish_planner.build_plan(goal)
        return ExecutionPlan(goal=goal, workflow_name="final-contract", nodes=[
            ExecutionNode(node_id="generate-image", skill_name="fixture.generate"),
        ])


class StageRuntimeBoundary:
    """Replace external generation, keep the real runner, review service and controls."""

    def __init__(self, scope, transport):
        self.scope = scope
        self.transport = transport
        self.completed_renders = []
        self.dispatches = []

    def __call__(self, root, **values):
        inspect.signature(build_runtime).bind(root, **values)
        output_root = values["output_root"]
        run_id = values["run_id"]
        service = DiscordHumanReviewService(output_root, feedback_process=self.transport)
        social = AgentSocialSkills(ToolRegistry(), output_root, prompt_engine=CaptionBoundary(), discord_review=service)
        registry = SkillRegistry()

        def generate(context):
            paths = []
            for index in range(2 if self.scope == "anchor_set" else 1):
                image = output_root / f"{run_id}-{index}.png"
                Image.new("RGB", (64, 40), "blue").save(image)
                paths.append(str(image))
            return SkillResult(status="success", outputs={"media_paths": paths})

        def render(context):
            self.completed_renders.append(run_id)
            return SkillResult(status="success", outputs={"media_paths": context.state["user-review"]["media_paths"]})

        registry.register("fixture.generate", generate, "Image generation boundary")
        registry.register("review.assets.select", social.select_best_assets, "Real user review")
        registry.register("fixture.render", render, "Video generation boundary")
        registry.register("publish.media.ingest", social.ingest_media, "Real publish ingest")

        def forbidden_dispatch(context):
            self.dispatches.append(run_id)
            raise AssertionError("An abandoned or rejected review must not dispatch publication")

        for skill in ("publish.media.process", "publish.caption.prepare", "publish.social.dispatch", "agent.summary.persist"):
            registry.register(skill, forbidden_dispatch, "Unapproved publication boundary")
        memory = RunMemory()
        runner = WorkflowRunner(registry, run_memory=memory, logger=values["logger"], recorder=values["recorder"])
        planner = FinalPlanBoundary(self.scope, root) if self.scope == "final_media" else StagePlanBoundary(self.scope)
        return planner, runner, memory


class NotificationBoundary:
    def notify(self, *args, **kwargs):
        inspect.signature(DiscordRunNotificationService.notify).bind(self, *args, **kwargs)
        return {"status": "skipped", "reason": "Local acceptance fixture"}


@pytest.mark.parametrize("entry", ["manual", "scheduler"])
@pytest.mark.parametrize("scope", ["first_frame", "last_frame", "anchor_set", "reference", "assets"])
def test_user_given_intermediate_review_when_regenerated_then_entry_point_finishes_a_fresh_full_run(entry, scope, tmp_path, monkeypatch, capsys, review_environment):
    """User Given a manual or scheduled intermediate review When 重新生成 is clicked and the next review is accepted Then a fresh run finishes, the abandoned run never renders video, and both review decisions remain inspectable."""
    config_path = tmp_path / "fixture.yaml"
    shutil.copytree(Path(__file__).resolve().parents[2] / "configs", tmp_path / "configs")
    config_path.write_text(yaml.safe_dump({
        "character": {"name": "Fixture"},
        "generation": {"subject_mode": "single", "output_dir": str(tmp_path / "output")},
    }), encoding="utf-8")
    transport = ReviewTransportBoundary(["重新生成", "Accept"])
    runtime = StageRuntimeBoundary(scope, transport)
    completed = []

    def workflow(request):
        from dataclasses import replace
        request = replace(request, repo_root=tmp_path, generation=replace(
            request.generation, prompt="Fixture scene", preferred_generation_type="text2img", news_driven=False,
        ))
        result = run_character_workflow(request, runtime_builder=runtime)
        completed.append(result)
        return result

    inspect.signature(run_character_workflow).bind(CharacterWorkflowRequest(tmp_path, config_path), runtime_builder=runtime)
    if entry == "manual":
        run_media_interface.main(["--config", str(config_path), "--prompt", "Fixture scene", "--no-publish"], workflow_runner=workflow)
        result = json.loads(capsys.readouterr().out)
    else:
        monkeypatch.setenv("SCHEDULER_CONFIG", str(config_path))
        monkeypatch.setenv("SCHEDULER_PUBLISH_AFTER_GENERATE", "false")
        result = run_scheduled_job(load_scheduler_config(), workflow_runner=workflow)
    abandoned = completed[0]
    assert abandoned["status"] == "regeneration_requested"
    assert abandoned["failure_reason"] is None
    assert abandoned["stage_status"]["review"]["status"] == "regeneration_requested"
    assert result["status"] == "success"
    assert result["run_id"] != abandoned["run_id"]
    assert abandoned["run_id"] not in runtime.completed_renders
    assert result["run_id"] in runtime.completed_renders
    assert all("重新生成" in review["labels"] for review in transport.reviewed)
    manifest = tmp_path / "logs" / "runs" / abandoned["run_id"] / "run_manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "regeneration_requested"


def test_user_given_a_fixed_seed_when_regenerated_then_the_next_run_can_generate_new_images(capsys):
    """User Given the initial run uses a fixed seed When 重新生成 is requested Then the initial seed is honored and subsequent runs use fresh sampling."""
    workflow = WorkflowBoundary()
    run_media_interface.main(["--character", "kirby", "--seed", "123"], workflow_runner=workflow)
    assert workflow.requests[0].generation.seed == 123
    assert workflow.requests[-1].generation.seed is None


@pytest.mark.parametrize("terminal_status", ["success", "failed", "awaiting_user_action"])
def test_user_given_repeated_regeneration_when_a_terminal_result_arrives_then_reruns_stop(terminal_status, capsys):
    """User Given multiple explicit regeneration requests When the fresh run completes, fails or needs user action Then the session returns that result without automatically launching more generation."""
    workflow = WorkflowBoundary(["regeneration_requested", "regeneration_requested", terminal_status])
    run_media_interface.main(["--character", "kirby"], workflow_runner=workflow)
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == terminal_status
    assert result["run_id"] == workflow.completed_runs[-1]["run_id"]
    assert all(run["status"] == "regeneration_requested" for run in workflow.completed_runs[:-1])


def test_user_given_final_publish_review_when_regenerated_then_abandoned_media_never_publish(tmp_path, capsys, review_environment):
    """User Given a final publish review When 重新生成 is clicked and the fresh review is rejected Then a new full run is delivered for review and neither run dispatches any publication."""
    from dataclasses import replace

    shutil.copytree(Path(__file__).resolve().parents[2] / "configs", tmp_path / "configs")
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(yaml.safe_dump({
        "character": {"name": "Fixture"},
        "generation": {"subject_mode": "single", "output_dir": str(tmp_path / "output")},
        "social_media": {"platforms": {"twitter": {"enabled": True}}},
    }), encoding="utf-8")
    transport = ReviewTransportBoundary(["重新生成", "Reject"])
    runtime = StageRuntimeBoundary("final_media", transport)
    completed = []

    def workflow(request):
        request = replace(request, repo_root=tmp_path, generation=replace(
            request.generation, prompt="Fixture scene", preferred_generation_type="text2img", news_driven=False,
        ))
        result = run_character_workflow(request, runtime_builder=runtime, notification_service=NotificationBoundary())
        completed.append(result)
        return result

    run_media_interface.main(["--config", str(config_path), "--prompt", "Fixture scene"], workflow_runner=workflow)
    result = json.loads(capsys.readouterr().out)
    assert completed[0]["status"] == "regeneration_requested"
    assert completed[0]["stage_status"]["publish"]["status"] == "regeneration_requested"
    assert result["run_id"] != completed[0]["run_id"]
    assert result["status"] == "failed"
    assert transport.reviewed[0]["media_paths"] != transport.reviewed[-1]["media_paths"]
    assert runtime.dispatches == []
