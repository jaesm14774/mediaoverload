from __future__ import annotations

import json
import logging
from hashlib import sha1

from agentic.memory import PortfolioMemory, RunMemory
from agentic.memory.run import MemoryEvent
from agentic.memory.portfolio import PortfolioRecord
from agentic.runtime.contracts import ExecutionPlan, RunRecord, RunState, SkillContext, SkillResult, WorkflowRunResult
from agentic.runtime.creativity import FeedbackRanker, RetryPolicy
from agentic.runtime.observability import RunRecorder
from agentic.runtime.registry import SkillRegistry


class WorkflowRunner:
    def __init__(
        self,
        skill_registry: SkillRegistry,
        run_memory: RunMemory | None = None,
        portfolio_memory: PortfolioMemory | None = None,
        retry_policy: RetryPolicy | None = None,
        feedback_ranker: FeedbackRanker | None = None,
        logger: logging.Logger | None = None,
        recorder: RunRecorder | None = None,
    ) -> None:
        self.skill_registry = skill_registry
        self.run_memory = run_memory
        self.portfolio_memory = portfolio_memory
        self.retry_policy = retry_policy or RetryPolicy()
        self.feedback_ranker = feedback_ranker
        self.logger = logger
        self.recorder = recorder or getattr(logger, "run_recorder", None)

    def run(self, plan: ExecutionPlan) -> WorkflowRunResult:
        graph = plan.as_graph()
        ordered_nodes = graph.topological_order()
        state = RunState(
            goal={
                "prompt": plan.goal.prompt,
                "media_type": plan.goal.media_type,
                "duration_seconds": plan.goal.duration_seconds,
                "style": plan.goal.style,
                "auto_download_assets": plan.goal.auto_download_assets,
                "constraints": plan.goal.constraints,
            },
            metadata=plan.metadata,
        )
        records: list[RunRecord] = []

        for node in ordered_nodes:
            skill = self.skill_registry.get(node.skill_name)
            context = SkillContext(plan=plan, node=node, state=state)
            attempts = 0
            while True:
                attempts += 1
                self._log(f"node.start | id={node.node_id} | skill={node.skill_name} | attempt={attempts} | stage={node.stage or 'runtime'}")
                if self.recorder:
                    self.recorder.record_event(
                        "node.started",
                        node_id=node.node_id,
                        skill_name=node.skill_name,
                        attempt=attempts,
                        stage=node.stage or "runtime",
                    )
                try:
                    result = skill.handler(context)
                except Exception as exc:
                    result = SkillResult(
                        status="failed",
                        outputs={},
                        logs=[f"{type(exc).__name__}: {exc}"],
                    )
                self._log(
                    f"node.end | id={node.node_id} | skill={node.skill_name} | status={result.status} | attempt={attempts}"
                )
                state[node.node_id] = result.outputs
                records.append(
                    RunRecord(
                        node_id=node.node_id,
                        skill_name=node.skill_name,
                        status=result.status,
                        attempt=attempts,
                        outputs=result.outputs,
                        metrics=result.metrics,
                        logs=result.logs,
                    )
                )
                if self.recorder:
                    self.recorder.record_node(
                        node_id=node.node_id,
                        skill_name=node.skill_name,
                        status=result.status,
                        attempt=attempts,
                        outputs=result.outputs,
                        metrics=result.metrics,
                        logs=result.logs,
                    )
                if self.run_memory:
                    self.run_memory.record(
                        MemoryEvent(
                            node_id=node.node_id,
                            skill_name=node.skill_name,
                            status=result.status,
                            attempt=attempts,
                            outputs=result.outputs,
                            metrics=result.metrics,
                            logs=result.logs,
                        )
                    )
                if self.feedback_ranker:
                    feedback = self.feedback_ranker.evaluate(node.node_id, result)
                    if feedback:
                        state.add_feedback(feedback)
                self._capture_prompt_metadata(state, node.node_id, result.outputs, attempts)
                if result.status == "success":
                    break
                if not self.retry_policy.should_retry(result.status, attempts):
                    run_result = WorkflowRunResult(
                        workflow_name=plan.workflow_name,
                        status="failed",
                        records=records,
                        state=state,
                    )
                    self._record_portfolio(plan, run_result)
                    self._record_workflow_result(run_result)
                    return run_result

        run_result = WorkflowRunResult(
            workflow_name=plan.workflow_name,
            status="success",
            records=records,
            state=state,
        )
        self._record_portfolio(plan, run_result)
        self._record_workflow_result(run_result)
        return run_result

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _record_portfolio(self, plan: ExecutionPlan, result: WorkflowRunResult) -> None:
        if not self.portfolio_memory:
            return
        goal_signature = self._goal_signature(plan)
        metrics = {}
        if result.records:
            metrics = result.records[-1].metrics
        notes = [record.logs for record in result.records if record.logs]
        flat_notes = [log for entry in notes for log in entry]
        record = PortfolioRecord(
            goal_signature=goal_signature,
            workflow_name=plan.workflow_name,
            status=result.status,
            metrics=metrics,
            notes=flat_notes,
        )
        self.portfolio_memory.append(record)

    def _record_workflow_result(self, result: WorkflowRunResult) -> None:
        if self.recorder:
            self.recorder.record_workflow_result(result.workflow_name, result.to_dict())

    @staticmethod
    def _goal_signature(plan: ExecutionPlan) -> str:
        digest_source = json.dumps(
            {
                "prompt": plan.goal.prompt,
                "media_type": plan.goal.media_type,
                "style": plan.goal.style,
                "metadata": plan.metadata,
            },
            sort_keys=True,
        )
        return sha1(digest_source.encode("utf-8")).hexdigest()

    @staticmethod
    def _capture_prompt_metadata(state: RunState, node_id: str, outputs: dict[str, object], attempts: int) -> None:
        prompt_mode = outputs.get("prompt_mode")
        if isinstance(prompt_mode, str) and prompt_mode:
            state.set_node_prompt_mode(node_id, prompt_mode)

        lineage_keys = (
            "original_prompt",
            "revised_prompt",
            "review_notes",
            "selected_assets",
            "selected_count",
            "rejected_assets",
            "rejected_asset_details",
            "selection_rationale",
            "regeneration_notes",
            "failure_tags",
            "retry_direction",
            "retry_intensity",
            "publish_ready",
            "media_paths",
            "caption",
            "platform_bundle",
            "caption_strategy",
            "dispatch_ready",
            "review_mode",
            "reviewer",
            "review_session_id",
            "review_session_path",
            "edited_review_text",
            "fallback_reason",
            "manager_error",
            "llm_backend",
        )
        lineage_payload = {key: outputs[key] for key in lineage_keys if key in outputs}
        if attempts > 1:
            lineage_payload["retry_count"] = attempts - 1
        if lineage_payload:
            lineage_payload["node_id"] = node_id
            if isinstance(prompt_mode, str) and prompt_mode:
                lineage_payload["prompt_mode"] = prompt_mode
            state.add_prompt_lineage(lineage_payload)
