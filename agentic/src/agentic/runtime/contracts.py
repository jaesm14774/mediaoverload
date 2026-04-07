from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class ContractError(RuntimeError):
    """Raised when execution graph contracts are violated."""


@dataclass(slots=True)
class GoalRequest:
    prompt: str
    media_type: str = "long_video"
    duration_seconds: int = 30
    style: str = "cinematic surreal"
    auto_download_assets: bool = False
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionNode:
    node_id: str
    skill_name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    tool_name: str | None = None
    stage: str | None = None


@dataclass(slots=True)
class ExecutionPlan:
    goal: GoalRequest
    workflow_name: str
    nodes: list[ExecutionNode]
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": asdict(self.goal),
            "workflow_name": self.workflow_name,
            "metadata": self.metadata,
            "description": self.description,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "skill_name": node.skill_name,
                    "inputs": node.inputs,
                    "depends_on": node.depends_on,
                    "tags": node.tags,
                    "tool_name": node.tool_name,
                    "stage": node.stage,
                }
                for node in self.nodes
            ],
        }

    def as_graph(self) -> "ExecutionGraph":
        return ExecutionGraph(self.nodes)


@dataclass(slots=True)
class ExecutionGraph:
    nodes: list[ExecutionNode]

    def _index(self) -> dict[str, ExecutionNode]:
        return {node.node_id: node for node in self.nodes}

    def topological_order(self) -> list[ExecutionNode]:
        graph = self._index()
        visited: set[str] = set()
        visiting: set[str] = set()
        ordered: list[ExecutionNode] = []

        def dfs(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                raise ContractError(f"Cyclic dependency detected at node '{node_id}'")
            visiting.add(node_id)
            node = graph.get(node_id)
            if node is None:
                raise ContractError(f"Node '{node_id}' referenced in graph but not defined")
            for dependency in node.depends_on:
                if dependency not in graph:
                    raise ContractError(f"Node '{node_id}' depends on unknown node '{dependency}'")
                dfs(dependency)
            visiting.remove(node_id)
            visited.add(node_id)
            ordered.append(node)

        for node in self.nodes:
            dfs(node.node_id)
        return ordered


@dataclass(slots=True)
class SkillResult:
    status: str
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunRecord:
    node_id: str
    skill_name: str
    status: str
    attempt: int = 1
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WorkflowRunResult:
    workflow_name: str
    status: str
    records: list[RunRecord]
    state: "RunState"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "status": self.status,
            "state": self.state.to_dict(),
            "records": [
                {
                    "node_id": record.node_id,
                    "skill_name": record.skill_name,
                    "status": record.status,
                    "attempt": record.attempt,
                    "outputs": record.outputs,
                    "metrics": record.metrics,
                    "logs": record.logs,
                }
                for record in self.records
            ],
        }


@dataclass(slots=True)
class SkillContext:
    plan: ExecutionPlan
    node: ExecutionNode
    state: "RunState"


@dataclass(slots=True)
class RunState:
    goal: dict[str, Any]
    metadata: dict[str, Any]
    node_outputs: dict[str, Any] = field(default_factory=dict)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    prompt_lineage: list[dict[str, Any]] = field(default_factory=list)
    node_prompt_modes: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key == "goal":
            return self.goal
        if key == "metadata":
            return self.metadata
        return self.node_outputs[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.node_outputs[key] = value

    def add_feedback(self, payload: dict[str, Any]) -> None:
        self.feedback.append(payload)

    def add_prompt_lineage(self, payload: dict[str, Any]) -> None:
        self.prompt_lineage.append(payload)

    def set_node_prompt_mode(self, node_id: str, prompt_mode: str) -> None:
        self.node_prompt_modes[node_id] = prompt_mode

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "metadata": self.metadata,
            "node_outputs": self.node_outputs,
            "feedback": self.feedback,
            "prompt_lineage": self.prompt_lineage,
            "node_prompt_modes": self.node_prompt_modes,
        }


class SkillHandler(Protocol):
    def __call__(self, context: SkillContext) -> SkillResult: ...


class ToolHandler(Protocol):
    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]: ...
