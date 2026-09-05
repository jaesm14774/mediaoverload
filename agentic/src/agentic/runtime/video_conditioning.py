"""Provider-neutral long-video conditioning contracts.

Workflow JSON files declare which logical recipes they implement.  The planner
samples those recipes and passes a logical conditioning plan to the renderer;
provider-specific node names stay in the workflow/tool binding layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RecipeContract:
    """The input shape and state semantics of one workflow recipe."""

    name: str
    anchor_positions: tuple[str, ...] = ()
    reference_types: tuple[str, ...] = ()
    continuation: str = "none"
    render_tool: str = "comfy.workflow.image_to_video"
    reference_selection_limit: int = 0
    reference_minimum: int = 0
    reference_maximum: int = 0

    @classmethod
    def from_dict(cls, name: str, payload: Mapping[str, Any] | None) -> "RecipeContract":
        values = dict(payload or {})
        anchors = values.get("anchors", values.get("anchor_positions", ()))
        references = values.get("references", values.get("reference_types", ()))
        if isinstance(anchors, str):
            anchors = [anchors]
        if isinstance(references, str):
            references = [references]
        limit = int(values.get("reference_selection_limit") or values.get("reference_maximum") or 0)
        maximum = int(values.get("reference_maximum") or limit or 0)
        minimum = int(values.get("reference_minimum") or (1 if references else 0))
        return cls(
            name=str(name),
            anchor_positions=tuple(str(item) for item in anchors or ()),
            reference_types=tuple(str(item) for item in references or ()),
            continuation=str(values.get("continuation") or "none"),
            render_tool=str(values.get("render_tool") or "comfy.workflow.image_to_video"),
            reference_selection_limit=max(0, limit),
            reference_minimum=max(0, minimum),
            reference_maximum=max(0, maximum),
        )

    @property
    def requires_references(self) -> bool:
        return bool(self.reference_types or self.reference_minimum)

    @property
    def requires_first(self) -> bool:
        return "first" in self.anchor_positions

    @property
    def requires_last(self) -> bool:
        return "last" in self.anchor_positions

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "anchors": list(self.anchor_positions),
            "references": list(self.reference_types),
            "continuation": self.continuation,
            "render_tool": self.render_tool,
            "reference_selection_limit": self.reference_selection_limit,
            "reference_minimum": self.reference_minimum,
            "reference_maximum": self.reference_maximum,
        }


@dataclass(frozen=True, slots=True)
class WorkflowCapability:
    workflow_name: str
    provider: str = ""
    recipes: Mapping[str, RecipeContract] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest: Any) -> "WorkflowCapability":
        raw = dict(getattr(manifest, "conditioning", {}) or {})
        raw_recipes = raw.get("recipes", {})
        if not isinstance(raw_recipes, Mapping):
            raw_recipes = {}
        recipes = {
            str(name): RecipeContract.from_dict(str(name), value if isinstance(value, Mapping) else {})
            for name, value in raw_recipes.items()
        }
        return cls(
            workflow_name=str(getattr(manifest, "name", "")),
            provider=str(raw.get("provider") or ""),
            recipes=recipes,
        )

    def supports(self, recipe_name: str) -> bool:
        return str(recipe_name) in self.recipes


@dataclass(frozen=True, slots=True)
class ConditioningPlan:
    """Logical segment inputs resolved by the planner before rendering."""

    recipe: str
    workflow_name: str
    anchor_nodes: Mapping[str, str] = field(default_factory=dict)
    reference_node: str = ""
    continuation_node: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "workflow_name": self.workflow_name,
            "anchors": dict(self.anchor_nodes),
            "reference_node": self.reference_node,
            "continuation_node": self.continuation_node,
        }


def capabilities_from_manifests(manifests: list[Any]) -> dict[str, WorkflowCapability]:
    capabilities: dict[str, WorkflowCapability] = {}
    for manifest in manifests:
        capability = WorkflowCapability.from_manifest(manifest)
        if capability.recipes:
            capabilities[capability.workflow_name] = capability
    return capabilities


def recipe_candidates(
    capabilities: Mapping[str, WorkflowCapability],
    *,
    preferred_workflows: list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[WorkflowCapability]]:
    preferred = [str(name) for name in (preferred_workflows or ()) if str(name).strip()]
    ordered = sorted(
        capabilities.values(),
        key=lambda item: (preferred.index(item.workflow_name) if item.workflow_name in preferred else len(preferred), item.workflow_name),
    )
    result: dict[str, list[WorkflowCapability]] = {}
    for capability in ordered:
        for recipe_name in capability.recipes:
            result.setdefault(recipe_name, []).append(capability)
    return result




def production_recipe_sequence(
    segment_count: int,
    eligible: Mapping[str, list[WorkflowCapability]],
    *,
    use_reference_bundle: bool = False,
) -> list[str]:
    """Build a deterministic sequence for a publishable story assembly.

    The production route uses a stable editorial rhythm. I2V carries the
    actual tail-to-next-segment handoff; FL2V is reserved for deliberate state
    transitions and Ref2VA is used only for the opening identity lock when the
    caller supplied approved references.
    """

    count = max(1, int(segment_count))

    def supported(name: str) -> bool:
        return bool(eligible.get(name))

    if not supported("anchor_first"):
        raise ValueError("Production long-video route requires an H3 I2V workflow with first-frame conditioning")

    transition_indices: set[int] = set()
    if count >= 3 and supported("anchor_first_last"):
        transition_indices.update({max(1, count // 3), max(1, (2 * count) // 3), count - 1})

    sequence: list[str] = []
    for index in range(count):
        if index == 0 and use_reference_bundle and supported("reference_bundle"):
            sequence.append("reference_bundle")
        elif index in transition_indices:
            sequence.append("anchor_first_last")
        else:
            sequence.append("anchor_first")
    return sequence
