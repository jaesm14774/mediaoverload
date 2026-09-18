import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, SkillContext
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.prompting import build_game_sprite_reference_context, normalize_dynamic_sprite_beats
from agentic.runtime.planner import TaskPlanner
from agentic.app.character_workflow import (
    _load_global_routing_config,
    _route_generation_from_character_config,
    build_goal_payload_from_character_config,
    load_character_config,
)
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.tools.sprite_adapter import SpriteAdapter
from character_workflow_helpers import make_character_workflow_request


def test_template_motion_plan_is_open_ended_and_choreographed_on_technical_grid() -> None:
    goal = GoalRequest(
        prompt="a jelly star folds into a paper plane and circles back",
        media_type="game_sprite",
        style="bright toy game art",
        constraints={"character": "jelly star"},
    )

    plan = LLMPromptEngine(mode="template").build_dynamic_sprite_motion_plan(goal)

    assert plan["layout_mode"] == "single_motion"
    assert plan["motion_plan_mode"] == "choreographed_action_graph"
    assert plan["grid"] == {"rows": 4, "columns": 4}
    assert len(plan["frame_map"]) == 16
    assert 4 <= len(plan["beats"]) <= 8
    assert plan["video_duration_seconds"] == 8.0
    assert plan["chroma_color"] == plan["background_color"]
    assert plan["chroma_color"] != "#ff00ff"
    assert all({"cause", "transition", "body_change", "spatial_change"} <= set(beat) for beat in plan["beats"])
    assert "paper plane" in plan["video_prompt"]
    assert "attack" not in plan["video_prompt"].lower()
    assert "no floor" in plan["image_prompt"].lower()
    assert "ground plane" in plan["video_prompt"].lower()
    assert plan["identity_repair_applied"] == "none"


def test_game_sprite_reference_pack_is_ephemeral_inspiration_not_a_moveset() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    goal = GoalRequest(
        prompt="a jelly star folds into a paper plane and circles back",
        media_type="game_sprite",
        constraints={
            "sprite_reference_seed": "test-seed",
            "strategy_context": _load_global_routing_config(repo_root)["strategy_contexts"]["game_sprite"],
        },
    )

    context = build_game_sprite_reference_context(goal, limit=5)

    assert context["pack_version"] == "game_sprite_action_references_v1"
    assert "crisp 2d pixel-art game asset" in context["visual_contract"].casefold()
    assert len(context["references"]) == 5
    assert "optional inspiration only" in context["prompt_text"]
    assert "do not assign or register abilities" in context["prompt_text"]
    assert len({item["id"] for item in context["references"]}) == 5


def test_llm_sprite_prompt_receives_reference_notes_without_character_moveset() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    class FakeTextModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def chat_completion(self, messages: list[dict], **_kwargs: object) -> str:
            self.calls.append({"messages": messages})
            beats = [
                {
                    "time_start": index / 4,
                    "time_end": (index + 1) / 4,
                    "purpose": f"beat_{index + 1}",
                    "action": f"the subject continues beat {index + 1}",
                    "body_change": "the silhouette changes with the force",
                    "spatial_change": "the subject follows one readable path",
                    "cause": "the previous beat supplies momentum",
                    "transition": "the motion continues directly",
                }
                for index in range(4)
            ]
            return json.dumps(
                {
                    "motion_name": "reference_inspired_motion",
                    "layout_mode": "single_motion",
                    "motion_plan_mode": "choreographed_action_graph",
                    "creative_twist": "a redirected physical consequence",
                    "background_color": "cyan",
                    "image_prompt": "one clean jelly star",
                    "video_prompt": "The jelly star follows one continuous physical action.",
                    "negative_prompt": "clutter",
                    "animation_kind": "one_shot",
                    "fps": 12,
                    "video_duration_seconds": 8,
                    "beats": beats,
                }
            )

    class FakeManager:
        def __init__(self) -> None:
            self.text_model = FakeTextModel()

    manager = FakeManager()
    plan = LLMPromptEngine(mode="llm", manager=manager).build_dynamic_sprite_motion_plan(
        GoalRequest(
            prompt="a jelly star folds into a paper plane and circles back",
            media_type="game_sprite",
            style="bright toy game art",
            constraints={
                "character": "jelly star",
                "sprite_reference_seed": "test-seed",
                "strategy_context": _load_global_routing_config(repo_root)["strategy_contexts"]["game_sprite"],
            },
        )
    )

    prompt = str(manager.text_model.calls[0]["messages"][1]["content"])
    assert "Game action references for optional inspiration only" in prompt
    assert "do not assign or register abilities" in prompt
    assert "action_reference_pack" in plan
    assert len(plan["action_references"]) == 8
    assert "idle, walk, jump, attack" not in prompt.lower()


def test_template_motion_plan_uses_resolved_subject_as_identity_anchor() -> None:
    plan = LLMPromptEngine(mode="template").build_dynamic_sprite_motion_plan(
        GoalRequest(
            prompt="Kirby folds a glowing leaf and returns",
            media_type="game_sprite",
            style="bright toy game art",
            constraints={"character": "Waddle Dee"},
        )
    )

    assert "one single Waddle Dee" in plan["image_prompt"]
    assert "same Waddle Dee" in plan["video_prompt"]


def test_dynamic_sprite_normalizes_second_based_llm_timeline_as_one_sequence() -> None:
    fallback = [
        {
            "time_start": 0.0,
            "time_end": 1.0,
            "purpose": "fallback",
            "action": "fallback action",
            "body_change": "fallback body change",
            "spatial_change": "fallback spatial change",
            "cause": "fallback cause",
            "transition": "fallback transition",
        }
    ] * 4
    beats = normalize_dynamic_sprite_beats(
        [
            {
                "time_start": index,
                "time_end": index + 1,
                "purpose": f"beat_{index}",
                "action": "continue the action",
                "body_change": "change pose",
                "spatial_change": "move forward",
                "cause": "previous beat",
                "transition": "next beat",
            }
            for index in range(4)
        ],
        fallback,
    )

    assert beats[0]["time_start"] == 0.0
    assert beats[0]["time_end"] == 0.25
    assert beats[-1]["time_end"] == 1.0
    assert all(beats[index]["time_start"] >= beats[index - 1]["time_end"] for index in range(1, 4))


def test_dynamic_sprite_recovers_when_schema_clamps_later_beats_to_one() -> None:
    fallback = [
        {
            "time_start": index / 6,
            "time_end": (index + 1) / 6,
            "purpose": f"fallback_{index}",
            "action": "fallback action",
            "body_change": "fallback body change",
            "spatial_change": "fallback spatial change",
            "cause": "fallback cause",
            "transition": "fallback transition",
        }
        for index in range(6)
    ]
    beats = normalize_dynamic_sprite_beats(
        [
            {
                "time_start": 0.0 if index == 0 else 1.0,
                "time_end": 0.5 if index == 0 else 1.0,
                "purpose": f"beat_{index}",
                "action": "continue the action",
                "body_change": "change pose",
                "spatial_change": "move forward",
                "cause": "previous beat",
                "transition": "next beat",
            }
            for index in range(6)
        ],
        fallback,
    )

    assert [(beat["time_start"], beat["time_end"]) for beat in beats] == [
        (0.0, 0.1667),
        (0.1667, 0.3333),
        (0.3333, 0.5),
        (0.5, 0.6667),
        (0.6667, 0.8333),
        (0.8333, 1.0),
    ]


def test_dynamic_sprite_identity_risk_catches_model_rephrasing() -> None:
    assert LLMPromptEngine._dynamic_sprite_identity_risk(
        "The body compresses vertically and twists into a tight, springy spiral."
    )
    assert LLMPromptEngine._dynamic_sprite_identity_risk(
        "The character compresses its round body, then the body coils tightly around a star."
    )
    assert not LLMPromptEngine._dynamic_sprite_identity_risk(
        "The character tilts, hops twice, and circles one separate prop before settling."
    )
    assert not LLMPromptEngine._dynamic_sprite_identity_risk(
        "The character follows a spiral path around a separate glowing thread while keeping its original silhouette."
    )


def test_dynamic_sprite_action_diversity_risk_only_catches_collapsed_beats() -> None:
    repeated = [{"action": "continue the same readable physical action"} for _ in range(5)]
    varied = [{"action": f"visible action {index}"} for index in range(5)]

    assert LLMPromptEngine._dynamic_sprite_action_diversity_risk(repeated)
    assert not LLMPromptEngine._dynamic_sprite_action_diversity_risk(varied)


def test_dynamic_sprite_identity_repair_fails_closed_without_fixed_actions() -> None:
    class StubEngine(LLMPromptEngine):
        def _require_manager(self):
            return object()

        def _chat_json_with_recorder(self, _manager, _system, _prompt, *, schema_name, **_kwargs):
            if schema_name == "dynamic_game_sprite_motion":
                return {
                    "motion_name": "unsafe_generated_motion",
                    "layout_mode": "single_motion",
                    "image_prompt": "one clean jelly star",
                    "video_prompt": "The body compresses its round body into a tight spring coil.",
                    "negative_prompt": "no clutter",
                    "animation_kind": "one_shot",
                    "fps": 12,
                    "video_duration_seconds": 5,
                    "frame_map": [
                        {"index": index, "beat": f"beat_{index}", "description": "motion"}
                        for index in range(16)
                    ],
                }
            raise RuntimeError("identity repair unavailable")

        def _mark_llm_payload(self, payload):
            return payload

    plan = StubEngine(mode="llm").build_dynamic_sprite_motion_plan(
        GoalRequest(
            prompt="a jelly star compresses into a spring and returns",
            media_type="game_sprite",
            style="bright toy game art",
            constraints={"character": "jelly star"},
        )
    )

    assert plan["identity_repair_applied"] == "contract_repair"
    assert "compresses into a spring" not in plan["video_prompt"].lower()
    assert "idle" not in plan["video_prompt"].lower()
    assert "attack" not in plan["video_prompt"].lower()


def test_template_fallback_sanitizes_risky_identity_prompt() -> None:
    plan = LLMPromptEngine(mode="template").build_dynamic_sprite_motion_plan(
        GoalRequest(
            prompt="a jelly star compresses into a spring and returns",
            media_type="game_sprite",
            style="bright toy game art",
            constraints={"character": "jelly star"},
        )
    )

    assert plan["identity_repair_applied"] == "contract_repair"
    assert "compresses into a spring" not in plan["video_prompt"].lower()
    assert "one single jelly star" in plan["image_prompt"]


def test_game_sprite_plan_has_no_human_review_or_fixed_action_nodes() -> None:
    from agentic.assets.registry import AssetRegistry

    planner = TaskPlanner(AssetRegistry(Path(__file__).resolve().parents[2]))
    goal = planner.create_goal(
        "a jelly star folds into a paper plane and circles back",
        "game_sprite",
        8,
        "bright toy game art",
        False,
        {"character": "jelly star"},
    )

    plan = planner.build_plan(goal)

    assert plan.workflow_name == "game_sprite_v1"
    assert plan.metadata["motion_source"] == "llm_dynamic"
    assert plan.metadata["human_review"] is False
    assert [node.node_id for node in plan.nodes] == [
        "sprite-motion-plan",
        "sprite-image-assets",
        "sprite-master-image",
        "sprite-video-assets",
        "sprite-motion-video",
        "sprite-package",
    ]
    video_node = next(node for node in plan.nodes if node.node_id == "sprite-motion-video")
    assert video_node.inputs["length"] == 192


def test_game_sprite_h3_render_forwards_sampling_steps(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeTools:
        @staticmethod
        def call(name: str, payload: dict[str, object]) -> dict[str, object]:
            calls.append((name, payload))
            return {"saved_files": []}

    skills = AgentMediaSkills(FakeTools(), tmp_path / "runs")
    context = SkillContext(
        plan=ExecutionPlan(
            goal=GoalRequest(prompt="a generated sprite motion", media_type="game_sprite"),
            workflow_name="game_sprite_v1",
            nodes=[],
        ),
        node=ExecutionNode(
            node_id="sprite-motion-video",
            skill_name="media.image.animate",
            inputs={
                "workflow_name": "minimax_h3_lowvram_i2v",
                "image_path": str(tmp_path / "master.png"),
                "width": 608,
                "height": 352,
                "length": 124,
                "steps": 16,
            },
        ),
        state={},
    )

    result = skills.animate_image(context)

    assert result.status == "success"
    assert calls[0][0] == "comfy.workflow.image_to_video"
    assert calls[0][1]["length"] == 124
    assert calls[0][1]["steps"] == 16


def test_kirby_yaml_exposes_game_sprite_as_an_explicit_additive_route() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    kirby_config = repo_root / "configs" / "characters" / "kirby.yaml"
    payload = build_goal_payload_from_character_config(
        make_character_workflow_request(
            repo_root,
            kirby_config,
            prompt="Kirby folds a glowing leaf into a tiny glider and returns",
            preferred_generation_type="game_sprite",
            publish_after_generate=False,
        )
    )

    assert payload["source_generation_type"] == "game_sprite"
    assert payload["media_type"] == "game_sprite"
    assert payload["duration_seconds"] == 8
    assert payload["constraints"]["sprite_video_length"] == 192
    assert payload["constraints"]["sprite_chroma_color"] == "random"
    assert payload["constraints"]["image_workflow_name"] == "krea2_turbo"
    assert payload["constraints"]["video_workflow_name"] == "minimax_h3_lowvram_i2v"
    assert payload["constraints"]["sprite_source_width"] == 1024
    assert payload["constraints"]["sprite_source_height"] == 576
    assert payload["constraints"]["sprite_h3_model_profile"] == "q2"
    assert payload["constraints"]["strategy_context"]["reference_pack_version"] == "game_sprite_action_references_v1"
    assert len(payload["constraints"]["strategy_context"]["creative_inspiration"]) == 25

    native_payload = build_goal_payload_from_character_config(
        make_character_workflow_request(
            repo_root,
            kirby_config,
            prompt="Kirby enters a fifteen second story",
            preferred_generation_type="native_h3_story",
            publish_after_generate=False,
        )
    )
    assert native_payload["constraints"]["native_h3_recipe"]["require_human_review"] is True


def test_kirby_yaml_can_weight_game_sprite_for_random_selection(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    kirby_config = repo_root / "configs" / "characters" / "kirby.yaml"
    config = load_character_config(kirby_config)

    assert config["generation"]["generation_type_weights"]["game_sprite"] == 3

    config["generation"]["generation_type_weights"] = {"game_sprite": 1}
    result = _route_generation_from_character_config(
        repo_root,
        config,
        character_name="Kirby",
        style="polished 2D anime",
        prompt="Kirby turns a falling star into a little glider",
        preferred_generation_type=None,
        requested_duration_seconds=None,
        rng=random.Random(0),
        routing_history_path=tmp_path / "routing.json",
    )

    assert result["generation_type"] == "game_sprite"
    assert result["selection_source"] == "weighted_random"


def test_sprite_adapter_builds_transparent_4x4_gif_atlas(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_paths = []
    for index in range(16):
        image = Image.new("RGB", (128, 96), (255, 0, 255))
        draw = ImageDraw.Draw(image)
        left = 18 + index * 4
        draw.ellipse((left, 30, left + 28, 58), fill=(30, 180, 255))
        path = source_dir / f"frame-{index:02d}.png"
        image.save(path)
        source_paths.append(str(path))

    result = SpriteAdapter().build_from_frames(
        source_paths,
        output_dir=str(tmp_path / "sprite"),
        cell_width=32,
        cell_height=32,
        fps=12,
        loop=True,
        key_color="#ff00ff",
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    atlas = Image.open(result["atlas_path"])
    gif = Image.open(result["gif_path"])

    assert manifest["grid"] == {"rows": 4, "columns": 4}
    assert manifest["frame_count"] == 16
    assert manifest["alpha"] == "binary"
    assert manifest["qa"]["passed"] is True
    assert atlas.size == (128, 128)
    assert atlas.mode == "RGBA"
    assert atlas.getpixel((0, 0))[3] == 0
    assert gif.n_frames == 16
    assert gif.info["loop"] == 0
    assert gif.convert("RGBA").getpixel((0, 0))[3] == 0


def test_sprite_adapter_removes_h3_shifted_border_background(tmp_path: Path) -> None:
    source_dir = tmp_path / "shifted-background"
    source_dir.mkdir()
    source_paths = []
    for index in range(16):
        background = (32, 180, 48) if index % 2 else (168, 54, 42)
        image = Image.new("RGB", (128, 96), background)
        draw = ImageDraw.Draw(image)
        left = 24 + index * 3
        draw.ellipse((left, 28, left + 30, 58), fill=(30, 160, 240))
        path = source_dir / f"frame-{index:02d}.png"
        image.save(path)
        source_paths.append(str(path))

    result = SpriteAdapter().build_from_frames(
        source_paths,
        output_dir=str(tmp_path / "shifted-output"),
        cell_width=64,
        cell_height=64,
    )

    atlas = Image.open(result["atlas_path"]).convert("RGBA")
    assert atlas.getpixel((0, 0))[3] == 0
    assert all(check for check in result["qa"]["checks"].values())


def test_sprite_adapter_keeps_technical_frame_contracts(tmp_path: Path) -> None:
    source_dir = tmp_path / "disappearing-subject"
    source_dir.mkdir()
    source_paths = []
    for index in range(16):
        image = Image.new("RGB", (128, 96), (255, 0, 255))
        draw = ImageDraw.Draw(image)
        size = 30 if index != 8 else 3
        draw.ellipse((48, 42, 48 + size, 42 + size), fill=(30, 180, 255))
        path = source_dir / f"frame-{index:02d}.png"
        image.save(path)
        source_paths.append(str(path))

    result = SpriteAdapter().build_from_frames(
        source_paths,
        output_dir=str(tmp_path / "disappearing-output"),
        cell_width=64,
        cell_height=64,
    )
    assert result["qa"]["passed"]
