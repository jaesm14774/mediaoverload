import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.llm_engine import LLMPromptEngine
from provider_fakes import ScriptedLLMManager
from agentic.runtime.prompting import build_game_sprite_reference_context, normalize_dynamic_sprite_beats
from agentic.runtime.planner import TaskPlanner
from agentic.app.generation_routing import _load_global_routing_config, _route_generation_from_character_config
from agentic.app.character_payload import build_goal_payload_from_character_config
from agentic.app.character_config import load_character_config
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.tools.sprite_adapter import SpriteAdapter
from character_workflow_helpers import make_character_workflow_request


def test_user_given_sprite_prompt_when_planned_then_direct_prompt_and_render_metadata_are_returned() -> None:
    """User Given a game-sprite prompt When its render plan is prepared Then the prompt passes through with atlas metadata."""

    goal = GoalRequest(
        prompt="a jelly star folds into a paper plane and circles back",
        media_type="game_sprite",
        style="bright toy game art",
        constraints={"character": "jelly star"},
    )

    plan = LLMPromptEngine(mode="template").build_dynamic_sprite_motion_plan(goal)

    assert plan["layout_mode"] == "single_motion"
    assert plan["motion_plan_mode"] == "direct_prompt"
    assert plan["grid"] == {"rows": 4, "columns": 4}
    assert len(plan["frame_map"]) == 16
    assert plan["beats"][0]["action"] == goal.prompt
    assert plan["video_duration_seconds"] > 0
    assert "chroma_color" not in plan
    assert "background_color" not in plan
    assert "chroma_threshold" not in plan
    assert "chroma-key" not in plan["image_prompt"].casefold()
    assert "chroma-key" not in plan["video_prompt"].casefold()
    assert isinstance(plan["image_prompt"], str) and plan["image_prompt"].strip()
    assert isinstance(plan["video_prompt"], str) and plan["video_prompt"].strip()


def test_user_given_the_game_sprite_route_when_its_context_is_built_then_it_carries_the_pixel_art_contract_and_no_list_of_moves() -> None:
    """User Given the game-sprite route When its context is built Then it carries the pixel-art look and no fixed list of moves to copy."""

    repo_root = Path(__file__).resolve().parents[2]
    strategy_context = _load_global_routing_config(repo_root)["strategy_contexts"]["game_sprite"]
    goal = GoalRequest(
        prompt="a jelly star folds into a paper plane and circles back",
        media_type="game_sprite",
        constraints={"strategy_context": strategy_context},
    )

    context = build_game_sprite_reference_context(goal)

    assert "crisp 2d pixel-art game asset" in context["visual_contract"].casefold()
    assert "creative_inspiration" not in strategy_context
    assert set(context) == {"pack_version", "visual_contract"}


def test_user_given_a_news_happening_when_sprite_prompts_are_prepared_then_the_model_is_asked_to_act_it_out_not_to_fight() -> None:
    """User Given a run whose news happening is mapped to a scene When sprite prompts are prepared Then the model is asked to act out that happening as an everyday action and is told it is never an attack, with no list of game moves."""

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
                "news_source_anchors": [
                    {"source_excerpt": "x", "visual_mapping": "the jelly star hugs an empty pot"}
                ],
                "strategy_context": _load_global_routing_config(repo_root)["strategy_contexts"]["game_sprite"],
            },
        )
    )

    prompt = str(manager.text_model.calls[0]["messages"][1]["content"])
    system = str(manager.text_model.calls[0]["messages"][0]["content"])
    assert "the jelly star hugs an empty pot" in prompt
    assert "never an attack" in system
    assert "inspiration notes" not in prompt.casefold()
    assert "chroma-key" not in prompt.casefold()
    assert "atlas" not in prompt.casefold()
    assert "-second clip" not in prompt.casefold()
    assert "do not assign or register abilities" not in prompt
    assert "action_reference_pack" in plan
    assert "chroma_color" not in plan
    assert "chroma_threshold" not in plan
    assert "action_references" not in plan
    assert "idle, walk, jump, attack" not in prompt.lower()


def test_user_given_a_crowded_news_scene_when_sprite_prompts_are_requested_then_the_model_is_told_to_draw_one_character_in_one_frame() -> None:
    """User Given a news scene with several props When the sprite prompts are requested Then the model must reduce it to one character, one motif and one still frame."""

    class FakeTextModel:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []

        def chat_completion(self, messages: list[dict], **_kwargs: object) -> str:
            self.calls.append(messages)
            return json.dumps({"image_prompt": "Kirby stands ready", "video_prompt": "Kirby leaps."})

    class FakeManager:
        def __init__(self) -> None:
            self.text_model = FakeTextModel()

    manager = FakeManager()
    LLMPromptEngine(mode="llm", manager=manager).build_dynamic_sprite_motion_plan(
        GoalRequest(
            prompt="Kirby casts a ballot beside a flag, a gift box and a shaken building",
            media_type="game_sprite",
            style="watercolor",
            constraints={"character": "Kirby"},
        )
    )

    system_prompt = str(manager.text_model.calls[0][0]["content"]).casefold()
    assert "exactly one character" in system_prompt
    assert "one still frame" in system_prompt
    assert "storyboard" in system_prompt  # named so that it is forbidden
    assert "reduced to that one action and at most one prop" in system_prompt


def test_user_given_a_model_keyframe_prompt_when_compiled_then_single_subject_and_game_art_lead_the_prompt() -> None:
    """User Given a model keyframe prompt that mixes in a style When it is compiled Then the one-character game-art contract precedes the creative text so the style cannot turn it into a sheet or a painting."""

    from agentic.runtime.prompting import compile_dynamic_sprite_image_prompt

    compiled = compile_dynamic_sprite_image_prompt(
        "Kirby in watercolor beside a ballot box", "Crisp 2D pixel-art game asset"
    ).casefold()

    assert "exactly one character" in compiled
    assert "sprite sheet" in compiled and "never" in compiled
    creative = compiled.index("kirby in watercolor")
    assert compiled.index("exactly one character") < creative
    assert compiled.index("crisp 2d pixel-art game asset") < creative


def test_user_given_a_keyframe_prompt_when_compiled_then_the_backdrop_is_a_flat_green_the_pipeline_can_cut_out() -> None:
    """User Given any creative keyframe text When it is compiled Then the leading and closing clauses both demand one character on a flat pure-green backdrop so a scenic idea cannot survive."""

    from agentic.runtime.prompting import compile_dynamic_sprite_image_prompt

    compiled = compile_dynamic_sprite_image_prompt(
        "Kirby stands at the torn edge of a photograph of a building", "Crisp 2D pixel-art game asset"
    ).casefold()

    creative = compiled.index("kirby stands at the torn edge")
    assert "flat" in compiled[:creative] and "green" in compiled[:creative]
    assert "two thirds of the frame height" in compiled[:creative]
    assert "green" in compiled[creative + 30 :]


def test_user_given_a_video_prompt_when_compiled_then_the_green_backdrop_must_stay_unchanged() -> None:
    """User Given a motion prompt When it is compiled Then the clip is told to keep the green backdrop solid so every frame can be cut out."""

    from agentic.runtime.prompting import compile_dynamic_sprite_video_prompt

    compiled = compile_dynamic_sprite_video_prompt("Kirby spins", "Crisp 2D pixel-art game asset").casefold()

    assert "green" in compiled and "unchanged" in compiled
    assert "same position" in compiled and "face stays visible" in compiled
    assert "explosions" in compiled  # named so that they are forbidden


def test_user_given_a_photographic_news_run_when_sprite_prompts_are_requested_then_the_model_sees_the_headline_motif_but_no_photo_style() -> None:
    """User Given a news run whose random style is photographic When sprite prompts are requested Then the model receives the headline and subject but no photographic style direction."""

    class FakeTextModel:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []

        def chat_completion(self, messages: list[dict], **_kwargs: object) -> str:
            self.calls.append(messages)
            return json.dumps({"image_prompt": "Kirby stands ready", "video_prompt": "Kirby leaps."})

    class FakeManager:
        def __init__(self) -> None:
            self.text_model = FakeTextModel()

    manager = FakeManager()
    LLMPromptEngine(mode="llm", manager=manager).build_dynamic_sprite_motion_plan(
        GoalRequest(
            prompt="A real photograph of a modern building with a tiny drawn character at its torn edge",
            media_type="game_sprite",
            style="photograph with a tiny drawn character",
            constraints={
                "character": "Kirby",
                "news_context": {"title": "Typhoon turns away from the coast", "keyword": "typhoon"},
            },
        )
    )

    user_prompt = str(manager.text_model.calls[0][1]["content"]).casefold()
    assert "typhoon turns away from the coast" in user_prompt
    assert "kirby" in user_prompt
    assert "photograph" not in user_prompt


def test_template_motion_plan_includes_selected_subject_as_reference_context() -> None:
    """User Given a selected subject When a template plan is built Then it is included as reference context."""

    plan = LLMPromptEngine(mode="template").build_dynamic_sprite_motion_plan(
        GoalRequest(
            prompt="Kirby folds a glowing leaf and returns",
            media_type="game_sprite",
            style="bright toy game art",
            constraints={"character": "Waddle Dee"},
        )
    )

    assert "waddle dee" in plan["image_prompt"].lower()
    assert "kirby folds a glowing leaf and returns" in plan["video_prompt"].lower()


def test_dynamic_sprite_normalizes_second_based_llm_timeline_as_one_sequence() -> None:
    """User Given second-based motion beats When normalized Then one contiguous unit interval is returned."""

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
    """User Given collapsed later beats When normalized Then the fallback timeline stays contiguous and bounded."""

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

    assert len(beats) == len(fallback)
    assert beats[0]["time_start"] == 0.0
    assert beats[-1]["time_end"] == 1.0
    assert all(0.0 <= beat["time_start"] < beat["time_end"] <= 1.0 for beat in beats)
    assert all(beats[index]["time_start"] >= beats[index - 1]["time_end"] for index in range(1, len(beats)))


def test_user_given_a_model_motion_prompt_when_planned_then_it_is_not_rewritten_by_a_creative_gate() -> None:
    """User Given a model motion prompt When the plan is normalized Then the motion prompt is preserved."""
    manager = ScriptedLLMManager([{'motion_name': 'unsafe_generated_motion', 'layout_mode': 'single_motion', 'image_prompt': 'one clean jelly star', 'video_prompt': 'The body compresses its round body into a tight spring coil.', 'negative_prompt': 'no clutter', 'animation_kind': 'one_shot', 'fps': 12, 'video_duration_seconds': 5, 'frame_map': [{'index': index, 'beat': f'beat_{index}', 'description': 'motion'} for index in range(16)]}])
    plan = LLMPromptEngine(mode='llm', manager=manager).build_dynamic_sprite_motion_plan(GoalRequest(prompt='a jelly star compresses into a spring and returns', media_type='game_sprite', style='bright toy game art', constraints={'character': 'jelly star'}))
    assert 'compresses its round body into a tight spring coil' in plan['video_prompt'].lower()
    assert plan['negative_prompt'] == ''


def test_user_given_a_game_sprite_prompt_when_no_model_is_available_then_fallback_keeps_the_prompt() -> None:
    """User Given a game-sprite prompt When no model is available Then the fallback keeps the prompt."""

    plan = LLMPromptEngine(mode="template").build_dynamic_sprite_motion_plan(
        GoalRequest(
            prompt="a jelly star compresses into a spring and returns",
            media_type="game_sprite",
            style="bright toy game art",
            constraints={"character": "jelly star"},
        )
    )

    assert "compresses into a spring" in plan["video_prompt"].lower()
    assert "a jelly star compresses into a spring and returns" in plan["image_prompt"].lower()


def test_game_sprite_plan_has_no_human_review_or_fixed_action_nodes() -> None:
    """User Given a game-sprite goal When the route is planned Then the additive route exposes its required lifecycle."""

    from agentic.assets.registry import AssetRegistry

    planner = TaskPlanner(AssetRegistry(Path(__file__).resolve().parents[2]))
    goal = planner.create_goal(
        "a jelly star folds into a paper plane and circles back",
        "game_sprite",
        8,
        "bright toy game art",
        {"character": "jelly star"},
    )

    plan = planner.build_plan(goal)

    assert plan.workflow_name == "game_sprite_v1"
    assert plan.metadata["motion_source"] == "llm_dynamic"
    assert plan.metadata["human_review"] is False
    node_ids = {node.node_id for node in plan.nodes}
    assert {
        "sprite-motion-plan",
        "sprite-image-assets",
        "sprite-master-image",
        "sprite-video-assets",
        "sprite-motion-video",
        "sprite-package",
    } <= node_ids
    video_node = next(node for node in plan.nodes if node.node_id == "sprite-motion-video")
    # A 16-frame atlas does not need a ten-second clip or a full-size canvas.
    assert video_node.inputs["length"] < 240
    assert max(video_node.inputs["width"], video_node.inputs["height"]) <= 640
    assert not {"sprite-motion-editorial-pacing", "sprite-motion-nvidia-vfx"} & node_ids
    package_node = next(node for node in plan.nodes if node.node_id == "sprite-package")
    assert package_node.inputs["video_node"] == "sprite-motion-video"


def test_game_sprite_h3_render_leaves_the_distilled_schedule_to_the_route(tmp_path: Path) -> None:
    """User Given a sprite video node When it renders Then the configured sampling contract reaches the media tool."""

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
                "workflow_name": "minimax_h3_i2va",
                "image_path": str(tmp_path / "master.png"),
                "width": 608,
                "height": 352,
                "length": 124,
            },
        ),
        state=RunState(goal={}, metadata={}, node_outputs={}),
    )

    result = skills.animate_image(context)

    assert result.status == "success"
    assert len(calls) == 1
    tool_name, payload = calls[0]
    assert tool_name == "comfy.render_h3"
    assert payload["length"] == 124
    assert "steps" not in payload and "model_profile" not in payload


def test_kirby_yaml_exposes_game_sprite_as_an_explicit_additive_route() -> None:
    """User Given the Kirby configuration When game-sprite generation is requested Then its ComfyUI workflow and inspiration context reach the goal."""

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
    assert payload["duration_seconds"] > 0
    assert payload["constraints"]["sprite_video_length"] == round(payload["duration_seconds"] * 24)
    assert "sprite_chroma_color" not in payload["constraints"]
    assert "sprite_chroma_threshold" not in payload["constraints"]
    assert payload["constraints"]["image_workflow_name"] == "krea2_turbo"
    assert payload["constraints"]["video_workflow_name"] == "minimax_h3_i2va"
    assert payload["constraints"]["sprite_source_width"] == payload["constraints"]["canvas_width"]
    assert payload["constraints"]["sprite_source_height"] == payload["constraints"]["canvas_height"]
    video_width = payload["constraints"]["sprite_video_width"]
    video_height = payload["constraints"]["sprite_video_height"]
    canvas_width = payload["constraints"]["canvas_width"]
    canvas_height = payload["constraints"]["canvas_height"]
    assert min(video_width, video_height) >= 256
    assert video_width % 32 == 0 and video_height % 32 == 0
    assert video_width * video_height < canvas_width * canvas_height
    # The clip keeps the keyframe's shape; H3 stretches a mismatched first frame.
    assert abs(video_width / video_height - canvas_width / canvas_height) < 0.1
    assert payload["constraints"]["strategy_context"]["reference_pack_version"] == "game_sprite_action_references_v1"
    assert "creative_inspiration" not in payload["constraints"]["strategy_context"]

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
    )

    assert result["generation_type"] == "game_sprite"
    assert result["selection_source"] == "weighted_random"


def _green_backdrop_frames(directory: Path, backdrops: list[tuple[int, int, int]]) -> list[str]:
    directory.mkdir()
    paths = []
    for index, backdrop in enumerate(backdrops):
        image = Image.new("RGB", (128, 96), backdrop)
        draw = ImageDraw.Draw(image)
        left = 30 + index * 2
        draw.ellipse((left, 24, left + 44, 70), fill=(250, 150, 190), outline=(120, 30, 70), width=2)
        draw.ellipse((left + 12, 38, left + 18, 48), fill=(20, 20, 60))
        path = directory / f"frame-{index + 1:02d}.png"
        image.save(path)
        paths.append(str(path))
    return paths


def test_user_given_a_character_on_a_flat_green_backdrop_when_packaged_then_the_sprite_is_a_transparent_pixel_art_atlas(
    tmp_path: Path,
) -> None:
    """User Given sixteen frames of one character on a flat green backdrop When packaging runs Then the atlas shows only the character on transparency with a small palette and the preview is enlarged without blur."""
    paths = _green_backdrop_frames(tmp_path / "source", [(0, 255, 0)] * 16)

    result = SpriteAdapter().build_from_frames(
        paths, output_dir=str(tmp_path / "sprite"), cell_width=32, cell_height=32, fps=12, loop=True
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    atlas = Image.open(result["atlas_path"])
    gif = Image.open(result["gif_path"])
    first_cell = atlas.crop((0, 0, 32, 32))
    opaque_colors = {
        pixel[:3] for pixel in first_cell.getdata() if pixel[3] == 255
    } | {pixel[:3] for pixel in atlas.getdata() if pixel[3] == 255}

    assert manifest["grid"] == {"rows": 4, "columns": 4}
    assert manifest["frame_count"] == 16
    assert manifest["alpha"] == "binary"
    assert manifest["qa"]["passed"] is True
    assert atlas.size == (128, 128)
    assert first_cell.getpixel((0, 0))[3] == 0
    assert first_cell.getchannel("A").getbbox() is not None
    assert not any(color[1] > 200 and color[0] < 80 for color in opaque_colors)
    assert len(opaque_colors) <= 32
    assert gif.n_frames == 16
    assert gif.info["loop"] == 0
    assert gif.size[0] > 32 and gif.size[1] > 32
    assert gif.convert("RGBA").getpixel((0, 0))[3] == 0


def test_user_given_a_backdrop_whose_green_drifts_between_frames_when_packaged_then_every_frame_is_cut_out(
    tmp_path: Path,
) -> None:
    """User Given a video whose green backdrop shifts shade from frame to frame When packaging runs Then no frame keeps a backdrop."""
    shades = [(32, 180, 48), (60, 225, 90), (0, 255, 0), (20, 200, 30)] * 4
    paths = _green_backdrop_frames(tmp_path / "drift", shades)

    result = SpriteAdapter().build_from_frames(
        paths, output_dir=str(tmp_path / "drift-output"), cell_width=64, cell_height=64
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    for record in manifest["frames"]:
        frame = Image.open(record["path"]).convert("RGBA")
        assert frame.getpixel((1, 1))[3] == 0
        assert frame.getpixel((32, 32))[3] == 255
    assert all(check for check in result["qa"]["checks"].values())


def test_user_given_one_screen_filling_burst_when_packaged_then_the_character_stays_large_in_the_other_frames(
    tmp_path: Path,
) -> None:
    """User Given a skill where one frame flashes a screen-filling burst When packaging runs Then the character keeps a readable size in the ordinary frames instead of shrinking to fit the burst."""
    directory = tmp_path / "burst"
    directory.mkdir()
    paths = []
    for index in range(16):
        image = Image.new("RGB", (192, 192), (0, 255, 0))
        draw = ImageDraw.Draw(image)
        if index == 9:
            draw.ellipse((4, 4, 188, 188), fill=(255, 140, 40))
        draw.ellipse((78, 80, 118, 120), fill=(250, 150, 190), outline=(120, 30, 70), width=2)
        path = directory / f"frame-{index + 1:02d}.png"
        image.save(path)
        paths.append(str(path))

    result = SpriteAdapter().build_from_frames(
        paths, output_dir=str(tmp_path / "burst-output"), cell_width=128, cell_height=128
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    ordinary = Image.open(manifest["frames"][0]["path"]).convert("RGBA")
    bbox = ordinary.getchannel("A").getbbox()
    assert bbox is not None
    assert (bbox[2] - bbox[0]) >= 0.5 * 128


def test_user_given_a_frame_without_any_green_backdrop_when_packaged_then_it_stays_visible(
    tmp_path: Path,
) -> None:
    """User Given one solid red frame among green-backdrop frames When packaging runs Then the red frame is not erased."""
    source_dir = tmp_path / "varied"
    paths = _green_backdrop_frames(source_dir, [(0, 255, 0)] * 16)
    Image.new("RGB", (128, 96), (224, 14, 0)).save(paths[4])

    result = SpriteAdapter().build_from_frames(
        paths, output_dir=str(tmp_path / "varied-output"), cell_width=32, cell_height=32
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    fifth_frame = Image.open(manifest["frames"][4]["path"]).convert("RGBA")
    assert result["qa"]["passed"] is True
    assert fifth_frame.getpixel((16, 16))[3] == 255
    assert len(manifest["frames"]) == 16


def test_user_given_sprite_touches_a_non_floor_edge_when_packaged_then_discord_can_review_it(tmp_path: Path) -> None:
    """User Given a generated frame touches an edge When it is packaged Then it remains available for Discord review."""

    source_dir = tmp_path / "clipped-subject"
    source_dir.mkdir()
    source_paths = []
    for index in range(16):
        image = Image.new("RGB", (128, 96), (255, 0, 255))
        draw = ImageDraw.Draw(image)
        left = 0 if index == 7 else 40
        draw.rectangle((left, 24, left + 28, 60), fill=(30, 180, 255))
        path = source_dir / f"frame-{index:02d}.png"
        image.save(path)
        source_paths.append(str(path))

    result = SpriteAdapter().build_from_frames(
        source_paths,
        output_dir=str(tmp_path / "clipped-output"),
        cell_width=64,
        cell_height=64,
    )

    assert result["qa"]["passed"] is True
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    packaged_frame = Image.open(manifest["frames"][7]["path"]).convert("RGBA")
    assert packaged_frame.getchannel("A").getbbox() is not None


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


def test_user_given_a_packaged_sprite_when_it_goes_to_review_then_the_clip_shows_the_cut_out_sprite_on_a_plain_backdrop_not_the_green_screen(
    tmp_path: Path,
) -> None:
    """User Given a sprite rendered on a green screen When it is packaged and its media is collected for review Then the only clip offered is a looping video of the cut-out sprite on a plain backdrop, with no green in it."""
    from agentic.app.character_publishing import collect_media_paths_from_run_result
    from agentic.tools.ffmpeg_adapter import FFmpegAdapter

    paths = _green_backdrop_frames(tmp_path / "source", [(0, 255, 0)] * 16)
    result = SpriteAdapter().build_from_frames(
        paths, output_dir=str(tmp_path / "sprite"), cell_width=32, cell_height=32, fps=12, loop=True
    )
    showcase = Path(result["showcase_path"])
    raw_clip = tmp_path / "videos" / "h3_i2va_01.mp4"
    raw_clip.parent.mkdir()
    raw_clip.write_bytes(showcase.read_bytes())

    offered = collect_media_paths_from_run_result(
        {
            "state": {
                "node_outputs": {
                    "sprite-motion-video": {"saved_files": [str(raw_clip)], "video_path": str(raw_clip)},
                    "sprite-package": {**result, "final_video_path": str(showcase), "source_video_path": str(raw_clip)},
                }
            }
        }
    )

    assert offered == [str(showcase)]
    probe = FFmpegAdapter().probe_media(str(showcase))
    assert float(probe["duration"]) >= 4
    frame = tmp_path / "frame.png"
    FFmpegAdapter().extract_frames_at(str(showcase), [str(frame)], [1.0])
    pixels = list(Image.open(frame).convert("RGB").getdata())
    assert not any(green > 200 and red < 80 and blue < 80 for red, green, blue in pixels)
    assert Image.open(frame).convert("RGB").getpixel((2, 2))[0] > 200
