from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.drama import (
    ContinuityContract,
    DialogueCue,
    DramaPlan,
    DramaPlanError,
    DramaScene,
    SfxCue,
    compile_drama_plan,
)
from agentic.runtime.registry import ToolRegistry
from agentic.skills.editing import EditingSkills


def build_scene(
    scene_id: str,
    *,
    source: str | None = "/tmp/scene.mp4",
    transition: str = "cut",
    duration: float = 2.0,
) -> DramaScene:
    return DramaScene(
        scene_id=scene_id,
        beat="HOOK" if scene_id == "S01" else "PAYOFF",
        duration_seconds=duration,
        objective="protect the prop",
        start_state="the hero is reaching for the prop",
        action_beats=("the prop moves", "the hero reacts"),
        end_state="the hero holds the prop safely",
        next_hook="the next beat raises the consequence",
        cause="the prop slips",
        effect="the hero changes position",
        character_ids=("hero",),
        prop_ids=("prop",),
        location="kitchen",
        selected_source_path=source,
        motion="none",
        transition_to_next=transition,
        transition_duration_seconds=0.2,
        dialogue=(DialogueCue("hero", "I have it", start_seconds=0.3, duration_seconds=0.6),),
        sfx=(SfxCue("impact", start_seconds=1.1, duration_seconds=0.25),),
        continuity=ContinuityContract(
            character_ids=("hero",),
            prop_ids=("prop",),
            location="kitchen",
            style_anchor="warm pastel 2D",
        ),
    )


def build_plan(*scenes: DramaScene) -> DramaPlan:
    return DramaPlan(
        plan_id="episode-001",
        title="The runaway prop",
        premise="A small mistake creates a physical comedy payoff.",
        objective="recover the prop and end in a readable loop",
        scenes=tuple(scenes),
        character_ids=("hero",),
        prop_ids=("prop",),
        style="warm pastel 2D",
        output_width=576,
        output_height=1024,
        fps=24,
        target_duration_seconds=4.0,
        variant_seed=7,
    )


class DramaPlanTests(unittest.TestCase):
    def test_round_trip_preserves_story_contract_and_cues(self) -> None:
        plan = build_plan(build_scene("S01"), build_scene("S02"))
        restored = DramaPlan.from_dict(plan.to_dict())

        self.assertEqual(restored, plan)
        self.assertEqual(restored.scenes[0].dialogue[0].text, "I have it")
        self.assertEqual(restored.scenes[0].sfx[0].name, "impact")

    def test_compile_hard_cut_drama_to_existing_edit_plan(self) -> None:
        compiled = compile_drama_plan(build_plan(build_scene("S01"), build_scene("S02")))

        self.assertEqual(compiled.profile, "motion_cut_v1")
        self.assertEqual(compiled.transitions, ())
        self.assertEqual([clip.label for clip in compiled.clips], ["S01:HOOK", "S02:PAYOFF"])
        self.assertEqual(compiled.metadata["story_contract"][0]["start_state"], "the hero is reaching for the prop")
        self.assertEqual(compiled.metadata["dialogue_cue_count"], 2)
        self.assertEqual(compiled.target_duration_seconds, 4.0)

    def test_hard_cut_allows_zero_transition_duration(self) -> None:
        first = replace(build_scene("S01"), transition_duration_seconds=0.0)
        build_plan(first, build_scene("S02")).validate()

    def test_compile_fadeblack_drama_to_chapter_profile(self) -> None:
        compiled = compile_drama_plan(
            build_plan(build_scene("S01", transition="fadeblack"), build_scene("S02"))
        )

        self.assertEqual(compiled.profile, "chapter_dip_v1")
        self.assertEqual(compiled.transitions[0].name, "fadeblack")
        self.assertEqual(compiled.transitions[0].duration_seconds, 0.2)

    def test_requires_selected_asset_only_for_compilation(self) -> None:
        plan = build_plan(build_scene("S01", source=None))

        plan.validate()
        with self.assertRaisesRegex(DramaPlanError, "selected_source_path"):
            compile_drama_plan(plan)

    def test_rejects_mixed_transition_families(self) -> None:
        plan = build_plan(
            build_scene("S01", transition="cut"),
            build_scene("S02", transition="fadeblack"),
            build_scene("S03"),
        )

        with self.assertRaisesRegex(DramaPlanError, "mixed hard cuts"):
            plan.validate()

    def test_rejects_unknown_character_and_missing_story_state(self) -> None:
        unknown_character = build_scene("S01")
        unknown_character = replace(unknown_character, character_ids=("villain",))
        with self.assertRaisesRegex(DramaPlanError, "unknown character"):
            build_plan(unknown_character).validate()

        incomplete = build_scene("S01")
        incomplete = DramaScene(
            scene_id=incomplete.scene_id,
            beat=incomplete.beat,
            duration_seconds=incomplete.duration_seconds,
            objective=incomplete.objective,
            start_state="",
            action_beats=incomplete.action_beats,
            end_state=incomplete.end_state,
            next_hook=incomplete.next_hook,
            cause=incomplete.cause,
            effect=incomplete.effect,
            character_ids=incomplete.character_ids,
            prop_ids=incomplete.prop_ids,
        )
        with self.assertRaisesRegex(DramaPlanError, "scene.start_state"):
            build_plan(incomplete).validate()

    def test_audio_cues_accept_audio_files_and_reject_media_files(self) -> None:
        scene = build_scene("S01")
        scene = replace(
            scene,
            dialogue=(replace(scene.dialogue[0], audio_path="/tmp/dialogue.mp3"),),
            sfx=(replace(scene.sfx[0], audio_path="/tmp/impact.wav"),),
        )
        build_plan(scene).validate()

        invalid = replace(scene.dialogue[0], audio_path="/tmp/dialogue.png")
        with self.assertRaisesRegex(DramaPlanError, "dialogue.audio_path"):
            build_plan(replace(scene, dialogue=(invalid,))).validate()

    def test_editing_skill_accepts_drama_plan_and_persists_it(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            class FakeTools:
                def __init__(self) -> None:
                    self.compose_plan = None
                    self.qa_inputs = None

                def call(self, name: str, payload: dict[str, object]) -> dict[str, object]:
                    if name == "media.compose_edit":
                        self.compose_plan = payload["edit_plan"]
                        for key in ("output_path", "manifest_path", "contact_sheet_path"):
                            path = Path(str(payload[key]))
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(b"candidate")
                        return {
                            "video_path": str(payload["output_path"]),
                            "manifest_path": str(payload["manifest_path"]),
                            "contact_sheet_path": str(payload["contact_sheet_path"]),
                        }
                    if name == "media.video_qa":
                        self.qa_inputs = payload
                        return {"passed": True}
                    if name == "media.materialize_edit":
                        return {
                            "video_path": str(payload["output_path"]),
                            "manifest_path": str(payload["manifest_path"]),
                            "contact_sheet_path": str(payload["contact_sheet_path"]),
                        }
                    raise AssertionError(name)

            fake_tools = FakeTools()
            skills = EditingSkills(fake_tools, root)  # type: ignore[arg-type]
            plan = build_plan(build_scene("S01"), build_scene("S02"))
            node = ExecutionNode(
                node_id="compose-edit",
                skill_name="media.video.compose_timeline",
                inputs={"drama_plan": plan.to_dict()},
            )
            context = SkillContext(
                plan=ExecutionPlan(
                    goal=GoalRequest(prompt="make a short drama", media_type="image_sequence_edit"),
                    workflow_name="image_sequence_edit_v1",
                    nodes=[node],
                ),
                node=node,
                state=RunState(goal={}, metadata={}),
            )

            result = skills.compose_timeline(context)

            self.assertEqual(result.status, "success")
            persisted = Path(result.outputs["drama_plan_path"])
            self.assertTrue(persisted.is_file())
            self.assertEqual(json.loads(persisted.read_text(encoding="utf-8"))["plan_id"], "episode-001")
            self.assertEqual(self._profile(fake_tools.compose_plan), "motion_cut_v1")
            self.assertFalse(fake_tools.qa_inputs["require_audio"])
            self.assertFalse(fake_tools.qa_inputs["require_stereo_audio"])

    @staticmethod
    def _profile(raw_plan: object) -> str:
        if not isinstance(raw_plan, dict):
            raise AssertionError("compose tool did not receive an EditPlan")
        return str(raw_plan["profile"])


if __name__ == "__main__":
    unittest.main()
