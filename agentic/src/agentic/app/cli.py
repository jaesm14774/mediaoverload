from __future__ import annotations

import argparse
import os
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic media runtime")
    parser.add_argument("--goal", help="High-level goal for the runtime")
    parser.add_argument("--media-type", help="Target media type")
    parser.add_argument("--duration-seconds", type=int, default=30, help="Target duration")
    parser.add_argument("--style", default="cinematic surreal", help="Preferred visual style")
    parser.add_argument("--execute", action="store_true", help="Execute the generated plan")
    parser.add_argument("--output-dir", help="Optional output directory for artifact-producing workflows")
    parser.add_argument("--width", type=int, help="Output video width in pixels for image-to-video workflows")
    parser.add_argument("--height", type=int, help="Output video height in pixels for image-to-video workflows")
    parser.add_argument("--longvideo-steps", type=int, help="Sampling steps for long-video H3 workflows")
    parser.add_argument("--comfy-host", help="Override ComfyUI host")
    parser.add_argument("--comfy-port", type=int, help="Override ComfyUI port")
    parser.add_argument(
        "--comfy-root",
        default=os.environ.get("COMFYUI_ROOT", r"D:\ComfyUI_windows_portable"),
        help="ComfyUI root for asset checks (default: COMFYUI_ROOT or D:\\ComfyUI_windows_portable)",
    )
    parser.add_argument("--input-image", help="Input image path for img2img/i2v/upscale workflows")
    parser.add_argument("--input-video", help="Input video path for frame extraction workflows")
    parser.add_argument(
        "--reference-video",
        help="Reference video path or URL used to extract pacing, shot, and motion evidence for remix planning",
    )
    parser.add_argument(
        "--reference-video-depth",
        choices=("standard", "deep"),
        default="standard",
        help="Reference-video analysis depth (both modes currently use local structural evidence; deep is reserved for richer analysis)",
    )
    parser.add_argument(
        "--reference-keyframes",
        type=int,
        default=12,
        help="Number of evenly spaced reference keyframes to extract (2-20)",
    )
    parser.add_argument("--input-dir", help="Input directory for publish/review workflows")
    parser.add_argument("--media-path", action="append", dest="media_paths", help="Explicit media path for publish/review workflows; repeatable")
    parser.add_argument("--edit-input", action="append", dest="edit_input_paths", help="Ordered image/video input for image_sequence_edit; repeatable")
    parser.add_argument("--edit-input-root", action="append", dest="edit_input_roots", help="Approved root for image_sequence_edit inputs; repeatable")
    parser.add_argument("--drama-plan", help="JSON DramaPlan file for image_sequence_edit; compiles scenes into the deterministic timeline editor")
    parser.add_argument(
        "--edit-profile",
        choices=("baseline_concat", "motion_cut_v1", "xfade_clean_v1", "chapter_dip_v1", "editorial_kinetic_v1"),
        help="Timeline profile for image_sequence_edit",
    )
    parser.add_argument("--edit-transition-duration", type=float, help="Transition overlap in seconds for timeline editing")
    parser.add_argument("--edit-variant-seed", type=int, help="Deterministic variation seed for timeline editing")
    parser.add_argument("--edit-require-audio", action="store_true", help="Require an audio stream in image_sequence_edit QA")
    parser.add_argument("--edit-analyze-audio", action="store_true", help="Run loudness/silence analysis in image_sequence_edit QA")
    parser.add_argument("--text", help="Text payload for TTS-style workflows")
    parser.add_argument("--character", help="Optional character or subject identity for agentic planning")
    parser.add_argument("--character-config", help="YAML production profile for serial review, video speed and social platforms; the approved card supplies the actual character identity")
    parser.add_argument(
        "--episode-topic",
        help="Optional topic for serial story planning; it is not added to public captions",
    )
    parser.add_argument("--platform", action="append", dest="platforms", help="Target publish platform; repeatable")
    parser.add_argument(
        "--publish-mode",
        choices=("live", "safe_poc"),
        default="",
        help="Publishing policy: live or safe_poc (YouTube private, Facebook draft, Instagram container-only)",
    )
    parser.add_argument("--hashtag", action="append", dest="hashtags", help="Hashtag for publish captions; repeatable")
    parser.add_argument("--caption-prefix", help="Optional caption prefix for publish workflows")
    parser.add_argument("--selection-limit", type=int, help="Review selection cap for publish/review workflows")
    parser.add_argument("--review-notes", help="Optional review feedback used to trigger refine/regenerate branches")
    parser.add_argument("--enable-review-loop", action="store_true", help="Force planner retry/review branches where supported")
    parser.add_argument("--dry-run-publish", action="store_true", help="Keep publish/review dispatch in dry-run mode")
    parser.add_argument("--serial-mode", action="store_true", help="Generate or continue a story-first approved series with per-segment human gates")
    parser.add_argument("--serial-auto-next", action="store_true", help="Resume blocked work first; otherwise continue an eligible existing series with an approved episode outline")
    parser.add_argument("--serial-require-delivery-receipts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--series-id", help="Stable ID for an explicit serial video series")
    parser.add_argument(
        "--serial-story-draft",
        action="store_true",
        help="Create a full series story treatment before choosing a character",
    )
    parser.add_argument(
        "--serial-news-inspiration",
        choices=("random", "none"),
        default="random",
        help="Use one random traceable local news item as fictional inspiration, or explicitly disable news",
    )
    parser.add_argument(
        "--serial-story-complexity",
        choices=("simple", "cinematic"),
        default="simple",
        help="Keep serial animation simple and clear by default, or opt into cinematic complexity",
    )
    parser.add_argument(
        "--serial-story-review-plan-id",
        help="Story plan ID to accept or reject after reviewing its Markdown file",
    )
    parser.add_argument(
        "--serial-story-review-decision",
        choices=("accepted", "rejected"),
        help="Human decision for a pending series story treatment",
    )
    parser.add_argument(
        "--serial-story-reviewed-file",
        help="Exact story file reviewed by the creator; required when accepting",
    )
    parser.add_argument(
        "--serial-story-review-notes",
        default="",
        help="Optional creator notes for the story decision",
    )
    parser.add_argument(
        "--serial-character-draft",
        action="store_true",
        help="Create a character identity card from the latest approved series story",
    )
    parser.add_argument(
        "--serial-character-preferences",
        default="",
        help="Optional creator preferences for the character; omit to let the model invent one",
    )
    parser.add_argument(
        "--serial-character-card-review-id",
        help="Character card ID to accept or reject after reviewing its Markdown file",
    )
    parser.add_argument(
        "--serial-character-card-review-decision",
        choices=("accepted", "rejected"),
        help="Human decision for a pending character identity card",
    )
    parser.add_argument(
        "--serial-character-card-name",
        help="Final character name to lock when approving a card",
    )
    parser.add_argument(
        "--serial-character-card-reviewed-file",
        help="Exact character card file reviewed by the creator",
    )
    parser.add_argument(
        "--serial-character-card-review-notes",
        default="",
        help="Optional creator notes for the character card decision",
    )
    parser.add_argument(
        "--serial-episode-plan-draft",
        action="store_true",
        help="Draft and review an episode story, scene and shot outline from the approved series materials",
    )
    parser.add_argument(
        "--serial-episode-plan-review-id",
        help="Episode outline ID to accept or reject after reviewing its Markdown file",
    )
    parser.add_argument(
        "--serial-episode-plan-review-decision",
        choices=("accepted", "rejected"),
        help="Human decision for a pending episode outline",
    )
    parser.add_argument(
        "--serial-episode-plan-reviewed-file",
        help="Exact episode outline file reviewed by the creator",
    )
    parser.add_argument(
        "--serial-episode-plan-review-notes",
        default="",
        help="Optional creator notes for the episode outline decision",
    )
    parser.add_argument(
        "--serial-character-reference",
        action="append",
        dest="serial_character_references",
        default=[],
        help="Reference image to lock when approving a character card; serial generation uses the approved paths and hashes",
    )
    parser.add_argument("--serial-episode-number", type=int, help="Episode to create or re-render; defaults to the next ready episode")
    parser.add_argument("--serial-publish-episode-number", type=int, help="Run final Discord approval and social dispatch for an already-ready episode")
    parser.add_argument("--serial-resume-episode-number", type=int, help="Resume the current blocked episode after its accepted segments are reviewed")
    parser.add_argument("--serial-review-episode-number", type=int, help="Current episode whose timed-out segment review is being recorded")
    parser.add_argument("--serial-review-segment-number", type=int, help="Segment reviewed by the creator")
    parser.add_argument("--serial-review-decision", choices=("accepted", "rejected"), help="Creator's decision for one QA-passed serial MP4")
    parser.add_argument("--serial-reviewed-video", help="Exact MP4 path inspected by the creator")
    parser.add_argument(
        "--discord-review-timeout-seconds",
        type=int,
        help="Maximum seconds to wait for a required Discord human review",
    )
    args = parser.parse_args(argv)
    if args.media_type is None:
        args.media_type = "long_video" if args.serial_mode else "image"
    story_review_mode = bool(args.serial_story_review_plan_id)
    character_review_mode = bool(args.serial_character_card_review_id)
    character_workflow_mode = bool(args.serial_character_draft or character_review_mode)
    episode_outline_review_mode = bool(args.serial_episode_plan_review_id)
    episode_outline_workflow_mode = bool(args.serial_episode_plan_draft or episode_outline_review_mode)
    story_workflow_mode = bool(args.serial_story_draft or story_review_mode)
    onboarding_workflow_mode = (
        story_workflow_mode or character_workflow_mode or episode_outline_workflow_mode
    )
    has_story_review_fields = any(
        value is not None
        for value in (
            args.serial_story_review_decision,
            args.serial_story_reviewed_file,
        )
    ) or bool(args.serial_story_review_notes)
    has_character_review_fields = any(
        value is not None
        for value in (
            args.serial_character_card_review_decision,
            args.serial_character_card_name,
            args.serial_character_card_reviewed_file,
        )
    ) or bool(args.serial_character_references)
    has_character_review_fields = has_character_review_fields or bool(
        args.serial_character_card_review_notes
    )
    has_episode_outline_review_fields = any(
        value is not None
        for value in (
            args.serial_episode_plan_review_decision,
            args.serial_episode_plan_reviewed_file,
        )
    ) or bool(args.serial_episode_plan_review_notes)
    if (
        not args.goal
        and not args.serial_auto_next
        and not onboarding_workflow_mode
        and not (args.serial_mode and args.series_id)
    ):
        parser.error("--goal is required unless --serial-auto-next is selected")
    if sum((
        bool(args.serial_story_draft),
        story_review_mode,
        bool(args.serial_character_draft),
        character_review_mode,
        bool(args.serial_episode_plan_draft),
        episode_outline_review_mode,
    )) > 1:
        parser.error("Run one story-first onboarding stage at a time")
    if onboarding_workflow_mode and args.serial_mode:
        parser.error("Story-first onboarding stages are separate from video generation; omit --serial-mode")
    if character_workflow_mode and not str(args.series_id or "").strip():
        parser.error("Character card stages require --series-id from the approved story")
    if args.serial_character_draft and (has_character_review_fields or args.serial_character_card_review_decision):
        parser.error("Create a character card draft or review a saved card, not both")
    if character_review_mode and args.serial_character_card_review_decision is None:
        parser.error("Character card review requires --serial-character-card-review-decision accepted|rejected")
    if has_character_review_fields and not character_review_mode:
        parser.error("Character card decision, name, file, and references require --serial-character-card-review-id")
    if (
        character_review_mode
        and args.serial_character_card_review_decision == "accepted"
        and not str(args.serial_character_card_reviewed_file or "").strip()
    ):
        parser.error("Approving a character card requires --serial-character-card-reviewed-file")
    if (
        character_review_mode
        and args.serial_character_card_review_decision == "accepted"
        and not str(args.serial_character_card_name or "").strip()
    ):
        parser.error("Approving a character card requires --serial-character-card-name")
    if (
        character_review_mode
        and args.serial_character_card_review_decision == "accepted"
        and not args.serial_character_references
    ):
        parser.error("Approving a character card requires at least one --serial-character-reference image")
    if episode_outline_workflow_mode and not str(args.series_id or "").strip():
        parser.error("Episode outline stages require --series-id from the approved story")
    if args.serial_episode_plan_draft and args.serial_episode_number is None:
        parser.error("Episode outline drafting requires --serial-episode-number")
    if args.serial_episode_plan_draft and args.serial_episode_number is not None and args.serial_episode_number < 1:
        parser.error("--serial-episode-number must be positive")
    if args.serial_episode_plan_draft and not str(args.episode_topic or "").strip():
        parser.error("Episode outline drafting requires --episode-topic")
    if episode_outline_review_mode and args.serial_episode_plan_review_decision is None:
        parser.error("Episode outline review requires --serial-episode-plan-review-decision accepted|rejected")
    if has_episode_outline_review_fields and not episode_outline_review_mode:
        parser.error("Episode outline decision, file, and notes require --serial-episode-plan-review-id")
    if (
        episode_outline_review_mode
        and args.serial_episode_plan_review_decision == "accepted"
        and not str(args.serial_episode_plan_reviewed_file or "").strip()
    ):
        parser.error("Approving an episode outline requires --serial-episode-plan-reviewed-file")
    if args.serial_story_draft and args.serial_news_inspiration == "none" and not str(args.goal or "").strip():
        parser.error("--serial-news-inspiration none requires a creator story idea in --goal")
    if story_review_mode and args.serial_story_review_decision is None:
        parser.error("Story review requires --serial-story-review-decision accepted|rejected")
    if has_story_review_fields and not story_review_mode:
        parser.error("Story review decision, file, and notes require --serial-story-review-plan-id")
    if (
        story_review_mode
        and args.serial_story_review_decision == "accepted"
        and not str(args.serial_story_reviewed_file or "").strip()
    ):
        parser.error("Accepting a story requires --serial-story-reviewed-file")
    if onboarding_workflow_mode and any(
        value is not None
        for value in (
            args.serial_episode_number if not args.serial_episode_plan_draft else None,
            args.serial_publish_episode_number,
            args.serial_resume_episode_number,
            args.serial_review_episode_number,
        )
    ):
        parser.error("Story planning and episode lifecycle options cannot be combined")
    if args.serial_mode:
        if not args.series_id and not args.serial_auto_next:
            parser.error("--serial-mode requires --series-id")
        if args.series_id and args.serial_auto_next:
            parser.error("--serial-auto-next selects the series; omit --series-id")
        if args.media_type.strip().lower() != "long_video":
            parser.error("--serial-mode requires --media-type long_video")
        if not (args.character or args.series_id):
            parser.error("--serial-mode requires --character for auto-next or --series-id for a saved series")
        if args.serial_episode_number is not None and args.serial_episode_number < 1:
            parser.error("--serial-episode-number must be positive")
        if args.serial_publish_episode_number is not None and args.serial_publish_episode_number < 1:
            parser.error("--serial-publish-episode-number must be positive")
        if args.serial_resume_episode_number is not None and args.serial_resume_episode_number < 1:
            parser.error("--serial-resume-episode-number must be positive")
        if args.serial_review_episode_number is not None and args.serial_review_episode_number < 1:
            parser.error("--serial-review-episode-number must be positive")
        if args.serial_review_segment_number is not None and args.serial_review_segment_number < 1:
            parser.error("--serial-review-segment-number must be positive")
        lifecycle_modes = sum(
            value is not None
            for value in (
                args.serial_episode_number,
                args.serial_publish_episode_number,
                args.serial_resume_episode_number,
                args.serial_review_episode_number,
            )
        )
        if lifecycle_modes > 1:
            parser.error("Rerender, resume, and manual review episode options are mutually exclusive")
        if (
            args.serial_publish_episode_number is None
            and args.serial_resume_episode_number is None
            and args.serial_review_episode_number is None
            and not str(args.episode_topic or "").strip()
            and not args.series_id
        ):
            if not args.serial_auto_next:
                parser.error("New serial episodes require --episode-topic")
        if args.serial_auto_next and any(
            value is not None
            for value in (
                args.serial_episode_number,
                args.serial_publish_episode_number,
                args.serial_resume_episode_number,
                args.serial_review_episode_number,
            )
        ):
            parser.error("--serial-auto-next cannot be combined with episode-specific lifecycle options")
        has_review_inputs = any(
            value is not None
            for value in (
                args.serial_review_segment_number,
                args.serial_review_decision,
                args.serial_reviewed_video,
            )
        )
        if (has_review_inputs or args.serial_review_episode_number is not None) and not all(
            value is not None
            for value in (
                args.serial_review_episode_number,
                args.serial_review_segment_number,
                args.serial_review_decision,
                args.serial_reviewed_video,
            )
        ):
            parser.error("Manual serial review requires an episode, segment, decision, and exact reviewed MP4 path")
        if args.serial_review_episode_number is not None and args.execute:
            parser.error("Manual serial review records one decision; do not combine it with --execute")
    elif not onboarding_workflow_mode and (
        (args.series_id and not onboarding_workflow_mode)
        or (args.serial_news_inspiration != "random" and not args.serial_story_draft)
        or (args.serial_character_references and not character_review_mode)
        or args.serial_character_preferences
        or args.serial_character_card_name
        or args.serial_character_card_reviewed_file
        or args.serial_character_card_review_decision is not None
        or args.serial_character_card_review_notes
        or args.serial_episode_plan_draft
        or args.serial_episode_plan_review_id
        or args.serial_episode_plan_review_decision is not None
        or args.serial_episode_plan_reviewed_file
        or args.serial_episode_plan_review_notes
        or (args.serial_episode_number is not None and not episode_outline_workflow_mode)
        or args.serial_publish_episode_number is not None
        or args.serial_resume_episode_number is not None
        or args.serial_review_episode_number is not None
        or args.serial_auto_next
        or args.serial_review_segment_number is not None
        or args.serial_review_decision is not None
        or args.serial_reviewed_video is not None
    ):
        parser.error("Serial series options require --serial-mode")
    if args.discord_review_timeout_seconds is not None and args.discord_review_timeout_seconds < 1:
        parser.error("--discord-review-timeout-seconds must be positive")
    if args.media_type.strip().lower() == "long_video" and not (args.serial_mode or onboarding_workflow_mode):
        parser.error("--media-type long_video requires --serial-mode")
    return args
