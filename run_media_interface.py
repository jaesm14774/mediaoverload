import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
for candidate in (REPO_ROOT, AGENTIC_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from agentic.app.character_requests import (
    CharacterGenerationOptions,
    CharacterReviewOptions,
    CharacterRuntimeOptions,
    CharacterWorkflowRequest,
)
from agentic.app.character_workflow import dumps_result, run_character_workflow
from agentic.tools.social_services import complete_facebook_profile_handoff


def _default_comfy_root() -> str:
    configured = os.environ.get("COMFYUI_ROOT", "").strip()
    if configured:
        return configured
    container_root = Path("/comfyui")
    if container_root.is_dir():
        return str(container_root)
    return r"D:\ComfyUI_windows_portable"


def _resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config).resolve()
    if args.character:
        config_path = REPO_ROOT / "configs" / "characters" / f"{args.character.lower()}.yaml"
        if config_path.exists():
            return config_path
        raise ValueError(f"Character config not found: {config_path}")
    raise ValueError("Must provide --config or --character argument")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Character media interface backed by agentic runtime")
    parser.add_argument("--config", type=str, help="Path to character config file")
    parser.add_argument("--character", type=str, help="Character name")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--seed", type=int, help="Optional deterministic seed passed to the image and video workflows")
    parser.add_argument("--generation-type", type=str, help="Override generation type, for example text2longvideo")
    parser.add_argument(
        "--duration-seconds",
        type=int,
        help="Requested clip duration; 5 seconds uses one clear action, 15 seconds uses a compact native H3 story",
    )
    parser.add_argument("--reference-video", type=str, help="Optional local reference video or URL used as structural visual evidence")
    parser.add_argument("--reference-video-depth", choices=("standard", "deep"), default="standard")
    parser.add_argument("--reference-keyframes", type=int, default=12, help="Number of structural reference keyframes (2-20)")
    parser.add_argument("--dry-run-publish", action="store_true", help="Run publish stage in dry-run mode")
    parser.add_argument(
        "--publish-mode",
        choices=("live", "safe_poc"),
        default="",
        help="Publishing policy: live or safe_poc (YouTube private, Facebook draft, Instagram container-only)",
    )
    parser.add_argument(
        "--publish-platform",
        action="append",
        dest="publish_platforms",
        help="Restrict publishing to a platform; repeat for multiple platforms",
    )
    parser.add_argument(
        "--facebook-profile-share-url",
        type=str,
        default="",
        help="Optional public HTTPS media URL used to create a Facebook Profile link-share dialog",
    )
    parser.add_argument(
        "--complete-facebook-profile-handoff",
        type=str,
        help="Complete a Facebook Profile handoff artifact with the actual Facebook post URL",
    )
    parser.add_argument(
        "--facebook-profile-post-url",
        type=str,
        default="",
        help="Actual Facebook Profile post URL used with --complete-facebook-profile-handoff",
    )
    parser.add_argument("--no-publish", action="store_true", help="Skip publish stage after generation")
    parser.add_argument("--no-review", action="store_true", help="Disable human review for this run; auto Ref2VA keeps generated references without a Discord selection gate")
    parser.add_argument("--stage-probe", action="store_true", help="Run the real multi-stage graph with six image candidates and vision-LLM auto-selection; never use this as publish approval")
    parser.add_argument("--news-driven", action="store_true", help="Require a fresh unseen news item for this run")
    parser.add_argument("--enable-review-loop", action="store_true", help="Enable retry/review branches where supported")
    parser.add_argument("--review-notes", type=str, default="", help="Review notes for planner retry branches")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--comfy-host", type=str, help="Override ComfyUI host")
    parser.add_argument("--comfy-port", type=int, help="Override ComfyUI port")
    parser.add_argument(
        "--comfy-root",
        type=str,
        default=_default_comfy_root(),
        help="ComfyUI root for asset checks (default: COMFYUI_ROOT or D:\\ComfyUI_windows_portable)",
    )
    parser.add_argument("--auto-download-assets", action="store_true", help="Allow automatic workflow asset preparation")
    args = parser.parse_args()

    if args.complete_facebook_profile_handoff:
        if not args.facebook_profile_post_url:
            parser.error("--facebook-profile-post-url is required with --complete-facebook-profile-handoff")
        print(
            dumps_result(
                complete_facebook_profile_handoff(
                    args.complete_facebook_profile_handoff,
                    args.facebook_profile_post_url,
                )
            )
        )
        return

    config_path = _resolve_config_path(args)
    request = CharacterWorkflowRequest(
        repo_root=REPO_ROOT,
        config_path=config_path,
        generation=CharacterGenerationOptions(
            prompt=args.prompt,
            temperature=args.temperature,
            preferred_generation_type=args.generation_type,
            duration_seconds=args.duration_seconds,
            output_dir=args.output_dir,
            news_driven=args.news_driven,
            reference_video_source=args.reference_video,
            reference_video_depth=args.reference_video_depth,
            reference_video_max_keyframes=args.reference_keyframes,
            seed=args.seed,
        ),
        review=CharacterReviewOptions(
            dry_run_publish=args.dry_run_publish,
            publish_mode=args.publish_mode,
            publish_platforms=tuple(args.publish_platforms or ()),
            facebook_profile_share_url=args.facebook_profile_share_url,
            publish_after_generate=not args.no_publish,
            enable_review_loop=args.enable_review_loop,
            review_notes=args.review_notes,
            no_review=args.no_review,
            stage_probe=args.stage_probe,
        ),
        runtime=CharacterRuntimeOptions(
            comfy_host=args.comfy_host,
            comfy_port=args.comfy_port,
            comfy_root=args.comfy_root,
            auto_download_assets=args.auto_download_assets,
        ),
    )
    result = run_character_workflow(request)
    print(dumps_result(result))


if __name__ == "__main__":
    main()
