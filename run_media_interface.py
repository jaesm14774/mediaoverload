import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
for candidate in (REPO_ROOT, AGENTIC_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from agentic.app.character_workflow import dumps_result, run_character_workflow


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
    parser.add_argument("--generation-type", type=str, help="Override legacy generation type, for example text2longvideo")
    parser.add_argument("--dry-run-publish", action="store_true", help="Run publish stage in dry-run mode")
    parser.add_argument("--no-publish", action="store_true", help="Skip publish stage after generation")
    parser.add_argument("--enable-review-loop", action="store_true", help="Enable retry/review branches where supported")
    parser.add_argument("--review-notes", type=str, default="", help="Review notes for planner retry branches")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--comfy-host", type=str, help="Override ComfyUI host")
    parser.add_argument("--comfy-port", type=int, help="Override ComfyUI port")
    parser.add_argument("--comfy-root", type=str, help="Override ComfyUI root for asset checks")
    parser.add_argument("--auto-download-assets", action="store_true", help="Allow automatic workflow asset preparation")
    args = parser.parse_args()

    config_path = _resolve_config_path(args)
    result = run_character_workflow(
        REPO_ROOT,
        config_path,
        prompt=args.prompt,
        temperature=args.temperature,
        preferred_generation_type=args.generation_type,
        dry_run_publish=args.dry_run_publish,
        publish_after_generate=not args.no_publish,
        output_dir=args.output_dir,
        enable_review_loop=args.enable_review_loop,
        review_notes=args.review_notes,
        comfy_host=args.comfy_host,
        comfy_port=args.comfy_port,
        comfy_root=args.comfy_root,
        auto_download_assets=args.auto_download_assets,
    )
    print(dumps_result(result))


if __name__ == "__main__":
    main()
