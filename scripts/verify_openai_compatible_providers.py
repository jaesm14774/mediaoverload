"""Smoke-test configured OpenAI-compatible providers without exposing API keys."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.runtime.model_backends import (  # noqa: E402
    ModelConfig,
    OpenRouterModel,
    build_model,
    provider_credentials_present,
    provider_default_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify configured free OpenAI-compatible LLM providers.")
    parser.add_argument(
        "--provider",
        action="append",
        choices=("openrouter", "gemini", "groq", "mistral"),
        help="Provider to test; repeatable. Defaults to OpenRouter plus approved auxiliaries.",
    )
    parser.add_argument("--modality", choices=("text", "vision"), default="text")
    parser.add_argument("--model", action="append", help="Optional model override, aligned with --provider")
    parser.add_argument("--image", help="Local image path for a vision smoke test")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.modality == "vision" and not args.image:
        print("vision smoke tests require --image", file=sys.stderr)
        return 2
    providers = args.provider or ["openrouter", "gemini", "groq", "mistral"]
    results: list[dict[str, object]] = []
    for index, provider in enumerate(providers):
        model_name = args.model[index] if args.model and index < len(args.model) else ""
        if not model_name:
            model_name = (
                OpenRouterModel.FREE_VISION_MODELS[0]
                if args.modality == "vision"
                else OpenRouterModel.FREE_TEXT_MODELS[0]
            ) if provider == "openrouter" else provider_default_model(provider, args.modality)
        if not model_name:
            results.append({"provider": provider, "status": "skip", "reason": "no_model_for_modality"})
            continue
        if provider != "openrouter" and not provider_credentials_present(provider):
            results.append({"provider": provider, "model": model_name, "status": "skip", "reason": "missing_api_key"})
            continue
        started = time.perf_counter()
        try:
            model = (
                OpenRouterModel(ModelConfig(model_name=model_name))
                if provider == "openrouter"
                else build_model(provider, ModelConfig(model_name=model_name))
            )
            response = model.chat_completion(
                [{"role": "user", "content": "Reply with exactly: provider smoke test ok"}],
                images=[args.image] if args.image else None,
                request_timeout=args.timeout,
                max_retries=1,
            )
            results.append(
                {
                    "provider": provider,
                    "model": model_name,
                    "status": "ok",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "response_preview": response[:120],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "provider": provider,
                    "model": model_name,
                    "status": "error",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
    return 0 if all(item["status"] in {"ok", "skip"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
