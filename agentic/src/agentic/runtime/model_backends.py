from __future__ import annotations

import base64
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import requests
from dotenv import load_dotenv


STATIC_MODEL_MODES = {"structured", "prompt_only", "reasoning_off"}
DEFAULT_STATIC_MODEL_POOLS = {
    "text": [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ],
    "vision": [
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
    ],
}
DEFAULT_STATIC_MODEL_MODES = {
    "nvidia/nemotron-3-ultra-550b-a55b:free": "structured",
    "nvidia/nemotron-3-super-120b-a12b:free": "structured",
    "google/gemma-4-26b-a4b-it:free": "structured",
    "nvidia/nemotron-3-nano-30b-a3b:free": "reasoning_off",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "reasoning_off",
    "nvidia/nemotron-nano-12b-v2-vl:free": "prompt_only",
}
_STATIC_MODEL_CONFIG_CACHE: dict[str, Any] | None = None


def _load_project_env() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    for env_path in (repo_root / "media_overload.env", repo_root / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


def _load_static_model_config() -> dict[str, Any]:
    global _STATIC_MODEL_CONFIG_CACHE
    if _STATIC_MODEL_CONFIG_CACHE is not None:
        return _STATIC_MODEL_CONFIG_CACHE
    repo_root = Path(__file__).resolve().parents[4]
    configured_path = os.environ.get("AGENTIC_OPENROUTER_MODEL_CONFIG", "").strip()
    config_path = Path(configured_path) if configured_path else repo_root / "configs" / "openrouter_models.yaml"
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config: dict[str, Any] = {}
    try:
        import yaml

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config = loaded
    except Exception:
        config = {}
    _STATIC_MODEL_CONFIG_CACHE = config
    return config


def static_openrouter_models(modality: str) -> list[str]:
    normalized = "vision" if modality.lower() in {"vision", "image", "multimodal"} else "text"
    configured = _load_static_model_config().get(normalized)
    models: list[str] = []
    if isinstance(configured, list):
        for item in configured:
            raw_id = item.get("id") if isinstance(item, dict) else item
            model_id = str(raw_id or "").strip()
            if model_id and model_id not in models:
                models.append(model_id)
    return models or list(DEFAULT_STATIC_MODEL_POOLS[normalized])


def static_openrouter_model_modes(modality: str | None = None) -> dict[str, str]:
    modes = dict(DEFAULT_STATIC_MODEL_MODES)
    config = _load_static_model_config()
    modalities = (modality,) if modality in {"text", "vision"} else ("text", "vision")
    for configured_modality in modalities:
        entries = config.get(configured_modality)
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            mode = str(item.get("mode") or "").strip().lower()
            if model_id and mode in STATIC_MODEL_MODES:
                modes[model_id] = mode
    return modes


@dataclass(slots=True)
class ModelConfig:
    model_name: str
    temperature: float = 0.3
    max_tokens: int | None = None


class ChatModel(Protocol):
    def chat_completion(self, messages: list[dict], images: list[str] | None = None, **kwargs) -> str: ...


class FallbackChatModel:
    def __init__(self, primary: ChatModel, fallback: ChatModel) -> None:
        self._primary = primary
        self._fallback = fallback

    def chat_completion(self, messages: list[dict], images: list[str] | None = None, **kwargs) -> str:
        try:
            return self._primary.chat_completion(messages, images=images, **kwargs)
        except Exception as primary_exc:
            try:
                return self._fallback.chat_completion(messages, images=images, **kwargs)
            except Exception as fallback_exc:
                raise ValueError(
                    f"Fallback LLM failed after primary error ({type(primary_exc).__name__}: {primary_exc}). "
                    f"Fallback: {type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc


class OllamaModel:
    def __init__(self, config: ModelConfig) -> None:
        import ollama

        self.config = config
        self.client = ollama.Client(host="http://127.0.0.1:11434")

    def chat_completion(self, messages: list[dict], images: list[str] | None = None, **kwargs) -> str:
        request_messages = [dict(message) for message in messages]
        if images:
            request_messages[-1]["images"] = images
        response = self.client.chat(
            model=self.config.model_name,
            messages=request_messages,
            options={"temperature": self.config.temperature},
        )
        return str(response.message.content)


class GeminiModel:
    def __init__(self, config: ModelConfig) -> None:
        from google import genai
        from google.genai import types

        _load_project_env()
        self.config = config
        self.client = genai.Client(
            api_key=os.environ["gemini_api_token"],
            http_options=types.HttpOptions(timeout=300000),
        )

    def chat_completion(self, messages: list[dict], images: list[str] | None = None, **kwargs) -> str:
        combined_prompt = self._combine_messages(messages)
        if images:
            from PIL import Image

            contents = [combined_prompt, *[Image.open(image_path) for image_path in images]]
        else:
            contents = combined_prompt
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
        )
        return str(response.text)

    @staticmethod
    def _combine_messages(messages: list[dict]) -> str:
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if role == "system":
                parts.append(f"Instructions: {content}")
            else:
                parts.append(content)
        return "\n".join(parts).strip()


class OpenRouterModelCatalog:
    """Optional diagnostic discovery; normal scheduler runs use the static YAML pool."""

    ENDPOINT = "https://openrouter.ai/api/v1/models"
    EXCLUDED_TERMS = (
        "content-safety",
        "safety",
        "guard",
        "safeguard",
        "lyria",
        "alpha",
        "beta",
        "preview",
    )
    _cache: ClassVar[dict[str, tuple[float, list[dict[str, Any]]]]] = {}

    @classmethod
    def candidates(
        cls,
        modality: str = "text",
        *,
        limit: int | None = None,
        ttl_seconds: int | None = None,
        force_refresh: bool = False,
    ) -> list[str]:
        normalized = "vision" if modality.lower() in {"vision", "image", "multimodal"} else "text"
        ttl = max(0, int(ttl_seconds if ttl_seconds is not None else os.environ.get("AGENTIC_OPENROUTER_MODEL_CACHE_TTL_SECONDS", "21600")))
        now = time.time()
        cached = cls._cache.get(normalized)
        if not force_refresh and cached and now - cached[0] < ttl:
            entries = cached[1]
        else:
            entries = cls._fetch(normalized)
            cls._cache[normalized] = (now, entries)
        pool_size = int(limit or os.environ.get("AGENTIC_OPENROUTER_FREE_POOL_SIZE", "5"))
        pool_size = max(1, pool_size)
        selected = [str(item["id"]) for item in entries[:pool_size] if str(item.get("id") or "").strip()]
        if os.environ.get("AGENTIC_OPENROUTER_RANDOMIZE_MODELS", "true").lower() in {"1", "true", "yes"}:
            random.SystemRandom().shuffle(selected)
        return selected

    @classmethod
    def _fetch(cls, modality: str) -> list[dict[str, Any]]:
        try:
            response = requests.get(
                cls.ENDPOINT,
                headers={"Accept": "application/json", "User-Agent": "MediaOverload/agentic"},
                timeout=(5, 15),
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise RuntimeError(f"OpenRouter model catalog unavailable: {exc}") from exc
        raw_models = body.get("data", []) if isinstance(body, dict) else []
        if not isinstance(raw_models, list):
            raise RuntimeError("OpenRouter model catalog returned an invalid data list")
        eligible = [model for model in raw_models if isinstance(model, dict) and cls._is_eligible(model, modality)]
        eligible.sort(key=cls._score, reverse=True)
        return eligible

    @classmethod
    def _is_eligible(cls, model: dict[str, Any], modality: str) -> bool:
        model_id = str(model.get("id") or "").strip().lower()
        label = " ".join(str(model.get(key) or "") for key in ("name", "description", "id")).lower()
        if not model_id or model_id == "openrouter/free" or any(term in label for term in cls.EXCLUDED_TERMS):
            return False
        pricing = model.get("pricing") or {}
        if not isinstance(pricing, dict):
            return False
        try:
            if float(pricing.get("prompt", 1)) != 0.0 or float(pricing.get("completion", 1)) != 0.0:
                return False
        except (TypeError, ValueError):
            return False
        architecture = model.get("architecture") or {}
        if not isinstance(architecture, dict):
            architecture = {}
        input_modalities = {str(item).lower() for item in architecture.get("input_modalities", []) if item}
        output_modalities = {str(item).lower() for item in architecture.get("output_modalities", []) if item}
        if output_modalities and "text" not in output_modalities:
            return False
        if modality == "vision" and input_modalities and not ({"image", "video"} & input_modalities):
            return False
        if modality == "text" and input_modalities and "text" not in input_modalities:
            return False
        return True

    @staticmethod
    def _score(model: dict[str, Any]) -> tuple[float, float, float]:
        supported = {str(item).lower() for item in model.get("supported_parameters", []) if item}
        context_length = float(model.get("context_length") or 0)
        created = float(model.get("created") or 0)
        structured_bonus = 1.0 if {"response_format", "structured_outputs"} & supported else 0.0
        return (structured_bonus, min(context_length, 2_000_000) / 2_000_000, created)


class OpenRouterModel:
    FREE_TEXT_MODELS = list(DEFAULT_STATIC_MODEL_POOLS["text"])
    FREE_VISION_MODELS = list(DEFAULT_STATIC_MODEL_POOLS["vision"])

    def __init__(self, config: ModelConfig) -> None:
        _load_project_env()
        self.config = config
        self.last_success_model: str = ""
        self.api_key = (
            os.environ.get("open_router_token")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENROUTER_API_TOKEN")
        )
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set one of: open_router_token, OPENROUTER_API_KEY, OPENROUTER_API_TOKEN."
            )
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mediaoverload",
            "X-Title": "MediaOverload",
        }

    @classmethod
    def get_random_free_text_model(cls) -> str:
        try:
            return random.choice(static_openrouter_models("text"))
        except (RuntimeError, ValueError):
            return random.choice(DEFAULT_STATIC_MODEL_POOLS["text"])

    @classmethod
    def get_random_free_vision_model(cls) -> str:
        try:
            return random.choice(static_openrouter_models("vision"))
        except (RuntimeError, ValueError):
            return random.choice(DEFAULT_STATIC_MODEL_POOLS["vision"])

    def chat_completion_single_model(
        self,
        model_name: str,
        messages: list[dict],
        images: list[str] | None = None,
        max_retries: int = 5,
        initial_retry_delay: float = 3.0,
        **kwargs: object,
    ) -> str:
        payload = {
            "model": model_name,
            "messages": self._process_messages_with_images(messages, images),
            "temperature": self.config.temperature,
        }
        for key, value in kwargs.items():
            if key not in {"images", "max_retries", "initial_retry_delay"}:
                payload[key] = value

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=(10, 30))
                response.raise_for_status()
                body = response.json()
                message = body["choices"][0]["message"]
                text = self._extract_message_text(message)
                self.last_success_model = model_name
                return text
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_error = exc
                if attempt >= max_retries - 1:
                    break
                import time

                time.sleep(initial_retry_delay * (1.5**attempt))
        raise ValueError(
            f"OpenRouter API call failed for model {model_name!r} after {max_retries} attempts: {last_error}"
        )

    def chat_completion(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_retries: int = 5,
        initial_retry_delay: float = 3.0,
        **kwargs: object,
    ) -> str:
        return self.chat_completion_single_model(
            self.config.model_name,
            messages,
            images=images,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            **kwargs,
        )

    @staticmethod
    def _extract_message_text(message: dict[str, object]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
                    continue
                if item.get("type") == "text":
                    nested_text = item.get("content")
                    if isinstance(nested_text, str) and nested_text:
                        text_parts.append(nested_text)
            merged = "\n".join(part for part in text_parts if part).strip()
            if merged:
                return merged
        raise ValueError(f"OpenRouter response did not contain text content: {message}")

    @staticmethod
    def _process_messages_with_images(messages: list[dict], images: list[str] | None = None) -> list[dict]:
        if not images:
            return [dict(message) for message in messages]

        processed: list[dict] = []
        for message in messages:
            copied = dict(message)
            if copied.get("role") != "user":
                processed.append(copied)
                continue
            parts: list[dict[str, object]] = []
            content = copied.get("content")
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            for image_path in images:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": OpenRouterModel._encode_image_to_base64(image_path)},
                    }
                )
            copied["content"] = parts
            processed.append(copied)
        return processed

    @staticmethod
    def _encode_image_to_base64(image_path: str) -> str:
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        suffix = Path(image_path).suffix.lower()
        mime_type = mime_types.get(suffix, "image/jpeg")
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"


class OpenRouterRotatingModel:
    """同一請求內：單一 model 重試後仍失敗則換池中下一個 model。"""

    def __init__(
        self,
        config: ModelConfig,
        candidate_models: list[str],
        *,
        max_models_per_call: int | None = None,
        model_modes: dict[str, str] | None = None,
        random_each_call: bool = True,
    ) -> None:
        if not candidate_models:
            raise ValueError("OpenRouterRotatingModel requires a non-empty candidate_models list.")
        limit = max_models_per_call if max_models_per_call and max_models_per_call > 0 else len(candidate_models)
        self._candidates = candidate_models[:limit]
        self._inner = OpenRouterModel(ModelConfig(model_name=self._candidates[0], temperature=config.temperature))
        self._model_modes = dict(model_modes or {})
        self._random_each_call = random_each_call
        self.last_success_model: str = ""

    def chat_completion(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        **kwargs: object,
    ) -> str:
        last_error: Exception | None = None
        candidates = list(self._candidates)
        if self._random_each_call and len(candidates) > 1:
            random.SystemRandom().shuffle(candidates)
        for model_name in candidates:
            model_kwargs = dict(kwargs)
            mode = self._model_modes.get(model_name, "structured")
            if mode == "prompt_only":
                model_kwargs.pop("response_format", None)
            elif mode == "reasoning_off":
                model_kwargs["reasoning"] = {"enabled": False}
            try:
                text = self._inner.chat_completion_single_model(
                    model_name,
                    messages,
                    images=images,
                    max_retries=1,
                    initial_retry_delay=0.5,
                    **model_kwargs,
                )
                self.last_success_model = model_name
                return text
            except ValueError as exc:
                last_error = exc
                continue
        raise ValueError(f"OpenRouter: exhausted configured model candidates. Last error: {last_error}")


@dataclass(slots=True)
class AgenticLLMManager:
    text_model: ChatModel
    vision_model: ChatModel


def build_model(provider: str, config: ModelConfig) -> ChatModel:
    normalized = provider.lower()
    if normalized == "ollama":
        return OllamaModel(config)
    if normalized == "gemini":
        return GeminiModel(config)
    if normalized == "openrouter":
        return OpenRouterModel(config)
    raise ValueError(f"Unsupported model provider: {provider}")
