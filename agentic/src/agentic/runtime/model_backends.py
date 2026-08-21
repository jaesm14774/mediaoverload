from __future__ import annotations

import base64
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


STATIC_MODEL_MODES = {"structured", "prompt_only", "reasoning_off"}
MAX_PROVIDER_RETRIES = 5
MAX_OPENROUTER_MODELS_PER_CALL = 5
MAX_IMAGE_BYTES = 25 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
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


class ProviderRequestError(RuntimeError):
    """A provider request or provider response failed and may be retried/fallbacked."""


class ProviderConfigurationError(ValueError):
    """The local request or provider configuration is invalid."""


def _configured_timeout(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name, str(default)).strip()
    try:
        return max(0.1, float(raw))
    except (TypeError, ValueError):
        return max(0.1, float(default))


def _remaining_deadline(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0.1:
        raise ProviderRequestError("LLM provider total timeout exceeded before the next request.")
    return remaining


class FallbackChatModel:
    """Try the primary model, then configured auxiliary models in order."""

    def __init__(self, primary: ChatModel, fallback: ChatModel | None = None, *additional: ChatModel) -> None:
        models = [primary]
        if fallback is not None:
            models.append(fallback)
        models.extend(additional)
        self._models = models
        self._primary = models[0]
        # Keep the old attribute for callers that inspect the first fallback.
        self._fallback = models[1] if len(models) > 1 else None
        self.last_success_model: str = ""
        self.last_attempt_model: str = ""

    def chat_completion(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        *,
        _deadline: float | None = None,
        **kwargs,
    ) -> str:
        errors: list[str] = []
        for index, model in enumerate(self._models):
            if _deadline is not None and time.monotonic() >= _deadline:
                raise ProviderRequestError("LLM provider fallback chain exceeded its total timeout.")
            self.last_attempt_model = self._model_id(model)
            attempt_deadline = _deadline
            if _deadline is not None:
                # Reserve an equal share of the remaining chain budget for
                # every provider still available.  Without this slice, an
                # OpenRouter pool can consume the entire deadline before
                # Gemini/Groq/Mistral ever gets a chance to run.
                remaining = _deadline - time.monotonic()
                providers_left = max(1, len(self._models) - index)
                attempt_deadline = time.monotonic() + (remaining / providers_left)
                attempt_deadline = min(attempt_deadline, _deadline)
            try:
                result = model.chat_completion(messages, images=images, _deadline=attempt_deadline, **kwargs)
                self.last_success_model = self._model_id(model)
                return result
            except ProviderRequestError as exc:
                errors.append(f"{self.last_attempt_model or type(model).__name__}: {type(exc).__name__}: {exc}")
            except (requests.RequestException, TimeoutError, ConnectionError) as exc:
                errors.append(f"{self.last_attempt_model or type(model).__name__}: {type(exc).__name__}: {exc}")
            else:
                raise
        raise ProviderRequestError("All configured LLM providers failed: " + " | ".join(errors))

    @staticmethod
    def _model_id(model: Any) -> str:
        for attribute in ("last_success_model", "last_attempt_model"):
            raw_value = getattr(model, attribute, "")
            if isinstance(raw_value, str) and raw_value.strip():
                return raw_value.strip()
        config = getattr(model, "config", None)
        raw_model_name = getattr(config, "model_name", "")
        return raw_model_name.strip() if isinstance(raw_model_name, str) else ""


OPENAI_COMPATIBLE_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    # These are deliberately limited to providers with an official OpenAI-
    # compatible endpoint and a documented free path for this project.
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": ("GEMINI_API_KEY", "gemini_api_token"),
        "text_model": "gemini-3.5-flash",
        "vision_model": "gemini-3.5-flash",
        "supports_vision": True,
        "supports_response_format": True,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": ("GROQ_API_KEY",),
        # Groq retired llama-3.3-70b-versatile in the current runtime window;
        # keep the text default aligned with the supported qwen fallback that
        # is already used by the vision route.
        "text_model": "qwen/qwen3.6-27b",
        "vision_model": "qwen/qwen3.6-27b",
        "supports_vision": True,
        # llama-3.3-70b-versatile accepts JSON prompting but rejects the
        # OpenAI `json_schema` response format with HTTP 400.  Omit the
        # structured-output parameter and keep the existing JSON validator.
        "supports_response_format": False,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": ("MISTRAL_API_KEY",),
        "text_model": "mistral-small-latest",
        "vision_model": "",
        "supports_vision": False,
        # The shared OpenAI json_schema envelope is rejected by the current
        # Mistral compatibility endpoint for several nested pipeline schemas.
        # Keep strict JSON prompting and local validation as the portable path.
        "supports_response_format": False,
    },
}


def provider_spec(provider: str) -> dict[str, Any]:
    normalized = str(provider or "").strip().lower()
    try:
        return OPENAI_COMPATIBLE_PROVIDER_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}") from exc


def provider_default_model(provider: str, modality: str = "text") -> str:
    spec = provider_spec(provider)
    key = "vision_model" if modality.lower() in {"vision", "image", "multimodal"} else "text_model"
    return str(spec.get(key) or "").strip()


def provider_credentials_present(provider: str) -> bool:
    _load_project_env()
    return any(bool(os.environ.get(name, "").strip()) for name in provider_spec(provider)["api_key_env"])


class OpenAICompatibleModel:
    """Small OpenAI Chat Completions adapter shared by cloud providers.

    Providers only need to supply a base URL, an environment-variable key list,
    and a model id. The request/response contract remains the same for
    OpenRouter, Gemini, Groq, and Mistral.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        provider_name: str,
        base_url: str,
        api_key_env: tuple[str, ...] | list[str],
        default_headers: dict[str, str] | None = None,
        supports_vision: bool = True,
        supports_response_format: bool = True,
    ) -> None:
        _load_project_env()
        self.config = config
        self.provider_name = str(provider_name).strip().lower()
        self.api_key_env = tuple(api_key_env)
        self.api_key = next(
            (os.environ.get(name, "").strip() for name in self.api_key_env if os.environ.get(name, "").strip()),
            "",
        )
        if not self.api_key:
            names = ", ".join(self.api_key_env)
            raise ProviderConfigurationError(f"{self.provider_name} API key not found. Set one of: {names}.")
        self.base_url = self._resolve_endpoint(base_url)
        self.headers = {
            key: value
            for key, value in (default_headers or {}).items()
            if key.lower() not in {"authorization", "content-type"}
        }
        # Authentication and content type are owned by this adapter. Provider
        # metadata may add headers, but cannot redirect or de-authenticate it.
        self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.headers["Content-Type"] = "application/json"
        self.supports_vision = bool(supports_vision)
        self.supports_response_format = bool(supports_response_format)
        self.last_success_model: str = ""
        self.last_attempt_model: str = ""

    @classmethod
    def from_provider(cls, provider: str, config: ModelConfig) -> "OpenAICompatibleModel":
        normalized = str(provider or "").strip().lower()
        spec = provider_spec(normalized)
        return cls(
            config,
            provider_name=normalized,
            base_url=str(spec["base_url"]),
            api_key_env=tuple(spec["api_key_env"]),
            supports_vision=bool(spec.get("supports_vision", True)),
            supports_response_format=bool(spec.get("supports_response_format", True)),
        )

    @staticmethod
    def _resolve_endpoint(base_url: str) -> str:
        endpoint = str(base_url or "").rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ProviderConfigurationError("OpenAI-compatible provider endpoint must use HTTPS.")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        return f"{endpoint}/chat/completions"

    def chat_completion_single_model(
        self,
        model_name: str,
        messages: list[dict],
        images: list[str] | None = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        request_timeout: float | None = None,
        _deadline: float | None = None,
        **kwargs: object,
    ) -> str:
        response_validator = kwargs.pop("_response_validator", None)
        if images and not self.supports_vision:
            raise ProviderConfigurationError(f"{self.provider_name} does not support vision requests.")
        payload: dict[str, object] = {
            "model": model_name,
            "messages": self._process_messages_with_images(messages, images),
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        for key, value in kwargs.items():
            if key not in {"images", "max_retries", "initial_retry_delay", "request_timeout", "max_models_per_call", "_deadline"}:
                if key == "response_format" and not self.supports_response_format:
                    continue
                payload[key] = value

        last_error: Exception | None = None
        self.last_attempt_model = model_name
        attempt_limit = min(MAX_PROVIDER_RETRIES, max(1, int(max_retries)))
        for attempt in range(attempt_limit):
            response = None
            try:
                remaining = _remaining_deadline(_deadline)
                configured_read_timeout = (
                    max(0.1, float(request_timeout))
                    if request_timeout is not None
                    else _configured_timeout("AGENTIC_LLM_REQUEST_TIMEOUT_SECONDS", 30.0)
                )
                read_timeout = min(configured_read_timeout, remaining) if remaining is not None else configured_read_timeout
                connect_timeout = min(10.0, read_timeout)
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=(connect_timeout, read_timeout),
                    stream=_deadline is not None,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                last_error = exc
                if _deadline is not None and time.monotonic() >= _deadline:
                    raise ProviderRequestError(
                        f"{self.provider_name} request exceeded its total deadline for model {model_name!r}"
                    ) from exc
                if attempt >= attempt_limit - 1 or not self._is_retryable_error(exc):
                    break
                retry_delay = self._retry_delay(response, attempt, initial_retry_delay)
                if _deadline is not None:
                    retry_delay = min(retry_delay, max(0.0, _deadline - time.monotonic()))
                time.sleep(retry_delay)
                continue
            try:
                if _deadline is None:
                    body = response.json()
                else:
                    body = self._read_streamed_json(response, _deadline)
                message = body["choices"][0]["message"]
                text = self._extract_message_text(message)
            except requests.RequestException as exc:
                last_error = exc
                if _deadline is not None and time.monotonic() >= _deadline:
                    raise ProviderRequestError(
                        f"{self.provider_name} response exceeded its total deadline for model {model_name!r}"
                    ) from exc
                if attempt >= attempt_limit - 1:
                    break
                retry_delay = self._retry_delay(response, attempt, initial_retry_delay)
                if _deadline is not None:
                    retry_delay = min(retry_delay, max(0.0, _deadline - time.monotonic()))
                time.sleep(retry_delay)
                continue
            except (KeyError, ValueError, TypeError) as exc:
                raise ProviderRequestError(
                    f"{self.provider_name} returned an invalid chat completion for model {model_name!r}: {exc}"
                ) from exc
            if response_validator is not None:
                try:
                    response_validator(text)
                except Exception as exc:
                    last_error = ProviderRequestError(
                        f"{self.provider_name} returned invalid model output for model {model_name!r}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if attempt >= attempt_limit - 1:
                        break
                    retry_delay = self._retry_delay(response, attempt, initial_retry_delay)
                    if _deadline is not None:
                        retry_delay = min(retry_delay, max(0.0, _deadline - time.monotonic()))
                    time.sleep(retry_delay)
                    continue
            self.last_success_model = model_name
            return text
        detail = ""
        response = getattr(last_error, "response", None)
        if response is not None:
            try:
                body = str(response.text or "").strip()
                if body:
                    detail = f"; response={body[:1000]}"
            except Exception:
                pass
        raise ProviderRequestError(
            f"{self.provider_name} API call failed for model {model_name!r} after {attempt_limit} attempts: {last_error}{detail}"
        )

    @staticmethod
    def _read_streamed_json(response: requests.Response, deadline: float) -> Any:
        chunks: list[bytes] = []
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if time.monotonic() >= deadline:
                    raise requests.Timeout("Provider response exceeded its total deadline.")
                if isinstance(chunk, str):
                    chunks.append(chunk.encode("utf-8"))
                elif chunk:
                    chunks.append(bytes(chunk))
        finally:
            response.close()
        encoding = getattr(response, "encoding", None) or "utf-8"
        return json.loads(b"".join(chunks).decode(encoding))

    def chat_completion(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        request_timeout: float | None = None,
        _deadline: float | None = None,
        **kwargs: object,
    ) -> str:
        return self.chat_completion_single_model(
            self.config.model_name,
            messages,
            images=images,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            request_timeout=request_timeout,
            _deadline=_deadline,
            **kwargs,
        )

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            return True
        return int(status_code) in {408, 409, 425, 429} or int(status_code) >= 500

    @staticmethod
    def _retry_delay(response: object, attempt: int, initial_retry_delay: float) -> float:
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
        try:
            if retry_after is not None:
                return min(60.0, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
        return min(60.0, max(0.0, float(initial_retry_delay)) * (1.5**attempt))

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
        raise ValueError(f"{message}")

    @classmethod
    def _process_messages_with_images(cls, messages: list[dict], images: list[str] | None = None) -> list[dict]:
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
            elif isinstance(content, list):
                parts.extend(item for item in content if isinstance(item, dict))
            for image_path in images:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": cls._encode_image_to_base64(image_path)},
                    }
                )
            copied["content"] = parts
            processed.append(copied)
        return processed

    @staticmethod
    def _encode_image_to_base64(image_path: str) -> str:
        path = Path(image_path).expanduser()
        if path.is_symlink():
            raise ProviderConfigurationError(f"Refusing symlink image path: {image_path}")
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ProviderConfigurationError(f"Image path is not readable: {image_path}") from exc
        repo_root = Path(__file__).resolve().parents[4]
        allowed_roots = [repo_root]
        for raw_root in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(","):
            if raw_root.strip():
                allowed_roots.append(Path(raw_root.strip()).expanduser().resolve())
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise ProviderConfigurationError(f"Image path is outside allowed roots: {image_path}")
        if not resolved.is_file() or resolved.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            raise ProviderConfigurationError(f"Unsupported image file: {image_path}")
        if resolved.stat().st_size > MAX_IMAGE_BYTES:
            raise ProviderConfigurationError(f"Image file is too large: {image_path}")
        mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
        mime_type = mime_types[resolved.suffix.lower()]
        encoded = base64.b64encode(resolved.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"


class OllamaModel:
    def __init__(self, config: ModelConfig) -> None:
        import ollama

        self.config = config
        self._ollama = ollama
        self.host = "http://127.0.0.1:11434"
        self.request_timeout = _configured_timeout("AGENTIC_OLLAMA_REQUEST_TIMEOUT_SECONDS", 30.0)
        self.client = ollama.Client(
            host=self.host,
            timeout=self.request_timeout,
        )

    def chat_completion(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        _deadline: float | None = None,
        **kwargs,
    ) -> str:
        remaining = _remaining_deadline(_deadline)
        client = self.client
        if remaining is not None and remaining < self.request_timeout:
            client = self._ollama.Client(host=self.host, timeout=remaining)
        request_messages = [dict(message) for message in messages]
        if images:
            request_messages[-1]["images"] = images
        response = client.chat(
            model=self.config.model_name,
            messages=request_messages,
            options={"temperature": self.config.temperature},
        )
        return str(response.message.content)


class GeminiModel(OpenAICompatibleModel):
    def __init__(self, config: ModelConfig) -> None:
        spec = provider_spec("gemini")
        super().__init__(
            config,
            provider_name="gemini",
            base_url=str(spec["base_url"]),
            api_key_env=tuple(spec["api_key_env"]),
            supports_vision=bool(spec.get("supports_vision", True)),
            supports_response_format=bool(spec.get("supports_response_format", True)),
        )


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


class OpenRouterModel(OpenAICompatibleModel):
    FREE_TEXT_MODELS = list(DEFAULT_STATIC_MODEL_POOLS["text"])
    FREE_VISION_MODELS = list(DEFAULT_STATIC_MODEL_POOLS["vision"])

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(
            config,
            provider_name="openrouter",
            base_url="https://openrouter.ai/api/v1/chat/completions",
            api_key_env=("open_router_token", "OPENROUTER_API_KEY", "OPENROUTER_API_TOKEN"),
            default_headers={
                "HTTP-Referer": "https://github.com/mediaoverload",
                "X-Title": "MediaOverload",
            },
            supports_vision=True,
            supports_response_format=True,
        )

    def chat_completion(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        max_retries: int = MAX_PROVIDER_RETRIES,
        initial_retry_delay: float = 1.0,
        request_timeout: float | None = None,
        _deadline: float | None = None,
        **kwargs: object,
    ) -> str:
        return super().chat_completion(
            messages,
            images=images,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            request_timeout=request_timeout,
            _deadline=_deadline,
            **kwargs,
        )

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
        self._candidates = candidate_models[: min(limit, MAX_OPENROUTER_MODELS_PER_CALL)]
        self._inner = OpenRouterModel(ModelConfig(model_name=self._candidates[0], temperature=config.temperature))
        self._model_modes = dict(model_modes or {})
        self._random_each_call = random_each_call
        self.last_success_model: str = ""
        self.last_attempt_model: str = ""

    def chat_completion(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        _deadline: float | None = None,
        **kwargs: object,
    ) -> str:
        last_error: Exception | None = None
        candidates = list(self._candidates)
        if self._random_each_call and len(candidates) > 1:
            random.SystemRandom().shuffle(candidates)
        requested_limit = kwargs.pop("max_models_per_call", None)
        if requested_limit is not None:
            candidates = candidates[: min(MAX_OPENROUTER_MODELS_PER_CALL, max(1, int(requested_limit)))]
        for model_name in candidates:
            _remaining_deadline(_deadline)
            self.last_attempt_model = model_name
            model_kwargs = dict(kwargs)
            call_max_retries = max(1, int(model_kwargs.pop("max_retries", 1)))
            call_initial_retry_delay = float(model_kwargs.pop("initial_retry_delay", 0.5))
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
                    max_retries=call_max_retries,
                    initial_retry_delay=call_initial_retry_delay,
                    _deadline=_deadline,
                    **model_kwargs,
                )
                self.last_success_model = model_name
                return text
            except ProviderRequestError as exc:
                last_error = exc
                continue
        raise ProviderRequestError(f"OpenRouter: exhausted configured model candidates. Last error: {last_error}")


@dataclass(slots=True)
class AgenticLLMManager:
    text_model: ChatModel
    vision_model: ChatModel


def build_model(provider: str, config: ModelConfig) -> ChatModel:
    normalized = str(provider or "").strip().lower()
    if normalized == "ollama":
        return OllamaModel(config)
    if normalized == "openrouter":
        return OpenRouterModel(config)
    if normalized in OPENAI_COMPATIBLE_PROVIDER_SPECS:
        if normalized == "gemini":
            return GeminiModel(config)
        return OpenAICompatibleModel.from_provider(normalized, config)
    raise ValueError(f"Unsupported model provider: {provider}")
