from __future__ import annotations

import base64
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests
from dotenv import load_dotenv


def _load_project_env() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    for env_path in (repo_root / "media_overload.env", repo_root / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


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


class OpenRouterModel:
    FREE_TEXT_MODELS = [
        "qwen/qwen3.6-plus:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "z-ai/glm-4.5-air:free",
        "minimax/minimax-m2.5:free",
        "openrouter/free"
    ]
    FREE_VISION_MODELS = [
        "qwen/qwen3.6-plus:free",
        "openrouter/free"
    ]

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
        return random.choice(cls.FREE_TEXT_MODELS)

    @classmethod
    def get_random_free_vision_model(cls) -> str:
        return random.choice(cls.FREE_VISION_MODELS)

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
    ) -> None:
        if not candidate_models:
            raise ValueError("OpenRouterRotatingModel requires a non-empty candidate_models list.")
        limit = max_models_per_call if max_models_per_call and max_models_per_call > 0 else len(candidate_models)
        self._candidates = candidate_models[:limit]
        self._inner = OpenRouterModel(ModelConfig(model_name=self._candidates[0], temperature=config.temperature))
        self.last_success_model: str = ""

    def chat_completion(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        **kwargs: object,
    ) -> str:
        last_error: Exception | None = None
        for model_name in self._candidates:
            try:
                text = self._inner.chat_completion_single_model(
                    model_name,
                    messages,
                    images=images,
                    **kwargs,
                )
                self.last_success_model = model_name
                return text
            except ValueError as exc:
                last_error = exc
                continue
        raise ValueError(f"OpenRouter: exhausted text model candidates. Last error: {last_error}")


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
