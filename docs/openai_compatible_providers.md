# OpenAI and OpenAI-compatible provider setup

The default configuration routes both text and vision LLM requests directly to
OpenAI `gpt-6-luna`. Set `AGENTIC_REASONING_EFFORT` to choose one effort for
both routes. The shared Chat Completions adapter sends that value as
`reasoning_effort` for OpenAI and omits `temperature` from its requests.

## Providers

| Provider | API key | Default text model | Default vision model | Role |
| --- | --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `gpt-6-luna` | `gpt-6-luna` | Direct API, text and vision |
| OpenRouter | `open_router_token` or `OPENROUTER_API_KEY` | repo free pool | repo verified vision pool | Optional primary or fallback |
| Gemini | `GEMINI_API_KEY` or `gemini_api_token` | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` | Optional vision fallback |
| Groq | `GROQ_API_KEY` | `qwen/qwen3.8-27b` | `qwen/qwen3.8-27b` | Optional text/vision fallback |
| Mistral | `MISTRAL_API_KEY` | `ministral-14b-2512` | none | Optional text fallback |

The provider registry and shared implementation live in
`agentic/src/agentic/runtime/model_backends.py`. All cloud providers use the
same Chat Completions request/response path. OpenAI accepts `none`, `low`,
`medium`, `high`, `xhigh`, and `max` as reasoning-effort values for GPT-6 Luna.

## Configure GPT-6 Luna for all LLM requests

Put the API key in the private `media_overload.env` file or set it in the
current PowerShell session. Never commit the private file or print the key.

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
AGENTIC_TEXT_MODEL_PROVIDER=openai
AGENTIC_TEXT_MODEL=gpt-6-luna
AGENTIC_VISION_MODEL_PROVIDER=openai
AGENTIC_VISION_MODEL=gpt-6-luna
AGENTIC_REASONING_EFFORT=xhigh
AGENTIC_PROVIDER_FALLBACK_ENABLED=false
AGENTIC_TEXT_ALLOW_FALLBACK=false
AGENTIC_VISION_ALLOW_FALLBACK=false
```

Change `AGENTIC_REASONING_EFFORT` to another value supported by the selected
OpenAI model. The single setting applies to both text and vision routes. Keep
fallback disabled when every LLM request must use GPT-6 Luna; an enabled
fallback may send requests to another provider and model.

## Optional cross-provider fallback

Enable this only when recovery through a different model is acceptable:

```dotenv
AGENTIC_PROVIDER_FALLBACK_ENABLED=true
AGENTIC_TEXT_FALLBACK_PROVIDERS=gemini,mistral
AGENTIC_TEXT_FALLBACK_MODELS=gemini-3.5-flash-lite,ministral-14b-2512
AGENTIC_VISION_FALLBACK_PROVIDERS=gemini,groq
AGENTIC_VISION_FALLBACK_MODELS=gemini-3.5-flash-lite,qwen/qwen3.8-27b
```

OpenAI recommends the Responses API for its newest reasoning and tool features.
This project currently uses text/JSON Chat Completions without model function
calling, so the existing adapter remains the smaller integration path. See
[GPT-6 Luna API documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
and the [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning).

## Optional provider keys

Create keys in the providers' official consoles and add them only if enabling
those providers:

- [Google AI Studio API keys](https://aistudio.google.com/apikey)
- [Groq API keys](https://console.groq.com/keys)
- [Mistral API keys](https://admin.mistral.ai/)

The repo cannot complete account verification, email confirmation, CAPTCHA,
phone verification, or payment checks. If a provider key is absent, that
provider is recorded as skipped. API keys are not included in recorded request
or response metadata.
