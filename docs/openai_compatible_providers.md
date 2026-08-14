# OpenAI-compatible provider setup

MediaOverload uses OpenRouter as the primary provider. Gemini, Groq, and
Mistral are approved auxiliary providers. They are attempted only after the
OpenRouter model or free-model rotation is exhausted, and only when the
corresponding API key is configured.

## Approved providers

| Provider | API key | Default text model | Default vision model | Role |
| --- | --- | --- | --- | --- |
| OpenRouter | `open_router_token` or `OPENROUTER_API_KEY` | repo free pool | repo verified vision pool | Primary |
| Gemini | `GEMINI_API_KEY` or `gemini_api_token` | `gemini-3.5-flash` | `gemini-3.5-flash` | Vision fallback |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | `qwen/qwen3.6-27b` | Fast text/vision fallback |
| Mistral | `MISTRAL_API_KEY` | `mistral-small-latest` | none | Fast text fallback |

The provider registry and the shared implementation live in
`agentic/src/agentic/runtime/model_backends.py`. All four cloud providers use
the same `chat.completions.create`-style payload and return a normalized text
result.

## Obtain keys

Create the keys from the providers' official consoles:

- [Google AI Studio API keys](https://aistudio.google.com/apikey)
- [Groq API keys](https://console.groq.com/keys)
- [Mistral API keys](https://admin.mistral.ai/)

The repo cannot complete account verification, email confirmation, CAPTCHA,
phone verification, or payment checks. After creating a key, put it in the
private `media_overload.env` file or set it in the current PowerShell session.
Never commit that file or print the key.

```powershell
$env:GEMINI_API_KEY = "..."
$env:GROQ_API_KEY = "..."
$env:MISTRAL_API_KEY = "..."
```

## Enable the auxiliary chain

```dotenv
AGENTIC_TEXT_MODEL_PROVIDER=openrouter
AGENTIC_VISION_MODEL_PROVIDER=openrouter
AGENTIC_PROVIDER_FALLBACK_ENABLED=true
AGENTIC_TEXT_FALLBACK_PROVIDERS=groq,mistral
AGENTIC_TEXT_FALLBACK_MODELS=llama-3.3-70b-versatile,mistral-small-latest
AGENTIC_VISION_FALLBACK_PROVIDERS=gemini,groq
AGENTIC_VISION_FALLBACK_MODELS=gemini-3.5-flash,qwen/qwen3.6-27b
```

If a key is absent, the provider is recorded as skipped and the request stays
on the OpenRouter path. The chain records the concrete successful model in
`last_success_model` so run reports can distinguish a free-model failure from
an auxiliary recovery.

## Verify before running workflows

Text smoke test:

```powershell
python scripts/verify_openai_compatible_providers.py --json
```

Vision smoke test:

```powershell
python scripts/verify_openai_compatible_providers.py --modality vision --image D:\MediaOverload\caption_compare\inputs\kirby_generated_image.png --json
```

The script reports `ok`, `skip`, or `error` per provider and never prints API
keys. A provider should remain disabled if its smoke test fails or if its model
does not pass the same image/video evidence comparison used for the OpenRouter
vision pool.
