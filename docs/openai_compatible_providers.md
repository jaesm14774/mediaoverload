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
| Cloudflare Workers AI | `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` | `@cf/openai/gpt-oss-120b` | none | Free-tier text fallback |
| NVIDIA NIM | `NVIDIA_API_KEY` | `deepseek-ai/deepseek-v4.1-flash` | none | Free-tier text fallback |
| AMD Token Factory | `RADEON_API_KEY` | `DeepSeek-V4-Flash` | none | Free-tier text fallback |

Cloudflare, NVIDIA and AMD are text-only: do not select them as the vision
provider, because the vision route would then have no model, and they are
ignored in vision fallback lists. Only the first five distinct
provider/model entries of a fallback list are used, and an entry that is later
skipped (missing key, no vision support) still counts toward that limit. Each attempt gets an equal share of the
remaining request deadline, so a longer chain gives every model, the primary
included, less time. Cloudflare is skipped unless both its token and account
id are set. None of these providers is sent the `json_schema` response format;
JSON is validated and repaired locally.

The Cloudflare and NVIDIA default model ids answered a live chat request on 2026-11-13. NVIDIA's
`/v1/models` list includes models that return 404 or HTTP 410 (end of life)
for a given account, so test a model before configuring it. Cloudflare's
advertised 10,000 Neurons/day and NVIDIA's rate limits are not verified. Free
quotas change and NVIDIA may log prompts, so check each console and send no
sensitive content.

The provider registry and shared implementation live in
`agentic/src/agentic/runtime/model_backends.py`. All cloud providers use the
same Chat Completions request/response path. OpenAI accepts `none`, `low`,
`medium`, `high`, `xhigh`, and `max` as reasoning-effort values for GPT-6 Luna.

## AMD Token Factory

Free OpenAI-compatible LLM API from AMD Radeon Cloud (China site only; a free
trial and development service, not for production, no reselling).

1. Sign in at <https://developer.amd.com.cn/radeon/tokenfactory>, open
   **Token Factory**, and use **Show key** on the *Your API key* card. The key
   starts with `rc-` and works for every model.
2. Set `RADEON_API_KEY` in the private `media_overload.env`.
3. List the exact model ids your key can call (use them as
   `AGENTIC_TEXT_FALLBACK_MODELS` entries):

```bash
curl https://developer.amd.com.cn/radeon/api/v1/models -H "Authorization: Bearer $RADEON_API_KEY"
```

Catalog chat models include `DeepSeek-V4-Flash`, `DeepSeek-V4.1-Flash`,
`Qwen3.8-Flash-Next`, `Qwen3.8-27B`, `GLM-5.3-Flash`, `MiMo-V2.6-Flash` and
`Gemma-4-31B-IT`. One key serves all of them, so `amd` may appear more than
once in a fallback list with different models (an entry is dropped only when
the same provider and model repeat). Those entries share one rate limit, so
they only help when a single model fails (retired, unknown, overloaded); they
do not help against an account-wide 429. Interleave other providers between
them.

To use AMD as the primary text provider instead, set
`AGENTIC_TEXT_MODEL_PROVIDER=amd` and `AGENTIC_TEXT_MODEL=DeepSeek-V4-Flash`
while keeping a vision-capable provider for `AGENTIC_VISION_MODEL_PROVIDER`.

Limits (per the platform's rate-limit page): 30 requests/minute per key, 20
requests/minute per account, 8 concurrent requests per key, plus a daily point
cap that resets at 00:00 Beijing time. A 429 carries `Retry-After`; the shared
adapter waits for it (up to 60 seconds, bounded by the attempt's deadline
share) and retries the same model before moving to the next entry. Qwen3.8,
GLM-5.3-Flash and MiMo-V2.6-Flash think by default, which uses more of the
allowance, adds latency, and nothing strips that reasoning before JSON
validation, so prefer `DeepSeek-V4-Flash` for JSON-heavy steps. The model ids
and limits come from the platform documentation and were not exercised with a
live key.

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
AGENTIC_TEXT_FALLBACK_PROVIDERS=amd,nvidia,amd,cloudflare,gemini
AGENTIC_TEXT_FALLBACK_MODELS=DeepSeek-V4-Flash,deepseek-ai/deepseek-v4.1-flash,Qwen3.8-27B,@cf/openai/gpt-oss-120b,gemini-3.5-flash-lite
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
- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/)
- [NVIDIA NIM API keys](https://build.nvidia.com/)
- [AMD Token Factory](https://developer.amd.com.cn/radeon/tokenfactory) ([docs](https://amd-aim.github.io/radeon-cloud-docs/guides/token-factory/))

The repo cannot complete account verification, email confirmation, CAPTCHA,
phone verification, or payment checks. If a provider key is absent, that
provider is recorded as skipped. API keys are not included in recorded request
or response metadata.
