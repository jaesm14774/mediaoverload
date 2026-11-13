"""Cloudflare Workers AI and NVIDIA NIM as free-tier chat providers.

These scenarios verify how the shared OpenAI-compatible adapter is configured
for each provider.  The wire contract itself is covered by
test_provider_http_contract.py; availability of the real cloud services and
their free quotas is not verified here.
"""

import pytest

from agentic.runtime.llm_manager_adapter import _add_auxiliary_fallbacks
from agentic.runtime.model_backends import (
    FallbackChatModel,
    ModelConfig,
    OpenAICompatibleModel,
    ProviderConfigurationError,
    build_model,
    provider_credentials_present,
    provider_default_model,
)

ACCOUNT_ID = "0123456789abcdef0123456789abcdef"
CLOUDFLARE_ENV = ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID")
NVIDIA_ENV = ("NVIDIA_API_KEY",)


AMD_ENV = ("RADEON_API_KEY",)
AMD_MODELS = [
    "DeepSeek-V4-Flash",
    "DeepSeek-V4.1-Flash",
    "Qwen3.8-27B",
    "GLM-5.3-Flash",
    "MiMo-V2.6-Flash",
]


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    # Blank, not deleted: the project env loader fills only absent variables,
    # so a developer's real keys in media_overload.env must not leak in.
    for name in (*CLOUDFLARE_ENV, *NVIDIA_ENV, *AMD_ENV):
        monkeypatch.setenv(name, "")


def test_user_given_a_radeon_key_when_selecting_amd_then_requests_go_to_token_factory(
    monkeypatch,
):
    """User Given an AMD Token Factory key When the amd provider is selected Then the adapter targets Token Factory's chat completions endpoint with a default text model."""
    monkeypatch.setenv("RADEON_API_KEY", "rc-test")
    model = build_model("amd", ModelConfig(provider_default_model("amd", "text")))
    assert isinstance(model, OpenAICompatibleModel)
    assert model.base_url == "https://developer.amd.com.cn/radeon/api/v1/chat/completions"
    assert model.config.model_name
    assert model.headers["Authorization"] == "Bearer rc-test"


def test_user_given_five_models_on_one_free_provider_when_the_text_chain_is_built_then_all_five_are_available_for_recovery(
    monkeypatch,
):
    """User Given one AMD key and five distinct AMD models as fallbacks When the text chain is built Then all five are recorded as available for recovery, in the configured order."""
    monkeypatch.setenv("RADEON_API_KEY", "rc-test")
    backend = {
        "allow_text_fallback": True,
        "text_fallback_providers": ["amd"] * len(AMD_MODELS),
        "text_fallback_models": AMD_MODELS,
    }
    chain = _add_auxiliary_fallbacks(object(), backend, "text")
    assert isinstance(chain, FallbackChatModel)
    assert backend["text_fallback_skipped"] == []
    assert backend["text_fallback_candidates"] == [f"amd:{name}" for name in AMD_MODELS]


def test_user_given_more_than_five_fallbacks_when_the_text_chain_is_built_then_at_least_five_are_used_and_the_rest_are_ignored(
    monkeypatch,
):
    """User Given eight distinct fallback entries with credentials When the text chain is built Then at least five join the chain, in the configured order, and the list is bounded."""
    monkeypatch.setenv("RADEON_API_KEY", "rc-test")
    models = [*AMD_MODELS, "Gemma-4-31B-IT", "Qwen3.8-Flash-Next", "MiniCPM5-2B"]
    backend = {
        "allow_text_fallback": True,
        "text_fallback_providers": ["amd"] * len(models),
        "text_fallback_models": models,
    }
    _add_auxiliary_fallbacks(object(), backend, "text")
    used = backend["text_fallback_candidates"]
    assert 5 <= len(used) < len(models)
    assert used == [f"amd:{name}" for name in models[: len(used)]]


def test_user_given_five_providers_with_one_missing_key_when_the_text_chain_is_built_then_the_remaining_four_are_still_used(
    monkeypatch,
):
    """User Given five fallback providers where one has no key When the text chain is built Then the keyless one is skipped and the other four still join the chain."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("RADEON_API_KEY", "rc-test")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "mi-test")
    providers = ["groq", "cloudflare", "nvidia", "amd", "mistral"]
    backend = {
        "allow_text_fallback": True,
        "text_fallback_providers": providers,
        "text_fallback_models": [provider_default_model(p, "text") for p in providers],
    }
    chain = _add_auxiliary_fallbacks(object(), backend, "text")
    assert isinstance(chain, FallbackChatModel)
    assert backend["text_fallback_skipped"] == ["groq:missing_api_key"]
    assert len(backend["text_fallback_candidates"]) == 5

def test_user_given_an_nvidia_key_when_selecting_nvidia_then_requests_go_to_the_nim_chat_endpoint(
    monkeypatch,
):
    """User Given an NVIDIA API key When the nvidia provider is selected Then the adapter targets NVIDIA's chat completions endpoint with a default text model."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    model = build_model("nvidia", ModelConfig(provider_default_model("nvidia", "text")))
    assert isinstance(model, OpenAICompatibleModel)
    assert model.base_url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert model.config.model_name
    assert model.headers["Authorization"] == "Bearer nvapi-test"


def test_user_given_a_cloudflare_token_and_account_when_selecting_cloudflare_then_requests_go_to_that_accounts_endpoint(
    monkeypatch,
):
    """User Given a Cloudflare token and account id When the cloudflare provider is selected Then the adapter targets that account's Workers AI chat completions endpoint."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID)
    model = build_model(
        "cloudflare", ModelConfig(provider_default_model("cloudflare", "text"))
    )
    assert isinstance(model, OpenAICompatibleModel)
    assert model.base_url == (
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1/chat/completions"
    )
    assert model.config.model_name
    assert model.headers["Authorization"] == "Bearer cf-test"


def test_user_given_a_cloudflare_token_without_an_account_when_selecting_cloudflare_then_the_missing_setting_is_named(
    monkeypatch,
):
    """User Given a Cloudflare token but no account id When the cloudflare provider is selected Then the error names the missing setting and the provider is not reported as usable."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test")
    assert not provider_credentials_present("cloudflare")
    with pytest.raises(ProviderConfigurationError, match="CLOUDFLARE_ACCOUNT_ID"):
        build_model(
            "cloudflare", ModelConfig(provider_default_model("cloudflare", "text"))
        )


@pytest.mark.parametrize("account_id", ["../evil", "abc/def", "a b", "x@evil.com"])
def test_user_given_a_malformed_cloudflare_account_when_selecting_cloudflare_then_it_is_rejected_before_any_endpoint_is_built(
    monkeypatch, account_id
):
    """User Given a malformed Cloudflare account id When the cloudflare provider is selected Then it is rejected instead of altering the request path."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", account_id)
    with pytest.raises(ProviderConfigurationError, match="CLOUDFLARE_ACCOUNT_ID"):
        build_model(
            "cloudflare", ModelConfig(provider_default_model("cloudflare", "text"))
        )


def test_user_given_free_tier_fallbacks_without_keys_when_the_text_chain_is_built_then_they_are_skipped_and_the_primary_is_kept():
    """User Given cloudflare and nvidia fallbacks but no credentials When the text chain is built Then both are recorded as skipped and the primary model is unchanged."""
    primary = object()
    backend = {
        "allow_text_fallback": True,
        "text_fallback_providers": ["cloudflare", "nvidia"],
        "text_fallback_models": [
            provider_default_model("cloudflare", "text"),
            provider_default_model("nvidia", "text"),
        ],
    }
    assert _add_auxiliary_fallbacks(primary, backend, "text") is primary
    assert backend["text_fallback_skipped"] == [
        "cloudflare:missing_api_key",
        "nvidia:missing_api_key",
    ]


def test_user_given_free_tier_fallbacks_with_keys_when_the_text_chain_is_built_then_both_join_the_chain_in_order(
    monkeypatch,
):
    """User Given cloudflare and nvidia fallbacks with credentials When the text chain is built Then both providers are available for recovery after the primary."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    backend = {
        "allow_text_fallback": True,
        "text_fallback_providers": ["cloudflare", "nvidia"],
        "text_fallback_models": [
            provider_default_model("cloudflare", "text"),
            provider_default_model("nvidia", "text"),
        ],
    }
    chain = _add_auxiliary_fallbacks(object(), backend, "text")
    assert isinstance(chain, FallbackChatModel)
    assert backend["text_fallback_skipped"] == []
    assert backend["text_fallback_candidates"] == [
        f"cloudflare:{provider_default_model('cloudflare', 'text')}",
        f"nvidia:{provider_default_model('nvidia', 'text')}",
    ]


@pytest.mark.parametrize("explicit_model", [False, True])
@pytest.mark.parametrize("provider", ["cloudflare", "nvidia", "amd"])
def test_user_given_a_text_only_free_provider_when_it_is_listed_as_a_vision_fallback_then_the_vision_chain_ignores_it(
    monkeypatch, provider, explicit_model
):
    """User Given a text-only free provider with credentials, with or without an explicit model When it is configured as a vision fallback Then the vision chain keeps its primary and records the text-only provider as skipped."""
    monkeypatch.setenv("RADEON_API_KEY", "rc-test")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    backend = {
        "allow_vision_fallback": True,
        "vision_fallback_providers": [provider],
    }
    if explicit_model:
        backend["vision_fallback_models"] = [provider_default_model(provider, "text")]
    primary = object()
    assert _add_auxiliary_fallbacks(primary, backend, "vision") is primary
    assert backend["vision_fallback_skipped"]
