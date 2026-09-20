"""LLM client factories.

This module is the single provider swap point: nothing else in the codebase
may import the ``openai`` or ``azure-identity`` SDKs directly.
"""

from typing import Any

from openai import AzureOpenAI, OpenAI

from triage.config import Settings, get_settings


def _client_kwargs(settings: Settings, azure_deployment: str | None) -> dict[str, Any]:
    if settings.llm_provider == "azure":
        kwargs: dict[str, Any] = {
            "azure_endpoint": settings.azure_openai_endpoint,
            "api_version": settings.azure_openai_api_version,
            "azure_deployment": azure_deployment,
        }
        if settings.azure_openai_api_key:
            kwargs["api_key"] = settings.azure_openai_api_key
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
        return kwargs
    return {
        "base_url": settings.openai_base_url,
        "api_key": settings.openai_api_key,
    }


def get_chat_client() -> OpenAI:
    """Return a client for chat completions.

    On Azure the client is bound to ``AZURE_OPENAI_CHAT_DEPLOYMENT``; on an
    OpenAI-compatible endpoint pass ``OPENAI_CHAT_MODEL`` per request.
    """
    settings = get_settings()
    kwargs = _client_kwargs(settings, settings.azure_openai_chat_deployment)
    if settings.llm_provider == "azure":
        return AzureOpenAI(**kwargs)
    return OpenAI(**kwargs)


def get_embedding_client() -> OpenAI:
    """Return a client for embeddings.

    On Azure the client is bound to ``AZURE_OPENAI_EMBEDDING_DEPLOYMENT``; on an
    OpenAI-compatible endpoint pass ``OPENAI_EMBEDDING_MODEL`` per request.
    """
    settings = get_settings()
    kwargs = _client_kwargs(settings, settings.azure_openai_embedding_deployment)
    if settings.llm_provider == "azure":
        return AzureOpenAI(**kwargs)
    return OpenAI(**kwargs)
