"""LLM provider boundary.

Everything outside this file talks to ChatModel / EmbeddingModel and never
names a provider, deployment, or model. The ``openai`` and ``azure-identity``
SDKs may be imported only in this file.
"""

from typing import Any, TypeVar

from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel

from triage.config import Settings, get_settings

# Scope for Entra ID tokens, used when AZURE_OPENAI_API_KEY is not configured.
_ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"

_S = TypeVar("_S", bound=BaseModel)


def _build_client(settings: Settings) -> OpenAI:
    if settings.llm_provider == "azure":
        kwargs: dict[str, Any] = {
            "azure_endpoint": settings.azure_openai_endpoint,
            "api_version": settings.azure_openai_api_version,
        }
        if settings.azure_openai_api_key:
            kwargs["api_key"] = settings.azure_openai_api_key
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(), _ENTRA_SCOPE
            )
        return AzureOpenAI(**kwargs)
    return OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)


def probe_entra_credential() -> None:
    """Acquire an Entra ID token or raise; used by config.validate_provider()."""
    from azure.identity import DefaultAzureCredential

    DefaultAzureCredential().get_token(_ENTRA_SCOPE)


class ChatModel:
    """Chat completions bound to one deployment/model at construction."""

    def __init__(self, settings: Settings) -> None:
        self._client = _build_client(settings)
        self._model = (
            settings.azure_openai_chat_deployment
            if settings.llm_provider == "azure"
            else settings.openai_chat_model
        )

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return response.choices[0].message.content or ""

    def complete_structured(self, messages: list[dict], schema: type[_S], **kwargs: Any) -> _S:
        # Both providers expose the same .parse() path; the only provider
        # difference — deployment name vs model name — is already absorbed by
        # self._model at construction.
        response = self._client.chat.completions.parse(
            model=self._model, messages=messages, response_format=schema, **kwargs
        )
        message = response.choices[0].message
        if message.parsed is None:
            raise ValueError(f"structured output not parsed (refusal: {message.refusal})")
        return message.parsed


class EmbeddingModel:
    """Embeddings bound to one deployment/model at construction."""

    def __init__(self, settings: Settings) -> None:
        self._client = _build_client(settings)
        self._model = (
            settings.azure_openai_embedding_deployment
            if settings.llm_provider == "azure"
            else settings.openai_embedding_model
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def get_chat_model() -> ChatModel:
    return ChatModel(get_settings())


def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel(get_settings())
