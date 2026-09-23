"""LLM provider boundary.

Everything outside this file talks to ChatModel / EmbeddingModel and never
names a provider, deployment, or model. The ``openai`` and ``azure-identity``
SDKs may be imported only in this file.
"""

import math
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    OpenAI,
    RateLimitError,
)
from openai.types import CompletionUsage
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from pydantic import BaseModel

from triage.config import Settings, get_settings
from triage.models import Category

# Scope for Entra ID tokens, used when AZURE_OPENAI_API_KEY is not configured.
_ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"

_S = TypeVar("_S", bound=BaseModel)


class LLMError(Exception):
    """Base class for provider errors."""


class TransientLLMError(LLMError):
    """A provider failure worth retrying: rate limit, timeout or connection error."""


_TRANSIENT_SDK_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)

# Pricing for gpt-4o (Standard tier, Sweden Central)
GPT4O_INPUT_COST_PER_TOKEN = 2.50 / 1_000_000
GPT4O_OUTPUT_COST_PER_TOKEN = 10.00 / 1_000_000


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate call cost in USD from token counts."""
    return (prompt_tokens * GPT4O_INPUT_COST_PER_TOKEN) + (
        completion_tokens * GPT4O_OUTPUT_COST_PER_TOKEN
    )


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


class CategoryProbe(BaseModel):
    """Lean schema for category-only classification probes."""

    category: Category


class ChatModel:
    """Chat completions bound to one deployment/model at construction."""

    def __init__(self, settings: Settings) -> None:
        self._client = _build_client(settings)
        self._model = (
            settings.azure_openai_chat_deployment
            if settings.llm_provider == "azure"
            else settings.openai_chat_model
        )
        # Attribute, not a changed return type: keeps complete() signatures
        # stable; callers that need usage read it right after the call.
        self.last_usage: CompletionUsage | None = None

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=messages, **kwargs
            )
        except _TRANSIENT_SDK_ERRORS as exc:
            raise TransientLLMError(str(exc)) from exc
        self.last_usage = response.usage
        return response.choices[0].message.content or ""

    def complete_structured(self, messages: list[dict], schema: type[_S], **kwargs: Any) -> _S:
        # Both providers expose the same .parse() path; the only provider
        # difference — deployment name vs model name — is already absorbed by
        # self._model at construction.
        try:
            response = self._client.chat.completions.parse(
                model=self._model, messages=messages, response_format=schema, **kwargs
            )
        except _TRANSIENT_SDK_ERRORS as exc:
            raise TransientLLMError(str(exc)) from exc
        self.last_usage = response.usage
        message = response.choices[0].message
        if message.parsed is None:
            raise ValueError(f"structured output not parsed (refusal: {message.refusal})")
        return message.parsed

    def classify_category_with_confidence(
        self,
        text: str,
        system_prompt: str = (
            "You are an SAP AMS ticket classifier. "
            "Classify the issue into the single best category."
        ),
    ) -> tuple[Category, float, dict[str, float]]:
        """Classify ticket text and extract normalized confidence across category alternatives.

        Returns: (predicted_category, normalized_confidence, distribution_dict)
        """
        # Request up to 10 top logprobs at API version 2024-10-21
        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Ticket:\n{text}"},
            ],
            response_format=CategoryProbe,
            max_tokens=60,
            logprobs=True,
            top_logprobs=10,
        )
        self.last_usage = response.usage

        choice = response.choices[0]
        parsed: CategoryProbe = choice.message.parsed
        tokens = choice.logprobs.content

        # 1. Locate the first value token (immediately following the colon '":"' or ':')
        colon_idx = next(
            (idx for idx, item in enumerate(tokens) if '":"' in item.token or ":" in item.token),
            -1,
        )
        if colon_idx == -1 or colon_idx + 1 >= len(tokens):
            # Fallback if structure is irregular
            return parsed.category, 1.0, {parsed.category.value: 1.0}

        decision_token = tokens[colon_idx + 1]

        # 2. Map of 12 distinct category prefix keywords
        category_prefixes = {
            "access": Category.ACCESS_AUTHORIZATION,
            "password": Category.PASSWORD_ACCOUNT,
            "short": Category.SHORT_DUMP,
            "performance": Category.PERFORMANCE,
            "batch": Category.BATCH_JOB,
            "interface": Category.INTERFACE_IDOC,
            "pricing": Category.PRICING_SALES,
            "invoice": Category.INVOICE_POSTING,
            "purchasing": Category.PURCHASING,
            "master": Category.MASTER_DATA,
            "output": Category.OUTPUT_PRINTING,
            "transport": Category.TRANSPORT_CHANGE,
        }

        # 3. Accumulate probabilities for each valid category candidate at this branch
        candidate_probs: dict[str, float] = {}

        if decision_token.top_logprobs:
            for alt in decision_token.top_logprobs:
                clean_tok = alt.token.strip().lower().lstrip('"').lstrip("_")
                for prefix, cat in category_prefixes.items():
                    if clean_tok.startswith(prefix) or prefix.startswith(clean_tok):
                        p = math.exp(alt.logprob)
                        candidate_probs[cat.value] = candidate_probs.get(cat.value, 0.0) + p
                        break

        # 4. Fallback if winning token was somehow not captured in top_logprobs
        winner_val = parsed.category.value
        if winner_val not in candidate_probs:
            candidate_probs[winner_val] = math.exp(decision_token.logprob)

        # 5. Renormalize across valid candidates to eliminate length and grammar bias
        total_mass = sum(candidate_probs.values())
        if total_mass > 0:
            norm_dist = {k: v / total_mass for k, v in candidate_probs.items()}
        else:
            norm_dist = {winner_val: 1.0}

        normalized_confidence = norm_dist.get(winner_val, 1.0)
        return parsed.category, normalized_confidence, norm_dist


class EmbeddingModel:
    """Embeddings bound to one deployment/model at construction."""

    def __init__(self, settings: Settings) -> None:
        self._client = _build_client(settings)
        self._model = (
            settings.azure_openai_embedding_deployment
            if settings.llm_provider == "azure"
            else settings.openai_embedding_model
        )
        self.last_usage: EmbeddingUsage | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        self.last_usage = response.usage
        return [item.embedding for item in response.data]


def get_chat_model() -> ChatModel:
    return ChatModel(get_settings())


def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel(get_settings())
