from functools import lru_cache
from typing import Literal

from pydantic import ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["azure", "openai_compatible"]

_AZURE_REQUIRED = (
    "azure_openai_endpoint",
    "azure_openai_chat_deployment",
    "azure_openai_embedding_deployment",
)

_OPENAI_COMPATIBLE_REQUIRED = (
    "openai_base_url",
    "openai_api_key",
    "openai_chat_model",
    "openai_embedding_model",
)

# GA default: earliest stable Azure OpenAI API version supporting structured
# outputs (chat.completions.parse / response_format=json_schema) on both
# classic (*.openai.azure.com) and Foundry (*.services.ai.azure.com) endpoints.
_DEFAULT_API_VERSION = "2024-10-21"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: ProviderName

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = _DEFAULT_API_VERSION
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_chat_model: str | None = None
    openai_embedding_model: str | None = None

    # Azure AI Search (optional retrieval backend)
    azure_search_endpoint: str | None = None
    azure_search_key: str | None = None
    azure_search_index_name: str = "tickets-history"

    @field_validator("azure_openai_api_version", mode="before")
    @classmethod
    def _default_api_version(cls, value: object) -> object:
        # An empty value in .env should mean "use the default", not "".
        return value or _DEFAULT_API_VERSION

    @model_validator(mode="after")
    def _require_provider_settings(self) -> "Settings":
        required = _AZURE_REQUIRED if self.llm_provider == "azure" else _OPENAI_COMPATIBLE_REQUIRED
        missing = [name.upper() for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(
                f"LLM_PROVIDER={self.llm_provider!r} requires the following "
                f"settings, which are missing or empty: {', '.join(missing)}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_provider() -> None:
    """Fail fast at startup with a readable message if the provider is unusable."""
    try:
        settings = get_settings()
    except ValidationError as exc:
        raise RuntimeError(f"invalid triage configuration:\n{exc}") from exc
    if settings.llm_provider == "azure" and not settings.azure_openai_api_key:
        from triage.llm import probe_entra_credential

        try:
            probe_entra_credential()
        except Exception as exc:
            raise RuntimeError(
                "LLM_PROVIDER='azure' with an empty AZURE_OPENAI_API_KEY "
                "authenticates via Entra ID, but no credential could acquire a "
                f"token ({type(exc).__name__}: {exc}). Run `az login`, configure "
                "workload/managed identity, or set AZURE_OPENAI_API_KEY."
            ) from exc
