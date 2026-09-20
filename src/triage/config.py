from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["azure", "openai_compatible"]

_AZURE_REQUIRED = (
    "azure_openai_endpoint",
    "azure_openai_api_version",
    "azure_openai_chat_deployment",
    "azure_openai_embedding_deployment",
)

_OPENAI_COMPATIBLE_REQUIRED = (
    "openai_base_url",
    "openai_api_key",
    "openai_chat_model",
    "openai_embedding_model",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: ProviderName

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_chat_model: str | None = None
    openai_embedding_model: str | None = None

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
