from unittest.mock import patch

import pytest
from pydantic import ValidationError

from triage.config import Settings, get_settings, validate_provider

_AZURE_ENV = {
    "LLM_PROVIDER": "azure",
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "test-key",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-4o",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
}

_OPENAI_COMPATIBLE_ENV = {
    "LLM_PROVIDER": "openai_compatible",
    "OPENAI_BASE_URL": "http://localhost:11434/v1",
    "OPENAI_API_KEY": "test-key",
    "OPENAI_CHAT_MODEL": "qwen2.5",
    "OPENAI_EMBEDDING_MODEL": "nomic-embed-text",
}

_ALL_KEYS = set(_AZURE_ENV) | set(_OPENAI_COMPATIBLE_ENV)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for key in _ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Point pydantic-settings at a non-existent file so a real .env on the
    # machine can never leak into tests.
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / "no-such.env")
    get_settings.cache_clear()


def _set_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_loads_azure_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, _AZURE_ENV)

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "azure"
    assert settings.azure_openai_endpoint == "https://example.openai.azure.com/"
    assert settings.azure_openai_chat_deployment == "gpt-4o"


def test_loads_openai_compatible_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, _OPENAI_COMPATIBLE_ENV)

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openai_compatible"
    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.openai_embedding_model == "nomic-embed-text"


def test_missing_provider_fails_loudly() -> None:
    with pytest.raises(ValidationError, match="llm_provider"):
        Settings(_env_file=None)


def test_missing_azure_key_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {k: v for k, v in _AZURE_ENV.items() if k != "AZURE_OPENAI_CHAT_DEPLOYMENT"}
    _set_env(monkeypatch, env)

    with pytest.raises(ValidationError, match="AZURE_OPENAI_CHAT_DEPLOYMENT"):
        Settings(_env_file=None)


def test_missing_openai_key_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {k: v for k, v in _OPENAI_COMPATIBLE_ENV.items() if k != "OPENAI_API_KEY"}
    _set_env(monkeypatch, env)

    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(_env_file=None)


def test_api_version_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {k: v for k, v in _AZURE_ENV.items() if k != "AZURE_OPENAI_API_VERSION"}
    _set_env(monkeypatch, env)

    assert Settings(_env_file=None).azure_openai_api_version == "2024-10-21"


def test_api_version_defaults_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, {**_AZURE_ENV, "AZURE_OPENAI_API_VERSION": ""})

    assert Settings(_env_file=None).azure_openai_api_version == "2024-10-21"


def test_validate_provider_passes_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, _AZURE_ENV)

    validate_provider()


def test_validate_provider_raises_readable_error_on_missing_config() -> None:
    with pytest.raises(RuntimeError, match="invalid triage configuration"):
        validate_provider()


def test_validate_provider_raises_readable_error_when_entra_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = {k: v for k, v in _AZURE_ENV.items() if k != "AZURE_OPENAI_API_KEY"}
    _set_env(monkeypatch, env)

    with (
        patch("triage.llm.probe_entra_credential", side_effect=Exception("no creds")),
        pytest.raises(RuntimeError, match="Entra ID"),
    ):
        validate_provider()
