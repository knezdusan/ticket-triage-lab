from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from triage.config import Settings
from triage.llm import ChatModel, EmbeddingModel


class SampleOutput(BaseModel):
    label: str
    score: float


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

_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


def _settings(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_chat_model_azure_sends_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, _AZURE_ENV)
    with patch("triage.llm.AzureOpenAI") as mock_cls:
        client = mock_cls.return_value
        model = ChatModel(settings)

        mock_cls.assert_called_once_with(
            azure_endpoint="https://example.openai.azure.com/",
            api_version="2024-10-21",
            api_key="test-key",
        )
        model.complete(_MESSAGES)
        client.chat.completions.create.assert_called_once_with(model="gpt-4o", messages=_MESSAGES)


def test_chat_model_azure_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, _AZURE_ENV)
    parsed = SampleOutput(label="bug", score=0.9)
    with patch("triage.llm.AzureOpenAI") as mock_cls:
        client = mock_cls.return_value
        client.chat.completions.parse.return_value.choices[0].message.parsed = parsed

        result = ChatModel(settings).complete_structured(_MESSAGES, SampleOutput)

        client.chat.completions.parse.assert_called_once_with(
            model="gpt-4o", messages=_MESSAGES, response_format=SampleOutput
        )
        assert result is parsed


def test_chat_model_azure_entra_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {k: v for k, v in _AZURE_ENV.items() if k != "AZURE_OPENAI_API_KEY"}
    settings = _settings(monkeypatch, env)
    with (
        patch("triage.llm.AzureOpenAI") as mock_cls,
        patch("azure.identity.DefaultAzureCredential"),
        patch("azure.identity.get_bearer_token_provider") as mock_provider,
    ):
        ChatModel(settings)

        kwargs = mock_cls.call_args.kwargs
        assert "api_key" not in kwargs
        assert kwargs["azure_ad_token_provider"] is mock_provider.return_value


def test_embedding_model_azure_sends_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, _AZURE_ENV)
    with patch("triage.llm.AzureOpenAI") as mock_cls:
        client = mock_cls.return_value
        client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        result = EmbeddingModel(settings).embed(["hello"])

        client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["hello"]
        )
        assert result == [[0.1, 0.2, 0.3]]


def test_chat_model_openai_sends_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, _OPENAI_COMPATIBLE_ENV)
    with patch("triage.llm.OpenAI") as mock_cls:
        client = mock_cls.return_value
        model = ChatModel(settings)

        mock_cls.assert_called_once_with(base_url="http://localhost:11434/v1", api_key="test-key")
        model.complete(_MESSAGES)
        client.chat.completions.create.assert_called_once_with(model="qwen2.5", messages=_MESSAGES)


def test_chat_model_openai_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, _OPENAI_COMPATIBLE_ENV)
    parsed = SampleOutput(label="feature", score=0.5)
    with patch("triage.llm.OpenAI") as mock_cls:
        client = mock_cls.return_value
        client.chat.completions.parse.return_value.choices[0].message.parsed = parsed

        result = ChatModel(settings).complete_structured(_MESSAGES, SampleOutput)

        client.chat.completions.parse.assert_called_once_with(
            model="qwen2.5", messages=_MESSAGES, response_format=SampleOutput
        )
        assert result is parsed


def test_embedding_model_openai_sends_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, _OPENAI_COMPATIBLE_ENV)
    with patch("triage.llm.OpenAI") as mock_cls:
        client = mock_cls.return_value
        client.embeddings.create.return_value.data = [MagicMock(embedding=[1.0])]

        result = EmbeddingModel(settings).embed(["x"])

        client.embeddings.create.assert_called_once_with(model="nomic-embed-text", input=["x"])
        assert result == [[1.0]]
