# ticket-triage-lab

Python 3.12 skeleton for a ticket-triage prototype. Scaffolding only — no business logic.

## Setup

```bash
uv sync          # creates .venv with Python 3.12 and installs dependencies
cp .env.example .env   # then fill in values for your provider
```

## Verify

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Environment variables

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `azure` or `openai_compatible` — selects the backend used by `triage.llm` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key; leave empty to use Entra ID via `azure-identity` |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI REST API version |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Azure deployment name for chat completions |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Azure deployment name for embeddings |
| `OPENAI_BASE_URL` | Base URL of an OpenAI-compatible API |
| `OPENAI_API_KEY` | API key for the OpenAI-compatible API |
| `OPENAI_CHAT_MODEL` | Chat model name |
| `OPENAI_EMBEDDING_MODEL` | Embedding model name |

Only the variables for the selected `LLM_PROVIDER` are required; missing values
raise a `ValidationError` naming every absent key.

## Layout

```
src/triage/config.py   # Settings (pydantic-settings, reads .env)
src/triage/llm.py      # get_chat_client() / get_embedding_client() — provider swap point
src/triage/models.py   # pydantic models
tests/                 # pytest
```

Nothing outside `src/triage/llm.py` imports the `openai` or `azure-identity` SDKs.
