"""Manual provider smoke check. Run: uv run python -m triage.smoke"""

import sys

from pydantic import BaseModel

from triage.config import validate_provider
from triage.llm import get_chat_model, get_embedding_model


class SmokeTriage(BaseModel):
    category: str
    is_urgent: bool


def main() -> None:
    try:
        validate_provider()
    except RuntimeError as exc:
        sys.exit(f"configuration error: {exc}")

    chat = get_chat_model()
    text = chat.complete([{"role": "user", "content": "Reply with exactly: provider ok"}])
    print(f"chat completion: {text}")

    parsed = chat.complete_structured(
        [
            {
                "role": "user",
                "content": (
                    "Triage this ticket: 'Production login is down for all users.' "
                    "Return a category and whether it is urgent."
                ),
            }
        ],
        SmokeTriage,
    )
    print(f"structured completion: {parsed!r}")

    embedding = get_embedding_model().embed(["ticket triage smoke test"])
    print(f"embedding dimension: {len(embedding[0])}")


if __name__ == "__main__":
    main()
