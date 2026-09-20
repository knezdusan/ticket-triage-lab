from pydantic import BaseModel


class TriagedTicket(BaseModel):
    """Placeholder output schema for a triaged support ticket."""

    category: str
    priority: str
    summary: str
