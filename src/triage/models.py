from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


#  -------------- Enums --------------
class TicketType(StrEnum):
    INCIDENT = "incident"
    SERVICE_REQUEST = "service_request"
    PROBLEM = "problem"
    CHANGE = "change"


class Level(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Tier(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class AssignmentGroup(StrEnum):
    SERVICE_DESK = "Service Desk"
    SECURITY = "Security"
    BASIS = "Basis"
    ABAP = "ABAP"
    FI = "FI"
    SD = "SD"
    MM = "MM"
    INTEGRATION = "Integration"


class Category(StrEnum):
    ACCESS_AUTHORIZATION = "access_authorization"
    PASSWORD_ACCOUNT = "password_account"
    SHORT_DUMP = "short_dump"
    PERFORMANCE = "performance"
    BATCH_JOB = "batch_job"
    INTERFACE_IDOC = "interface_idoc"
    PRICING_SALES = "pricing_sales"
    INVOICE_POSTING = "invoice_posting"
    PURCHASING = "purchasing"
    MASTER_DATA = "master_data"
    OUTPUT_PRINTING = "output_printing"
    TRANSPORT_CHANGE = "transport_change"


# --------- Helper Functions ---------


def derive_priority(impact: Level | str, urgency: Level | str) -> Priority:
    """
    Derives ticket priority (P1-P4) from Impact and Urgency levels.

    Matrix:
      Impact \\ Urgency | HIGH | MEDIUM | LOW
      ------------------+------+--------+-----
      HIGH              |  P1  |   P2   |  P3
      MEDIUM            |  P2  |   P3   |  P4
      LOW               |  P3  |   P4   |  P4
    """

    impact = str(impact).lower()
    urgency = str(urgency).lower()

    match (impact, urgency):
        case (Level.HIGH, Level.HIGH):
            return Priority.P1

        case (Level.HIGH, Level.MEDIUM) | (Level.MEDIUM, Level.HIGH):
            return Priority.P2

        case (Level.HIGH, Level.LOW) | (Level.MEDIUM, Level.MEDIUM) | (Level.LOW, Level.HIGH):
            return Priority.P3

        case (Level.MEDIUM, Level.LOW) | (Level.LOW, Level.MEDIUM) | (Level.LOW, Level.LOW):
            return Priority.P4

        case _:
            raise ValueError(f"Invalid combination: {impact}, {urgency}")


#  -------------- Pydantic Models --------------


class TicketInput(BaseModel):
    """Raw, unclassified ticket as it arrives into the service desk."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    ticket_id: str = Field(pattern=r"^(INC|REQ|PRB|CHG)\d{6}$")
    created_at: datetime
    short_description: str = Field(min_length=3, max_length=120)
    description: str = Field(max_length=4000)
    requester: str
    sap_module: str | None = None


class LabelledTicket(TicketInput):
    """TicketInput plus ground-truth labels. The answer key for synthetic data."""

    type: TicketType
    category: Category
    impact: Level
    urgency: Level
    priority: Priority
    assignment_group: AssignmentGroup
    tier: Tier
    resolution_notes: str | None = None

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.priority != derive_priority(self.impact, self.urgency):
            raise ValueError("Priority does not match impact x urgency")
        prefix_map = {
            "INC": TicketType.INCIDENT,
            "REQ": TicketType.SERVICE_REQUEST,
            "PRB": TicketType.PROBLEM,
            "CHG": TicketType.CHANGE,
        }
        if prefix_map.get(self.ticket_id[:3]) != self.type:
            raise ValueError(f"Ticket ID prefix does not match type {self.type}")
        return self


class GeneratedText(BaseModel):
    """Schema for LLM-generated ticket text (Structured Outputs compatible)."""

    model_config = ConfigDict(extra="forbid")

    short_description: str
    description: str


LabelField = Literal["type", "category", "priority", "assignment_group"]
LabelSource = Literal["rules", "similarity", "model"]
LABEL_FIELDS: tuple[LabelField, ...] = ("type", "category", "priority", "assignment_group")


class TriageVerdict(BaseModel):
    """Final output of the triage cascade: labels plus observability metadata."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    type: TicketType | None = None
    category: Category | None = None
    priority: Priority | None = None
    assignment_group: AssignmentGroup | None = None
    # Gate-3 derivation inputs — observability, not labels; not part of
    # label_source (they explain how `priority` was derived).
    impact: Level | None = None
    urgency: Level | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    decided_by: Literal["rules", "similarity", "model", "abstain", "cascade"]
    label_source: dict[LabelField, LabelSource] = Field(default_factory=dict)
    rationale: str = Field(max_length=300)
    similar_ticket_ids: list[str] = Field(default_factory=list)
    cost_usd: float = Field(ge=0.0, default=0.0)
    latency_ms: float = Field(ge=0.0, default=0.0)

    @model_validator(mode="after")
    def check_abstain_consistency(self) -> Self:
        labels = [self.type, self.category, self.priority, self.assignment_group]
        if self.decided_by == "abstain":
            if any(label is not None for label in labels):
                raise ValueError("Abstaining verdict must have all labels set to None")
        elif all(label is None for label in labels):
            raise ValueError("A decided verdict must set at least one label")
        populated = {f for f, v in zip(LABEL_FIELDS, labels, strict=True) if v is not None}
        if set(self.label_source) != populated:
            raise ValueError("label_source must name exactly the populated labels")
        return self
