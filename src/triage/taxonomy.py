"""Sampling rules for synthetic ticket generation, as plain data.

Code chooses every label; the model only writes ticket text. Every weight is a
probability share within its distribution — datagen.py does the sampling.
"""

from datetime import date

from triage.models import AssignmentGroup, Category, Level, Priority, TicketType, Tier

# ---------------------------------------------------------------- type mix

TYPE_MIX: dict[TicketType, float] = {
    TicketType.INCIDENT: 0.45,
    TicketType.SERVICE_REQUEST: 0.45,
    TicketType.PROBLEM: 0.05,
    TicketType.CHANGE: 0.05,
}

# ------------------------------------------------------- priority | type

PRIORITY_MIX_BY_TYPE: dict[TicketType, dict[Priority, float]] = {
    TicketType.INCIDENT: {
        Priority.P1: 0.05,
        Priority.P2: 0.20,
        Priority.P3: 0.55,
        Priority.P4: 0.20,
    },
    TicketType.SERVICE_REQUEST: {
        Priority.P1: 0.0,
        Priority.P2: 0.03,
        Priority.P3: 0.37,
        Priority.P4: 0.60,
    },
    TicketType.PROBLEM: {
        Priority.P1: 0.0,
        Priority.P2: 0.30,
        Priority.P3: 0.60,
        Priority.P4: 0.10,
    },
    TicketType.CHANGE: {
        Priority.P1: 0.0,
        Priority.P2: 0.20,
        Priority.P3: 0.50,
        Priority.P4: 0.30,
    },
}

# Impact x urgency cells that produce each priority (see derive_priority matrix).
# Once priority is chosen, impact/urgency is a uniform pick from its row.
PRIORITY_CELLS: dict[Priority, list[tuple[Level, Level]]] = {
    Priority.P1: [(Level.HIGH, Level.HIGH)],
    Priority.P2: [(Level.HIGH, Level.MEDIUM), (Level.MEDIUM, Level.HIGH)],
    Priority.P3: [
        (Level.HIGH, Level.LOW),
        (Level.MEDIUM, Level.MEDIUM),
        (Level.LOW, Level.HIGH),
    ],
    Priority.P4: [
        (Level.MEDIUM, Level.LOW),
        (Level.LOW, Level.MEDIUM),
        (Level.LOW, Level.LOW),
    ],
}

# ------------------------------------------------- categories per type

CATEGORIES_BY_TYPE: dict[TicketType, dict[Category, float]] = {
    TicketType.INCIDENT: {
        Category.SHORT_DUMP: 0.20,
        Category.PERFORMANCE: 0.15,
        Category.BATCH_JOB: 0.15,
        Category.INTERFACE_IDOC: 0.10,
        Category.PRICING_SALES: 0.10,
        Category.INVOICE_POSTING: 0.10,
        Category.PURCHASING: 0.05,
        Category.MASTER_DATA: 0.05,
        Category.OUTPUT_PRINTING: 0.05,
        Category.ACCESS_AUTHORIZATION: 0.05,
    },
    TicketType.SERVICE_REQUEST: {
        Category.PASSWORD_ACCOUNT: 0.30,
        Category.ACCESS_AUTHORIZATION: 0.30,
        Category.MASTER_DATA: 0.15,
        Category.OUTPUT_PRINTING: 0.10,
        Category.TRANSPORT_CHANGE: 0.10,
        Category.PURCHASING: 0.05,
    },
    TicketType.PROBLEM: {
        Category.PERFORMANCE: 0.35,
        Category.SHORT_DUMP: 0.25,
        Category.INTERFACE_IDOC: 0.25,
        Category.BATCH_JOB: 0.15,
    },
    TicketType.CHANGE: {
        Category.TRANSPORT_CHANGE: 0.60,
        Category.ACCESS_AUTHORIZATION: 0.20,
        Category.MASTER_DATA: 0.20,
    },
}

# -------------------------------------- assignment groups per category
# Weights are probabilities. The first key is the default group; a second
# entry is a legitimate alternate (e.g. a short dump can land with ABAP or
# Basis depending on whether it looks like code or runtime).

ASSIGNMENT_GROUPS_BY_CATEGORY: dict[Category, dict[AssignmentGroup, float]] = {
    Category.ACCESS_AUTHORIZATION: {
        AssignmentGroup.SECURITY: 0.85,
        AssignmentGroup.SERVICE_DESK: 0.15,
    },
    Category.PASSWORD_ACCOUNT: {AssignmentGroup.SERVICE_DESK: 1.0},
    Category.SHORT_DUMP: {
        AssignmentGroup.ABAP: 0.70,
        AssignmentGroup.BASIS: 0.30,
    },
    Category.PERFORMANCE: {
        AssignmentGroup.BASIS: 0.70,
        AssignmentGroup.ABAP: 0.30,
    },
    Category.BATCH_JOB: {AssignmentGroup.BASIS: 1.0},
    Category.INTERFACE_IDOC: {
        AssignmentGroup.INTEGRATION: 0.80,
        AssignmentGroup.ABAP: 0.20,
    },
    Category.PRICING_SALES: {AssignmentGroup.SD: 1.0},
    Category.INVOICE_POSTING: {AssignmentGroup.FI: 1.0},
    Category.PURCHASING: {AssignmentGroup.MM: 1.0},
    Category.MASTER_DATA: {},  # group follows sap_module — see MASTER_DATA_GROUP_BY_MODULE
    Category.OUTPUT_PRINTING: {
        AssignmentGroup.BASIS: 0.60,
        AssignmentGroup.SERVICE_DESK: 0.40,
    },
    Category.TRANSPORT_CHANGE: {AssignmentGroup.BASIS: 1.0},
}

# master_data is special: the assignment group is not sampled, it follows the
# ticket's SAP module — material/vendor master -> MM, customer/vendor master
# in finance -> FI, sales master data -> SD. The module list for this
# category must therefore never contain None.
MASTER_DATA_GROUP_BY_MODULE: dict[str, AssignmentGroup] = {
    "MM": AssignmentGroup.MM,
    "FI": AssignmentGroup.FI,
    "SD": AssignmentGroup.SD,
}

# ------------------------------------------------ sap_module per category
# None means "no module named on the ticket" — common for account and
# cross-module issues.

SAP_MODULES_BY_CATEGORY: dict[Category, list[str | None]] = {
    Category.ACCESS_AUTHORIZATION: ["SD", "MM", "FI", None],
    Category.PASSWORD_ACCOUNT: [None],
    Category.SHORT_DUMP: ["SD", "MM", "FI", None],
    Category.PERFORMANCE: [None, "SD", "MM", "FI"],
    Category.BATCH_JOB: [None, "SD", "MM", "FI"],
    Category.INTERFACE_IDOC: ["SD", "MM", "FI"],
    Category.PRICING_SALES: ["SD"],
    Category.INVOICE_POSTING: ["FI"],
    Category.PURCHASING: ["MM"],
    Category.MASTER_DATA: ["MM", "FI", "SD"],
    Category.OUTPUT_PRINTING: [None],
    Category.TRANSPORT_CHANGE: [None],
}

# ------------------------------------------------------------------- tier
# Base tier follows the assignment group; problems always go one tier deeper
# (L3 — root-cause analysis is developer/vendor work), applied in datagen.

TIER_BY_GROUP: dict[AssignmentGroup, Tier] = {
    AssignmentGroup.SERVICE_DESK: Tier.L1,
    AssignmentGroup.SECURITY: Tier.L1,
    AssignmentGroup.FI: Tier.L2,
    AssignmentGroup.SD: Tier.L2,
    AssignmentGroup.MM: Tier.L2,
    AssignmentGroup.BASIS: Tier.L2,
    AssignmentGroup.INTEGRATION: Tier.L2,
    AssignmentGroup.ABAP: Tier.L3,
}

PROBLEM_TIER: Tier = Tier.L3

# -------------------------------------------------------------- ticket id

PREFIX_BY_TYPE: dict[TicketType, str] = {
    TicketType.INCIDENT: "INC",
    TicketType.SERVICE_REQUEST: "REQ",
    TicketType.PROBLEM: "PRB",
    TicketType.CHANGE: "CHG",
}

# ------------------------------------------------------------ created_at

DATE_RANGE: tuple[date, date] = (date(2026, 7, 1), date(2026, 9, 22))

# Probability of keeping a sampled weekend day; weekdays dominate.
WEEKEND_PROB = 0.15

# Hour weights: business hours 08-17 dominate, shoulders low, night rare.
HOUR_WEIGHTS: dict[int, float] = (
    {h: 1.0 for h in range(0, 7)}
    | {h: 3.0 for h in (7, 18, 19)}
    | {h: 8.0 for h in range(8, 18)}
    | {h: 1.0 for h in range(20, 24)}
)

# ------------------------------------------------------------ near-dupes

# Share of tickets that paraphrase an earlier ticket's labels.
DUPLICATE_RATE = 0.10

# ------------------------------------------- level -> business facts
# Phrasings the generator weaves into ticket text. They describe the
# situation, never the label: no "impact"/"urgency"/"priority" words and
# no P-codes appear here.

IMPACT_SITUATIONS: dict[Level, list[str]] = {
    Level.HIGH: [
        "the whole company is blocked",
        "an entire site is blocked",
        "everyone in the department is stopped",
        "all users on the system are hitting this",
    ],
    Level.MEDIUM: [
        "a team of several users is affected",
        "several colleagues hit this, but a workaround exists",
        "one group's work is disrupted while others are fine",
    ],
    Level.LOW: [
        "only one user is affected",
        "a single colleague reports this",
        "only my own work is affected",
    ],
}

URGENCY_SITUATIONS: dict[Level, list[str]] = {
    Level.HIGH: [
        "work has stopped completely right now",
        "there is a hard deadline today",
        "a hard closing deadline is blocked until this is fixed",
        "production is standing still while this is broken",
    ],
    Level.MEDIUM: [
        "work is slowing down because of this",
        "a deadline later this week is at risk",
        "people can still work but everything takes much longer",
    ],
    Level.LOW: [
        "no deadline is at risk",
        "this can wait, there is no time pressure",
        "it is annoying but nothing is blocked",
    ],
}

# Extra prompt sentence appended only when urgency is low — models otherwise
# tend to write low-urgency tickets that still sound time-pressured.
LOW_URGENCY_GUARD = (
    "The text must not mention any deadline, must not ask for urgent or "
    "immediate action, and must read as something that can wait."
)

# Same problem on the impact axis: low/medium-impact tickets still get written
# as site- or company-wide outages, which breaks the impact x priority signal.
LOW_IMPACT_GUARD = (
    "Exactly one person is affected. The text must not claim a team, "
    "department, site or company is affected, and must not say the system "
    "or production is down."
)

MEDIUM_IMPACT_GUARD = (
    "A team or several colleagues are affected — not the whole company or "
    "site — and other groups are working normally."
)

# ----------------------------------------------------------- writing style
# style key -> instruction injected into the generation prompt.

WRITING_STYLES: dict[str, str] = {
    "terse": "a terse one-liner; almost no detail",
    "rambling": "long and rambling, several paragraphs, includes irrelevant context",
    "forwarded_email": "a forwarded email with a greeting and a signature block",
    "error_paste": "mostly pasted error text or a log fragment with very little prose",
    "non_native": "non-native English with small typos and awkward phrasing",
    "escalation": "a frustrated manager escalation; impatient tone, mentions a deadline",
}

STYLE_MIX: dict[str, float] = {
    "terse": 0.25,
    "rambling": 0.15,
    "forwarded_email": 0.15,
    "error_paste": 0.15,
    "non_native": 0.15,
    "escalation": 0.15,
}

# -------------------------------------------------------------- requesters
# Synthetic people only.

REQUESTERS: list[str] = [
    "m.petrovic",
    "a.schroeder",
    "j.okafor",
    "l.moreau",
    "s.andersson",
    "k.tanaka",
    "r.garcia",
    "e.nowak",
    "d.kovacs",
    "n.vandenberg",
    "p.lindqvist",
    "t.nakamura",
    "h.meier",
    "c.dubois",
    "f.rossi",
    "o.yilmaz",
    "b.kaminski",
    "g.fernandez",
    "u.sato",
    "w.jansen",
    "i.petrova",
    "q.holm",
    "v.silva",
    "y.tanaka",
    "z.horvat",
    "m.abebe",
    "a.kaur",
    "j.molnar",
    "l.nilsen",
    "s.osei",
    "k.ramos",
    "r.tan",
    "e.wolf",
    "d.almeida",
    "n.costea",
    "p.fiala",
    "t.groeneveld",
    "h.ito",
    "c.krause",
    "f.larsen",
]

# -------------------------------------------------------- resolution notes
# Canned resolved-ticket notes, used for the history split only.

RESOLUTION_NOTES_BY_CATEGORY: dict[Category, list[str]] = {
    Category.ACCESS_AUTHORIZATION: [
        "Added missing authorization object via PFCG role change.",
        "Role extended after manager approval; access verified with user.",
    ],
    Category.PASSWORD_ACCOUNT: [
        "Password reset and account unlocked; user confirmed login.",
        "Reset via SU01 after identity check.",
    ],
    Category.SHORT_DUMP: [
        "Fixed in custom report; transport moved to PRD.",
        "OSS note applied; dump no longer reproducible.",
    ],
    Category.PERFORMANCE: [
        "Stale statistics updated; runtime back to normal.",
        "Missing index created on the affected table.",
    ],
    Category.BATCH_JOB: [
        "Job restarted after work process was freed.",
        "Variant corrected and job rescheduled.",
    ],
    Category.INTERFACE_IDOC: [
        "Failed IDocs reprocessed in BD87 after partner profile fix.",
        "RFC destination corrected; queue drained.",
    ],
    Category.PRICING_SALES: [
        "Condition record corrected in VK12.",
        "Pricing analysis pointed to a missing access sequence entry.",
    ],
    Category.INVOICE_POSTING: [
        "Posting period opened; invoice posted successfully.",
        "Tax code mapping fixed; document posted.",
    ],
    Category.PURCHASING: [
        "Purchase requisition released after approval fix.",
        "Vendor master block removed; PO created.",
    ],
    Category.MASTER_DATA: [
        "Master record corrected and change document retained.",
        "Duplicate record merged; requester informed.",
    ],
    Category.OUTPUT_PRINTING: [
        "Spool request reprocessed; output delivered.",
        "Frontend printer mapping fixed.",
    ],
    Category.TRANSPORT_CHANGE: [
        "Transport released and imported to QAS.",
        "Import rerun after dependency transport landed.",
    ],
}
