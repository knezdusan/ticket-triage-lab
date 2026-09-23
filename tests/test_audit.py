"""Fixture checks for the audit regexes — the only way to know the ruler
is straight. Half genuine wide-scope claims, half lookalikes that describe
a single person or are just salutations."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "audit_data", Path(__file__).parents[1] / "scripts" / "audit_data.py"
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
LOW_IMPACT_RE = _audit._LOW_IMPACT_RE

# Genuine multi-person / wide-scope claims — must be flagged.
WIDE_SCOPE = [
    "Our team is unable to post any invoices since the patch.",
    "The whole department is blocked by this error.",
    "All users in finance cannot log in this morning.",
    "Everyone in purchasing sees the same dump.",
    "Users across the company are hitting this.",
    "Production is down since this morning.",
    "The system is down for all of us.",
    "A company-wide outage is blocking all postings.",
    "Several colleagues cannot open VA01 at all.",
    "The department cannot work until this is fixed.",
]

# Salutations and singular references — must NOT be flagged.
SINGLE_PERSON = [
    "Hello Team,\nmy VA01 session hangs every time I save.",
    "Dear Support Team, I am the only one affected by this.",
    "One team member cannot print delivery notes.",
    "I am the only member of the team who sees this dump.",
    "A colleague in our team reports a short dump on save.",
    "My colleague cannot access the site master data.",
    "It is annoying but only affects me; there is no deadline.",
    "Hi all, just my own login fails with a password error.",
]


@pytest.mark.parametrize("text", WIDE_SCOPE)
def test_low_impact_regex_flags_wide_scope(text):
    assert LOW_IMPACT_RE.search(text), text


@pytest.mark.parametrize("text", SINGLE_PERSON)
def test_low_impact_regex_ignores_single_person(text):
    assert not LOW_IMPACT_RE.search(text), text
