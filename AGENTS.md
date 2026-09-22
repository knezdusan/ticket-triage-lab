# AGENTS.md

Training prototype: three-gate AI ticket triage for an SAP AMS service desk. Python 3.12, Pydantic v2, Azure OpenAI.
Full context, data model, milestones and decision log: `docs/BLUEPRINT.md`. Read it before any non-trivial task.

## Commands (run from repo root)
- Setup: `uv sync`
- Lint: `uv run ruff check .` and `uv run ruff format --check .`
- Test: `uv run pytest`
- Live check (human only, needs credentials): `uv run python -m triage.smoke`

Done means all three lint/test commands pass. Never report work as finished without running them.

## Never
- Run git commands. Hand the human the exact commands instead.
- Run `smoke.py` or anything else that needs real credentials.
- Read, print, or write `.env` values.
- Edit files outside those named in the task.
- Use or invent real customer or employer data. All tickets are synthetic.

## Reporting
End every task with: what changed (file by file), the lint/test output, and every decision the task did not specify.
If a change affects milestones or architecture, update `docs/BLUEPRINT.md` §9 or §10 in the same change.
