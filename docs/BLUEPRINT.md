# BLUEPRINT — ticket-triage-lab

> Single source of truth for this project. Read it in full before doing any work.
> Humans and agents both work from this file. When reality changes, update this file in the same change.

Last updated: 2026-09-21

---

## 1. What this is

A small, working prototype of **AI-based ticket triage** for an SAP Application Management Services (AMS) desk.

Given a support ticket's free text, it predicts **what kind of ticket it is, what it is about, how urgent it is, and who should handle it** — using the cheapest method that is confident enough, and escalating to a language model only for what remains.

It exists for one reason: to build working, from-memory fluency in the tools and domain of an AI Forward Deployed Engineer role (Python, Pydantic, Azure OpenAI, Azure AI Search, Azure AI Document Intelligence, evaluation, ITSM) before 1 October 2026.

**It is a training lab, not a product.** Correctness of the learning matters more than polish of the result. Nothing here uses real customer data — all tickets are synthetic.

---

## 2. Who writes what — the ownership rule

The point of the project is recall under pressure, not a finished artefact. So some parts must be written by hand.

| Owner | Parts |
|---|---|
| **Human only** (agent may review, explain, and write tests — never the implementation) | Python idiom exercises · every Pydantic model in `models.py` · the three gate functions · the evaluation harness logic |
| **Agent** | Scaffolding, config, packaging, synthetic data generation, Azure provisioning scripts, test plumbing, CLI wiring, README, refactors the human asks for |

If an agent directive touches a human-only part, the agent stops and says so instead of writing it.

---

## 3. Stack — fixed

| Concern | Choice | Note |
|---|---|---|
| Language | Python 3.12 (pinned `>=3.12,<3.13`) | managed by `uv` |
| Env and dependencies | `uv` | `uv.lock` and `.python-version` are tracked |
| Validation / schemas | Pydantic v2, pydantic-settings | the Zod equivalent |
| LLM + embeddings | `openai` SDK v3 via `AzureOpenAI` | `chat.completions.parse` for structured output |
| Auth | API key; Entra ID fallback via `azure-identity` | fallback only when no key is set |
| Retrieval | Azure AI Search (planned) | fallback: Postgres + pgvector |
| Documents | Azure AI Document Intelligence (planned) | one long document only |
| Tests / lint | pytest, ruff (line length 100; E, F, I, UP, B) | |

**Not allowed without an explicit decision recorded in §10:** LangChain, Semantic Kernel, LlamaIndex, FastAPI, or any other orchestration or web framework. We want to see the raw mechanics first.

---

## 4. Azure environment

Personal subscription, used only for this lab. Nothing connects to any employer system.

| Item | Value |
|---|---|
| Resource group | `rg-preboarding-test` (delete as a whole when the lab ends) |
| Canonical resource | `knezdusan-0285-resource` (Foundry), Sweden Central |
| Endpoint used by code | `https://knezdusan-0285-resource.openai.azure.com/` — the **Azure OpenAI** endpoint |
| Not used by code | `*.services.ai.azure.com` — the Foundry **project** endpoint (Foundry SDK / agents) |
| Chat deployment | gpt-4o, Standard, ~50k TPM |
| Embedding deployment | `text-embedding-3-small`, Standard, 120k TPM, 1536 dimensions |
| API version | `2024-10-21` (GA, supports structured output) — verified live |
| Budget | $30/month alert at 80% — **an alert, not a cap** |
| Unused | `oai-test-dusan-01` (classic Azure OpenAI) — leave alone, goes with the group |

**Deployment name ≠ model name.** Azure routes by the name you gave the deployment. Always check the exact string in Foundry → Models → Deployments.

---

## 5. Architecture — the three-gate cascade

The core idea: most tickets can be triaged cheaply. Only the ambiguous remainder should cost a model call. Each gate either returns a confident verdict or passes the ticket down.

```
ticket ──► Gate 1: rules ──confident──► verdict (decided_by="rules")
               │ not sure
               ▼
           Gate 2: similarity ──confident──► verdict (decided_by="similarity")
               │ not sure
               ▼
           Gate 3: model ──► verdict (decided_by="model")  [may abstain]
```

**Gate 1 — deterministic rules.** Keyword and pattern matching: transaction codes, "short dump", "access to", "password", "authorisation", known phrases. Near-zero cost. Only answers when a rule is unambiguous.

**Gate 2 — embedding similarity.** Embed the ticket, find the nearest resolved, already-labelled tickets. If the top neighbours agree above a similarity threshold, adopt their labels. The threshold is **measured from the data, never guessed** (lesson carried over from VectorMatch).

**Gate 3 — language model.** Structured output validated against a Pydantic schema, with the retrieved neighbours given as examples. Output that fails validation is rejected, not patched. The model may abstain; abstention routes to a human.

**Why this shape:** in AMS the margin is the gap between a fixed contract price and engineer hours. Cost per ticket is the business metric, so the architecture has to report it.

---

## 6. Data model

Field names follow ITSM vocabulary. Final shapes are written by hand in `models.py` (see §2) — this section is the intent, not the code.

**Ticket (input)**
`ticket_id` · `created_at` · `short_description` · `description` · `requester` · `sap_module` (optional) · `tcode` (optional)

**Ground truth labels (synthetic data only)**
`type` — incident | service_request | problem | change
`category`, `subcategory` — from a fixed taxonomy
`impact`, `urgency` — high | medium | low
`priority` — P1–P4, **derived** from impact × urgency, never set directly
`assignment_group` — e.g. Basis, FI, SD, MM, ABAP, Security, Service Desk
`tier` — L1 | L2 | L3
`resolution_notes` — for resolved tickets

**TriageVerdict (output)**
`type` · `category` · `priority` · `assignment_group` · `confidence` (0–1) · `decided_by` (rules | similarity | model | abstain) · `rationale` (short) · `similar_ticket_ids` · `cost_usd` · `latency_ms`

### Synthetic data rules
- ~300 tickets. Realistic, not dramatic: mostly short, badly written, often duplicated.
- Target priority shape: P1 ≈ 2%, P2 ≈ 10%, P3 and P4 the rest. (Working assumption, not an NTT figure.)
- Service requests and incidents dominate; problems and changes are rare.
- SAP flavour: tcodes, short dumps (ST22), SAP Notes, transports DEV → QAS → PRD.
- Split: resolved history (Gate 2 neighbours) / held-out evaluation set. The evaluation set is never used for retrieval or threshold tuning.

---

## 7. Evaluation — what "good" means

The harness runs the full cascade over the held-out set and reports:

- Accuracy per field (type, category, priority, assignment group)
- Per gate: share of tickets it decided, and its accuracy on those
- Abstention rate
- Cost per ticket and total; latency per ticket
- Confusion on priority — **a P1 predicted as P3 is far worse than the reverse**; report it separately

A result is only reported with its sample size. No accuracy figure on fewer than ~30 cases without saying so.

---

## 8. Code layout and conventions

```
src/triage/
  config.py     Settings (pydantic-settings), validate_provider()
  llm.py        ChatModel, EmbeddingModel, get_chat_model(), get_embedding_model()
  models.py     Pydantic models (human-written)
  smoke.py      live end-to-end check: uv run python -m triage.smoke
  (planned) rules.py · similarity.py · classifier.py · cascade.py · retrieval.py · evaluate.py
data/           synthetic tickets (planned)
tests/          no network calls, ever
```

**Rules that do not bend**
1. The `openai` and `azure` SDKs are imported **only in `llm.py`** (and later, the retrieval module for Azure AI Search). Everything else goes through `ChatModel` / `EmbeddingModel`.
2. Callers never pass a model or deployment name.
3. Model output is untrusted input: validate, reject on failure, never patch.
4. Tests mock the SDK. No network in `pytest`. Live checks live in `smoke.py`.
5. Type hints everywhere.
6. All green before any report: `uv run ruff check . && uv run ruff format --check . && uv run pytest`

**Agent working rules**
- The agent does not run git. It hands over the exact commands; the human commits.
- The agent does not touch files outside those named in a directive.
- The agent reports every decision the directive did not specify.
- The agent never runs `smoke.py` or anything else that needs real credentials.

---

## 9. Milestones

| Day | Block | Status |
|---|---|---|
| Sun 20 Sep | Azure account + budget alert · repo scaffold · provider boundary · ITSM vocabulary | **Done** — smoke test green against live Azure |
| Mon 21 | Python idiom and type hints (keyboard, no agent) | In progress |
| Tue 22 | Pydantic models (human) · synthetic ticket generator (agent) | — |
| Wed 23 | Azure OpenAI patterns · review and fix synthetic data | — |
| Thu 24 | Gate 1 rules · Gate 2 similarity | — |
| Fri 25 | Gate 3 model classification · cascade wiring | — |
| Sat 26 | Retrieval on Azure AI Search (fallback pgvector) | — |
| Sun 27 | **Rest** | — |
| Mon 28 | Evaluation harness | — |
| Tue 29 | Document Intelligence (half day) · SAP BTP / ServiceNow orientation | Droppable |
| Wed 30 | README, tidy, delete nothing yet, rest | — |

**If behind, drop in this order:** Document Intelligence → Azure AI Search (use pgvector) → BTP reading. Never drop: Pydantic, the three gates, evaluation.

---

## 10. Decisions log

| # | Decision | Why |
|---|---|---|
| D1 | Python 3.12, uv, hatchling, src layout | stable, reproducible, matches common Python practice |
| D2 | No orchestration framework | learn the raw mechanics before any abstraction; framework choice at the employer is unknown |
| D3 | Provider boundary: `ChatModel` / `EmbeddingModel` bind the model at construction | callers stay provider-agnostic; swap Azure ↔ OpenAI-compatible in config only |
| D4 | API version `2024-10-21` | earliest GA version with structured output; verified live 20 Sep |
| D5 | Foundry resource is canonical; code uses the `*.openai.azure.com` endpoint | Foundry is the target platform; the `AzureOpenAI` client needs the OpenAI endpoint |
| D6 | API key auth, Entra fallback | simplest for a personal lab; Entra path mirrors enterprise practice |
| D7 | Empty env values fall back to defaults | avoids a silent `""` overriding a working default |
| D8 | Repo and directory named `ticket-triage-lab` | not named after any employer |
| D9 | Fixed label taxonomy as `StrEnum`s in `models.py`: `TicketType` (4), `Level` for both impact and urgency (3), `Priority` (4), `Tier` (3), `AssignmentGroup` (8, display values), `Category` (12, snake_case) | one shared enum set is the contract the gates emit and the evaluation scores against; string values keep JSON output human-readable |
| D10 | TypeSafe (Jev) not adopted; its design ideas adopted instead — decomposed judgments, priority derived in code, confidence from probability spread rather than self-report; optional comparison experiment Tue 29 if time allows | reviewed 22 Sep: not NTT's stack, early access, data-handling unlikely to be approved for customer tickets, weaker on non-native English and long inputs |

---

## 11. Out of scope

Real ticket data of any kind · ServiceNow integration · any UI · deployment or hosting · agents or multi-step tool use · fine-tuning · anything that touches an employer system or account.

---

## 12. Glossary (short)

**Incident** broken thing, restore fast · **Service request** standard ask, nothing broken · **Problem** root cause behind incidents · **Change** planned modification with approval · **P1–P4** priority from impact × urgency · **L1/L2/L3** service desk / specialists / developers or vendor · **SLA** contracted response and resolution times · **Transition** handover of a customer estate to the AMS provider · **RCA** root cause analysis · **Known error** cause known, workaround documented · **Tcode** SAP transaction code · **Short dump** ABAP runtime crash (ST22) · **Transport** package carrying a change DEV → QAS → PRD
