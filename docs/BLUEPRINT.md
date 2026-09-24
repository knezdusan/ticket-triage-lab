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

**Every accuracy figure is reported against its ceiling.** `triage.ceiling` computes, from `taxonomy.py` alone, the best achievable assignment-group accuracy (~88% — multi-group categories cap it) and type accuracy (~90% — problem/incident and change/service_request are textually indistinguishable). A raw accuracy number without its ceiling is meaningless here.

---

## 8. Code layout and conventions

```
src/triage/
  config.py     Settings (pydantic-settings), validate_provider()
  llm.py        ChatModel, EmbeddingModel, TransientLLMError, get_chat_model(), get_embedding_model()
  models.py     Pydantic models (human-written)
  utils.py      retry(), chunks()
  taxonomy.py   synthetic-data sampling rules as plain data
  datagen.py    synthetic ticket generator: uv run python -m triage.datagen [--dry-run] [--n N]
  smoke.py      live end-to-end check: uv run python -m triage.smoke
  (planned) rules.py · similarity.py · classifier.py · cascade.py · retrieval.py · evaluate.py
data/           tickets.jsonl · history.jsonl · eval.jsonl · clusters.json · REPORT.md (written by datagen)
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
| Tue 22 | Pydantic models (human) · synthetic ticket generator (agent) | Models done (32 tests green) · generator + priority-signal phrasings + P1/P2 stress set, dry-run verified; live generation pending |
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
| D11 | Datagen: all labels sampled in code (seed 42, `taxonomy.py` weights); the model writes only ticket text via `GeneratedText`; leak check rejects text mentioning priority/impact/urgency or P1–P4; ~10% near-duplicate clusters stay atomic across the history/eval split; SDK rate-limit/timeout/connection errors surface as `TransientLLMError` | keeps labels honest (no model self-labelling), reproducible plans, and prevents near-dupes leaking from eval into Gate 2 retrieval |
| D12 | Datagen carries impact/urgency into text as business-fact phrasings (`IMPACT_SITUATIONS` / `URGENCY_SITUATIONS`), stored on the plan; `--stress N` emits a separate P1/P2-only `eval_p1.jsonl`, IDs continuing the main sequence, excluded from all counts | without the severity signal in text, priority is unpredictable and the eval metric meaningless; the stress set measures P1 confusion without distorting the realistic distribution |
| D13 | master_data's assignment group follows the sampled sap_module (MM→MM, FI→FI, SD→SD), never drawn independently; `LOW_URGENCY_GUARD` prompt sentence appended only for low-urgency tickets; `DATE_RANGE` capped at 2026-09-22; `triage.ceiling` computes label-noise accuracy ceilings; `scripts/audit_data.py` audits generated text offline | fixes label/text coherence bugs found on the first live run; ceilings keep eval honest; the audit catches verbatim phrasing leaks and pressure words in low-urgency tickets before data review |
| D14 | Stress set composition fixed by construction: ceil(N/2) P1 incidents (high x high) + floor(N/2) P2 across P2-capable types weighted by TYPE_MIX | renormalised sampling made P1 nearly vanish (3 of 40); a set meant to measure severe-end confusion must not leave P1 to chance |
| D15 | Impact guards mirror the urgency guard: LOW_IMPACT_GUARD (one person only, no team/site/company, nothing "down") and MEDIUM_IMPACT_GUARD (a team, not site-wide, others work normally), appended in _prompt per impact level | audit found 74/122 low-impact tickets claiming wider scope — impact text inflates, which caps derivable priority accuracy |
| D16 | Logprobs work with structured output | Verified live 23 Sep on Azure `2024-10-21` gpt-4o. Pydantic `parse` natively returns logprobs. Grammar-constrained syntax tokens (`{"`, `category`) are 100%; the first distinguishing value token yields genuine confidence when normalized across category candidates (73%–100%). Friday Gate 3 will use native logprob confidence. |
| D17 | Schema constraints silently accepted | Azure OpenAI does NOT return HTTP 400 for unsupported validation constraints (`max_length`, `pattern`, `ge/le`). It silently accepts the schema and relies entirely on client-side Pydantic enforcement. Model-facing schemas must stay lean (`Gate3Output`), validating into `TriageVerdict` on our side. |
| D18 | Measured Gate 3 cost ($1.74 / 1k) | Measured across real SAP tickets: mean 386 input tokens, 78 output tokens per ticket. Using Sweden Central Standard `gpt-4o` rates ($2.50/M input, $10.00/M output), cost is $0.00174 per ticket ($1.74 per 1,000; $174 per 100k). Replaces the initial ~800 token / $3.00 estimate. |
| D19 | Deployment capacity is 300 RPM (50k TPM) | Read from Azure Foundry deployment page: 50k TPM corresponds to 300 RPM. Because short completions release reserved tokens immediately upon response termination, burst capacity is high (absorbed 60 concurrent calls). Recommended safe sustained worker pool is 5–10 concurrent requests. |
| D20 | Retry decorator lacks exponential backoff | The current `@retry` decorator retries immediately without delay or inspecting Azure's `retry-after` header. Documented as a prototype limitation; safe for controlled lab runs, but exponential backoff + jitter is required for unattended production workloads. |
| D21 | Gate 1 measured: 2.5% coverage @ 100% precision | `scripts/eval_gate1.py` over `history.jsonl` (240 tickets). Baseline with helpdesk vocabulary in the veto (`error`, `unable`, `fail`): 1.7% coverage, 75% precision. Fix: split veto into urgency/scope (`_URGENCY_VETO`) vs second-domain markers (`_DOMAIN_CONFLICT`). The ceiling is pattern recall, not the veto: most `access_authorization` tickets are phrased as incidents ("unable to access", "authorization issue"), not provisioning requests — widening patterns adds more P3 mislabels than P4 gains. `access_request` confidence set to 0.85: only ~80% of access tickets route to Security (rest to Service Desk). |
| D22 | Gate 2 first sweep: similarity lacks dynamic range here | Leave-one-out sweep over 240 embedded history tickets: top-1 cosine scores span ~0.65–0.88 with mass at 0.70–0.85 — the corpus is too domain-homogeneous for distance alone to separate. Best row (k=3, θ=0.75): 5.8% coverage @ 50% novel full-verdict precision; θ≥0.88 never fires. Per-field decomposition pending before design changes — hypothesis: `priority` (impact×urgency narrative) is what embeddings can't carry; `category`/`assignment_group` (topic) may be fine. |
| D23 | Cascade assembly-line & decomposed priority rubric | Shifted from all-or-nothing gates to sequential field filling with validated `label_source` provenance. Gate 2 fills category at 95.9% accuracy (k=1, θ=0.75). Injecting explicit ITIL impact/urgency rubrics into Gate 3 jumped priority accuracy from 43.3% to 66.7% (40.4% → 64.9% model-source). Diagnostic split revealed impact at 91.2% vs urgency at 61.4% (bounded by synthetic label noise where 18% of low-urgency tickets contain urgency language). Assignment group reached 80.0% against the 88.0% ceiling. Cascade cost: $2.33 / 1,000 tickets; mean latency 1548ms. |

---

## 11. Out of scope

Real ticket data of any kind · ServiceNow integration · any UI · deployment or hosting · agents or multi-step tool use · fine-tuning · anything that touches an employer system or account.

---

## 12. Glossary (short)

**Incident** broken thing, restore fast · **Service request** standard ask, nothing broken · **Problem** root cause behind incidents · **Change** planned modification with approval · **P1–P4** priority from impact × urgency · **L1/L2/L3** service desk / specialists / developers or vendor · **SLA** contracted response and resolution times · **Transition** handover of a customer estate to the AMS provider · **RCA** root cause analysis · **Known error** cause known, workaround documented · **Tcode** SAP transaction code · **Short dump** ABAP runtime crash (ST22) · **Transport** package carrying a change DEV → QAS → PRD
