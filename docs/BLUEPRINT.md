# BLUEPRINT — ticket-triage-lab

> Single source of truth for this project. Read it in full before doing any work.
> Humans and agents both work from this file. When reality changes, update this file in the same change.

Last updated: 2026-09-26

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
  llm.py        ChatModel, EmbeddingModel, TransientLLMError, cost_usd
  models.py     Pydantic models (human-written): TicketInput, LabelledTicket, TriageVerdict
  rules.py      Gate 1: deterministic rules, urgency veto & domain collision split
  similarity.py Gate 2: NumPy cosine similarity index, .npz caching, by_id lookup, partial unbundled emission
  classifier.py Gate 3: structured output via dynamic schemas, ITIL decomposed rubric, few-shot examples, temp=0.0
  cascade.py    Assembly line pipeline: rules -> similarity -> model, provenance merge
  evaluate.py   Wilson score intervals, 4x4 confusion matrix, severe over/under-calls, ceiling ratios
  search_index.py Azure AI Search backend mirroring SimilarityIndex (schema + upload + top-k); owns azure-search-documents SDK
  ceiling.py    label-noise accuracy ceilings computed from taxonomy.py
  probe.py      structured-output / logprob probes used on Wed 23
  utils.py      retry(), chunks()
  taxonomy.py   synthetic-data sampling rules as plain data
  datagen.py    synthetic ticket generator: uv run python -m triage.datagen [--dry-run] [--n N]
  smoke.py      live end-to-end check: uv run python -m triage.smoke
scripts/
  eval_cascade.py  full cascade runner over eval.jsonl / eval_p1.jsonl, Wilson intervals + confusion matrix
  eval_gate1.py    Gate 1 coverage/precision over history.jsonl
  eval_gate2.py    Gate 2 leave-one-out calibration sweep across (k, theta)
  audit_urgency.py per-ticket urgency adjudication dump -> data/eval_verdicts.jsonl
  audit_data.py    offline data audit for phrasing leaks and guardrail compliance
  verify_azure_search.py live parity check: Azure AI Search vs NumPy index (needs credentials)
data/           tickets.jsonl · history.jsonl · eval.jsonl · eval_p1.jsonl · history_index.npz · clusters.json · eval_verdicts.jsonl · REPORT.md
tests/          165 offline unit tests (100% mocked, zero network calls)
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
| Tue 22 | Pydantic models (human) · synthetic ticket generator (agent) | **Done** — models (32 tests green) · 240 history + 60 eval + 60 P1/P2 stress tickets generated live |
| Wed 23 | Azure OpenAI patterns · review and fix synthetic data | **Done** — structured outputs + logprobs verified live (D16–D18); data audit fixes (D12–D15) |
| Thu 24 | Gate 1 rules · Gate 2 similarity | **Done** — Gate 1 100% precision @ ~2.5% coverage; Gate 2 leave-one-out calibration sweep (k=1, θ=0.75 → 95.9% category accuracy) |
| Fri 25 | Gate 3 model classification · cascade wiring · Evaluation harness (pulled forward) | **Done** — assembly-line cascade with validated `label_source`, dynamic schema factory, ITIL decomposed rubric, 95% Wilson intervals, 4x4 confusion matrix, P1 stress validation (96.7% recall), temperature pinned to 0.0 |
| Sat 26 | Retrieval on Azure AI Search (fallback pgvector) | **Done** — `AzureSearchIndex` built + verified live (D26): 9/9 top-3 overlap vs numpy, cosine score formula `1/(2−cos)` confirmed, deliberately not wired into cascade |
| Sun 27 | **Rest** | — |
| Mon 28 | Evaluation harness (pulled forward to Fri 25) | **Completed early** — Mon freed for Document Intelligence (half day) + re-run eval to confirm temp=0.0 determinism |
| Tue 29 | README (pulled forward) · SAP BTP / ServiceNow orientation | Moved up from Wed 30 |
| Wed 30 | Buffer / tidy, delete nothing yet, rest | — |

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
| D21 | Gate 1 urgency veto and domain conflict split | Separated non-routine language into `_URGENCY_VETO` (severity words like "outage", "production halted") and `_DOMAIN_CONFLICT` (mentions of other tcodes/modules like "ME21N", "IDoc"). Normal helpdesk problem verbs ("unable", "error", "cannot") were removed from the veto after measuring that they collapsed coverage to ~0%. Result: 100% precision at ~2.5% honest coverage across all 4 fields. The ceiling is pattern recall, not the veto: most `access_authorization` tickets are phrased as incidents ("unable to access"), not provisioning requests — widening patterns adds more P3 mislabels than P4 gains. |
| D22 | Sequential assembly-line cascade & validated provenance | Shifted from all-or-nothing gates to an assembly-line cascade where gates collaborate. `TriageVerdict` was relaxed to allow partial label states and gained `decided_by="cascade"`. `label_source` was introduced as a validated Pydantic invariant (must name exactly the populated labels). The Gate 2 sweep first showed similarity lacks dynamic range on this corpus (top-1 cosines span ~0.65–0.88, mass at 0.70–0.85 — too homogeneous for distance alone to separate), so Gate 2 unbundled its emissions: k=1, θ=0.75 emits category at 95.9% accuracy while leaving priority open. Gate 3 dynamically constructs Pydantic schemas (`create_model`) omitting locked fields, structurally preventing the LLM from overriding earlier gates. Azure D17 schema silence handled via defensive `rationale[:300]` truncation. |
| D23 | Decomposed priority rubric & ground-truth label noise | Priority is never asked directly of Gate 3; it predicts `impact: Level` and `urgency: Level`, and Python mechanically computes `derive_priority()`. Injecting explicit ITIL rubrics into `_SYSTEM_PROMPT` jumped priority from 43.3% to 66.7% (40.4% → 64.9% model-source). Diagnostic split showed impact at 93.0% vs urgency at 61.4%. Manual adjudication of 21 urgency misses revealed that ~half were synthetic label faults caused by a structural conflict in `datagen.py` (`IMPACT_SITUATIONS[high]` generates complete-stoppage text despite a sampled `urgency=low`). Real model error isolated to scope-to-urgency bleed (addressed via anti-bleed prompt lines); effective urgency accuracy ≈79%. |
| D24 | Few-shot in-context learning null result | Passed Gate 2's consulted historical neighbours (incl. dissenters — contrastive signal) into Gate 3's prompt with their resolved labels via `SimilarityIndex.by_id` O(1) lookup. Measurement showed no statistically detectable accuracy change (all deltas ≤1.8 pts on n=57–60, well inside the ±12-pt Wilson interval) while cost rose +23% ($2.33 → $2.86/1k). Kept in code as structural grounding, but documented as an empirical null result consistent with the adjudication finding: retrieval context cannot overcome ground-truth label noise; remaining model error is ~5 tickets, below eval resolution. Side effect: `eval_gate2.py` twin% now means "twin among consulted neighbours" — a more honest memorization measure. |
| D25 | Statistical evaluation harness, non-determinism fix, and P1 safety | Implemented `evaluate.py`: 95% Wilson score confidence intervals, 4x4 confusion matrix, ceiling ratios, severe under-call detection (P1→P3/P4, P2→P4), and severe over-call detection (P3/P4→P1, P4→P2). Discovered evaluation non-determinism caused a ±6.6-point sampling swing between identical runs; pinned `temperature=0.0` at the Gate 3 call site for reproducible evaluations. Realistic eval (n=60): assignment group 78.3% (89.0% of the 88.0% ceiling); type 71.7% (79.7% of the 90.0% ceiling); category 95.0%; priority 68.3% with 0 severe under-calls and 4 severe over-calls. P1 stress set (n=60): 96.7% P1 recall (29/30 caught as P1, 1 as P2, none dropped below P2), 82.9% P1 precision — graded honestly on a set containing no low-severity tickets, so it partly rewards the model's escalation bias; 1 severe under-call across 120 total evaluations. |
| D26 | Azure AI Search backend verified at parity | `search_index.py` mirrors `SimilarityIndex` (create schema → upload → top-k) behind the same SDK-boundary rule as `llm.py`; `azure-search-documents` imported only there. Live verification (`scripts/verify_azure_search.py`, 240 docs): **9/9 top-3 rank overlap** vs exact NumPy — HNSW is effectively exact at this scale. Score semantics confirmed empirically, not from docs: Azure rescales cosine as `1/(2−cos)` — recover raw cosine as `2−1/score` (e.g. 0.7620 → 0.8077); values NOT comparable to numpy cosine, ordering identical. Latency ~700–1200ms hosted vs <3ms local — hosted buys scale/ops, not speed. Gotcha fixed live: `Edm.DateTimeOffset` rejects naive ISO strings; `_to_document` appends `Z` (dataset timestamps are UTC). Kept as a second interchangeable backend, deliberately NOT wired into the cascade — swapping the default buys nothing at n=240. |

---

## 11. Out of scope

Real ticket data of any kind · ServiceNow integration · any UI · deployment or hosting · agents or multi-step tool use · fine-tuning · anything that touches an employer system or account.

---

## 12. Glossary (short)

**Incident** broken thing, restore fast · **Service request** standard ask, nothing broken · **Problem** root cause behind incidents · **Change** planned modification with approval · **P1–P4** priority from impact × urgency · **L1/L2/L3** service desk / specialists / developers or vendor · **SLA** contracted response and resolution times · **Transition** handover of a customer estate to the AMS provider · **RCA** root cause analysis · **Known error** cause known, workaround documented · **Tcode** SAP transaction code · **Short dump** ABAP runtime crash (ST22) · **Transport** package carrying a change DEV → QAS → PRD
