# ticket-triage-lab: AI-Based Ticket Triage for SAP AMS

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev/)
[![Azure OpenAI](https://img.shields.io/badge/azure--openai-2024--10--21-orange.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![Azure AI Search](https://img.shields.io/badge/azure--ai--search-HNSW-purple.svg)](https://azure.microsoft.com/en-us/products/ai-services/ai-search)
[![Azure Document Intelligence](https://img.shields.io/badge/azure--docintel-markdown-blueviolet.svg)](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)
[![Tests](https://img.shields.io/badge/tests-182%20passed-brightgreen.svg)](tests/)

A working prototype of automated ticket triage for an SAP Application Management Services (AMS) desk, built in ten days to develop hands-on fluency with Azure OpenAI, vector retrieval, structured outputs, and statistical evaluation.

**All tickets in this repository are synthetic.** The measurements are honest, the failures are analysed, and the architectural limitations are documented.

Given raw, unstructured ticket text, the system predicts:

1. **Ticket type** — `incident`, `service_request`, `problem`, `change`
2. **Category** — 12 fixed SAP domain categories (`batch_job`, `short_dump`, `invoice_posting`, and so on)
3. **Priority** — P1 to P4, derived deterministically from ITIL impact × urgency
4. **Assignment group** — `Basis`, `ABAP`, `FI`, `SD`, `MM`, `Security`, `Integration`, `Service Desk`

Built without high-level orchestration frameworks (no LangChain, LlamaIndex, or Semantic Kernel) in order to learn the raw mechanics: Pydantic v2 data contracts, native Azure SDK clients, empirical calibration sweeps, and a statistical evaluation harness.

---

## 1. What I found

The most valuable outcomes were not the accuracy figures but the results that contradicted what I expected going in.

**1. Embedding retrieval beat the LLM on every contested field.**
Where both competed on the held-out set, vector retrieval outperformed `gpt-4o`: type 90.3% (n=31) against 46.2% (n=26), assignment group 87.9% (n=33) against 62.5% (n=24). For category, retrieval reached 95.9% (n=49) while the model scored 87.5% on the remaining 8 tickets — too small a sample to carry weight, so type and group carry the evidence. The expensive generative gate was the weakest one precisely because it only inherited the boundary cases retrieval declined to settle.

**2. Priority was a rubric problem, not a reasoning problem.**
`gpt-4o` scored 43.3% on priority until I realised it was being asked to apply a scale it had never been shown. Writing the impact and urgency definitions into the system prompt — one user versus a team versus a site; work stopped now versus a deadline this week — moved priority to 66.7%, a gain of 23.4 points, settling at 68.3% on the deterministic run. Deriving priority in Python from the two axes also absorbs error: an off-by-one urgency slip (low impact × medium urgency instead of low × low) still resolves to the correct P4.

**3. Few-shot examples were an expensive null result.**
Passing Gate 2's retrieved neighbours into Gate 3's prompt is standard practice and seemed certain to help. Measured, accuracy moved by at most 1.8 points — well inside the ±12-point confidence interval at this sample size — while cost rose 20% ($2.33 → $2.80 per 1,000 tickets). Retrieval context could not overcome noise in the ground truth.

**4. The evaluation itself was non-deterministic, and the noise exceeded the signal.**
Identical code over the same 60 tickets produced a 6.6-point swing in headline accuracy before `temperature=0.0` was pinned. That swing was larger than any prompt change being measured. On a small evaluation set, sampling variance dwarfs tuning deltas.

**5. Roughly half the urgency errors were bad labels, not bad predictions.**
Hand-adjudicating 21 urgency misses found 10 were faults in my own synthetic data, caused by a conflict inside the generator: the high-impact phrasing block produced complete-stoppage text even when urgency was sampled as low. The model was reading the text correctly; the label disagreed with it. Effective urgency accuracy is closer to 79% than the 61% first reported.

**6. Azure silently accepts schema constraints it does not enforce.**
Azure OpenAI accepts JSON Schema validation keywords (`max_length`, `pattern`, numeric bounds) without an HTTP 400 and then ignores them. There is no error to catch. Client-side Pydantic validation and defensive truncation are not optional.

**7. Azure AI Search does not return raw cosine.**
Against exact NumPy search over the same 240 vectors, Azure AI Search produced identical top-3 rankings on every query (9 of 9) — at this index size HNSW is effectively exact, and divergence would only be expected at much larger scale. The scores differ, though. Verified empirically on API version `2024-10-21`: Azure computes relevance from cosine distance as `score = 1 / (2 - cos)`, so raw cosine is recovered as `cos = 2 - 1/score`, matching NumPy to within 1e-4 across all test queries.

**8. Letting the service do the layout work beat parsing it myself.**
Parsing Document Intelligence's raw layout AST revealed that Azure emits every table cell twice — once in the structured grid and again as a standalone paragraph. A 5×5 SLA table produced 25 spurious micro-paragraphs, and semantic search latched onto a fragment containing only "P1", detached from its resolution time. I solved it with span-containment geometry, then deleted that code: requesting Markdown output (`output_content_format=DocumentContentFormat.MARKDOWN`) lets Azure resolve reading order and table structure server-side. Chunking becomes semantic header splitting, chunk count fell from 31 to 5, and SLA tables stay intact inside their section.

---

## 2. Metrics

All figures come from the deterministic run (`temperature=0.0`), reported with 95% Wilson score intervals so the sample size stays visible.

### Realistic distribution — held-out evaluation set, n=60

| Field | Accuracy | 95% interval | Against ceiling |
|---|---|---|---|
| Category | **95.0%** | 86.3% – 98.3% | no computed ceiling |
| Assignment group | **78.3%** | 66.4% – 86.9% | 89.0% of the 88.0% ceiling |
| Ticket type | **71.7%** | 59.2% – 81.5% | 79.7% of the 90.0% ceiling |
| Priority | **68.3%** | 55.8% – 78.7% | derived from impact × urgency |
| All four correct | **38.3%** | 27.1% – 51.0% | joint verdict |

Ceilings are computed from the label taxonomy itself, not estimated. Assignment group caps at 88% because several categories legitimately route to two teams; type caps at 90% because problem/incident and change/service_request are not distinguishable from text alone.

### Priority confusion and safety

- **Severe under-calls:** 0 on the realistic set; 1 across 120 total evaluations (one P2 called as P4). Burying an active outage in a low-priority queue is the failure that costs an SLA breach.
- **Severe over-calls:** 4 on the realistic set (P3 tickets escalated to P1). Over-calling costs engineer hours and unnecessary incident bridges. The system errs toward escalation — safer in AMS, but not free.

### High-severity stress set — n=60 (30 P1, 30 P2)

- **P1 recall: 96.7%** — 29 of 30 caught as P1, one slipped to P2, none dropped to P3 or P4.
- **P1 precision: 82.9%**
- **Caveat:** this set contains no P3 or P4 tickets. A model biased toward escalation scores well on it by construction, so the figure partly rewards the bias visible in the confusion matrix above. The realistic set is the honest severity measure.

### Cost and latency

- **$2.80 per 1,000 tickets** — $0.0028 per ticket, about $280 per 100,000.
- **Mean latency 1,317 ms** (p50 1,317 ms, p95 1,780 ms).
- *Illustrative only:* at €50/hour and three minutes of manual triage, 100,000 tickets is roughly 5,000 engineer hours. Against that assumption the API spend is under 0.1% of the labour cost. The assumption is mine, not a measured figure.

---

## 3. Architecture

The design moved from an all-or-nothing funnel — where an uncertain gate discards everything it knew — to an assembly line, where each gate locks the fields it can prove and passes the rest downstream.

```
  ticket text
       |
       v
  +---------------------------+
  | Gate 1 — rules            |   unambiguous routine request?
  | regex, <1ms, $0           |   yes -> complete verdict, done
  +---------------------------+
       | no / partial
       v
  +---------------------------+
  | Gate 2 — vector retrieval |   top-1 similarity >= 0.75?
  | Azure AI Search / NumPy   |   yes -> locks category (95.9%)
  +---------------------------+   priority left open: not predictable from vectors
       | remaining fields
       v
  +---------------------------+
  | Gate 3 — gpt-4o           |   receives locked fields as context
  | structured output         |   predicts what is missing, plus impact & urgency
  +---------------------------+
       |
       v
  priority = derive_priority(impact, urgency)   <- computed in Python, never asked
       |
       v
  TriageVerdict(decided_by="cascade", label_source={...})
```

### Three invariants

**Validated provenance.** Every populated label names the gate that produced it, enforced as a Pydantic invariant — `label_source` must match the populated fields exactly, or the verdict does not validate. Auditability is not optional under an SLA contract.

```json
{
  "ticket_id": "INC000055",
  "category": "batch_job",
  "assignment_group": "Basis",
  "type": "incident",
  "priority": "P3",
  "decided_by": "cascade",
  "label_source": {
    "category": "similarity",
    "assignment_group": "similarity",
    "type": "model",
    "priority": "model"
  }
}
```

**Structural anchoring.** Rather than asking the model not to overwrite earlier decisions, Gate 3 builds its response schema at runtime with `create_model`, omitting the locked fields entirely. The model cannot contradict Gate 2 because the fields are not in the schema it is answering.

**Priority is computed, not guessed.** Gate 3 returns impact and urgency; Python derives priority from the matrix below — the same mechanism ServiceNow uses.

| impact \ urgency | high | medium | low |
|---|---|---|---|
| **high** | P1 | P2 | P3 |
| **medium** | P2 | P3 | P4 |
| **low** | P3 | P4 | P4 |

---

## 4. Running it

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/), an Azure OpenAI deployment (`gpt-4o` and `text-embedding-3-small`), and — for the optional verification scripts — Azure AI Search and Azure AI Document Intelligence on their free tiers.

```bash
# install
git clone https://github.com/<your-username>/ticket-triage-lab.git
cd ticket-triage-lab
uv sync

# configure
cp .env.example .env      # then fill in your Azure endpoints and keys

# tests and lint — fully offline, no network calls
uv run pytest
uv run ruff check .
uv run ruff format --check .

# reproduce the cascade evaluation (deterministic)
uv run python scripts/eval_cascade.py

# reproduce the high-severity stress run
uv run python scripts/eval_cascade.py --eval data/eval_p1.jsonl

# optional live checks (need Azure credentials)
uv run python scripts/verify_azure_search.py
uv run python scripts/verify_docintel.py
```

---

## 5. What I would do differently for production

Written as open questions rather than conclusions — the answers depend on facts about a real estate that a prototype cannot supply.

### Gate 1 is the weakest part

Hardcoded regex over free text, at 2.5% coverage, is a learning exercise rather than production software. It will not survive multilingual tickets from thousands of users across multiple clients, and maintaining keyword lists in Python across customers is an operational burden nobody wants to own.

In a real ITSM platform that deterministic layer belongs somewhere else entirely. Rules fire on **structured metadata** — intake channel, monitoring alert, system ID, affected CI — not on narrative text, and they live in ServiceNow Business Rules or Flow Designer where the service desk can maintain them. Routine password resets and access requests are also largely deflected by self-service before a ticket exists at all; free text reaching a triage engine usually means self-service already failed.

### Supervised classification on closed ticket history

An AMS desk handling 100,000 tickets a year is already sitting on 100,000 labelled examples. For a high volume with a reasonably stable taxonomy, a trained classifier — fine-tuned transformer, or even TF-IDF with logistic regression — runs in single-digit milliseconds at essentially zero inference cost, against $2.80 per thousand for the LLM. The real cost moves to training and MLOps rather than per-call spend.

The obvious objection is that a new client has no clean training data on day one. A staged rollout answers it:

1. **Shadow.** Predict on live tickets, write the verdict to a field nobody routes on, measure agreement with human dispatchers for a few weeks. No risk, and it yields the first honest accuracy figure on real tickets.
2. **Suggest.** Pre-fill the verdict in the ServiceNow form for the engineer to accept or override. Every ticket now generates a deliberate supervised label — far better than a historical label produced by someone clicking through a form.
3. **Train.** Once enough verified examples accumulate, train the cheap classifier and make it the default; the LLM becomes the low-confidence fallback.
4. **Close the loop.** Overrides keep feeding the training set. Retrain on a schedule, monitor for drift.

**One caution, learned here the hard way:** do not train on raw historical labels without auditing them. Hand-adjudicating 21 tickets in this project showed roughly half the apparent errors were bad ground truth — in data I generated myself, with explicit guardrails. Real ticket history is worse: engineers re-route without changing the category, priority gets inflated by whoever phrased the request most urgently, and taxonomies drift as teams reorganise. A defensible starting filter is to train only on tickets that closed without re-assignment, and to sample-audit before trusting anything.

### Retrieval or training is a genuine trade-off, not a settled question

- **Retrieval** adapts instantly. A new category, a new SAP module, a new client needs only new vectors in an index — no retraining, no redeployment, and it degrades gracefully when a category has only a handful of examples.
- **A trained classifier** is cheaper per call and typically more accurate on a stable taxonomy, but needs an MLOps loop and probably one model per customer taxonomy, with the versioning, monitoring and rollback that implies.

The deciding factors are taxonomy stability and client count, and neither is knowable from outside. How often do client team structures actually change? That is a week-one question against real estates, not something to settle in advance.

### What is absent entirely

- **Feedback loop.** Nothing turns a human re-assignment into a new training example or retrieval entry.
- **Drift monitoring.** The evaluation harness runs offline. Production needs monitoring for distribution shift after a model update, a corpus change, or a taxonomy reorganisation.
- **Hierarchical classification.** Real ITSM taxonomies are trees — category → subcategory → item. The flat enum here is a simplification.
- **Resilience.** The retry decorator has no exponential backoff or jitter and ignores Azure's `Retry-After` header. Adequate for supervised lab runs; not for unattended batch work.

---

## 6. Limitations

- **Synthetic data.** All 300 tickets were generated with `gpt-4o`. A structural conflict was diagnosed in the high-impact / low-urgency cell, where stoppage phrasing was produced under a low-urgency label. This caps demonstrable urgency accuracy and is documented rather than fixed.
- **Sample size.** The evaluation set is 60 tickets, giving roughly a ±12-point margin at 95% confidence. Differences smaller than that are not measurable here. Meaningful resolution would need several hundred cases per slice.
- **Gate 3 confidence is a placeholder.** Gate 2 emits a similarity-derived confidence; Gate 3's is a hardcoded 0.80. Native logprob confidence was verified in isolated probes but not generalised across the dynamic multi-field schemas.
- **No real data.** No customer tickets, no production ServiceNow instance, no employer system was touched at any point.

---

## 7. Repository layout

```
data/
  tickets.jsonl          300 synthetic SAP tickets
  history.jsonl          240 resolved tickets — the Gate 2 index
  eval.jsonl             60 held-out evaluation tickets
  eval_p1.jsonl          60 high-severity stress tickets (30 P1, 30 P2)
  eval_verdicts.jsonl    persisted per-ticket verdicts from the eval run
  history_index.npz      cached 240×1536 embedding matrix
  clusters.json          category cluster assignments for the corpus
  REPORT.md              generated data-distribution report
  ams_sla_runbook.pdf    sample SLA runbook with a 5x5 table
  ams_sla_runbook.md     Markdown emitted by Document Intelligence

scripts/
  eval_cascade.py        cascade evaluation, Wilson intervals, confusion matrix
  eval_gate1.py          Gate 1 coverage and precision
  eval_gate2.py          leave-one-out calibration sweep across k and threshold
  audit_data.py          offline data audit for phrasing leaks
  audit_urgency.py       per-ticket urgency adjudication dump
  verify_azure_search.py live parity check against the NumPy index
  verify_docintel.py     live layout extraction and retrieval check
  generate_test_pdf.py   builds the sample SLA runbook

src/triage/
  config.py              settings and environment validation
  models.py              Pydantic domain schemas and ITIL enums
  llm.py                 Azure OpenAI SDK boundary, cost tracking
  rules.py               Gate 1 — deterministic rules
  similarity.py          Gate 2 — cosine index, partial emission
  classifier.py          Gate 3 — dynamic schemas, structured output
  cascade.py             assembly-line wiring and provenance merge
  evaluate.py            Wilson intervals, confusion matrix, ceiling ratios
  search_index.py        Azure AI Search HNSW backend
  documents.py           Document Intelligence Markdown chunking
  ceiling.py             accuracy ceilings computed from the taxonomy
  taxonomy.py            synthetic-data sampling rules
  datagen.py             synthetic ticket generator
  probe.py               live Azure capability probes
  smoke.py               live end-to-end connectivity check
  utils.py               shared helpers

tests/                   182 offline unit tests, zero network calls
docs/BLUEPRINT.md        design notes and the full decision log
```

---

## Licence

MIT. Built as personal preparation before starting an AI engineering role; not affiliated with or endorsed by any employer, and containing no proprietary data.
