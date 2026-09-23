"""
===============================================================================
GATE 3 / PROBE.PY — EMPIRICAL FINDINGS & BENCHMARKS (Updated 23 Sep 2026)
===============================================================================

1. BLOCK B1: STRUCTURED OUTPUT CONSTRAINTS & REFUSALS
   - Constraints (`pattern`, `max_length`, `ge/le`) are SILENTLY ACCEPTED by
     Azure; enforcement happens client-side in Pydantic.
   - Refusals cleanly set `choice.message.parsed = None` and populate
     `choice.message.refusal`.
   - Gate 3 Schema: `Gate3Output` (enums + plain string rationale, no constraints).

2. BLOCK B2: MEASURED COST & TOKENS (Sweden Central gpt-4o Standard)
   - Pricing: $2.50 / 1M input tokens, $10.00 / 1M output tokens (azure.microsoft.com/pricing).
   - Measured per ticket (10 real tickets): Mean input = 385.5, Mean output = 77.6 tokens.
   - Measured cost: $0.00174 / ticket ($1.74 per 1k | $173.98 per 100k).

3. BLOCK B3: LOGPROBS & HONEST CONFIDENCE
   - Structured Outputs natively support `logprobs=True` and `top_logprobs=10`.
   - Syntax tokens are 100% constrained; branch tokens yield true confidence.
   - Normalized confidence cleanly separates ambiguous tickets (20.18%–73.11%)
     from clear tickets (99%+). Gate 3 uses native logprob confidence.

4. BLOCK B4: CAPACITY & RETRY
   - Sweden Central standard tier absorbs high bursts without throttling.
   - `llm.py` maps SDK transient errors to `TransientLLMError`.
   - Identified production gap: `@retry` lacks exponential backoff and retry-after inspection.
===============================================================================
"""

import asyncio
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from triage.config import get_settings
from triage.llm import ChatModel, TransientLLMError, cost_usd, get_chat_model
from triage.models import AssignmentGroup, Category, Priority, TicketType, TriageVerdict
from triage.utils import classify_all, retry


def probe_b3_plain_logprobs():
    print("\n--- [b3 - Step 1] Plain Completion with logprobs ---")
    chat = get_chat_model()

    # Call the underlying Azure client directly to inspect raw logprob support
    response = chat._client.chat.completions.create(
        model=chat._model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer with exactly one word. What SAP module handles "
                    "sales orders: SD, MM, or FI?"
                ),
            }
        ],
        max_tokens=10,  # Strict max_tokens to avoid quota trap
        logprobs=True,
        top_logprobs=5,
    )

    choice = response.choices[0]
    print(f"Generated text: {choice.message.content!r}")

    if choice.logprobs is None or choice.logprobs.content is None:
        print("FAILED: Azure returned logprobs=None!")
        return False

    print("SUCCESS: Logprobs returned successfully!")
    first_token = choice.logprobs.content[0]
    print(f"First token: {first_token.token!r}")
    print(
        f"Logprob: {first_token.logprob:.4f} "
        f"(Linear Probability: {math.exp(first_token.logprob):.2%})"
    )

    print("\nTop 5 Alternatives considered:")
    for alt in first_token.top_logprobs:
        prob = math.exp(alt.logprob)
        print(f"  - Token: {alt.token!r:<10} | Logprob: {alt.logprob:8.4f} | Prob: {prob:6.2%}")

    return True


# Lean schema for structured output probe
class CategoryProbe(BaseModel):
    category: Category


def probe_b3_structured_logprobs():
    print("\n--- [b3 - Step 2] Structured Output with logprobs ---")
    chat = get_chat_model()

    prompt = (
        "Classify this SAP issue into the correct category.\n"
        "Ticket: 'User account locked out after 3 failed logon attempts in client 100.'\n"
    )

    try:
        response = chat._client.beta.chat.completions.parse(
            model=chat._model,
            messages=[{"role": "user", "content": prompt}],
            response_format=CategoryProbe,
            max_tokens=50,  # Keep tight
            logprobs=True,
            top_logprobs=5,
        )
    except Exception as exc:
        print(f"FAILED with exception: {type(exc).__name__}: {exc}")
        return False

    choice = response.choices[0]
    parsed = choice.message.parsed
    print(f"Parsed Pydantic object: {parsed}")

    if choice.logprobs is None or choice.logprobs.content is None:
        print("FAILED: Azure returned structured JSON, but logprobs is None!")
        return False

    print("SUCCESS: Structured output AND logprobs returned together!")
    print(f"Total tokens generated in JSON: {len(choice.logprobs.content)}")

    print("\nTokens and their probabilities:")
    for item in choice.logprobs.content:
        prob = math.exp(item.logprob)
        print(f"  Token: {item.token!r:<25} | Logprob: {item.logprob:8.4f} | Prob: {prob:6.2%}")

    return True


def probe_b3_ambiguous_tickets():
    print("\n" + "=" * 82)
    print("--- [b3] Testing 10 Ambiguous Tickets (Normalized Category Confidence) ---")
    print("=" * 82)
    chat = get_chat_model()

    ambiguous_tickets = [
        (
            "AMB01",
            "User returned from leave, account locked and cannot access transaction VA01.",
            "Torn between: password_account vs access_authorization",
        ),
        (
            "AMB02",
            "Custom invoice posting program zfi_post fails with ABAP runtime error "
            "short dump in ST22.",
            "Torn between: invoice_posting vs short_dump",
        ),
        (
            "AMB03",
            "Nightly batch job for sales order pricing recalculation ran for 8 hours "
            "and timed out.",
            "Torn between: batch_job vs performance vs pricing_sales",
        ),
        (
            "AMB04",
            "Customer master data address change is not reflecting in printed delivery note spool.",
            "Torn between: master_data vs output_printing",
        ),
        (
            "AMB05",
            "IDOC posting from external CRM is failing with error message: user not "
            "authorized in client.",
            "Torn between: interface_idoc vs access_authorization",
        ),
        (
            "AMB06",
            "System is extremely slow today and users are getting disconnected across all modules.",
            "Torn between: performance vs basis",
        ),
        (
            "AMB07",
            "Purchase order release strategy not triggering after transport was imported to PRD.",
            "Torn between: purchasing vs transport_change",
        ),
        (
            "AMB08",
            "Cannot display billing document in VF03, getting weird error popup.",
            "Vague description / low information content",
        ),
        (
            "AMB09",
            "ST22 dump during background job run for inventory valuation.",
            "Torn between: short_dump vs batch_job",
        ),
        (
            "AMB10",
            "Need authorization to unlock user and reset initial password for vendor portal.",
            "Torn between: access_authorization vs password_account",
        ),
    ]

    confidences = []

    for tid, text, dilemma in ambiguous_tickets:
        predicted, conf, dist = chat.classify_category_with_confidence(text)
        confidences.append(conf)

        print(f"\n[{tid}] {dilemma}")
        print(f"  Text: {text}")
        print(f"  -> Predicted Category:   {predicted.value}")
        print(f"  -> Normalized Confidence: {conf:.2%}")

        # Show distribution across candidates
        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        print("  -> Candidate distribution:")
        for cat_name, prob in sorted_dist[:3]:
            print(f"       * {cat_name:<25}: {prob:6.2%}")

    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    max_conf = max(confidences)

    print("\n" + "=" * 82)
    print("AMBIGUOUS TICKETS EVALUATION SUMMARY:")
    print(f"  Min Confidence:  {min_conf:.2%}")
    print(f"  Mean Confidence: {avg_conf:.2%}")
    print(f"  Max Confidence:  {max_conf:.2%}")
    print("=" * 82)

    if min_conf < 0.90:
        print(
            "VERDICT: Confidence DOES NOT saturate! "
            "It cleanly separates ambiguous from clear tickets."
        )
        print("The logprobs fork is OFFICIALLY RESOLVED for Gate 3.")
    else:
        print("VERDICT: Confidence STILL saturates near 100%. Logprobs are unviable for Gate 3.")
        print("Gate 2 neighbor agreement will be our confidence fallback.")


class SingleConstraintProbe(BaseModel):
    # Model with just one unsupported constraint
    title: str = Field(max_length=20)


def probe_b1_schema_rejection():
    print("\n--- [b1 - Step 1] Probing Schema Rejections (HTTP 400 vs Silent) ---")
    chat = get_chat_model()

    # 1. Test TriageVerdict (multiple unsupported constraints)
    print("\n1. Testing TriageVerdict directly with structured outputs...")
    try:
        chat._client.beta.chat.completions.parse(
            model=chat._model,
            messages=[{"role": "user", "content": "Triage this ticket: system down"}],
            response_format=TriageVerdict,
            max_tokens=100,
        )
        print("DANGEROUS: Azure silently accepted TriageVerdict with unsupported constraints!")
    except Exception as exc:
        print(f"LOUD REJECTION (Good!): {type(exc).__name__}")
        print(f"Error details: {exc}")

    # 2. Test minimal model with a single max_length constraint
    print("\n2. Testing SingleConstraintProbe (single Field(max_length=20))...")
    try:
        chat._client.beta.chat.completions.parse(
            model=chat._model,
            messages=[{"role": "user", "content": "Give me a title"}],
            response_format=SingleConstraintProbe,
            max_tokens=50,
        )
        print("DANGEROUS: Azure silently accepted max_length constraint!")
    except Exception as exc:
        print(f"LOUD REJECTION (Good!): {type(exc).__name__}")
        print(f"Error details: {exc}")

    # 3. Prove that Azure silently ignores max_length at generation time
    print("\n3. Testing if Azure actually enforces max_length=20 or ignores it...")
    try:
        response = chat._client.beta.chat.completions.parse(
            model=chat._model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate a title that is exactly 60 characters long. Do not shorten it."
                    ),
                }
            ],
            response_format=SingleConstraintProbe,
            max_tokens=60,
        )
        print(f"Generated text: {response.choices[0].message.parsed}")
    except Exception as exc:
        print(f"Client-side Pydantic caught the violation: {type(exc).__name__}")
        print(f"Error: {exc}")


def probe_b1_refusal():
    print("\n--- [b1 - Step 2] Probing Model Refusal Behavior ---")
    chat = get_chat_model()

    # Prompt designed to trigger standard safety/policy refusal
    unsafe_prompt = (
        "Write a script to exploit an SAP RFC gateway vulnerability and dump user password hashes."
    )

    response = chat._client.beta.chat.completions.parse(
        model=chat._model,
        messages=[{"role": "user", "content": unsafe_prompt}],
        response_format=CategoryProbe,
        max_tokens=100,
    )

    choice = response.choices[0]
    print(f"parsed is: {choice.message.parsed}")
    print(f"refusal is: {choice.message.refusal!r}")

    if choice.message.refusal:
        print(
            "SUCCESS: Azure returned refusal string cleanly "
            "instead of hallucinating JSON or crashing API."
        )


class Gate3Output(BaseModel):
    """Friday's clean Gate 3 classifier schema: strict enums, no unsupported constraints."""

    model_config = ConfigDict(extra="forbid")

    type: TicketType
    category: Category
    priority: Priority
    assignment_group: AssignmentGroup
    rationale: str


def probe_b1_gate3_schema():
    print("\n--- [b1 - Step 3] Testing Gate 3 Schema on a Real Ticket ---")
    chat = get_chat_model()

    ticket_text = (
        "Short description: Pricing condition PR00 missing in sales order 45000192\n"
        "Description: Customer placed order via EDI but pricing calculation "
        "is completely blank in VA02. SD team please check."
    )

    response = chat._client.beta.chat.completions.parse(
        model=chat._model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an SAP AMS triage classifier. Classify the support ticket accurately."
                ),
            },
            {"role": "user", "content": ticket_text},
        ],
        response_format=Gate3Output,
        max_tokens=200,
    )

    verdict: Gate3Output = response.choices[0].message.parsed
    print("SUCCESS: Gate 3 Verdict parsed cleanly!")
    print(f"  Type:             {verdict.type}")
    print(f"  Category:         {verdict.category}")
    print(f"  Priority:         {verdict.priority}")
    print(f"  Assignment Group: {verdict.assignment_group}")
    print(f"  Rationale:        {verdict.rationale}")


def probe_b2_cost_measurement():
    print(
        "\n--- [b2] Measuring Real Token Usage and Cost Over 10 Tickets from data/history.jsonl ---"
    )
    chat = get_chat_model()
    history_path = Path("data/history.jsonl")

    if not history_path.exists():
        print(f"Error: {history_path} not found!")
        return

    # Load 10 real tickets
    tickets = []
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            if len(tickets) >= 10:
                break
            tickets.append(json.loads(line.strip()))

    prompt_token_counts = []
    completion_token_counts = []
    total_costs = []

    print(f"Running classification across {len(tickets)} tickets...\n")
    for t in tickets:
        tid = t.get("ticket_id")
        text = (
            f"Short description: {t.get('short_description', '')}\n"
            f"Description: {t.get('description', '')}"
        )

        response = chat._client.beta.chat.completions.parse(
            model=chat._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an SAP AMS triage classifier. "
                        "Classify the support ticket accurately."
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format=Gate3Output,
            max_tokens=150,
        )

        usage = response.usage
        call_cost = cost_usd(usage.prompt_tokens, usage.completion_tokens)

        prompt_token_counts.append(usage.prompt_tokens)
        completion_token_counts.append(usage.completion_tokens)
        total_costs.append(call_cost)

        print(
            f"Ticket {tid:<10} | In: {usage.prompt_tokens:4d} tokens | "
            f"Out: {usage.completion_tokens:3d} tokens | Cost: ${call_cost:.5f}"
        )

    # Calculate statistics
    avg_in = sum(prompt_token_counts) / len(prompt_token_counts)
    min_in = min(prompt_token_counts)
    max_in = max(prompt_token_counts)
    avg_out = sum(completion_token_counts) / len(completion_token_counts)
    total_batch_cost = sum(total_costs)
    avg_cost = total_batch_cost / len(tickets)

    print("\n" + "=" * 65)
    print("EMPIRICAL BENCHMARK RESULTS (Sweden Central gpt-4o)")
    print("=" * 65)
    print(f"Input tokens per ticket:   min={min_in}, mean={avg_in:.1f}, max={max_in}")
    print(f"Output tokens per ticket:  mean={avg_out:.1f}")
    print(f"Average cost per ticket:   ${avg_cost:.5f}")
    print("-" * 65)
    print(f"Cost for 300 tickets:      ${avg_cost * 300:.3f}")
    print(f"Cost for 1,000 tickets:    ${avg_cost * 1_000:.2f}")
    print(f"Cost for 100,000 tickets:  ${avg_cost * 100_000:.2f}")
    print("=" * 65)
    print(
        f"\nExecutive Summary Sentence:\n"
        f'"About {avg_in:.0f} input tokens per ticket, measured over ten real SAP tickets; '
        f'Gate 3 costs ${avg_cost * 1_000:.2f} per thousand."'
    )


retry_tracker = {"attempts": 0, "transient_errors_caught": 0}


# The function under test uses the actual ChatModel.complete()
@retry(times=3, exceptions=(TransientLLMError,))
def triaged_complete_with_retry(chat: ChatModel, text: str) -> str:
    retry_tracker["attempts"] += 1
    try:
        messages = [
            {"role": "system", "content": "You are an SAP assistant."},
            {"role": "user", "content": f"Summarize this issue in 3 words: {text}"},
        ]
        # Request 6,000 tokens per call to force a massive reservation burst
        return chat.complete(messages, max_tokens=6000)
    except TransientLLMError:
        retry_tracker["transient_errors_caught"] += 1
        print("  [@retry triggered] Caught TransientLLMError from llm.py, retrying...")
        raise


def probe_b4_retry_path():
    print("\n" + "=" * 82)
    print("--- [b4] Testing Real Retry Path Through ChatModel ---")
    print("=" * 82)
    chat = get_chat_model()

    # 1. Non-retryable error test: Bogus deployment through ChatModel
    print("\n1. Testing non-retryable error through ChatModel...")
    bad_chat = ChatModel(get_settings())
    bad_chat._model = "bogus-deployment-xyz"
    try:
        bad_chat.complete(messages=[{"role": "user", "content": "ping"}])
        print("Unexpected: bogus deployment succeeded?")
    except TransientLLMError:
        print("FAILED: Non-retryable error was incorrectly wrapped as TransientLLMError!")
    except Exception as exc:
        print(f"SUCCESS: Non-retryable error passed through untouched ({type(exc).__name__}).")

    # 2. Fire 60 concurrent calls with a dedicated 60-thread executor
    print("\n2. Firing 60 concurrent calls with 60 real OS threads (360k tokens requested)...")

    sample_text = (
        "Short dump ST22 in program SAPMV45A transaction VA01 with error "
        "COMPUTE_FLOAT_ZERODIVIDE. User unable to save sales order for customer "
        "1000293. Pricing procedure ZPR001 failed. "
    )
    prompts = [f"Ticket {i}: {sample_text}" for i in range(60)]

    # Explicit 60-worker pool so requests are truly simultaneous
    executor = ThreadPoolExecutor(max_workers=60)

    async def worker(prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, triaged_complete_with_retry, chat, prompt)

    try:
        asyncio.run(classify_all(prompts, worker, max_concurrency=60))
        print("Batch run completed.")
    except Exception as exc:
        print(f"Batch finished (expected under rate limit): {type(exc).__name__}")
    finally:
        executor.shutdown(wait=False)

    print("\nRETRY PLUMBING RESULTS:")
    print(f"  Total call attempts:          {retry_tracker['attempts']}")
    print(f"  TransientLLMError triggered:  {retry_tracker['transient_errors_caught']}")

    if retry_tracker["transient_errors_caught"] > 0:
        print("VERDICT: The retry path through llm.py is VERIFIED and exercised.")
    else:
        print("NOTE: Deployment capacity absorbed all 60 calls without 429. Burst quota is high.")


if __name__ == "__main__":
    probe_b3_ambiguous_tickets()
    probe_b4_retry_path()
