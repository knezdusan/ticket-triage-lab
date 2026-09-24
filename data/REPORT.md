# Synthetic ticket data report

Scope: generated. Planned tickets: 300.

## Counts by type
- change: 14
- incident: 129
- problem: 13
- service_request: 144

## Counts by priority
- P1: 3
- P2: 25
- P3: 162
- P4: 110

## Counts by category
- access_authorization: 51
- batch_job: 19
- interface_idoc: 16
- invoice_posting: 23
- master_data: 43
- output_printing: 26
- password_account: 33
- performance: 18
- pricing_sales: 11
- purchasing: 10
- short_dump: 29
- transport_change: 21

## Counts by assignment_group
- ABAP: 26
- Basis: 79
- FI: 37
- Integration: 15
- MM: 33
- SD: 17
- Security: 41
- Service Desk: 52

## Counts by tier
- L1: 93
- L2: 173
- L3: 34

## Split
- history.jsonl: 240 tickets
- eval.jsonl: 60 tickets

## Near-duplicate clusters
- 26 clusters covering 59 tickets

## Priority stress set
- eval_p1.jsonl: 60 tickets — exactly 30 P1 and
  30 P2. Every P1 is an incident by construction
  (P1 requires impact=high x urgency=high, and only incidents carry
  P1 mass). Deliberately non-representative: exists only to measure
  priority confusion. Excluded from every distribution count above.
