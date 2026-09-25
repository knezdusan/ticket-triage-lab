"""Evaluation statistics: Wilson score intervals and result formatting.

Accuracy numbers on n=60 tickets are meaningless without a confidence
interval — a 2-point swing is noise, not signal. Wilson is preferred over
the normal approximation because it stays sane at the edges (p near 0 or 1)
and on small samples.
"""

import math

from triage.models import Priority


def wilson_interval(hits: int, total: int, z: float = 1.95996) -> tuple[float, float, float]:
    """Calculate the Wilson score confidence interval for a proportion.

    Args:
        hits: Number of successful outcomes (k >= 0).
        total: Total number of trials (n >= hits).
        z: Quantile for confidence level (default 1.95996 for 95% two-sided).

    Returns:
        tuple of (point_estimate, lower_bound, upper_bound), all in [0.0, 1.0].
        If total == 0, returns (0.0, 0.0, 0.0).
    """
    if total == 0:
        return (0.0, 0.0, 0.0)

    p = hits / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denom
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))


def format_interval(hits: int, total: int, z: float = 1.95996) -> str:
    """Format interval as a readable string: '65.0% [52.4% - 75.8%]'."""
    p, low, high = wilson_interval(hits, total, z)
    return f"{p:.1%} [{low:.1%} - {high:.1%}]"


# Off-by-one priority slips are routine; these pairs are SLA-breach territory.
_SEVERE_UNDERCALLS = {
    (Priority.P1, Priority.P3),
    (Priority.P1, Priority.P4),
    (Priority.P2, Priority.P4),
}

# Mirror image: a false P1 fires the Critical Incident Bridge — SMS alerts,
# L3 pages, emergency response. Expensive without being dangerous.
_SEVERE_OVERCALLS = {
    (Priority.P3, Priority.P1),
    (Priority.P4, Priority.P1),
    (Priority.P4, Priority.P2),
}


def compute_priority_confusion(
    truths: list[Priority],
    predictions: list[Priority],
) -> dict[Priority, dict[Priority, int]]:
    """Build a 4x4 confusion matrix mapping truth -> {pred: count}."""
    matrix = {t: {p: 0 for p in Priority} for t in Priority}
    for truth, pred in zip(truths, predictions, strict=True):
        matrix[truth][pred] += 1
    return matrix


def count_severe_undercalls(
    truths: list[Priority],
    predictions: list[Priority],
) -> int:
    """Count dangerous under-calls: P1 predicted as P3/P4, or P2 predicted as P4."""
    return sum(
        (truth, pred) in _SEVERE_UNDERCALLS for truth, pred in zip(truths, predictions, strict=True)
    )


def count_severe_overcalls(
    truths: list[Priority],
    predictions: list[Priority],
) -> int:
    """Count costly false alarms: P3/P4 predicted as P1, or P4 predicted as P2."""
    return sum(
        (truth, pred) in _SEVERE_OVERCALLS for truth, pred in zip(truths, predictions, strict=True)
    )


def format_priority_confusion(
    truths: list[Priority],
    predictions: list[Priority],
) -> str:
    """Render an ASCII table of Truth vs Predicted plus severe under-call count."""
    matrix = compute_priority_confusion(truths, predictions)
    col = max(len(p.name) for p in Priority) + 6
    lines = [" " * 9 + "".join(f"Pred {p.name}".rjust(col) for p in Priority)]
    lines += [
        f"Truth {t.name}".ljust(9) + "".join(str(matrix[t][p]).rjust(col) for p in Priority)
        for t in Priority
    ]
    severe = count_severe_undercalls(truths, predictions)
    over = count_severe_overcalls(truths, predictions)
    lines.append(f"Severe under-calls (P1->P3/P4, P2->P4): {severe}")
    lines.append(f"Severe over-calls  (P3/P4->P1, P4->P2): {over}")
    return "\n".join(lines)


def ceiling_ratio(observed_acc: float, ceiling_acc: float) -> float:
    """Return observed / ceiling, clamped to [0.0, 1.0]. 0.0 if ceiling <= 0."""
    if ceiling_acc <= 0.0:
        return 0.0
    return min(1.0, max(0.0, observed_acc / ceiling_acc))
