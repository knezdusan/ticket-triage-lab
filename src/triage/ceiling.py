"""Accuracy ceilings for the synthetic data, from taxonomy alone. No data read.

A perfect text classifier still cannot beat label noise: when a category can
land with more than one assignment group, the best strategy is to always pick
the most likely one. Same for type when two types are textually
indistinguishable. Run: uv run python -m triage.ceiling
"""

from triage.models import Category, TicketType
from triage.taxonomy import (
    ASSIGNMENT_GROUPS_BY_CATEGORY,
    CATEGORIES_BY_TYPE,
    TYPE_MIX,
)

# Pairs the blueprint treats as textually indistinguishable.
_INDISTINGUISHABLE: tuple[frozenset[TicketType], ...] = (
    frozenset({TicketType.PROBLEM, TicketType.INCIDENT}),
    frozenset({TicketType.CHANGE, TicketType.SERVICE_REQUEST}),
)


def category_marginal() -> dict[Category, float]:
    """P(category) under the type mix: sum_t P(t) * P(category|t)."""
    marginal = dict.fromkeys(Category, 0.0)
    for ticket_type, type_p in TYPE_MIX.items():
        for category, cat_p in CATEGORIES_BY_TYPE[ticket_type].items():
            marginal[category] += type_p * cat_p
    return marginal


def max_group_accuracy() -> tuple[float, dict[Category, float]]:
    """Ceiling on assignment_group accuracy.

    Per category the best predictor always outputs argmax_g P(g|c), scoring
    that weight. master_data is deterministic (group follows the module), so
    its ceiling is 1.0. Returns (total, per-category contributions).
    """
    per_category: dict[Category, float] = {}
    for category, cat_p in category_marginal().items():
        if cat_p == 0.0:
            continue
        if category is Category.MASTER_DATA:
            best = 1.0  # deterministic via MASTER_DATA_GROUP_BY_MODULE
        else:
            best = max(ASSIGNMENT_GROUPS_BY_CATEGORY[category].values())
        per_category[category] = cat_p * best
    return sum(per_category.values()), per_category


def max_type_accuracy() -> tuple[float, dict[frozenset[TicketType], float]]:
    """Ceiling on type accuracy given indistinguishable pairs.

    Within each merged pair the best predictor picks the more frequent type,
    scoring its share of the mix. Returns (total, per-pair contributions).
    """
    per_pair = {pair: max(TYPE_MIX[t] for t in pair) for pair in _INDISTINGUISHABLE}
    return sum(per_pair.values()), per_pair


def main() -> None:
    group_ceiling, per_category = max_group_accuracy()
    print("== assignment_group ceiling ==")
    for category, contribution in sorted(per_category.items()):
        groups = (
            "deterministic (module -> group)"
            if category is Category.MASTER_DATA
            else f"best = {max(ASSIGNMENT_GROUPS_BY_CATEGORY[category])} "
            f"-> {max(ASSIGNMENT_GROUPS_BY_CATEGORY[category].values()):.2f}"
        )
        cat_p = category_marginal()[category]
        print(f"  {category.value:22s} P(cat)={cat_p:.3f} x {groups} = {contribution:.4f}")
    print(f"  TOTAL: sum = {group_ceiling:.3f} ({group_ceiling:.1%})")

    type_ceiling, per_pair = max_type_accuracy()
    print("\n== type ceiling ==")
    for pair, contribution in per_pair.items():
        names = " vs ".join(sorted(t.value for t in pair))
        shares = " / ".join(f"{TYPE_MIX[t]:.2f}" for t in sorted(pair))
        print(f"  {names}: P = {shares} -> pick majority, adds {contribution:.2f}")
    print(f"  TOTAL: {type_ceiling:.2f} ({type_ceiling:.0%})")


if __name__ == "__main__":
    main()
