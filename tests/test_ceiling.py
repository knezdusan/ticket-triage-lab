"""Tests for src/triage/ceiling.py — accuracy ceilings from taxonomy alone."""

import pytest

from triage.ceiling import category_marginal, max_group_accuracy, max_type_accuracy
from triage.models import Category


def test_category_marginal_sums_to_one():
    assert sum(category_marginal().values()) == pytest.approx(1.0)


def test_max_group_accuracy():
    ceiling, per_category = max_group_accuracy()
    # Hand-computed: sum_c P(c) * max_g P(g|c), master_data deterministic.
    assert ceiling == pytest.approx(0.8801, abs=1e-3)
    assert 0.0 < ceiling < 1.0
    assert per_category[Category.MASTER_DATA] == pytest.approx(
        category_marginal()[Category.MASTER_DATA]
    )


def test_max_type_accuracy():
    ceiling, per_pair = max_type_accuracy()
    # incident(0.45) beats problem(0.05); service_request(0.45) beats change(0.05)
    assert ceiling == pytest.approx(0.90)
    assert len(per_pair) == 2
