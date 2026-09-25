"""Tests for the evaluation harness: statistics, intervals, and confusion reporting."""

from triage.evaluate import (
    ceiling_ratio,
    compute_priority_confusion,
    count_severe_overcalls,
    count_severe_undercalls,
    format_interval,
    format_priority_confusion,
    wilson_interval,
)
from triage.models import Priority


class TestWilsonInterval:
    def test_empty_total_returns_zeroes(self):
        p, low, high = wilson_interval(0, 0)
        assert p == 0.0
        assert low == 0.0
        assert high == 0.0

    def test_standard_case_bounds(self):
        # 39 out of 60 (our ~65% priority score)
        p, low, high = wilson_interval(39, 60)
        assert round(p, 4) == 0.65
        # 95% Wilson interval for 39/60 is approx [0.524, 0.758]
        assert 0.50 < low < 0.55
        assert 0.73 < high < 0.78
        assert low < p < high

    def test_zero_hits_upper_bound_non_zero(self):
        # 0 hits out of 30: point is 0, but upper bound must be > 0 (Wilson property)
        p, low, high = wilson_interval(0, 30)
        assert p == 0.0
        assert low == 0.0
        assert 0.05 < high < 0.15

    def test_perfect_hits_lower_bound_sub_one(self):
        # 30 hits out of 30: point is 1.0, lower bound must be < 1.0
        p, low, high = wilson_interval(30, 30)
        assert p == 1.0
        assert 0.85 < low < 0.95
        assert high == 1.0

    def test_formatting_output(self):
        formatted = format_interval(39, 60)
        assert "65.0%" in formatted
        assert "[" in formatted and "]" in formatted
        assert "-" in formatted


class TestPriorityConfusion:
    def test_compute_priority_confusion_structure(self):
        truths = [Priority.P1, Priority.P2, Priority.P3]
        preds = [Priority.P1, Priority.P3, Priority.P3]
        matrix = compute_priority_confusion(truths, preds)

        # All 4x4 cells exist
        for p1 in Priority:
            for p2 in Priority:
                assert p1 in matrix and p2 in matrix[p1]

        assert matrix[Priority.P1][Priority.P1] == 1
        assert matrix[Priority.P2][Priority.P3] == 1
        assert matrix[Priority.P3][Priority.P3] == 1
        assert matrix[Priority.P4][Priority.P4] == 0

    def test_count_severe_undercalls(self):
        truths = [
            Priority.P1,  # -> P3 (severe!)
            Priority.P1,  # -> P4 (severe!)
            Priority.P1,  # -> P2 (off by one, not severe undercall)
            Priority.P2,  # -> P4 (severe!)
            Priority.P2,  # -> P3 (off by one, not severe undercall)
            Priority.P3,  # -> P4 (off by one, not severe)
            Priority.P4,  # -> P1 (over-call, not severe undercall)
        ]
        preds = [
            Priority.P3,
            Priority.P4,
            Priority.P2,
            Priority.P4,
            Priority.P3,
            Priority.P4,
            Priority.P1,
        ]
        assert count_severe_undercalls(truths, preds) == 3

    def test_format_priority_confusion_renders(self):
        truths = [Priority.P1, Priority.P4]
        preds = [Priority.P1, Priority.P4]
        formatted = format_priority_confusion(truths, preds)

        assert "Truth P1" in formatted
        assert "Pred P1" in formatted
        assert "Severe under-calls" in formatted


class TestCeilingRatio:
    def test_standard_ratio(self):
        # 80.0% observed against 88.0% ceiling -> ~90.9%
        ratio = ceiling_ratio(0.80, 0.88)
        assert round(ratio, 3) == 0.909

    def test_clamps_to_one(self):
        # If observed exceeds ceiling due to sample noise, clamp to 1.0
        assert ceiling_ratio(0.95, 0.90) == 1.0

    def test_zero_ceiling_safe(self):
        assert ceiling_ratio(0.50, 0.0) == 0.0


class TestSevereOvercalls:
    def test_counts_false_alarms(self):
        truths = [
            Priority.P3,  # -> P1 (severe over-call!)
            Priority.P4,  # -> P1 (severe over-call!)
            Priority.P4,  # -> P2 (severe over-call!)
            Priority.P3,  # -> P2 (off-by-one over, not severe)
            Priority.P2,  # -> P1 (off-by-one over, not severe)
            Priority.P1,  # -> P4 (severe UNDER-call, not counted here)
            Priority.P4,  # -> P4 (correct)
        ]
        preds = [
            Priority.P1,
            Priority.P1,
            Priority.P2,
            Priority.P2,
            Priority.P1,
            Priority.P4,
            Priority.P4,
        ]
        assert count_severe_overcalls(truths, preds) == 3

    def test_matrix_renders_both_counters(self):
        formatted = format_priority_confusion([Priority.P3], [Priority.P1])
        assert "Severe under-calls" in formatted
        assert "Severe over-calls" in formatted
        assert ": 1" in formatted.split("over-calls")[1]
