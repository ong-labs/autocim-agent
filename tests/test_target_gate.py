"""nodes/evaluator.py's target_accuracy/target_energy_pj/target_latency_ms
gate: a candidate must meet every *configured* target in addition to
@verifier's own IR-drop/noise feasibility check to count as converged.
Before this, a hardware-feasible-but-real-accuracy-too-low candidate would
be reported "Converged (Done)" -- Research_Plan.md's own evaluator_node
sketch names "Target 지표 미달성" as a failure reason, so this closes a real
gap between the documented design intent and what was actually implemented.
"""

import pytest

from nodes.evaluator import MAX_RETRY_LIMIT, check_targets, evaluator_node
from tests.test_evaluator_candidate_history import _metrics


# --- check_targets (pure function) -------------------------------------------


def test_check_targets_returns_empty_when_nothing_configured():
    assert check_targets(0.1, 999.0, 999.0, None, None, None) == []


def test_check_targets_flags_accuracy_below_target():
    reasons = check_targets(0.5, None, None, target_accuracy=0.7, target_energy_pj=None, target_latency_ms=None)
    assert len(reasons) == 1
    assert "accuracy" in reasons[0]


def test_check_targets_passes_accuracy_at_or_above_target():
    assert check_targets(0.7, None, None, target_accuracy=0.7, target_energy_pj=None, target_latency_ms=None) == []
    assert check_targets(0.8, None, None, target_accuracy=0.7, target_energy_pj=None, target_latency_ms=None) == []


def test_check_targets_flags_energy_above_target():
    reasons = check_targets(None, 100.0, None, target_accuracy=None, target_energy_pj=50.0, target_latency_ms=None)
    assert len(reasons) == 1
    assert "energy_pj" in reasons[0]


def test_check_targets_flags_latency_above_target():
    reasons = check_targets(None, None, 5.0, target_accuracy=None, target_energy_pj=None, target_latency_ms=2.0)
    assert len(reasons) == 1
    assert "noc_latency_ms" in reasons[0]


def test_check_targets_reports_all_missed_targets_together():
    reasons = check_targets(0.1, 100.0, 5.0, target_accuracy=0.7, target_energy_pj=50.0, target_latency_ms=2.0)
    assert len(reasons) == 3


def test_check_targets_treats_missing_metric_as_not_met():
    """A None metric value can't be judged as satisfying a configured
    target -- same 'incomplete data is never silently fine' rule as
    validate_metrics, just for target achievement."""
    assert check_targets(None, None, None, target_accuracy=0.7, target_energy_pj=None, target_latency_ms=None) != []


# --- evaluator_node integration -----------------------------------------------


def test_hw_feasible_but_low_accuracy_does_not_converge(state_factory):
    """The exact scenario reported: verifier says IR-drop/noise are fine
    (is_converged=True in its own data), but real accuracy is far below a
    configured target -- the overall candidate must NOT be reported as
    converged."""
    state = state_factory(
        metrics_store=_metrics(accuracy=0.204, is_converged=True),
        iteration_count=1,
        target_accuracy=0.7,
    )

    update = evaluator_node(state)

    assert update["is_converged"] is False
    assert "accuracy" in update["failure_history"][0]["reason"]
    # still recorded as real, useful surrogate data -- just not converged
    assert update["candidate_history"][0]["accuracy"] == 0.204
    assert update["candidate_history"][0]["is_converged"] is False


def test_hw_feasible_and_target_accuracy_met_converges(state_factory):
    state = state_factory(
        metrics_store=_metrics(accuracy=0.8, is_converged=True),
        iteration_count=1,
        target_accuracy=0.7,
    )

    update = evaluator_node(state)

    assert update["is_converged"] is True


def test_no_targets_configured_preserves_prior_behavior(state_factory):
    """state_factory's default target_accuracy/target_energy_pj/target_latency_ms
    are all None -- convergence must depend only on @verifier's own
    is_converged flag, exactly as before this feature existed."""
    state = state_factory(metrics_store=_metrics(accuracy=0.01, is_converged=True), iteration_count=1)
    assert evaluator_node(state)["is_converged"] is True

    state = state_factory(metrics_store=_metrics(accuracy=0.99, is_converged=False), iteration_count=1)
    assert evaluator_node(state)["is_converged"] is False


def test_target_miss_uses_the_same_retry_and_hitl_flow_as_any_other_failure(state_factory):
    """A target miss must not bypass or duplicate the existing
    MAX_RETRY_LIMIT -> needs_hitl flow -- it's just another reason a
    candidate doesn't converge."""
    state = state_factory(
        metrics_store=_metrics(accuracy=0.1, is_converged=True),
        iteration_count=1,
        retry_count=MAX_RETRY_LIMIT - 1,
        target_accuracy=0.7,
    )

    update = evaluator_node(state)

    assert update["is_converged"] is False
    assert update["retry_count"] == MAX_RETRY_LIMIT
    assert update["needs_hitl"] is True


def test_hw_infeasible_reason_unaffected_by_unmet_targets(state_factory):
    """When @verifier itself reports not-converged (IR-drop/noise), the
    failure reason should stay about that -- not get confused with a
    target miss that was never actually evaluated."""
    state = state_factory(
        metrics_store=_metrics(accuracy=0.1, is_converged=False),
        iteration_count=1,
        target_accuracy=0.7,
    )

    update = evaluator_node(state)

    assert update["is_converged"] is False
    assert "verifier reported not converged" in update["failure_history"][0]["reason"]
