"""@precision_verifier node + precision_check_tool: mock Stage-2
high-fidelity re-check that graph.py now routes @evaluator's own
"Converged (Done)" edge through before a real END (CLAUDE.md 5.D Mock
First -- see nodes/precision_verifier.py's docstring)."""

from middleware import set_execution_context
from nodes.common import build_execution_context
from nodes.evaluator import MAX_RETRY_LIMIT, evaluator_router
from nodes.precision_verifier import precision_verifier_node
from tools.simulators import precision_check_tool

from tests.test_evaluator_candidate_history import _metrics


# --- precision_check_tool (wrap_tool_call layer) ----------------------------


def test_precision_check_tool_converges_when_no_targets_are_configured(state_factory, registered_hw_config):
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    set_execution_context(build_execution_context(state))

    layer_configs = [{"layer_name": "l0", "weight_bits": 6, "activation_bits": 6, "column_pruning_ratio": 0.0}]
    result = precision_check_tool(layer_configs=layer_configs, accuracy=0.8, noc_latency_ms=1.0)["precision_check"]

    assert result["status"] == "SUCCESS"
    assert result["data"]["is_converged"] is True
    assert result["data"]["precise_energy_pj"] > result["data"]["raw_energy_pj"]  # perturbed, per module docstring


def test_precision_check_tool_flags_a_target_the_precise_energy_number_now_misses(state_factory, registered_hw_config):
    """The precise energy number is deliberately higher than the fast
    approximation's (simulate_precise_energy's perturbation) -- a target set
    just above the raw number but below the precise one must now fail."""
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    set_execution_context(build_execution_context(state))

    layer_configs = [{"layer_name": "l0", "weight_bits": 6, "activation_bits": 6, "column_pruning_ratio": 0.0}]
    raw = precision_check_tool(layer_configs=layer_configs)["precision_check"]["data"]["raw_energy_pj"]

    result = precision_check_tool(
        layer_configs=layer_configs, accuracy=0.8, noc_latency_ms=1.0, target_energy_pj=raw
    )["precision_check"]

    assert result["data"]["is_converged"] is False
    assert any("energy_pj" in reason for reason in result["data"]["target_errors"])


def test_precision_check_tool_fails_gracefully_without_layer_configs(state_factory, registered_hw_config):
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    set_execution_context(build_execution_context(state))

    result = precision_check_tool()["precision_check"]  # missing required layer_configs -> ValidationError

    assert result["status"] == "FAILED"


# --- precision_verifier_node ------------------------------------------------


def test_precision_verifier_node_converges_and_updates_calibration_factors(state_factory, registered_hw_config):
    state = state_factory(
        hw_spec_id=registered_hw_config.hw_spec_id,
        metrics_store=_metrics(accuracy=0.8, is_converged=True),
        iteration_count=1,
    )

    update = precision_verifier_node(state)

    assert update["is_converged"] is True
    assert update["needs_hitl"] is False
    assert update["calibration_factors"][registered_hw_config.hw_spec_id] > 0
    provenance = update["calibration_provenance"][registered_hw_config.hw_spec_id]
    assert provenance["precision_verified"] is True
    assert provenance["raw_energy_pj"] > 0
    assert provenance["precise_energy_pj"] > provenance["raw_energy_pj"]
    assert "failure_history" not in update


def test_precision_verifier_node_un_converges_on_a_missed_target_and_bumps_retry_count(state_factory, registered_hw_config):
    """A candidate @evaluator's fast approximation judged converged can
    still fail here, since simulate_precise_energy's mock perturbation makes
    the precise energy number strictly higher -- this must use the exact
    same retry_count/failure_history bookkeeping as an ordinary @evaluator
    miss, not a bespoke path."""
    metrics = _metrics(accuracy=0.8, energy_pj=5.0, is_converged=True)
    raw_energy_pj = metrics["profiler"]["data"]["energy_pj"]
    state = state_factory(
        hw_spec_id=registered_hw_config.hw_spec_id,
        metrics_store=metrics,
        iteration_count=1,
        retry_count=0,
        target_energy_pj=raw_energy_pj,
    )

    update = precision_verifier_node(state)

    assert update["is_converged"] is False
    assert update["retry_count"] == 1
    assert update["needs_hitl"] is False
    assert "precision check" in update["failure_history"][0]["reason"]
    assert update["calibration_factors"][registered_hw_config.hw_spec_id] > 1.0
    # calibration_provenance must update even on a failed precision check --
    # the correction factor itself is still real/usable data, this candidate
    # just didn't meet the target with the corrected number.
    assert update["calibration_provenance"][registered_hw_config.hw_spec_id]["precision_verified"] is True


def test_precision_verifier_node_triggers_hitl_at_max_retry_limit(state_factory, registered_hw_config):
    metrics = _metrics(accuracy=0.8, energy_pj=5.0, is_converged=True)
    raw_energy_pj = metrics["profiler"]["data"]["energy_pj"]
    state = state_factory(
        hw_spec_id=registered_hw_config.hw_spec_id,
        metrics_store=metrics,
        iteration_count=1,
        retry_count=MAX_RETRY_LIMIT - 1,
        target_energy_pj=raw_energy_pj,
    )

    update = precision_verifier_node(state)

    assert update["retry_count"] == MAX_RETRY_LIMIT
    assert update["needs_hitl"] is True


# --- routing (evaluator_router reused for both edges, see graph.py) --------


def test_evaluator_router_reused_for_precision_verifier_output_routes_converged_to_done():
    state = {"is_converged": True, "needs_hitl": False}
    assert evaluator_router(state) == "Converged (Done)"


def test_evaluator_router_reused_for_precision_verifier_output_routes_failure_to_hitl_or_replan():
    assert evaluator_router({"is_converged": False, "needs_hitl": True}) == "HITL Interrupt"
    assert evaluator_router({"is_converged": False, "needs_hitl": False}) == "Re-plan with History"
