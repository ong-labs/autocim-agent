"""@evaluator's `candidate_history` bookkeeping (nodes/evaluator.py):
one entry per fully-evaluated iteration, with a real Pareto rank
(tools/search.py) computed against whatever history already exists in
state -- replacing @tuner's old hardcoded `pareto_rank: 1`, which a single
QAT trial has no basis to produce on its own.
"""

from nodes.evaluator import evaluator_node


_LAYER_CONFIGS = [
    {"layer_name": "conv", "weight_bits": 6, "activation_bits": 6, "column_pruning_ratio": 0.05},
    {"layer_name": "fc", "weight_bits": 4, "activation_bits": 4, "column_pruning_ratio": 0.15},
]


def _metrics(accuracy=0.8, energy_pj=5.0, noc_latency_ms=1.2, is_converged=True):
    return {
        "tuner": {
            "status": "SUCCESS",
            "data": {
                "accuracy": accuracy,
                "avg_weight_bits": 6,
                "avg_column_pruning_ratio": 0.1,
                "layer_configs": _LAYER_CONFIGS,
            },
            "error": None,
        },
        "mapper": {
            "status": "SUCCESS",
            "data": {"noc_latency_ms": noc_latency_ms, "buffer_delay_ms": 0.1, "tile_utilization": 0.5},
            "error": None,
        },
        "profiler": {
            "status": "SUCCESS",
            "data": {"energy_pj": energy_pj, "latency_ns": 1.0},
            "error": None,
        },
        "verifier": {
            "status": "SUCCESS",
            "data": {
                "ir_drop_error_pct": 0.1,
                "noise_margin_db": 10.0,
                "is_converged": is_converged,
                "noc_latency_ms": noc_latency_ms,
                "energy_pj": energy_pj,
            },
            "error": None,
        },
    }


def test_evaluator_records_one_candidate_history_entry_per_complete_iteration(state_factory):
    state = state_factory(metrics_store=_metrics(accuracy=0.8, energy_pj=5.0, noc_latency_ms=1.2), iteration_count=1)

    update = evaluator_node(state)

    assert len(update["candidate_history"]) == 1
    entry = update["candidate_history"][0]
    assert entry["accuracy"] == 0.8
    assert entry["energy_pj"] == 5.0
    assert entry["noc_latency_ms"] == 1.2
    assert entry["avg_weight_bits"] == 6
    assert entry["avg_column_pruning_ratio"] == 0.1
    assert entry["layer_configs"] == _LAYER_CONFIGS  # full per-stage record for the search surrogate
    assert entry["pareto_rank"] == 1  # no prior history to be dominated by


def test_evaluator_does_not_record_incomplete_iteration(state_factory):
    metrics = _metrics()
    del metrics["mapper"]  # simulate a failed/missing @mapper result
    state = state_factory(metrics_store=metrics, iteration_count=1)

    update = evaluator_node(state)

    assert update["candidate_history"] == []


def test_evaluator_computes_pareto_rank_against_existing_candidate_history(state_factory):
    dominating_entry = {
        "iteration": 1,
        "avg_weight_bits": 8,
        "avg_column_pruning_ratio": 0.0,
        "accuracy": 0.95,
        "energy_pj": 2.0,
        "noc_latency_ms": 0.5,
        "is_converged": True,
        "pareto_rank": 1,
    }
    # strictly worse than dominating_entry on every objective
    state = state_factory(
        metrics_store=_metrics(accuracy=0.6, energy_pj=6.0, noc_latency_ms=2.0),
        candidate_history=[dominating_entry],
        iteration_count=2,
    )

    update = evaluator_node(state)

    assert update["candidate_history"][0]["pareto_rank"] == 2


def test_evaluator_still_appends_candidate_history_on_non_convergence(state_factory):
    """A candidate that fails verifier's convergence check (e.g. IR-drop)
    still has real accuracy/energy/latency and is useful surrogate training
    data -- it must still be recorded, just with is_converged=False."""
    state = state_factory(metrics_store=_metrics(is_converged=False), iteration_count=1)

    update = evaluator_node(state)

    assert update["is_converged"] is False
    assert len(update["candidate_history"]) == 1
    assert update["candidate_history"][0]["is_converged"] is False
