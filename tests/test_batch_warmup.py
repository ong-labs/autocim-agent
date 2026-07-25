"""tools/batch_warmup.py: parallel evaluation of independent LHS warm-up
candidates through the real tuner/mapper/profiler/verifier tool functions
(bypassing wrap_tool_call's contextvar injection -- see module docstring).

Runs against the same autouse stub backends every other test in this suite
uses (stub_tuner_qat_backend, stub_planner_layer_groups) -- no network,
no real training, and no need for register_hw_config/set_execution_context
since this module never touches the middleware registry/contextvars at all.
"""

from concurrent.futures import ThreadPoolExecutor

from nodes.planner import real_stage_names, warmup_count
from tools.batch_warmup import run_parallel_warmup
from tools.search import compute_pareto_rank


def test_run_parallel_warmup_evaluates_every_warmup_candidate(good_hw_config):
    candidate_history, failure_history = run_parallel_warmup("resnet18", good_hw_config)

    n_stages = len(real_stage_names("resnet18"))
    expected_n = warmup_count(n_stages)
    assert len(candidate_history) + len(failure_history) == expected_n


def test_run_parallel_warmup_records_valid_multi_objective_data(good_hw_config):
    candidate_history, failure_history = run_parallel_warmup("resnet18", good_hw_config)

    assert failure_history == []  # good_hw_config converges -> no reason for any candidate to fail validation
    for entry in candidate_history:
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert entry["energy_pj"] > 0
        assert entry["noc_latency_ms"] >= 0
        assert entry["layer_configs"]
        assert entry["pareto_rank"] >= 1


def test_run_parallel_warmup_pareto_ranks_match_sequential_recomputation(good_hw_config):
    """The whole point of the second, sequential pass: ranks must be
    exactly what compute_pareto_rank would produce processing the same
    candidates one at a time in warm-up-index order -- proving the
    parallel execution didn't introduce order-dependent races."""
    candidate_history, _ = run_parallel_warmup("resnet18", good_hw_config)

    recomputed_history = []
    for entry in candidate_history:
        expected_rank = compute_pareto_rank(entry, recomputed_history)
        assert entry["pareto_rank"] == expected_rank
        recomputed_history.append(entry)


def test_run_parallel_warmup_still_records_non_converging_candidates(registered_bad_hw_config):
    """bad_hw_config's high wire resistance means verifier never reports
    is_converged=True -- but a non-converging candidate still has real,
    complete accuracy/energy/latency data (nodes/evaluator.py's own
    test_evaluator_still_appends_candidate_history_on_non_convergence
    establishes this for the sequential path), so it belongs in
    candidate_history with is_converged=False, not failure_history --
    failure_history is only for incomplete/schema-invalid results."""
    candidate_history, failure_history = run_parallel_warmup("resnet18", registered_bad_hw_config)

    n_stages = len(real_stage_names("resnet18"))
    assert len(candidate_history) == warmup_count(n_stages)
    assert failure_history == []
    for entry in candidate_history:
        assert entry["is_converged"] is False


def test_run_parallel_warmup_honors_calibration_factors(good_hw_config):
    """calibration_factors must actually reach @profiler's calculation, not
    be silently dropped for the parallel path -- otherwise warm-up energy
    figures would be uncalibrated even when the sequential path's would be."""
    uncalibrated, _ = run_parallel_warmup("resnet18", good_hw_config)
    calibrated, _ = run_parallel_warmup("resnet18", good_hw_config, calibration_factors={good_hw_config.hw_spec_id: 2.0})

    # Same candidates (deterministic seed) -> same layer_configs -> the only
    # difference should be the calibration multiplier applied to energy_pj.
    for u, c in zip(sorted(uncalibrated, key=lambda e: e["iteration"]), sorted(calibrated, key=lambda e: e["iteration"])):
        assert c["energy_pj"] == u["energy_pj"] * 2.0


def test_run_parallel_warmup_is_actually_concurrent(good_hw_config, monkeypatch):
    """Not just correct -- actually parallel: patches ThreadPoolExecutor to
    record the max_workers it was constructed with, confirming more than
    one worker is used when there's more than one warm-up candidate."""
    n_stages = len(real_stage_names("resnet18"))
    expected_n = warmup_count(n_stages)
    assert expected_n > 1  # sanity: this test is meaningless if there's only one candidate

    captured = {}
    real_executor = ThreadPoolExecutor

    class _RecordingExecutor(real_executor):
        def __init__(self, max_workers=None, *args, **kwargs):
            captured["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, *args, **kwargs)

    monkeypatch.setattr("tools.batch_warmup.ThreadPoolExecutor", _RecordingExecutor)

    run_parallel_warmup("resnet18", good_hw_config)

    assert captured["max_workers"] > 1
