"""Full-graph integration: @planner's search (tools/search.py) and
@evaluator's candidate_history bookkeeping actually work together across
real (stubbed) multiple iterations, not just in isolated unit tests."""

from graph import build_graph
from nodes.evaluator import MAX_RETRY_LIMIT


def test_candidate_history_accumulates_one_entry_per_iteration_and_planner_uses_warmup(
    state_factory, registered_bad_hw_config
):
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-search-e2e-thread"}}
    init_state = state_factory(hw_spec_id=registered_bad_hw_config.hw_spec_id)

    result = compiled.invoke(init_state, config=config)

    assert "__interrupt__" in result  # bad HW never converges -> HITL after MAX_RETRY_LIMIT
    assert len(result["candidate_history"]) == MAX_RETRY_LIMIT

    for entry in result["candidate_history"]:
        assert entry["pareto_rank"] >= 1
        assert entry["is_converged"] is False
        for key in ("avg_weight_bits", "avg_column_pruning_ratio", "accuracy", "energy_pj", "noc_latency_ms"):
            assert entry[key] is not None

    # MAX_RETRY_LIMIT (3) <= the 3-candidate LHS warm-up window, so every
    # iteration before this first HITL pause should have used warm-up
    # sampling, never the surrogate phase.
    planner_messages = [m["content"] for m in result["messages"] if "LHS warm-up" in m.get("content", "")]
    assert len(planner_messages) == MAX_RETRY_LIMIT
