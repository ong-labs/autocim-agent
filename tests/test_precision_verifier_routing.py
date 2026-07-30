"""graph.py: @evaluator's "Converged (Done)" edge must route through
@precision_verifier before a real END (not straight to END as before) --
see graph.py's module docstring and nodes/precision_verifier.py."""

from graph import build_graph


def test_converged_run_passes_through_precision_verifier_before_end(state_factory, registered_hw_config):
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-precision-verifier-thread"}}
    init_state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)

    chunks = list(compiled.stream(init_state, config=config, stream_mode="updates"))

    assert any("precision_verifier" in chunk for chunk in chunks), (
        "a converged candidate must visit precision_verifier -- evaluator's "
        "'Converged (Done)' edge must no longer go straight to END"
    )
    final_state = compiled.get_state(config).values
    assert "__interrupt__" not in final_state
    assert final_state["is_converged"] is True
    # precision_verifier_node's own calibration_factors update (see its
    # docstring) must have actually merged into state, not been dropped.
    assert final_state["calibration_factors"].get(registered_hw_config.hw_spec_id) is not None
