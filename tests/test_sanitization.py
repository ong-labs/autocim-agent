"""[checklist item 3] planner_node must sanitize human_overrides/needs_hitl/
retry_count exactly when it consumes an override, immediately after reading
it -- and must leave retry_count/needs_hitl untouched on ordinary loops
without an override.

The second half is a regression test: an earlier version of planner_node
reset retry_count on every call unconditionally, which erased @evaluator's
progress toward MAX_RETRY_LIMIT and made needs_hitl unreachable, looping
the graph forever (caught via GraphRecursionError while building graph.py).
"""

from nodes.evaluator import MAX_RETRY_LIMIT, evaluator_node
from nodes.planner import planner_node


def test_planner_sanitizes_when_overrides_present(state_factory):
    state = state_factory(
        iteration_count=3,
        human_overrides={"new_bounds": {"weight_bits_min": 2}},
        needs_hitl=True,
        retry_count=3,
    )

    update = planner_node(state)

    assert update["human_overrides"] == {}
    assert update["needs_hitl"] is False
    assert update["retry_count"] == 0
    assert update["iteration_count"] == 4
    assert update["messages"], "should log that an override was consumed"


def test_planner_leaves_retry_state_untouched_without_overrides(state_factory):
    state = state_factory(iteration_count=1, human_overrides={}, needs_hitl=False, retry_count=2)

    update = planner_node(state)

    assert update["iteration_count"] == 2
    assert "retry_count" not in update
    assert "needs_hitl" not in update
    assert "human_overrides" not in update
    # @planner's LLM call always logs its decision, even with no override to
    # sanitize -- only the *sanitization* keys above must stay absent.
    assert update["messages"], "should log the LLM's layer_configs proposal"
    assert update["planned_layer_configs"]


def test_planner_treats_empty_dict_override_as_no_override(state_factory):
    """An empty {} from hitl_node (researcher gave no bounds) is falsy, so
    it is NOT treated as a consumed override -- matches hitl_node's own
    `human_input.get("new_bounds", {})` contract."""
    state = state_factory(iteration_count=0, human_overrides={}, needs_hitl=True, retry_count=5)

    update = planner_node(state)

    assert "retry_count" not in update
    assert "needs_hitl" not in update


def test_search_bound_overrides_persist_across_iterations_after_hitl(state_factory, registered_hw_config):
    """Regression: an approved HITL override must keep constraining every
    later candidate's search_bounds()/_clamp_layer_configs() for the rest of
    the run, not just the one candidate right after approval -- otherwise
    the same convergence failure (and the same HITL suggestion) recurs every
    MAX_RETRY_LIMIT iterations forever.

    weight_bits_min=3 is deliberately within registered_hw_config's own
    ceiling (adc_bits=8/dac_bits=4 -> bits_max=4), so this only exercises
    persistence, not the separate hw-ceiling-clamp behavior covered by
    test_hitl_override_cannot_push_weight_bits_past_the_hw_ceiling below."""
    state = state_factory(
        hw_spec_id=registered_hw_config.hw_spec_id,
        iteration_count=3,
        human_overrides={"weight_bits_min": 3},
        needs_hitl=True,
        retry_count=3,
    )

    update = planner_node(state)
    assert update["search_bound_overrides"] == {"weight_bits_min": 3}

    # Next iteration: human_overrides is back to {} (as hitl_node/planner
    # leave it after consumption), but search_bound_overrides carries
    # forward -- simulate the same merge LangGraph's reducer performs.
    state = {**state, **update}
    state["human_overrides"] = {}

    update2 = planner_node(state)

    assert "search_bound_overrides" not in update2  # nothing new to persist
    for layer_config in update2["planned_layer_configs"]:
        assert layer_config["weight_bits"] >= 3


def test_hitl_override_cannot_push_weight_bits_past_the_hw_ceiling(state_factory, registered_hw_config):
    """Regression: a researcher's weight_bits_min override must be capped at
    hw_spec_id's own ceiling (min(adc_bits, dac_bits)), not forced through
    as-is -- neither @tuner (tools/simulators.py) nor the physics it calls
    (tools/cim_physics.py) validates weight_bits against dac_bits/adc_bits,
    so a raw pass-through here would silently simulate energy/accuracy for
    a crossbar configuration the hardware can't physically support (observed
    live: dac_bits=4 but a weight_bits_min=8 override still landed as
    weight_bits=8 in planned_layer_configs before this fix)."""
    state = state_factory(
        hw_spec_id=registered_hw_config.hw_spec_id,  # adc_bits=8, dac_bits=4 -> ceiling 4
        iteration_count=3,
        human_overrides={"weight_bits_min": 8},
        needs_hitl=True,
        retry_count=3,
    )

    update = planner_node(state)

    for layer_config in update["planned_layer_configs"]:
        assert layer_config["weight_bits"] <= 4


def test_retry_count_accumulates_across_replan_loops_until_needs_hitl(state_factory):
    """Regression guard: planner<->evaluator loops with no override must let
    retry_count climb every iteration until MAX_RETRY_LIMIT, not get reset
    to 0 by planner along the way."""
    state = state_factory(metrics_store={})  # empty -> evaluator treats every tool output as missing

    for expected_retry_count in range(1, MAX_RETRY_LIMIT + 1):
        state = {**state, **planner_node(state)}
        state = {**state, **evaluator_node(state)}
        assert state["retry_count"] == expected_retry_count
        assert state["needs_hitl"] == (expected_retry_count >= MAX_RETRY_LIMIT)

    assert state["needs_hitl"] is True
