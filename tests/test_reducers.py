"""[checklist item 2] namespaced_metrics_reducer must merge parallel
@mapper / @profiler fan-out results into metrics_store without either
namespace losing data or clobbering the other, regardless of arrival order.
"""

from nodes.mapper import mapper_node
from nodes.profiler import profiler_node
from nodes.tuner import tuner_node
from state import namespaced_metrics_reducer


def test_reducer_merges_disjoint_namespaces_without_mutating_inputs():
    left = {"mapper": {"status": "SUCCESS", "data": {"noc_latency_ms": 4.2}}}
    right = {"profiler": {"status": "SUCCESS", "data": {"energy_pj": 12.4}}}

    merged = namespaced_metrics_reducer(left, right)

    assert merged == {
        "mapper": {"status": "SUCCESS", "data": {"noc_latency_ms": 4.2}},
        "profiler": {"status": "SUCCESS", "data": {"energy_pj": 12.4}},
    }
    # neither input dict was mutated in place
    assert left == {"mapper": {"status": "SUCCESS", "data": {"noc_latency_ms": 4.2}}}
    assert right == {"profiler": {"status": "SUCCESS", "data": {"energy_pj": 12.4}}}


def test_reducer_result_is_independent_of_arrival_order():
    left = {"mapper": {"a": 1}}
    right = {"profiler": {"b": 2}}
    assert namespaced_metrics_reducer(left, right) == namespaced_metrics_reducer(right, left)


def test_reducer_handles_empty_sides():
    only_mapper = {"mapper": {"status": "SUCCESS"}}
    assert namespaced_metrics_reducer(only_mapper, {}) == only_mapper
    assert namespaced_metrics_reducer({}, only_mapper) == only_mapper
    assert namespaced_metrics_reducer({}, {}) == {}


def test_mapper_and_profiler_nodes_fan_out_without_clobbering(state_factory, registered_hw_config):
    """End-to-end at the node level: @mapper and @profiler both read the
    same post-@tuner state and each write only their own metrics_store
    namespace; folding their two partial updates in via
    namespaced_metrics_reducer must not lose @tuner's own entry or either
    parallel branch's data, in either arrival order.
    """
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    state["metrics_store"] = tuner_node(state)["metrics_store"]

    mapper_update = mapper_node(state)["metrics_store"]
    profiler_update = profiler_node(state)["metrics_store"]

    merged_mapper_first = namespaced_metrics_reducer(
        namespaced_metrics_reducer(state["metrics_store"], mapper_update), profiler_update
    )
    merged_profiler_first = namespaced_metrics_reducer(
        namespaced_metrics_reducer(state["metrics_store"], profiler_update), mapper_update
    )

    for merged in (merged_mapper_first, merged_profiler_first):
        assert set(merged.keys()) == {"tuner", "mapper", "profiler"}
        assert merged["tuner"]["status"] == "SUCCESS"
        assert merged["mapper"]["status"] == "SUCCESS"
        assert merged["profiler"]["status"] == "SUCCESS"
        assert "noc_latency_ms" in merged["mapper"]["data"]
        assert "energy_pj" in merged["profiler"]["data"]

    # fan-in result must not depend on which parallel branch merged first
    assert merged_mapper_first == merged_profiler_first
