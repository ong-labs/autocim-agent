"""@mapper/@profiler wiring: `layer_configs` threaded from @tuner's own
output (not directly from `state["planned_layer_configs"]` -- see
nodes/mapper.py's docstring for why), and `calibration_factors` applied to
@profiler's fast-approximation output (Research_Plan.md 3's calibration
loop)."""

from middleware import set_execution_context
from nodes.common import build_execution_context
from nodes.mapper import mapper_node
from nodes.profiler import profiler_node
from nodes.tuner import tuner_node
from tools.simulators import mapper_tool, profiler_tool


def test_tuner_node_reports_which_device_it_used(state_factory, registered_hw_config):
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)

    tuner_result = tuner_node(state)["metrics_store"]["tuner"]

    assert tuner_result["status"] == "SUCCESS"
    assert tuner_result["data"]["device"]  # tools/qat.py's _resolve_device always resolves to something real


def test_mapper_node_uses_layer_configs_from_tuner_output(state_factory, registered_hw_config):
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    state["metrics_store"] = tuner_node(state)["metrics_store"]

    mapper_result = mapper_node(state)["metrics_store"]["mapper"]

    assert mapper_result["status"] == "SUCCESS"
    assert mapper_result["data"]["tiles_needed"] == len(state["metrics_store"]["tuner"]["data"]["layer_configs"])


def test_profiler_node_uses_layer_configs_from_tuner_output(state_factory, registered_hw_config):
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    state["metrics_store"] = tuner_node(state)["metrics_store"]

    profiler_result = profiler_node(state)["metrics_store"]["profiler"]

    assert profiler_result["status"] == "SUCCESS"
    assert profiler_result["data"]["energy_pj"] > 0


def test_mapper_tool_fails_gracefully_without_layer_configs(state_factory, registered_hw_config):
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    set_execution_context(build_execution_context(state))

    result = mapper_tool(model_ref="some_model")  # missing required layer_configs -> ValidationError

    assert result["mapper"]["status"] == "FAILED"


def test_profiler_calibration_factor_scales_energy_and_latency(state_factory, registered_hw_config):
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    set_execution_context(build_execution_context(state))

    layer_configs = [{"layer_name": "l0", "weight_bits": 6, "activation_bits": 6, "column_pruning_ratio": 0.0}]

    uncalibrated = profiler_tool(model_ref="m", layer_configs=layer_configs)["profiler"]
    calibrated = profiler_tool(
        model_ref="m",
        layer_configs=layer_configs,
        calibration_factors={registered_hw_config.hw_spec_id: 2.0},
    )["profiler"]

    assert calibrated["data"]["energy_pj"] == round(uncalibrated["data"]["energy_pj"] * 2, 6)
    assert calibrated["data"]["latency_ns"] == round(uncalibrated["data"]["latency_ns"] * 2, 6)


def test_profiler_node_returns_calibration_factors_back_into_state(state_factory, registered_hw_config):
    """Regression test: profiler_node used to compute
    metrics_store.profiler.data.calibration_factors but never return a
    top-level `calibration_factors` key, so `state["calibration_factors"]`
    (the field @profiler itself reads on every call, via the merge_dicts
    reducer) never actually accumulated anything a previous iteration
    produced -- the calibration loop was silently a no-op."""
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id, calibration_factors={"unrelated_hw": 3.0})
    state["metrics_store"] = tuner_node(state)["metrics_store"]

    update = profiler_node(state)

    assert update["calibration_factors"].get(registered_hw_config.hw_spec_id) == 1.0
    # profiler_tool echoes the whole incoming dict back (plus its own
    # hw_spec_id's setdefault) rather than isolating just its own entry --
    # harmless through merge_dicts (state.py), but confirms this iteration's
    # update carries forward a factor set for a different hw_spec_id too.
    assert update["calibration_factors"].get("unrelated_hw") == 3.0


def test_profiler_node_uses_pre_seeded_calibration_factor_from_state(state_factory, registered_hw_config):
    """If tools/calibration.py already seeded a real factor for this
    hw_spec_id (main.py's build_initial_state), @profiler must apply it --
    not silently reset to 1.0."""
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id, calibration_factors={registered_hw_config.hw_spec_id: 5.0})
    state["metrics_store"] = tuner_node(state)["metrics_store"]

    baseline_state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id, calibration_factors={})
    baseline_state["metrics_store"] = state["metrics_store"]

    calibrated = profiler_node(state)["metrics_store"]["profiler"]["data"]
    uncalibrated = profiler_node(baseline_state)["metrics_store"]["profiler"]["data"]

    assert calibrated["energy_pj"] == round(uncalibrated["energy_pj"] * 5.0, 6)
