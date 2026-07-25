"""@profiler node: NeuroSim/CIM-Loop profiling.

Runs in parallel with @mapper after @tuner (fan-out). Writes only into its
own `metrics_store["profiler"]` namespace (see `namespaced_metrics_reducer`
in state.py). Hardware numbers (ADC/DAC resolution, crossbar geometry) are
resolved inside `profiler_tool` from the injected `HWConfig`. `layer_configs`
comes from @tuner's own output, matching @mapper's contract -- see
nodes/mapper.py's docstring for why.

Also returns `calibration_factors` back into state: `profiler_tool`
`setdefault`s an entry for this run's `hw_spec_id` (registering it at 1.0 --
uncalibrated -- the first time that hw_spec_id is seen, tools/calibration.py
seeds a real value before that via main.py's initial state instead), but
until this fix that computed dict was silently dropped -- `state["calibration_factors"]`
(the `merge_dicts`-reduced field @profiler itself reads on every call) never
actually accumulated anything a previous iteration produced.
"""

from typing import Any, Dict

from middleware import set_execution_context
from nodes.common import build_execution_context
from state import AutoCIMState
from tools.simulators import profiler_tool


def profiler_node(state: AutoCIMState) -> Dict[str, Any]:
    set_execution_context(build_execution_context(state))

    tuner_data = (state.get("metrics_store", {}).get("tuner") or {}).get("data") or {}
    model_ref = tuner_data.get("model_ref", state["model_id"])
    layer_configs = tuner_data.get("layer_configs", [])
    calibration_factors = state.get("calibration_factors") or None

    result = profiler_tool(model_ref=model_ref, layer_configs=layer_configs, calibration_factors=calibration_factors)
    updated_calibration_factors = (result.get("profiler") or {}).get("data") or {}
    return {
        "metrics_store": result,
        "calibration_factors": updated_calibration_factors.get("calibration_factors") or {},
    }
