"""@mapper node: Multi-Tile / NoC mapping.

Runs in parallel with @profiler after @tuner (fan-out from Research_Plan.md
2절's graph). Writes only into its own `metrics_store["mapper"]` namespace,
so this and @profiler's concurrent writes never race -- see
`namespaced_metrics_reducer` in state.py. Hardware numbers themselves
(crossbar geometry, NoC bandwidth) are resolved inside `mapper_tool` from
the injected `HWConfig`, not touched here. `layer_configs` comes from
@tuner's own output (`TunerToolOutput.data["layer_configs"]`), not directly
from `state["planned_layer_configs"]`, so @mapper always simulates against
the exact configuration @tuner actually used -- including when @tuner fell
back to its own default because no plan was available yet.
"""

from typing import Any, Dict

from middleware import set_execution_context
from nodes.common import build_execution_context
from state import AutoCIMState
from tools.simulators import mapper_tool


def mapper_node(state: AutoCIMState) -> Dict[str, Any]:
    set_execution_context(build_execution_context(state))

    tuner_data = (state.get("metrics_store", {}).get("tuner") or {}).get("data") or {}
    model_ref = tuner_data.get("model_ref", state["model_id"])
    layer_configs = tuner_data.get("layer_configs", [])

    result = mapper_tool(model_ref=model_ref, layer_configs=layer_configs)
    return {"metrics_store": result}
