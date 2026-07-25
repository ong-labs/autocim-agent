"""@tuner node: crossbar-aligned QAT/column-pruning, thin wrapper around `tuner_tool`.

Reads @planner's LLM-produced `layer_configs` candidate from
`state["planned_layer_configs"]` (nodes/planner.py). If it's absent -- cold
start before @planner has ever run, or @planner's own LLM call failed and it
already fell back itself -- this node falls back to the same
`default_layer_configs` baseline so @tuner never crashes for lack of a plan.
"""

from typing import Any, Dict

from middleware import set_execution_context
from nodes.common import build_execution_context, default_layer_configs
from state import AutoCIMState
from tools.simulators import tuner_tool

_DEFAULT_MAX_EPOCHS = 5


def tuner_node(state: AutoCIMState) -> Dict[str, Any]:
    set_execution_context(build_execution_context(state))

    layer_configs = state.get("planned_layer_configs") or default_layer_configs(state["hw_spec_id"])

    result = tuner_tool(
        layer_configs=layer_configs,
        max_epochs=_DEFAULT_MAX_EPOCHS,
    )
    return {"metrics_store": result}
