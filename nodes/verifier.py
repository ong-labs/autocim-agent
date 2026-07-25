"""@verifier node: fan-in synthesis of @mapper + @profiler (Research_Plan.md 2절).

By the time this node runs, `namespaced_metrics_reducer` guarantees both
`metrics_store["mapper"]` and `metrics_store["profiler"]` are present under
their own keys, however the parallel branches interleaved. This node just
reassembles them into `VerifierToolInput` and calls `verifier_tool`, which
does the actual HWConfig-aware IR-drop/noise check.
"""

from typing import Any, Dict

from middleware import set_execution_context
from nodes.common import build_execution_context
from schemas.tools import MapperToolOutput, ProfilerToolOutput
from state import AutoCIMState
from tools.simulators import verifier_tool

_MISSING_MAPPER = {"status": "FAILED", "data": None, "error": "mapper result missing from metrics_store"}
_MISSING_PROFILER = {"status": "FAILED", "data": None, "error": "profiler result missing from metrics_store"}


def verifier_node(state: AutoCIMState) -> Dict[str, Any]:
    set_execution_context(build_execution_context(state))

    metrics = state.get("metrics_store", {})
    mapper_result = MapperToolOutput(**(metrics.get("mapper") or _MISSING_MAPPER))
    profiler_result = ProfilerToolOutput(**(metrics.get("profiler") or _MISSING_PROFILER))

    result = verifier_tool(mapper_result=mapper_result, profiler_result=profiler_result)
    return {"metrics_store": result}
