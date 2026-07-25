"""Shared helpers for AutoCIM-Agent node functions."""

from typing import List

from middleware import get_hw_config
from schemas.config import ExecutionContext
from schemas.tools import LayerBitConfig
from state import AutoCIMState


def default_layer_configs(hw_spec_id: str) -> List[LayerBitConfig]:
    """Single-layer, hardware-bounded baseline candidate.

    Used by @tuner when @planner hasn't produced a `planned_layer_configs`
    proposal yet, and by @planner itself as the fallback when its LLM call
    fails -- one implementation so both fallback paths agree. Bit-width is
    bounded by the target hardware's converter resolution (`HWConfig.adc_bits`)
    rather than a hardcoded value (CLAUDE.md 5.C).
    """
    try:
        baseline_bits = min(8, get_hw_config(hw_spec_id).adc_bits)
    except Exception:
        # HWConfig not registered yet: fall back to a conservative default.
        # The downstream tool call's own wrap_tool_call will independently
        # re-resolve HWConfig and surface a FAILED result if it's genuinely
        # missing -- this fallback only keeps the caller from crashing first.
        baseline_bits = 8

    return [
        LayerBitConfig(
            layer_name="model_default",
            weight_bits=baseline_bits,
            activation_bits=baseline_bits,
            column_pruning_ratio=0.0,
        )
    ]


def run_id_for(state: AutoCIMState) -> str:
    """Deterministic run/session id from state, for anything (execution
    context, observability.py's structured-log correlation id) that needs
    to identify "this run" without LangGraph's real thread id, which lives
    in the invocation config and isn't threaded through to node functions.
    Descriptive metadata only, never used as an identity/security boundary
    anywhere in this codebase."""
    return f"{state['model_id']}::{state['hw_spec_id']}"


def build_execution_context(state: AutoCIMState) -> ExecutionContext:
    """Derive this node's ExecutionContext from state, for `middleware.set_execution_context`."""
    return ExecutionContext(
        session_id=run_id_for(state),
        model_id=state["model_id"],
        hw_spec_id=state["hw_spec_id"],
        iteration_count=state.get("iteration_count", 0),
    )
