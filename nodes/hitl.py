"""hitl_human_approval node.

Uses LangGraph's v0.2+ dynamic `interrupt()` standard (CLAUDE.md 5.B):
no `interrupt_before` in `compile()`, and no `graph.update_state()` call
here that would duplicate the state update `interrupt()`'s return value
already provides on resume (`Command(resume=...)`).
"""

from typing import Any, Dict

from langgraph.types import interrupt

from state import AutoCIMState


def hitl_node(state: AutoCIMState) -> Dict[str, Any]:
    human_input = interrupt(
        {
            "reason": "Convergence stalled -- researcher review requested",
            "iteration_count": state.get("iteration_count", 0),
            "retry_count": state.get("retry_count", 0),
            "failure_history": state.get("failure_history", []),
            "latest_metrics": state.get("metrics_store", {}),
            # hw_spec_id + targets: main.py's suggest_override_bounds() needs
            # these to size its suggestion against how far off target the
            # candidate actually is, capped at this hw_spec's real ADC/DAC
            # bit ceiling -- neither was in this payload before, so that
            # suggestion was a fixed step regardless of how big the miss was.
            "hw_spec_id": state.get("hw_spec_id"),
            "target_accuracy": state.get("target_accuracy"),
            "target_energy_pj": state.get("target_energy_pj"),
            "target_latency_ms": state.get("target_latency_ms"),
        }
    )

    new_overrides = human_input.get("new_bounds", {}) if isinstance(human_input, dict) else {}

    return {
        "human_overrides": new_overrides,
        "needs_hitl": False,
    }
