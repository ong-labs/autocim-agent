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
        }
    )

    new_overrides = human_input.get("new_bounds", {}) if isinstance(human_input, dict) else {}

    return {
        "human_overrides": new_overrides,
        "needs_hitl": False,
    }
