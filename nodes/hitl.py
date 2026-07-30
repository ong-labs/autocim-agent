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
            # Non-null only when this exact iteration's @planner LLM call
            # flagged something (state.py's planner_flagged_anomaly) -- this
            # may be why HITL fired now instead of at MAX_RETRY_LIMIT.
            "anomaly_note": state.get("planner_flagged_anomaly"),
            # Every bound a researcher has already approved this run
            # (state.py, nodes/planner.py merges each newly-approved
            # human_overrides into this) -- main.py's suggest_override_bounds()
            # needs this so a *later* HITL round's suggestion narrows from
            # what's already active, not from the hw's raw ceiling every
            # time (which could suggest something looser than an override
            # already approved, silently undoing it).
            "search_bound_overrides": state.get("search_bound_overrides", {}),
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
