"""@precision_verifier node: mock Stage-2 high-fidelity re-verification
(CLAUDE.md 5.D Mock First).

`graph.py` routes @evaluator's "Converged (Done)" edge here instead of
straight to `END` -- this node only ever runs on a candidate the fast-
approximation already judged converged, never on every candidate a search
tries, since a real precise simulator call (NeuroSim/CIM-Loop) is far more
expensive than `tools/cim_physics.py`'s analytical formulas.

If the precise check disagrees, this node un-converges the run using the
exact same `retry_count`/`needs_hitl` bookkeeping @evaluator itself uses on
an ordinary miss (`evaluator_router` is reused for the routing out of this
node too) -- a precision-check failure is just another kind of non-
convergence to that state machine, not a new bespoke failure path.
"""

from typing import Any, Dict

from middleware import set_execution_context
from nodes.common import build_execution_context, run_id_for
from nodes.evaluator import MAX_RETRY_LIMIT
from observability import log_event
from state import AutoCIMState
from tools.simulators import precision_check_tool


def precision_verifier_node(state: AutoCIMState) -> Dict[str, Any]:
    set_execution_context(build_execution_context(state))

    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)
    metrics = state.get("metrics_store", {})
    tuner_data = (metrics.get("tuner") or {}).get("data") or {}
    verifier_data = (metrics.get("verifier") or {}).get("data") or {}
    layer_configs = tuner_data.get("layer_configs") or []

    result = precision_check_tool(
        layer_configs=layer_configs,
        accuracy=tuner_data.get("accuracy"),
        noc_latency_ms=verifier_data.get("noc_latency_ms"),
        target_accuracy=state.get("target_accuracy"),
        target_energy_pj=state.get("target_energy_pj"),
        target_latency_ms=state.get("target_latency_ms"),
    )
    precision_data = (result.get("precision_check") or {}).get("data") or {}
    is_precisely_converged = bool(precision_data.get("is_converged"))

    # Merges into calibration_factors (state.py's merge_dicts reducer) the
    # same way tools/calibration.py's bootstrap_calibration_factors/
    # nodes/profiler.py's own update does -- this run's real precision
    # check refines the correction factor for every *later* candidate's
    # fast-approximation, in this run and (via main.py's GlobalHWStore
    # persistence, same as any other calibration_factors update) future
    # ones too.
    calibration_factor = precision_data.get("calibration_factor")
    calibration_factors = (
        {state["hw_spec_id"]: calibration_factor} if calibration_factor is not None else {}
    )
    # Paired provenance entry (state.py's calibration_provenance, same
    # merge_dicts reducer/shape as tools/calibration.py's bootstrap_*
    # functions) -- without this, tools/dashboard.py's
    # _calibration_section_html would see a non-None factor but a still-None
    # provenance entry and misreport this hw_spec_id as "uncalibrated".
    # `"precision_verified": True` keeps this branch distinguishable from a
    # real literature-reference match (`describe_calibration`) or the
    # nearest-neighbor scaling-law fallback (`describe_approximate_calibration`,
    # `"approximate": True`) -- this one is a Mock First (CLAUDE.md 5.D)
    # placeholder re-check, not a citation.
    calibration_provenance = (
        {
            state["hw_spec_id"]: {
                "precision_verified": True,
                "raw_energy_pj": precision_data.get("raw_energy_pj"),
                "precise_energy_pj": precision_data.get("precise_energy_pj"),
                "iteration": iteration_count,
            }
        }
        if calibration_factor is not None
        else {}
    )

    if is_precisely_converged:
        log_event(
            run_id_for(state),
            node="precision_verifier",
            event="precision_check_evaluated",
            iteration=iteration_count,
            is_converged=True,
            precise_energy_pj=precision_data.get("precise_energy_pj"),
            calibration_factor=calibration_factor,
        )
        return {
            "is_converged": True,
            "needs_hitl": False,
            "calibration_factors": calibration_factors,
            "calibration_provenance": calibration_provenance,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"[precision_verifier] iteration {iteration_count}: precisely verified -- "
                        f"precise_energy_pj={precision_data.get('precise_energy_pj')}"
                    ),
                }
            ],
        }

    new_retry_count = retry_count + 1
    needs_hitl = new_retry_count >= MAX_RETRY_LIMIT
    reason = "precision check: " + "; ".join(precision_data.get("target_errors") or ["precise re-verification failed"])
    failure_entry = {"iteration": iteration_count, "retry_count": new_retry_count, "reason": reason}
    log_event(
        run_id_for(state),
        node="precision_verifier",
        event="precision_check_evaluated",
        iteration=iteration_count,
        is_converged=False,
        needs_hitl=needs_hitl,
        retry_count=new_retry_count,
        reason=reason,
        precise_energy_pj=precision_data.get("precise_energy_pj"),
        calibration_factor=calibration_factor,
    )
    return {
        "is_converged": False,
        "needs_hitl": needs_hitl,
        "retry_count": new_retry_count,
        "failure_history": [failure_entry],
        "calibration_factors": calibration_factors,
        "calibration_provenance": calibration_provenance,
        "messages": [
            {
                "role": "system",
                "content": f"[precision_verifier] iteration {iteration_count}: {reason} -- fast-approximation was too optimistic",
            }
        ],
    }
