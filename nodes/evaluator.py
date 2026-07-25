"""@evaluator node + router.

Research_Plan.md 2절: "Pydantic Schema 인터페이스 검증, State Store 이력 누적,
수렴도 평가 및 자율 피드백 라우팅". This node validates every tool output
produced this iteration against its `schemas/tools.py` model, accumulates
`failure_history` on non-convergence, and decides `is_converged`/
`needs_hitl`/`retry_count`; `evaluator_router` reads those flags to route
the corresponding conditional edge (Converged / HITL / Re-plan).

This is also where `candidate_history` (state.py) gets its one entry per
fully-evaluated iteration: @evaluator is the first node with the complete
multi-objective picture (accuracy from @tuner, energy/latency from
@verifier's synthesis of @mapper+@profiler), so it's the only place a real
Pareto rank (tools/search.py) can be computed -- @tuner's own QAT trial has
no basis to rank itself against candidates it never saw. @planner reads
`candidate_history` back to drive its warm-up/surrogate-guided search.

MAX_RETRY_LIMIT is a control-flow constant (loop-termination policy), not a
hardware parameter, so it is defined here rather than sourced from
`HWConfig` -- CLAUDE.md 5.C only prohibits hardcoding hardware specs.
"""

from typing import Any, Dict, List, Literal, Optional

from nodes.common import run_id_for
from observability import log_event
from schemas.tools import (
    MapperToolOutput,
    ProfilerToolOutput,
    ToolStatus,
    TunerToolOutput,
    VerifierToolOutput,
)
from state import AutoCIMState
from tools.search import compute_pareto_rank

MAX_RETRY_LIMIT = 3

_OUTPUT_SCHEMAS = (
    ("tuner", TunerToolOutput),
    ("mapper", MapperToolOutput),
    ("profiler", ProfilerToolOutput),
    ("verifier", VerifierToolOutput),
)


def validate_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Validation errors (empty if all four tools' outputs are present and
    schema-valid with `ToolStatus.SUCCESS`) -- pulled out of `evaluator_node`
    so `tools/batch_warmup.py`'s parallel warm-up runner validates each
    candidate's raw tool results with the exact same rule the sequential
    graph loop uses, rather than a second, potentially-diverging copy of
    this logic."""
    validation_errors = []
    for node_name, schema in _OUTPUT_SCHEMAS:
        raw = metrics.get(node_name)
        if raw is None:
            validation_errors.append(f"{node_name}: missing from metrics_store")
            continue
        try:
            parsed = schema(**raw)
        except Exception as exc:  # noqa: BLE001 -- record and keep validating the rest
            validation_errors.append(f"{node_name}: {exc}")
            continue
        if parsed.status != ToolStatus.SUCCESS:
            validation_errors.append(f"{node_name}: status={parsed.status}, error={parsed.error}")
    return validation_errors


def build_candidate_entry(
    history: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    iteration_count: int,
    is_converged: bool,
    has_validation_errors: bool,
) -> Optional[Dict[str, Any]]:
    """None when this iteration's multi-objective result is incomplete (a
    sub-tool failed/is missing, or is otherwise schema-invalid) -- an
    incomplete candidate has no trustworthy accuracy/energy/latency to rank
    or to feed the surrogate, so it must not be recorded into
    `candidate_history`. This is deliberately keyed off `has_validation_errors`
    rather than only "are these fields non-None": @verifier always returns
    ToolStatus.SUCCESS itself and echoes whatever @mapper/@profiler gave it,
    so a failed upstream tool doesn't reliably leave verifier's own fields
    None -- `validate_metrics`'s result is the authoritative signal that
    this iteration's numbers are trustworthy.

    Takes `history` (the accumulated `candidate_history` to rank against)
    directly rather than the full `AutoCIMState` -- `tools/batch_warmup.py`
    builds entries for candidates that were never part of any graph state,
    just a growing in-memory list, and this keeps that reuse a plain data
    dependency instead of requiring a fake state dict."""
    if has_validation_errors:
        return None

    tuner_data = (metrics.get("tuner") or {}).get("data") or {}
    verifier_data = (metrics.get("verifier") or {}).get("data") or {}

    candidate = {
        "iteration": iteration_count,
        "avg_weight_bits": tuner_data.get("avg_weight_bits"),
        "avg_column_pruning_ratio": tuner_data.get("avg_column_pruning_ratio"),
        # Full per-stage (weight_bits, column_pruning_ratio) record, the
        # exact config @tuner actually used this iteration -- @planner's
        # per-stage surrogate (tools.search.predict_accuracy_stages) needs
        # this, not just the across-stage averages above, to place this
        # candidate in the real 2*n_stages-dimensional search space.
        "layer_configs": tuner_data.get("layer_configs"),
        "accuracy": tuner_data.get("accuracy"),
        "energy_pj": verifier_data.get("energy_pj"),
        "noc_latency_ms": verifier_data.get("noc_latency_ms"),
        "is_converged": is_converged,
    }
    if any(candidate[key] is None for key in ("avg_weight_bits", "avg_column_pruning_ratio", "accuracy", "energy_pj", "noc_latency_ms")):
        return None

    candidate["pareto_rank"] = compute_pareto_rank(candidate, history)
    return candidate


def evaluator_node(state: AutoCIMState) -> Dict[str, Any]:
    metrics = state.get("metrics_store", {})
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)

    validation_errors = validate_metrics(metrics)

    verifier_data = (metrics.get("verifier") or {}).get("data") or {}
    is_converged = not validation_errors and bool(verifier_data.get("is_converged", False))

    candidate_entry = build_candidate_entry(
        state.get("candidate_history", []), metrics, iteration_count, is_converged, has_validation_errors=bool(validation_errors)
    )
    candidate_history: List[Dict[str, Any]] = [candidate_entry] if candidate_entry is not None else []

    if is_converged:
        log_event(
            run_id_for(state),
            node="evaluator",
            event="candidate_evaluated",
            iteration=iteration_count,
            is_converged=True,
            needs_hitl=False,
            accuracy=(candidate_entry or {}).get("accuracy"),
            energy_pj=(candidate_entry or {}).get("energy_pj"),
            noc_latency_ms=(candidate_entry or {}).get("noc_latency_ms"),
            pareto_rank=(candidate_entry or {}).get("pareto_rank"),
        )
        return {"is_converged": True, "needs_hitl": False, "candidate_history": candidate_history}

    new_retry_count = retry_count + 1
    needs_hitl = new_retry_count >= MAX_RETRY_LIMIT
    failure_entry = {
        "iteration": iteration_count,
        "retry_count": new_retry_count,
        "reason": "; ".join(validation_errors) if validation_errors else "verifier reported not converged",
    }
    log_event(
        run_id_for(state),
        node="evaluator",
        event="candidate_evaluated",
        iteration=iteration_count,
        is_converged=False,
        needs_hitl=needs_hitl,
        retry_count=new_retry_count,
        reason=failure_entry["reason"],
        accuracy=(candidate_entry or {}).get("accuracy"),
        energy_pj=(candidate_entry or {}).get("energy_pj"),
        noc_latency_ms=(candidate_entry or {}).get("noc_latency_ms"),
        pareto_rank=(candidate_entry or {}).get("pareto_rank"),
    )

    return {
        "is_converged": False,
        "needs_hitl": needs_hitl,
        "retry_count": new_retry_count,
        "failure_history": [failure_entry],
        "candidate_history": candidate_history,
    }


def evaluator_router(state: AutoCIMState) -> Literal["Converged (Done)", "HITL Interrupt", "Re-plan with History"]:
    if state.get("is_converged", False):
        return "Converged (Done)"
    if state.get("needs_hitl", False):
        return "HITL Interrupt"
    return "Re-plan with History"
