"""AutoCIMState and reducers for the LangGraph pipeline.

Reducers here exist to make parallel fan-in (mapper/profiler -> verifier)
safe: each node writes only into its own namespace of `metrics_store`, so
concurrent updates never overwrite each other (see CLAUDE.md 5.A).

`candidate_history` uses the same append-only `operator.add` reducer as
`failure_history`: @evaluator appends one entry per fully-evaluated
candidate (tools/search.py), and @planner reads the accumulated list back
to drive its warm-up/surrogate-guided search (nodes/planner.py).

`llm_usage` is the same append-only shape, one entry per real LLM call
@planner makes (llm.py's `invoke_with_retry`) -- since @planner calls the
LLM once per iteration, this is the only place total attempts/tokens/cost
across a whole run can be read back without re-deriving it from raw
provider logs.

`planner_decisions` is the same append-only shape, one entry per @planner
call, structurally recording *which* candidate was proposed and *why*
(LHS warm-up vs surrogate acquisition, LLM rationale) -- previously this
only existed as a free-text line in `messages`. `tools/dashboard.py` joins
this against `candidate_history` by `iteration` to show, per iteration,
both what was tried and what it measured.

`calibration_provenance` mirrors `calibration_factors` (`{hw_spec_id: {...}}`,
same `merge_dicts` reducer) but carries *why* a factor is what it is --
the matching literature reference's citation and stated uncertainty
(`tools.calibration.describe_calibration`) -- so `tools/dashboard.py` can
show a researcher which published number backs a candidate's energy
figure, not just an opaque multiplier. Empty for any `hw_spec_id` that
stayed uncalibrated (factor 1.0), same honesty rule as
`calibration_factors` itself.

`search_bound_overrides` accumulates every `human_overrides`/HITL `new_bounds`
a researcher has approved so far this session (nodes/planner.py merges each
newly-consumed `human_overrides` into it via the same `merge_dicts` reducer).
Unlike `human_overrides` itself -- which nodes/planner.py clears the same
step it reads it (checklist item 3 / CLAUDE.md 5.A), so the same HITL
interrupt doesn't re-fire forever -- the *bounds* a researcher already
approved keep constraining every later candidate's `search_bounds()` for the
rest of the run, not just the one candidate right after approval. A later
HITL round narrowing the same key (e.g. tightening `pruning_ratio_max`
again) overwrites that key's earlier value; other keys are untouched.

`target_accuracy`/`target_energy_pj`/`target_latency_ms` are optional,
run-wide multi-objective goals (nodes/evaluator.py: a candidate must meet
every *configured* one, in addition to @verifier's own IR-drop/noise
feasibility check, to count as converged) -- plain fields set once at
session start (main.py's CLI flags), not reduced, since they don't change
mid-run. `None` (the default) means that dimension isn't gated at all,
preserving prior behavior for any run that doesn't set them: before these
fields existed, @verifier's IR-drop/noise check was the *only* convergence
criterion, so a candidate with unacceptably low accuracy could still end
a run early -- these fields are the fix for that gap, not a new feature
invented speculatively (Research_Plan.md's own evaluator_node sketch names
"Target 지표 미달성" as a failure reason, implying an accuracy/energy/latency
target was always part of the intended design, just never implemented).

`planner_flagged_anomaly` is @planner's LLM-set early-warning signal
(schemas/planner.py's `PlannerLayerDecision.anomaly_note`) -- unlike
`human_overrides`, this needs no explicit reset: @planner sets it fresh
every iteration (`None` when nothing was flagged that iteration), so a
stale flag never leaks into a later one. @evaluator ORs it into
`needs_hitl` alongside the usual `retry_count >= MAX_RETRY_LIMIT` check,
so a researcher can be pulled in immediately on something the LLM finds
concerning, not only after 3 routine misses.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge two dicts, right taking precedence on key conflicts."""
    if not left:
        return right or {}
    if not right:
        return left or {}
    merged = left.copy()
    merged.update(right)
    return merged


def namespaced_metrics_reducer(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge node outputs into `metrics_store` under each node's own namespace.

    e.g. {"mapper": {...}} merged with {"profiler": {...}} ->
    {"mapper": {...}, "profiler": {...}}, so parallel writers never race.
    """
    merged = left.copy() if left else {}
    if not right:
        return merged

    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    return merged


class AutoCIMState(TypedDict):
    messages: Annotated[list, operator.add]
    failure_history: Annotated[list, operator.add]
    candidate_history: Annotated[List[Dict[str, Any]], operator.add]
    llm_usage: Annotated[List[Dict[str, Any]], operator.add]
    planner_decisions: Annotated[List[Dict[str, Any]], operator.add]
    metrics_store: Annotated[Dict[str, Any], namespaced_metrics_reducer]
    calibration_factors: Annotated[Dict[str, float], merge_dicts]
    calibration_provenance: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    human_overrides: Dict[str, Any]
    search_bound_overrides: Annotated[Dict[str, Any], merge_dicts]
    planned_layer_configs: List[Dict[str, Any]]
    planner_flagged_anomaly: Optional[str]
    model_id: str
    hw_spec_id: str
    target_accuracy: Optional[float]
    target_energy_pj: Optional[float]
    target_latency_ms: Optional[float]
    iteration_count: int
    retry_count: int
    is_converged: bool
    needs_hitl: bool
