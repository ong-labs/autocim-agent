"""@planner node.

Research_Plan.md 2/3 attribute surrogate-model construction (BO/NSGA-II
normalization, LHS sampling) and history-based search-bound updates to
@planner. `tools/search.py` implements the real (if minimal-dependency)
version of that: real n-dimensional Latin Hypercube Sampling for the first
`warmup_count(n_stages)` iterations, then an IDW-surrogate + random-search
UCB acquisition once real `candidate_history` (state.py, populated by
@evaluator) exists -- one independent (weight_bits, column_pruning_ratio)
pair *per real model stage* (`tools.qat.get_layer_groups`), not a single
flat point fanned out via a fixed rule: the search space is genuinely
`2 * len(stage_names)`-dimensional (12 for resnet18's 6 stages, up to 40 for
mobilenet_v2's 20). `points_to_stage_configs` only rounds/clamps each
stage's searched point into a `LayerBitConfig` -- it introduces no bias of
its own; whatever differences exist between stages come from the search
itself finding them independently, not from a hardcoded edge-stage rule.
The LLM sees the real stage names and this per-stage suggestion, and makes
the final call via a forced tool call (`PlannerLayerDecision`,
schemas/planner.py) -- never free-form text parsed for numbers, which is
what Research_Plan.md 1 names as the "수치 환각 (Hallucination)" failure mode
of using an LLM alone. Hardware bounds (`HWConfig.adc_bits`/`dac_bits`) are
likewise stated as a constraint for the LLM to respect -- @tuner's own
tool-call validation is still the actual enforcement point for those
bounds, the LLM is not trusted for that.

This module also owns the state-sanitization contract every other node
depends on: when `human_overrides` was just produced by `hitl_node`,
consume it and reset `needs_hitl`/`retry_count` in the same step so it
doesn't re-trigger the same HITL interrupt forever (checklist item 3 /
CLAUDE.md 5.A).

Sanitization only fires when there actually is an override to consume.
`retry_count` must otherwise survive ordinary "Re-plan with History" loops
untouched -- it is `@evaluator`'s counter towards `MAX_RETRY_LIMIT`
(nodes/evaluator.py); resetting it unconditionally on every planner call
would erase that counter every time evaluator increments it, so
`needs_hitl` could never become True and the graph would loop forever
(reproduced via GraphRecursionError while testing graph.py).
"""

import zlib
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from llm import LLMCallFailed, LLMCallRecord, check_budget, get_planner_chat_model, invoke_with_retry
from middleware import get_hw_config
from nodes.common import run_id_for
from observability import log_event
from schemas.planner import PlannerLayerDecision
from schemas.tools import LayerBitConfig
from state import AutoCIMState
from tools.qat import get_layer_groups
from tools.search import Bounds, StagePoint, propose_stage_candidates_via_surrogate, warmup_stage_candidate

_FALLBACK_STAGE_NAME = "model_default"

# Forcing tool_choice by name requires the exact name the schema converts
# to; deriving it from the class avoids the two ever drifting apart.
_TOOL_NAME = PlannerLayerDecision.__name__

# Search-space policy constants (not hardware specs -- CLAUDE.md 5.C only
# prohibits hardcoding hardware numbers, and the upper bit-width bound below
# is still derived from HWConfig, never a literal).
#
# Warm-up count scales with the real per-stage dimensionality
# (2 * n_stages): a flat 3 samples is nowhere near enough to seed a
# 12-40-dimensional search space (curse of dimensionality), but scaling
# unboundedly (e.g. the textbook ~10x-dimensions BO heuristic) would mean
# hundreds of real QAT trials at 37s-2min each before the surrogate phase
# even starts. `warmup_count` is the deliberate middle ground: at least
# one warm-up sample per stage, capped at `_MAX_WARMUP_CANDIDATES` so a
# 20-stage model (mobilenet_v2) still bounds warm-up wall-clock instead of
# scaling with model size.
_WARMUP_CANDIDATES = 3
_MAX_WARMUP_CANDIDATES = 12
_WEIGHT_BITS_MIN = 2
_PRUNING_RATIO_MAX = 0.5


def warmup_count(n_stages: int) -> int:
    return min(_MAX_WARMUP_CANDIDATES, max(_WARMUP_CANDIDATES, n_stages))


def search_seed_for(model_id: str, hw_spec_id: str) -> int:
    """Deterministic per-`(model_id, hw_spec_id)` seed for LHS warm-up
    sampling -- crc32, not Python's builtin `hash()`, since str hashing is
    randomized per-process (`PYTHONHASHSEED`) unless disabled, which would
    make "same warm-up index -> same point" false across runs/processes.
    Public (not just inlined in `_propose_stage_points`) so
    `tools/batch_warmup.py`'s parallel warm-up runner derives the exact
    same seed the sequential path would have used for this
    `(model_id, hw_spec_id)` -- computing all warm-up points up front must
    reproduce, not diverge from, what `_propose_stage_points` would have
    produced one at a time."""
    return zlib.crc32(f"{model_id}::{hw_spec_id}".encode())


_SYSTEM_PROMPT = (
    "You are the @planner agent in AutoCIM-Agent, a CIM-chip HW-SW "
    "co-design optimizer. You are given the target model's real stages, a "
    "statistically-proposed per-stage (weight_bits, column_pruning_ratio) "
    "search candidate -- from Latin Hypercube warm-up sampling or a "
    "surrogate-model acquisition over prior measured candidates -- plus the "
    "target hardware's bounds and recent failure history. Finalize the next "
    f"iteration's layer-wise quantization/pruning config by calling `{_TOOL_NAME}` "
    "-- a tool call only, never plain text. Rules:\n"
    "1. layer_configs must have exactly one entry per name in "
    "real_model_stages, using that exact string as layer_name.\n"
    "2. Every entry, including the 2nd/3rd/etc., must set all four fields: "
    "layer_name, weight_bits, activation_bits, column_pruning_ratio -- never "
    "omit a field on any entry after the first.\n"
    "3. Start from each stage's per_stage_suggested_candidate as-is, or "
    "adjust a stage within the stated hw bounds only if recent_failure_history "
    "shows a specific convergence problem it doesn't address (e.g. persistent "
    "IR-drop/noise failure -> prefer fewer weight_bits or higher pruning)."
)


def _clamp_bounds(lo: float, hi: float, override_lo: Any, override_hi: Any) -> Bounds:
    """Narrows (lo, hi) using override_lo/override_hi if they are numeric --
    never widens past the hw-derived bound, and if a bad pair would invert
    the range (lo > hi), ignores the override entirely and falls back to
    the original (lo, hi) rather than producing an unusable search range."""
    new_lo = max(lo, override_lo) if isinstance(override_lo, (int, float)) else lo
    new_hi = min(hi, override_hi) if isinstance(override_hi, (int, float)) else hi
    if new_lo > new_hi:
        return lo, hi
    return new_lo, new_hi


def search_bounds(hw_spec_id: str, overrides: Optional[Dict[str, Any]] = None) -> Tuple[Bounds, Bounds]:
    """`overrides` (from `human_overrides`/HITL `new_bounds`) is applied here
    as a hard clamp on the actual search range -- not just advisory text in
    the LLM prompt (`_build_user_prompt`'s `researcher_overrides` line still
    shows it too, but that alone never guaranteed compliance, especially
    from smaller/local tool-calling models). Recognized keys: weight_bits_min,
    weight_bits_max, pruning_ratio_min, pruning_ratio_max -- all optional,
    and each can only narrow its dimension's hw-derived bound, never exceed
    the physical adc_bits/dac_bits-derived ceiling."""
    try:
        hw = get_hw_config(hw_spec_id)
        bits_max = min(8, hw.adc_bits, hw.dac_bits)
    except Exception:
        bits_max = 8
    bits_max = max(bits_max, _WEIGHT_BITS_MIN + 1)

    bit_bounds = (_WEIGHT_BITS_MIN, bits_max)
    pruning_bounds = (0.0, _PRUNING_RATIO_MAX)
    if overrides:
        bit_bounds = _clamp_bounds(*bit_bounds, overrides.get("weight_bits_min"), overrides.get("weight_bits_max"))
        pruning_bounds = _clamp_bounds(*pruning_bounds, overrides.get("pruning_ratio_min"), overrides.get("pruning_ratio_max"))
    return bit_bounds, pruning_bounds


def real_stage_names(model_id: str) -> List[str]:
    """Real per-stage layer names for `model_id` (tools/qat.py), in true
    front-to-back architectural order -- `group_quantizable_layers` builds
    its dict while iterating `model.named_modules()`, which visits stages in
    definition/forward order, and dict insertion order is preserved; do NOT
    sort alphabetically here, or the per-stage search points from
    `_propose_stage_points` would land on the wrong stage names when zipped
    against them. Falls back to a single synthetic stage name for an
    unknown `model_id` so this module still works before @tuner's model
    registry covers it -- @tuner then applies that one entry's config
    uniformly (tools/qat.py's `default_config` fallback)."""
    try:
        return list(get_layer_groups(model_id).keys())
    except Exception:
        return [_FALLBACK_STAGE_NAME]


def _propose_stage_points(
    state: AutoCIMState, stage_names: List[str], overrides: Optional[Dict[str, Any]] = None
) -> Tuple[StagePoint, str]:
    """Returns one (weight_bits, column_pruning_ratio) point per real stage
    (in `stage_names` order) and a short tag describing which search phase
    produced it, for the LLM prompt/messages."""
    bit_bounds, pruning_bounds = search_bounds(state["hw_spec_id"], overrides)
    history = state.get("candidate_history", [])
    n_warmup = warmup_count(len(stage_names))
    seed = search_seed_for(state["model_id"], state["hw_spec_id"])

    if len(history) < n_warmup:
        point = warmup_stage_candidate(len(history), n_warmup, len(stage_names), bit_bounds, pruning_bounds, seed)
        return point, f"LHS warm-up candidate {len(history) + 1}/{n_warmup}"
    point = propose_stage_candidates_via_surrogate(history, stage_names, bit_bounds, pruning_bounds, seed)
    return point, f"surrogate-model acquisition over {len(history)} prior candidates"


def points_to_stage_configs(stage_points: StagePoint, stage_names: List[str]) -> List[LayerBitConfig]:
    """One `LayerBitConfig` per real stage: each stage's own independently
    searched (weight_bits, column_pruning_ratio) point, just rounded/clamped
    to valid values -- no per-stage bias is applied here, since the search
    itself (`_propose_stage_points`) already explored each stage
    independently."""
    configs = []
    for name, (bits, pruning) in zip(stage_names, stage_points):
        stage_bits = max(_WEIGHT_BITS_MIN, round(bits))
        stage_pruning = round(min(max(pruning, 0.0), _PRUNING_RATIO_MAX), 4)
        configs.append(
            LayerBitConfig(
                layer_name=name, weight_bits=stage_bits, activation_bits=stage_bits, column_pruning_ratio=stage_pruning
            )
        )
    return configs


def _clamp_layer_configs(
    layer_configs: List[LayerBitConfig], overrides: Optional[Dict[str, Any]]
) -> List[LayerBitConfig]:
    """Hard-clamps the LLM's (or fallback's) final weight_bits/
    column_pruning_ratio into `human_overrides` (HITL `new_bounds`) after
    the fact -- mirrors this module's existing "the LLM is not trusted for
    [bound] enforcement" policy (module docstring re: HWConfig.adc_bits/
    dac_bits) applied specifically to human_overrides, since a forced
    tool-calling response honoring the `researcher_overrides` prompt text
    was never guaranteed (weaker local/cloud models especially -- see
    README's provider table).

    Deliberately narrower than `search_bounds()`'s (hw ceiling + override)
    pair: this only enforces the override values themselves (a no-op when
    `overrides` is empty, as on every non-HITL iteration), never the
    hw-derived ceiling on its own -- that stays exclusively @tuner's tool-
    call validation's job, same as before this function existed (a
    conflicting/inverted override pair, e.g. min > max, is ignored for that
    one dimension rather than forced to a nonsensical single value).
    activation_bits is left untouched: overrides only name weight_bits_min/
    max, not activation_bits."""
    if not overrides:
        return layer_configs
    bits_lo, bits_hi = overrides.get("weight_bits_min"), overrides.get("weight_bits_max")
    if isinstance(bits_lo, (int, float)) and isinstance(bits_hi, (int, float)) and bits_lo > bits_hi:
        bits_lo = bits_hi = None
    pruning_lo, pruning_hi = overrides.get("pruning_ratio_min"), overrides.get("pruning_ratio_max")
    if isinstance(pruning_lo, (int, float)) and isinstance(pruning_hi, (int, float)) and pruning_lo > pruning_hi:
        pruning_lo = pruning_hi = None
    if bits_lo is None and bits_hi is None and pruning_lo is None and pruning_hi is None:
        return layer_configs

    clamped = []
    for lc in layer_configs:
        weight_bits = lc.weight_bits
        if isinstance(bits_lo, (int, float)):
            weight_bits = max(weight_bits, round(bits_lo))
        if isinstance(bits_hi, (int, float)):
            weight_bits = min(weight_bits, round(bits_hi))
        pruning = lc.column_pruning_ratio
        if isinstance(pruning_lo, (int, float)):
            pruning = max(pruning, pruning_lo)
        if isinstance(pruning_hi, (int, float)):
            pruning = min(pruning, pruning_hi)
        pruning = round(pruning, 4)
        if weight_bits == lc.weight_bits and pruning == lc.column_pruning_ratio:
            clamped.append(lc)
        else:
            clamped.append(lc.model_copy(update={"weight_bits": weight_bits, "column_pruning_ratio": pruning}))
    return clamped


def _describe_hw_bounds(hw_spec_id: str) -> str:
    try:
        hw = get_hw_config(hw_spec_id)
    except Exception:
        return f"hw_spec_id={hw_spec_id!r} (HWConfig not registered yet; assume a conservative 8-bit bound)"
    return (
        f"hw_spec_id={hw.hw_spec_id!r}, crossbar={hw.crossbar_rows}x{hw.crossbar_cols}, "
        f"num_tiles={hw.num_tiles}, adc_bits={hw.adc_bits}, dac_bits={hw.dac_bits}"
    )


def _build_user_prompt(
    state: AutoCIMState,
    overrides: Dict[str, Any],
    search_tag: str,
    stage_configs: List[LayerBitConfig],
) -> str:
    verifier_data = (state.get("metrics_store", {}).get("verifier") or {}).get("data") or {}
    suggested = ", ".join(
        f"{lc.layer_name}(weight_bits={lc.weight_bits}, column_pruning_ratio={lc.column_pruning_ratio})"
        for lc in stage_configs
    )
    lines = [
        f"model_id: {state['model_id']}",
        f"hardware: {_describe_hw_bounds(state['hw_spec_id'])}",
        f"real_model_stages (propose one layer_configs entry per stage, using these exact layer_name values): "
        f"{[lc.layer_name for lc in stage_configs]}",
        # Each stage below was searched independently ({search_tag}) -- not
        # one flat point fanned out by a fixed rule.
        f"per_stage_suggested_candidate ({search_tag}): {suggested}",
        f"iteration_count: {state.get('iteration_count', 0)}",
        f"recent_failure_history: {state.get('failure_history', [])[-3:]}",
        f"latest_verifier_metrics: {verifier_data}",
    ]
    if overrides:
        lines.append(
            f"researcher_overrides (already applied as hard bounds on per_stage_suggested_candidate above; "
            f"stay within them): {overrides}"
        )
    return "\n".join(lines)


def _extract_decision(response: Any) -> PlannerLayerDecision:
    for call in getattr(response, "tool_calls", None) or []:
        if call.get("name") == _TOOL_NAME:
            return PlannerLayerDecision(**call["args"])
    raise RuntimeError(f"planner LLM response had no {_TOOL_NAME!r} tool call: {response!r}")


def _propose_layer_configs(
    state: AutoCIMState,
    overrides: Dict[str, Any],
    search_tag: str,
    stage_configs: List[LayerBitConfig],
    iteration_count: int,
) -> Tuple[List[LayerBitConfig], str, "LLMCallRecord"]:
    """Real LLM call, retried on transient errors via `llm.invoke_with_retry`
    (rate limits/timeouts/5xx -- see llm.py), then forced through
    tool-calling. Raises `LLMCallFailed` -- carrying the real attempts/
    tokens/cost `LLMCallRecord` even for this failure, so a billable call
    that came back malformed still gets logged accurately -- on any failure
    the retry layer didn't absorb (including a well-formed response with no
    usable tool call); the caller must catch that and fall back to
    `stage_configs` directly. `get_planner_chat_model()` itself raising
    (e.g. missing API key/model env var) is not caught here -- that's a
    config problem retrying can't fix, and has no LLM call to build a
    record from; the caller handles it as a plain exception. This overall
    containment mirrors `middleware.wrap_tool_call`'s containment of
    backend-tool exceptions."""
    chat_model = get_planner_chat_model().bind_tools([PlannerLayerDecision], tool_choice=_TOOL_NAME)
    response, usage_record = invoke_with_retry(
        chat_model,
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(state, overrides, search_tag, stage_configs)),
        ],
        node="planner",
        iteration=iteration_count,
    )
    try:
        decision = _extract_decision(response)
    except Exception as exc:  # a real (billable) call came back malformed -- not a transient error to retry
        usage_record.succeeded = False
        usage_record.error = f"{type(exc).__name__}: {exc}"
        raise LLMCallFailed(usage_record) from exc
    return decision.layer_configs, decision.rationale, usage_record


def planner_node(state: AutoCIMState) -> Dict[str, Any]:
    iteration_count = state.get("iteration_count", 0) + 1
    overrides = state.get("human_overrides", {})  # read once, before clearing

    update: Dict[str, Any] = {"iteration_count": iteration_count}
    messages: List[Dict[str, Any]] = []

    if overrides:
        # Sanitize immediately after consuming (checklist item 3): prevents
        # this override / the HITL flag / the retry counter that triggered
        # it from leaking into the next iteration.
        update["human_overrides"] = {}
        update["needs_hitl"] = False
        update["retry_count"] = 0
        messages.append(
            {
                "role": "system",
                "content": f"[planner] applied human_overrides at iteration {iteration_count}: {overrides}",
            }
        )

    stage_names = real_stage_names(state["model_id"])
    stage_points, search_tag = _propose_stage_points(state, stage_names, overrides)
    stage_configs = points_to_stage_configs(stage_points, stage_names)

    budget_status = check_budget(state.get("llm_usage", []))
    if budget_status.exceeded:
        # Kill-switch: skip the LLM call entirely (never even constructs a
        # chat model) rather than retry/fallback after spending more --
        # retry/backoff (llm.invoke_with_retry) only contains failures
        # *within* one call, it has no concept of a cumulative per-run cap.
        used_llm = False
        usage_record = LLMCallRecord(
            node="planner", iteration=iteration_count, attempts=0, succeeded=False, latency_ms=0.0, error=budget_status.reason
        )
        layer_configs = stage_configs
        rationale = f"LLM call skipped: {budget_status.reason}"
        messages.append({"role": "system", "content": f"[planner] {rationale}; using the {search_tag} search point directly"})
    else:
        used_llm = True
        try:
            layer_configs, rationale, usage_record = _propose_layer_configs(
                state, overrides, search_tag, stage_configs, iteration_count
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"[planner] iteration {iteration_count} ({search_tag}): {rationale}",
                }
            )
        except LLMCallFailed as exc:  # a real LLM call happened (possibly after retries) but didn't pan out
            used_llm = False
            usage_record = exc.record
            layer_configs = stage_configs
            rationale = f"LLM proposal failed after {usage_record.attempts} attempt(s) ({usage_record.error})"
            messages.append({"role": "system", "content": f"[planner] {rationale}; using the {search_tag} search point directly"})
        except Exception as exc:  # no LLM call was ever made (e.g. get_planner_chat_model() config error)
            used_llm = False
            usage_record = LLMCallRecord(
                node="planner", iteration=iteration_count, attempts=0, succeeded=False, latency_ms=0.0, error=f"{type(exc).__name__}: {exc}"
            )
            layer_configs = stage_configs
            rationale = f"LLM proposal failed ({type(exc).__name__}: {exc})"
            messages.append({"role": "system", "content": f"[planner] {rationale}; using the {search_tag} search point directly"})

    # Re-clamped even on the non-LLM fallback paths (a no-op there, since
    # stage_configs already came from the override-aware bounds above) --
    # one enforcement point for every path instead of trusting each branch
    # to have respected overrides on its own.
    layer_configs = _clamp_layer_configs(layer_configs, overrides)
    layer_config_dicts = [lc.model_dump(mode="json") for lc in layer_configs]
    update["planned_layer_configs"] = layer_config_dicts
    update["llm_usage"] = [usage_record.to_dict()]
    decision_entry = {
        "iteration": iteration_count,
        "search_tag": search_tag,
        "used_llm": used_llm,
        "rationale": rationale,
        "layer_configs": layer_config_dicts,
    }
    update["planner_decisions"] = [decision_entry]
    log_event(
        run_id_for(state),
        node="planner",
        event="candidate_proposed",
        iteration=iteration_count,
        search_tag=search_tag,
        used_llm=used_llm,
        rationale=rationale,
        n_stages=len(stage_names),
    )
    update["messages"] = messages

    return update
