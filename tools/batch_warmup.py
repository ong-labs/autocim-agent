"""Parallel evaluation of independent LHS warm-up candidates.

nodes/planner.py's warm-up phase proposes exactly one candidate per graph
iteration (index = `len(candidate_history)`), so the sequential graph loop
evaluates all `warmup_count(n_stages)` warm-up candidates one real QAT
trial at a time -- even though they're mathematically independent:
`tools.search.latin_hypercube_stage_candidates` computes every warm-up
point up front from a deterministic seed, and nothing about candidate i
depends on candidate i-1's *result* the way the surrogate phase does.

`run_parallel_warmup` evaluates all of them concurrently via a thread
pool, through the same tuner -> mapper -> profiler -> verifier
computations nodes/tuner.py through nodes/verifier.py drive one at a time
-- PyTorch's C++/CUDA kernels (and NumPy's BLAS calls in the analytical
mapper/profiler math) release the GIL during the actual compute, so
CPU-bound QAT trials genuinely parallelize across threads, not just
I/O-bound work. `main.py` seeds `AutoCIMState.candidate_history` with the
results before its sequential graph loop starts, so @planner's search-space
history already reflects every warm-up candidate instead of the loop
re-discovering them one iteration at a time.

Two deliberate departures from the sequential path, both real behavior
differences, not incidental shortcuts:

- Bypasses `middleware.wrap_tool_call`'s `ExecutionContext`/`HWConfig`
  contextvar+registry injection -- calling `tuner_tool.__wrapped_tool__`
  etc. (the raw, undecorated functions; see `middleware.wrap_tool_call`'s
  docstring) with an explicitly-built `*ToolInput` instead. `contextvars`
  do not propagate to worker threads spawned by `ThreadPoolExecutor` (each
  starts with a fresh, empty context), and this path isn't LLM-driven, so
  there's no hallucination risk for `wrap_tool_call` to guard against here
  -- building each `*ToolInput` directly from values already in hand is
  simpler and thread-safe by construction, rather than fighting contextvar
  propagation.
- Skips the LLM confirmation step nodes/planner.py's sequential path makes
  every iteration: firing `warmup_count(n_stages)` real LLM calls
  concurrently would multiply cost/rate-limit exposure for what's meant to
  be a fast parallel pre-processing step (llm.py's retry/budget machinery
  is designed around one call per graph iteration, not a concurrent
  burst). Each candidate here uses its raw LHS point directly -- exactly
  `nodes/planner.py`'s own "LLM proposal failed, fall back to the search
  point" path, just taken deliberately instead of after a failed call. The
  LLM still confirms/adjusts every candidate normally once the sequential
  graph loop takes over after warm-up.

GPU note: if `tools.qat._resolve_device` resolves to a single GPU (the
common case), every worker thread's QAT trial targets that same GPU --
unlike CPU parallelism across cores, concurrent trials here compete for
one device's memory. Keep `max_workers` modest (or force
`AUTOCIM_QAT_DEVICE=cpu` for this step specifically) if warm-up starts
hitting out-of-memory errors; this module does not attempt to manage GPU
memory itself.

Known limitation, accepted rather than solved here: if a warm-up candidate
happens to converge, this module still records it (correctly, at real
pareto_rank 1) but does not short-circuit the run -- unlike the sequential
graph loop, which would immediately route to "Converged (Done)" the moment
one iteration converges. `main.py`'s sequential loop still runs its first
real (surrogate-phase) iteration after warm-up regardless. Solving that
properly means teaching `run_session` to recognize "warm-up already found
a converged candidate" and skip straight to done -- a real control-flow
change to `main.py` beyond this module's scope, not attempted here.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from nodes.evaluator import build_candidate_entry, validate_metrics
from nodes.planner import points_to_stage_configs, real_stage_names, search_bounds, search_seed_for, warmup_count
from schemas.config import ExecutionContext, HWConfig
from schemas.tools import MapperToolInput, ProfilerToolInput, ToolStatus, TunerToolInput, VerifierToolInput
from tools.search import latin_hypercube_stage_candidates
from tools.simulators import mapper_tool, profiler_tool, tuner_tool, verifier_tool

_DEFAULT_MAX_EPOCHS = 5  # mirrors nodes/tuner.py's _DEFAULT_MAX_EPOCHS -- kept in lockstep intentionally


def _failed_result_dict(error: str) -> Dict[str, Any]:
    return {"status": ToolStatus.FAILED.value, "data": None, "error": error}


def _evaluate_one_candidate(
    model_id: str, hw_config: HWConfig, layer_configs: List[Any], calibration_factors: Optional[Dict[str, float]]
) -> Dict[str, Any]:
    """Runs one candidate through the raw tuner/mapper/profiler/verifier
    functions (bypassing `wrap_tool_call`'s contextvar injection -- see
    module docstring) and returns a `metrics_store`-shaped dict (the same
    shape `evaluator_node` reads from `state["metrics_store"]`), so
    `validate_metrics`/`build_candidate_entry` work unmodified on it."""
    execution_context = ExecutionContext(
        session_id=f"{model_id}::{hw_config.hw_spec_id}",
        model_id=model_id,
        hw_spec_id=hw_config.hw_spec_id,
        iteration_count=0,
    )

    tuner_result = tuner_tool.__wrapped_tool__(
        TunerToolInput(
            execution_context=execution_context,
            hw_config=hw_config,
            layer_configs=layer_configs,
            max_epochs=_DEFAULT_MAX_EPOCHS,
        )
    )
    tuner_dict = tuner_result.model_dump(mode="json")

    if tuner_result.status != ToolStatus.SUCCESS:
        # Mirrors nodes/mapper.py's/nodes/profiler.py's own real behavior on
        # a failed @tuner: they never short-circuit (they'd just get an
        # empty layer_configs list and fail their own validation) -- this
        # produces the same net FAILED-with-reason effect without actually
        # re-running mapper/profiler against nothing.
        reason = f"tuner failed: {tuner_result.error}"
        return {
            "tuner": tuner_dict,
            "mapper": _failed_result_dict(reason),
            "profiler": _failed_result_dict(reason),
            "verifier": _failed_result_dict(reason),
        }

    tuner_data = tuner_result.data or {}
    model_ref = tuner_data.get("model_ref", model_id)
    tuned_layer_configs = tuner_data.get("layer_configs") or []

    mapper_result = mapper_tool.__wrapped_tool__(
        MapperToolInput(
            execution_context=execution_context,
            hw_config=hw_config,
            model_ref=model_ref,
            layer_configs=tuned_layer_configs,
        )
    )
    profiler_result = profiler_tool.__wrapped_tool__(
        ProfilerToolInput(
            execution_context=execution_context,
            hw_config=hw_config,
            model_ref=model_ref,
            layer_configs=tuned_layer_configs,
            calibration_factors=calibration_factors,
        )
    )
    verifier_result = verifier_tool.__wrapped_tool__(
        VerifierToolInput(
            execution_context=execution_context,
            hw_config=hw_config,
            mapper_result=mapper_result,
            profiler_result=profiler_result,
        )
    )

    return {
        "tuner": tuner_dict,
        "mapper": mapper_result.model_dump(mode="json"),
        "profiler": profiler_result.model_dump(mode="json"),
        "verifier": verifier_result.model_dump(mode="json"),
    }


def run_parallel_warmup(
    model_id: str,
    hw_config: HWConfig,
    calibration_factors: Optional[Dict[str, float]] = None,
    max_workers: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluates every LHS warm-up candidate for `(model_id, hw_config)`
    concurrently and returns `(candidate_history_entries, failure_entries)`
    -- ready to seed `AutoCIMState.candidate_history`/`failure_history`
    (state.py) before `main.py`'s sequential graph loop starts.
    `calibration_factors` should be the same dict `main.py` seeds into the
    initial state (`tools.calibration.bootstrap_calibration_factors`) --
    passed through to every trial's `profiler_tool` call so warm-up
    candidates' energy figures are corrected exactly like the sequential
    path's would be, not silently left uncalibrated.

    Pareto rank is computed in a second, sequential pass (by warm-up index,
    ascending -- the same order the sequential graph loop would have
    produced them in) *after* every parallel trial finishes: computing it
    incrementally *during* parallel execution would make each candidate's
    rank depend on how many sibling threads happened to finish first, a
    race with no well-defined answer. `max_workers` defaults to
    `min(n_warmup, cpu_count)` -- never more threads than there are
    candidates, and not more than this machine's real core count."""
    stage_names = real_stage_names(model_id)
    bit_bounds, pruning_bounds = search_bounds(hw_config.hw_spec_id)
    n_warmup = warmup_count(len(stage_names))
    seed = search_seed_for(model_id, hw_config.hw_spec_id)

    stage_points = latin_hypercube_stage_candidates(n_warmup, len(stage_names), bit_bounds, pruning_bounds, seed)
    layer_configs_per_candidate = [points_to_stage_configs(points, stage_names) for points in stage_points]

    resolved_max_workers = max_workers if max_workers is not None else min(n_warmup, os.cpu_count() or 1)
    resolved_max_workers = max(1, resolved_max_workers)

    with ThreadPoolExecutor(max_workers=resolved_max_workers) as executor:
        raw_metrics_by_index = list(
            executor.map(
                lambda layer_configs: _evaluate_one_candidate(model_id, hw_config, layer_configs, calibration_factors),
                layer_configs_per_candidate,
            )
        )

    candidate_history: List[Dict[str, Any]] = []
    failure_history: List[Dict[str, Any]] = []
    for warmup_index, metrics in enumerate(raw_metrics_by_index):
        validation_errors = validate_metrics(metrics)
        verifier_data = (metrics.get("verifier") or {}).get("data") or {}
        is_converged = not validation_errors and bool(verifier_data.get("is_converged", False))

        candidate_entry = build_candidate_entry(
            candidate_history, metrics, warmup_index + 1, is_converged, has_validation_errors=bool(validation_errors)
        )
        if candidate_entry is not None:
            candidate_history.append(candidate_entry)
        else:
            failure_history.append(
                {
                    "iteration": warmup_index + 1,
                    "retry_count": 0,
                    "reason": (
                        "; ".join(validation_errors) if validation_errors else "verifier reported not converged"
                    )
                    + " (parallel warm-up candidate)",
                }
            )

    return candidate_history, failure_history
