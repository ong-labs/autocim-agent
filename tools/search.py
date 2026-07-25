"""Real multi-candidate search for @planner (Research_Plan.md 2/3's "BO/NSGA-II
정규화 및 LHS 최소 샘플링 기반 대리 모델" and "정규화 다중 목적 탐색").

Previously @planner asked its LLM for a candidate with no statistical
grounding, and @tuner reported a hardcoded `pareto_rank: 1` that no single
QAT trial can actually justify (Pareto rank is inherently relative to other
evaluated candidates). This module is the missing piece underneath the LLM
call:

- `warmup_candidate`/`propose_via_surrogate`: the original 2D (single
  weight_bits, column_pruning_ratio point for the whole model) search --
  real Latin Hypercube Sampling for the first few iterations, then a
  lightweight real surrogate (inverse-distance-weighted regression +
  upper-confidence-bound acquisition) once there's history. Kept as-is for
  callers/tests that only need one flat point.
- `warmup_stage_candidate`/`propose_stage_candidates_via_surrogate`: the
  real per-stage generalization. A candidate is one (weight_bits,
  column_pruning_ratio) pair *per real model stage*
  (`tools.qat.get_layer_groups`), so the search space is
  `2 * len(stage_names)`-dimensional (12 for resnet18's 6 stages, up to 40
  for mobilenet_v2's 20) instead of a fixed 2D point that
  `nodes/planner.py` used to fan out to every stage via one hardcoded
  edge-stage-bias rule. `warmup_stage_candidate` still uses real LHS (now
  n-dimensional -- each of the `2 * n_stages` dimensions independently
  stratified, same construction as the 2D case). `propose_stage_candidates_via_surrogate`
  still fits the same real IDW surrogate over accumulated history, but
  since an exhaustive grid scan is `grid_size ** (2*n_stages)` --
  computationally infeasible past a handful of dimensions -- its
  acquisition step samples `n_candidates` random points from the space and
  takes the best-scoring one instead of scanning a grid. Still a real,
  minimal-dependency data-driven surrogate plus an explore/exploit
  acquisition rule (not LLM guesswork, not a GP/NSGA-II library), just one
  that actually scales to this space's real dimensionality.
- `compute_pareto_rank`: real non-dominated-sorting-style rank (NSGA-II's
  core ranking idea) against the actual accumulated `candidate_history`,
  computed once real multi-objective results (accuracy, energy, latency)
  exist for a candidate -- replacing the old hardcoded value.

@planner still makes the final call via forced tool-calling
(nodes/planner.py) -- this module produces the statistically-grounded
suggestion the LLM prompt is anchored to, not a replacement for it.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence, Tuple

Bounds = Tuple[float, float]


# ---------------------------------------------------------------------------
# Warm-up: real Latin Hypercube Sampling
# ---------------------------------------------------------------------------


def _latin_hypercube_1d(n_samples: int, bounds: Bounds, seed: int) -> List[float]:
    """One real LHS dimension: `n_samples` equal-width strata, one uniform
    sample per stratum, then shuffled -- every stratum is used exactly once,
    which is what distinguishes LHS from plain uniform random sampling."""
    low, high = bounds
    rng = random.Random(seed)
    width = (high - low) / n_samples
    points = [rng.uniform(low + i * width, low + (i + 1) * width) for i in range(n_samples)]
    rng.shuffle(points)
    return points


def latin_hypercube_candidates(
    n_samples: int, bit_bounds: Bounds, pruning_bounds: Bounds, seed: int
) -> List[Tuple[float, float]]:
    """`n_samples` (weight_bits, column_pruning_ratio) points via 2D LHS:
    each dimension independently stratified via `_latin_hypercube_1d`, then
    paired by index -- the standard LHS construction."""
    bits = _latin_hypercube_1d(n_samples, bit_bounds, seed)
    pruning = _latin_hypercube_1d(n_samples, pruning_bounds, seed + 1)
    return list(zip(bits, pruning))


def warmup_candidate(
    index: int, n_warmup: int, bit_bounds: Bounds, pruning_bounds: Bounds, seed: int
) -> Tuple[float, float]:
    """The `index`-th warm-up candidate. Deterministic in `(index, seed)`, so
    it never needs its own cursor in `AutoCIMState` -- calling this again
    with the same `index` (e.g. a HITL-resumed retry of the same iteration)
    reproduces the same point rather than silently sampling a new one."""
    candidates = latin_hypercube_candidates(n_warmup, bit_bounds, pruning_bounds, seed)
    return candidates[index % n_warmup]


# ---------------------------------------------------------------------------
# Per-stage (n-dimensional) warm-up: real n-D Latin Hypercube Sampling
# ---------------------------------------------------------------------------

StagePoint = List[Tuple[float, float]]  # one (weight_bits, column_pruning_ratio) pair per real stage


def latin_hypercube_stage_candidates(
    n_samples: int, n_stages: int, bit_bounds: Bounds, pruning_bounds: Bounds, seed: int
) -> List[StagePoint]:
    """`n_samples` candidates, each a list of `n_stages` independent
    (weight_bits, column_pruning_ratio) pairs -- real n-D LHS: each of the
    `2 * n_stages` dimensions (two per stage) is stratified independently
    via `_latin_hypercube_1d` with its own seed offset, exactly like
    `latin_hypercube_candidates`'s 2 dimensions, just `n_stages` times as
    many of them."""
    dims = [
        _latin_hypercube_1d(n_samples, bit_bounds if d % 2 == 0 else pruning_bounds, seed + d)
        for d in range(2 * n_stages)
    ]
    rows = zip(*dims)
    return [[(row[2 * s], row[2 * s + 1]) for s in range(n_stages)] for row in rows]


def warmup_stage_candidate(
    index: int, n_warmup: int, n_stages: int, bit_bounds: Bounds, pruning_bounds: Bounds, seed: int
) -> StagePoint:
    """The `index`-th per-stage warm-up candidate -- the n-D analogue of
    `warmup_candidate`, same determinism-in-`(index, seed)` guarantee."""
    candidates = latin_hypercube_stage_candidates(n_warmup, n_stages, bit_bounds, pruning_bounds, seed)
    return candidates[index % n_warmup]


# ---------------------------------------------------------------------------
# Surrogate-guided phase: IDW regression + UCB acquisition
# ---------------------------------------------------------------------------


def _normalize(value: float, bounds: Bounds) -> float:
    low, high = bounds
    return (value - low) / (high - low) if high > low else 0.0


def predict_accuracy(
    bits: float, pruning: float, history: Sequence[Dict[str, Any]], bit_bounds: Bounds, pruning_bounds: Bounds
) -> Tuple[float, float]:
    """Inverse-distance-weighted prediction of accuracy at `(bits, pruning)`
    from `history` entries (each needs `avg_weight_bits`,
    `avg_column_pruning_ratio`, `accuracy`), plus an uncertainty proxy
    (lower when close, well-sampled history exists nearby; ~1.0 with no
    history at all) used as the exploration term in `propose_via_surrogate`.
    """
    target = (_normalize(bits, bit_bounds), _normalize(pruning, pruning_bounds))
    weights: List[float] = []
    values: List[float] = []
    for entry in history:
        point = (
            _normalize(entry["avg_weight_bits"], bit_bounds),
            _normalize(entry["avg_column_pruning_ratio"], pruning_bounds),
        )
        distance = math.hypot(target[0] - point[0], target[1] - point[1])
        if distance < 1e-6:
            return entry["accuracy"], 0.0
        weights.append(1.0 / (distance**2))
        values.append(entry["accuracy"])

    if not weights:
        return 0.5, 1.0

    total_weight = sum(weights)
    predicted = sum(w * v for w, v in zip(weights, values)) / total_weight
    uncertainty = 1.0 / (1.0 + total_weight)
    return predicted, uncertainty


def propose_via_surrogate(
    history: Sequence[Dict[str, Any]],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
    grid_size: int = 9,
    exploration_weight: float = 0.3,
) -> Tuple[float, float]:
    """Scans a `grid_size` x `grid_size` grid over the valid search space and
    returns the point maximizing `predicted_accuracy + exploration_weight *
    uncertainty` -- a standard upper-confidence-bound acquisition rule over
    the IDW surrogate in `predict_accuracy`."""
    bit_lo, bit_hi = bit_bounds
    pruning_lo, pruning_hi = pruning_bounds

    best_score = -math.inf
    best_point = (bit_lo, pruning_lo)
    for i in range(grid_size):
        bits = bit_lo + (bit_hi - bit_lo) * i / (grid_size - 1)
        for j in range(grid_size):
            pruning = pruning_lo + (pruning_hi - pruning_lo) * j / (grid_size - 1)
            predicted, uncertainty = predict_accuracy(bits, pruning, history, bit_bounds, pruning_bounds)
            score = predicted + exploration_weight * uncertainty
            if score > best_score:
                best_score = score
                best_point = (bits, pruning)
    return best_point


# ---------------------------------------------------------------------------
# Per-stage (n-dimensional) surrogate-guided phase: IDW regression over the
# full per-stage vector + random-search UCB acquisition
# ---------------------------------------------------------------------------


def _stage_vector_from_layer_configs(
    layer_configs: Any, stage_names: Sequence[str]
) -> "StagePoint | None":
    """Recovers the (weight_bits, column_pruning_ratio) pair per stage from
    a `candidate_history` entry's stored `layer_configs` (the exact
    validated per-stage config @tuner used that iteration -- echoed through
    `TunerToolOutput.data`, see tools/simulators.py), in `stage_names`
    order. Returns None if this entry doesn't cover exactly `stage_names`
    (a different model_id's history, or an entry recorded before this field
    existed) -- such entries are real data, just not comparable point-for-
    point to the current search space, so `predict_accuracy_stages` skips
    them rather than guessing a partial vector."""
    if not layer_configs:
        return None
    by_name = {entry["layer_name"]: entry for entry in layer_configs}
    if set(by_name) != set(stage_names):
        return None
    return [(by_name[name]["weight_bits"], by_name[name]["column_pruning_ratio"]) for name in stage_names]


def _normalize_stage_point(point: StagePoint, bit_bounds: Bounds, pruning_bounds: Bounds) -> Tuple[float, ...]:
    flat: List[float] = []
    for bits, pruning in point:
        flat.append(_normalize(bits, bit_bounds))
        flat.append(_normalize(pruning, pruning_bounds))
    return tuple(flat)


def predict_accuracy_stages(
    point: StagePoint,
    history: Sequence[Dict[str, Any]],
    stage_names: Sequence[str],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
) -> Tuple[float, float]:
    """The n-dimensional analogue of `predict_accuracy`: same
    inverse-distance-weighted regression, just over the full
    `2 * len(stage_names)`-dimensional normalized point (built from each
    history entry's stored per-stage `layer_configs` via
    `_stage_vector_from_layer_configs`) instead of one flat 2D point."""
    target = _normalize_stage_point(point, bit_bounds, pruning_bounds)
    weights: List[float] = []
    values: List[float] = []
    for entry in history:
        vector = _stage_vector_from_layer_configs(entry.get("layer_configs"), stage_names)
        if vector is None:
            continue
        entry_point = _normalize_stage_point(vector, bit_bounds, pruning_bounds)
        distance = math.sqrt(sum((t - p) ** 2 for t, p in zip(target, entry_point)))
        if distance < 1e-6:
            return entry["accuracy"], 0.0
        weights.append(1.0 / (distance**2))
        values.append(entry["accuracy"])

    if not weights:
        return 0.5, 1.0

    total_weight = sum(weights)
    predicted = sum(w * v for w, v in zip(weights, values)) / total_weight
    uncertainty = 1.0 / (1.0 + total_weight)
    return predicted, uncertainty


def propose_stage_candidates_via_surrogate(
    history: Sequence[Dict[str, Any]],
    stage_names: Sequence[str],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
    seed: int,
    n_candidates: int = 200,
    exploration_weight: float = 0.3,
) -> StagePoint:
    """The n-dimensional analogue of `propose_via_surrogate`. An exhaustive
    `grid_size ** (2*len(stage_names))` scan is infeasible once
    `stage_names` has more than a couple of entries (12-40 real dimensions
    for this project's supported models), so this acquisition step instead
    draws `n_candidates` uniform-random points from the space (deterministic
    in `seed`) and keeps whichever maximizes `predicted_accuracy +
    exploration_weight * uncertainty` under `predict_accuracy_stages` -- the
    same UCB rule as the grid-scan version, over a random sample of the
    space instead of an exhaustive one."""
    rng = random.Random(seed)
    best_score = -math.inf
    best_point: StagePoint = [(bit_bounds[0], pruning_bounds[0]) for _ in stage_names]
    for _ in range(n_candidates):
        point = [(rng.uniform(*bit_bounds), rng.uniform(*pruning_bounds)) for _ in stage_names]
        predicted, uncertainty = predict_accuracy_stages(point, history, stage_names, bit_bounds, pruning_bounds)
        score = predicted + exploration_weight * uncertainty
        if score > best_score:
            best_score = score
            best_point = point
    return best_point


# ---------------------------------------------------------------------------
# Real Pareto rank (NSGA-II-style non-dominated sorting), computed online
# ---------------------------------------------------------------------------


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """`a` dominates `b`: at least as good on every objective (higher
    accuracy, lower energy, lower latency) and strictly better on at least
    one."""
    not_worse = a["accuracy"] >= b["accuracy"] and a["energy_pj"] <= b["energy_pj"] and a["noc_latency_ms"] <= b["noc_latency_ms"]
    strictly_better = a["accuracy"] > b["accuracy"] or a["energy_pj"] < b["energy_pj"] or a["noc_latency_ms"] < b["noc_latency_ms"]
    return not_worse and strictly_better


def compute_pareto_rank(candidate: Dict[str, Any], history: Sequence[Dict[str, Any]]) -> int:
    """1 = non-dominated (Pareto-optimal) against every entry in `history`;
    otherwise `1 + max(dominator's own pareto_rank)` -- an incremental
    approximation of full non-dominated sorting appropriate for the
    sequential (one-candidate-at-a-time) search loop this project runs,
    rather than batch-sorting a whole generation at once as textbook
    NSGA-II does."""
    dominating = [entry for entry in history if _dominates(entry, candidate)]
    if not dominating:
        return 1
    return 1 + max(entry.get("pareto_rank", 1) for entry in dominating)
