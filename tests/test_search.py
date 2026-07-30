"""tools/search.py: real LHS warm-up sampling, IDW-surrogate + UCB
acquisition, and NSGA-II-style online Pareto ranking -- the piece
underneath @planner's LLM call that was previously missing entirely
(the LLM just guessed numbers with no statistical grounding, and @tuner
reported a hardcoded pareto_rank: 1 with no basis to rank itself)."""

import math

from tools.search import (
    compute_pareto_rank,
    latin_hypercube_candidates,
    latin_hypercube_stage_candidates,
    predict_accuracy,
    predict_accuracy_stages,
    predict_energy_stages,
    predict_latency_stages,
    propose_stage_candidates_via_surrogate,
    propose_via_surrogate,
    warmup_candidate,
    warmup_stage_candidate,
)

BIT_BOUNDS = (2, 8)
PRUNING_BOUNDS = (0.0, 0.5)


# --- Latin Hypercube warm-up sampling ---------------------------------------


def test_latin_hypercube_candidates_stay_within_bounds():
    candidates = latin_hypercube_candidates(5, BIT_BOUNDS, PRUNING_BOUNDS, seed=42)

    assert len(candidates) == 5
    for bits, pruning in candidates:
        assert BIT_BOUNDS[0] <= bits <= BIT_BOUNDS[1]
        assert PRUNING_BOUNDS[0] <= pruning <= PRUNING_BOUNDS[1]


def test_latin_hypercube_candidates_deterministic_for_same_seed():
    a = latin_hypercube_candidates(5, BIT_BOUNDS, PRUNING_BOUNDS, seed=42)
    b = latin_hypercube_candidates(5, BIT_BOUNDS, PRUNING_BOUNDS, seed=42)
    assert a == b


def test_latin_hypercube_candidates_differ_for_different_seed():
    a = latin_hypercube_candidates(5, BIT_BOUNDS, PRUNING_BOUNDS, seed=1)
    b = latin_hypercube_candidates(5, BIT_BOUNDS, PRUNING_BOUNDS, seed=2)
    assert a != b


def test_warmup_candidate_same_index_and_seed_reproduces_same_point():
    a = warmup_candidate(1, 3, BIT_BOUNDS, PRUNING_BOUNDS, seed=7)
    b = warmup_candidate(1, 3, BIT_BOUNDS, PRUNING_BOUNDS, seed=7)
    assert a == b


def test_warmup_candidate_index_wraps_via_modulo():
    assert warmup_candidate(3, 3, BIT_BOUNDS, PRUNING_BOUNDS, seed=7) == warmup_candidate(
        0, 3, BIT_BOUNDS, PRUNING_BOUNDS, seed=7
    )


# --- IDW surrogate + UCB acquisition ----------------------------------------


def test_predict_accuracy_with_no_history_is_neutral_and_maximally_uncertain():
    predicted, uncertainty = predict_accuracy(4, 0.1, [], BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted == 0.5
    assert uncertainty == 1.0


def test_predict_accuracy_exact_match_returns_that_accuracy_with_no_uncertainty():
    history = [{"avg_weight_bits": 4, "avg_column_pruning_ratio": 0.1, "accuracy": 0.83}]
    predicted, uncertainty = predict_accuracy(4, 0.1, history, BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted == 0.83
    assert uncertainty == 0.0


def test_predict_accuracy_weights_nearby_history_more_than_distant_history():
    history = [
        {"avg_weight_bits": 4, "avg_column_pruning_ratio": 0.1, "accuracy": 0.9},  # near query
        {"avg_weight_bits": 2, "avg_column_pruning_ratio": 0.5, "accuracy": 0.1},  # far from query
    ]
    predicted, _ = predict_accuracy(4.2, 0.12, history, BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted > 0.5  # closer to the near (high-accuracy) point than the far (low) one


def test_propose_via_surrogate_favors_high_accuracy_region_under_pure_exploitation():
    good_point = (6.0, 0.05)
    bad_point = (2.0, 0.45)
    history = [
        {"avg_weight_bits": good_point[0], "avg_column_pruning_ratio": good_point[1], "accuracy": 0.9},
        {"avg_weight_bits": bad_point[0], "avg_column_pruning_ratio": bad_point[1], "accuracy": 0.1},
    ]

    proposed = propose_via_surrogate(history, BIT_BOUNDS, PRUNING_BOUNDS, grid_size=9, exploration_weight=0.0)

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    assert dist(proposed, good_point) < dist(proposed, bad_point)


# --- Per-stage (n-dimensional) LHS warm-up -----------------------------------

STAGE_NAMES = ["conv", "block1", "block2", "fc"]  # 4 stages -> 8 real dimensions


def test_latin_hypercube_stage_candidates_stay_within_bounds_per_stage():
    candidates = latin_hypercube_stage_candidates(5, len(STAGE_NAMES), BIT_BOUNDS, PRUNING_BOUNDS, seed=42)

    assert len(candidates) == 5
    for point in candidates:
        assert len(point) == len(STAGE_NAMES)
        for bits, pruning in point:
            assert BIT_BOUNDS[0] <= bits <= BIT_BOUNDS[1]
            assert PRUNING_BOUNDS[0] <= pruning <= PRUNING_BOUNDS[1]


def test_latin_hypercube_stage_candidates_deterministic_for_same_seed():
    a = latin_hypercube_stage_candidates(5, len(STAGE_NAMES), BIT_BOUNDS, PRUNING_BOUNDS, seed=42)
    b = latin_hypercube_stage_candidates(5, len(STAGE_NAMES), BIT_BOUNDS, PRUNING_BOUNDS, seed=42)
    assert a == b


def test_latin_hypercube_stage_candidates_are_independent_across_stages():
    """Real per-stage independence, not the old single-point-fanned-out-to-
    every-stage behavior: across the 5 warm-up samples, different stages
    should not all take the same value in lockstep."""
    candidates = latin_hypercube_stage_candidates(5, len(STAGE_NAMES), BIT_BOUNDS, PRUNING_BOUNDS, seed=42)
    stage0_bits = [point[0][0] for point in candidates]
    stage1_bits = [point[1][0] for point in candidates]
    assert stage0_bits != stage1_bits


def test_warmup_stage_candidate_same_index_and_seed_reproduces_same_point():
    a = warmup_stage_candidate(1, 3, len(STAGE_NAMES), BIT_BOUNDS, PRUNING_BOUNDS, seed=7)
    b = warmup_stage_candidate(1, 3, len(STAGE_NAMES), BIT_BOUNDS, PRUNING_BOUNDS, seed=7)
    assert a == b


def test_warmup_stage_candidate_index_wraps_via_modulo():
    a = warmup_stage_candidate(3, 3, len(STAGE_NAMES), BIT_BOUNDS, PRUNING_BOUNDS, seed=7)
    b = warmup_stage_candidate(0, 3, len(STAGE_NAMES), BIT_BOUNDS, PRUNING_BOUNDS, seed=7)
    assert a == b


# --- Per-stage IDW surrogate + NSGA-II multi-objective acquisition ----------


def _layer_configs(point):
    return [
        {"layer_name": name, "weight_bits": bits, "column_pruning_ratio": pruning}
        for name, (bits, pruning) in zip(STAGE_NAMES, point)
    ]


def test_predict_accuracy_stages_with_no_history_is_neutral_and_maximally_uncertain():
    point = [(4, 0.1)] * len(STAGE_NAMES)
    predicted, uncertainty = predict_accuracy_stages(point, [], STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted == 0.5
    assert uncertainty == 1.0


def test_predict_accuracy_stages_exact_match_returns_that_accuracy_with_no_uncertainty():
    point = [(4, 0.1), (6, 0.2), (2, 0.0), (8, 0.4)]
    history = [{"layer_configs": _layer_configs(point), "accuracy": 0.83}]
    predicted, uncertainty = predict_accuracy_stages(point, history, STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted == 0.83
    assert uncertainty == 0.0


def test_predict_accuracy_stages_skips_history_entries_covering_different_stages():
    """A history entry from a different model_id (different stage names)
    isn't directly comparable to the current search space -- it must be
    ignored rather than silently mismatched against the wrong stage."""
    history = [
        {
            "layer_configs": [
                {"layer_name": "only_stage_a", "weight_bits": 4, "column_pruning_ratio": 0.1},
                {"layer_name": "only_stage_b", "weight_bits": 6, "column_pruning_ratio": 0.2},
            ],
            "accuracy": 0.99,
        }
    ]
    point = [(4, 0.1)] * len(STAGE_NAMES)
    predicted, uncertainty = predict_accuracy_stages(point, history, STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted == 0.5  # neutral, exactly as if history were empty
    assert uncertainty == 1.0


def test_predict_energy_stages_exact_match_returns_that_energy_with_no_uncertainty():
    point = [(4, 0.1), (6, 0.2), (2, 0.0), (8, 0.4)]
    history = [{"layer_configs": _layer_configs(point), "energy_pj": 42.5}]
    predicted, uncertainty = predict_energy_stages(point, history, STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted == 42.5
    assert uncertainty == 0.0


def test_predict_energy_stages_with_no_history_is_zero_and_maximally_uncertain():
    """Unlike accuracy's 0.5 neutral default (a natural midpoint on
    [0, 1]), energy_pj/noc_latency_ms have no natural bounded range --
    0.0 is the documented neutral value, distinguishing this from
    predict_accuracy_stages's default."""
    point = [(4, 0.1)] * len(STAGE_NAMES)
    predicted, uncertainty = predict_energy_stages(point, [], STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted == 0.0
    assert uncertainty == 1.0


def test_predict_latency_stages_exact_match_returns_that_latency_with_no_uncertainty():
    point = [(4, 0.1), (6, 0.2), (2, 0.0), (8, 0.4)]
    history = [{"layer_configs": _layer_configs(point), "noc_latency_ms": 1.75}]
    predicted, uncertainty = predict_latency_stages(point, history, STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted == 1.75
    assert uncertainty == 0.0


def test_predict_stage_value_skips_entries_missing_that_objectives_key():
    """A candidate_history entry missing energy_pj (e.g. @tuner failed
    before @profiler ran) must not crash the energy surrogate -- it's
    simply excluded from that objective's fit, same treatment as a
    stage-name mismatch."""
    point = [(4, 0.1)] * len(STAGE_NAMES)
    history = [{"layer_configs": _layer_configs(point), "accuracy": 0.9}]  # no energy_pj key
    predicted, uncertainty = predict_energy_stages(point, history, STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS)
    assert predicted == 0.0  # neutral, exactly as if history were empty
    assert uncertainty == 1.0


def test_propose_stage_candidates_via_surrogate_favors_the_dominant_region_under_pure_exploitation():
    """Genuinely multi-objective now: `good_point` dominates `bad_point` on
    all three objectives (higher accuracy, lower energy, lower latency),
    so under pure exploitation (exploration_weight=0.0) every point NSGA-II
    finds on the surrogate's Pareto front should cluster near it -- not
    just accuracy, which is all the pre-NSGA-II single-objective UCB scan
    ever modeled (the real production gap this rewrite fixes)."""
    good_point = [(6.0, 0.05)] * len(STAGE_NAMES)
    bad_point = [(2.0, 0.45)] * len(STAGE_NAMES)
    history = [
        {"layer_configs": _layer_configs(good_point), "accuracy": 0.9, "energy_pj": 40.0, "noc_latency_ms": 1.0},
        {"layer_configs": _layer_configs(bad_point), "accuracy": 0.1, "energy_pj": 200.0, "noc_latency_ms": 8.0},
    ]

    proposed = propose_stage_candidates_via_surrogate(
        history, STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS, seed=42, n_candidates=100, exploration_weight=0.0
    )

    def dist(a, b):
        return math.sqrt(sum((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2 for i in range(len(a))))

    assert dist(proposed, good_point) < dist(proposed, bad_point)


def test_propose_stage_candidates_via_surrogate_deterministic_for_same_seed():
    history = [
        {"layer_configs": _layer_configs([(6.0, 0.05)] * len(STAGE_NAMES)), "accuracy": 0.9, "energy_pj": 40.0, "noc_latency_ms": 1.0}
    ]
    a = propose_stage_candidates_via_surrogate(history, STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS, seed=42)
    b = propose_stage_candidates_via_surrogate(history, STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS, seed=42)
    assert a == b


def test_propose_stage_candidates_via_surrogate_considers_energy_and_latency_not_just_accuracy():
    """The core regression this rewrite fixes: with two history points that
    trade off accuracy against energy/latency (neither dominates the
    other), the proposed candidate must not just chase the highest-accuracy
    point while ignoring energy/latency entirely -- it should land nearer
    the lower-energy/lower-latency point than pure accuracy-only search
    would, since energy_pj/noc_latency_ms are now real objectives too."""
    high_accuracy_expensive = [(7.0, 0.0)] * len(STAGE_NAMES)
    lower_accuracy_cheap = [(3.0, 0.3)] * len(STAGE_NAMES)
    history = [
        {
            "layer_configs": _layer_configs(high_accuracy_expensive),
            "accuracy": 0.9,
            "energy_pj": 300.0,
            "noc_latency_ms": 10.0,
        },
        {
            "layer_configs": _layer_configs(lower_accuracy_cheap),
            "accuracy": 0.6,
            "energy_pj": 30.0,
            "noc_latency_ms": 1.0,
        },
    ]

    # Pure exploitation, weighted purely toward accuracy under the OLD
    # single-objective surrogate would always land near high_accuracy_expensive.
    # Under genuine multi-objective NSGA-II, the returned Pareto-optimal set
    # should include points nearer the cheap/low-latency end too -- assert
    # that at least one of several seeded picks lands closer to the cheap
    # point than to the expensive one, proving energy/latency actually
    # shape the search instead of being invisible to it.
    def dist(a, b):
        return math.sqrt(sum((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2 for i in range(len(a))))

    proposals = [
        propose_stage_candidates_via_surrogate(
            history, STAGE_NAMES, BIT_BOUNDS, PRUNING_BOUNDS, seed=s, n_candidates=100, exploration_weight=0.0
        )
        for s in range(5)
    ]
    assert any(dist(p, lower_accuracy_cheap) < dist(p, high_accuracy_expensive) for p in proposals)


# --- Real Pareto rank (NSGA-II-style, computed online) ----------------------


def _candidate(accuracy, energy_pj, noc_latency_ms, pareto_rank=None):
    entry = {"accuracy": accuracy, "energy_pj": energy_pj, "noc_latency_ms": noc_latency_ms}
    if pareto_rank is not None:
        entry["pareto_rank"] = pareto_rank
    return entry


def test_pareto_rank_is_one_with_no_history():
    candidate = _candidate(accuracy=0.8, energy_pj=5.0, noc_latency_ms=1.0)
    assert compute_pareto_rank(candidate, []) == 1


def test_pareto_rank_is_one_when_not_dominated_by_any_history_entry():
    # candidate is better on accuracy but worse on energy than the one
    # history entry -- neither dominates the other, so still rank 1.
    history = [_candidate(accuracy=0.7, energy_pj=3.0, noc_latency_ms=1.0, pareto_rank=1)]
    candidate = _candidate(accuracy=0.9, energy_pj=6.0, noc_latency_ms=1.0)
    assert compute_pareto_rank(candidate, history) == 1


def test_pareto_rank_is_two_when_strictly_dominated_by_a_rank_one_entry():
    history = [_candidate(accuracy=0.9, energy_pj=3.0, noc_latency_ms=1.0, pareto_rank=1)]
    # worse (or equal) on every objective, strictly worse on accuracy -> dominated
    candidate = _candidate(accuracy=0.7, energy_pj=5.0, noc_latency_ms=1.0)
    assert compute_pareto_rank(candidate, history) == 2


def test_pareto_rank_builds_on_the_dominating_entrys_own_rank():
    history = [
        _candidate(accuracy=0.95, energy_pj=2.0, noc_latency_ms=1.0, pareto_rank=1),
        _candidate(accuracy=0.85, energy_pj=3.0, noc_latency_ms=1.0, pareto_rank=2),
    ]
    # dominated by the rank-2 entry (and transitively the rank-1 one)
    candidate = _candidate(accuracy=0.6, energy_pj=6.0, noc_latency_ms=2.0)
    assert compute_pareto_rank(candidate, history) == 3
