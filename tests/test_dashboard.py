"""tools/dashboard.py: joins candidate_history (what happened) against
planner_decisions (why it was tried) and llm_usage (real LLM cost) into a
self-contained HTML report -- a pure function of AutoCIMState, no I/O, so
tested purely against synthetic state (state_factory), no real graph run."""

from graph import build_graph
from tools.dashboard import render_dashboard_html

_PLANNER_DECISIONS = [
    {"iteration": 1, "search_tag": "LHS warm-up candidate 1/3", "used_llm": True, "rationale": "exploring the space"},
    {"iteration": 2, "search_tag": "surrogate-model acquisition over 1 prior candidates", "used_llm": False, "rationale": "LLM proposal failed"},
]

_CANDIDATE_HISTORY = [
    {
        "iteration": 1,
        "avg_weight_bits": 6,
        "avg_column_pruning_ratio": 0.1,
        "accuracy": 0.7,
        "energy_pj": 8.0,
        "noc_latency_ms": 1.5,
        "is_converged": False,
        "pareto_rank": 1,
    },
    {
        "iteration": 2,
        "avg_weight_bits": 8,
        "avg_column_pruning_ratio": 0.0,
        "accuracy": 0.9,
        "energy_pj": 4.0,
        "noc_latency_ms": 1.0,
        "is_converged": True,
        "pareto_rank": 1,
    },
]


def test_render_dashboard_html_with_no_candidates_yet(state_factory):
    state = state_factory()

    output = render_dashboard_html(state)

    assert "No evaluated candidates yet." in output
    assert "n/a (no AUTOCIM_LLM_COST_PER_1K_* configured)" in output


def test_render_dashboard_html_joins_candidate_history_and_planner_decisions_by_iteration(state_factory):
    state = state_factory(candidate_history=_CANDIDATE_HISTORY, planner_decisions=_PLANNER_DECISIONS)

    output = render_dashboard_html(state)

    assert "LHS warm-up candidate 1/3" in output
    assert "surrogate-model acquisition over 1 prior candidates" in output
    assert "exploring the space" in output
    assert "0.7000" in output  # iteration 1 accuracy
    assert "0.9000" in output  # iteration 2 accuracy


def test_render_dashboard_html_includes_pareto_chart_points_per_iteration(state_factory):
    state = state_factory(candidate_history=_CANDIDATE_HISTORY, planner_decisions=_PLANNER_DECISIONS)

    output = render_dashboard_html(state)

    assert "<svg" in output
    assert output.count("<circle") == 2  # one point per evaluated candidate


def test_render_dashboard_html_reports_estimated_llm_cost_when_configured(state_factory):
    usage = [
        {"node": "planner", "iteration": 1, "attempts": 1, "succeeded": True, "estimated_cost_usd": 0.02},
        {"node": "planner", "iteration": 2, "attempts": 2, "succeeded": False, "estimated_cost_usd": None},
    ]
    state = state_factory(candidate_history=_CANDIDATE_HISTORY, planner_decisions=_PLANNER_DECISIONS, llm_usage=usage)

    output = render_dashboard_html(state)

    assert "$0.0200" in output
    assert "2 node-invocation(s)" in output
    assert "1 failed after retries" in output


def test_render_dashboard_html_handles_a_candidate_with_no_matching_planner_decision(state_factory):
    """A candidate_history entry with no corresponding planner_decisions
    entry (e.g. state from before this feature existed) must not crash the
    join -- it should just render without a search phase/rationale."""
    state = state_factory(candidate_history=_CANDIDATE_HISTORY, planner_decisions=[])

    output = render_dashboard_html(state)

    assert "0.7000" in output
    assert "0.9000" in output


def test_calibration_section_warns_when_hw_spec_id_is_uncalibrated(state_factory):
    state = state_factory(calibration_factors={}, calibration_provenance={})

    output = render_dashboard_html(state)

    assert "uncalibrated" in output
    assert '<p class="calib-warning">' in output
    assert '<p class="calib-ok">' not in output


def test_calibration_section_shows_factor_and_citation_when_calibrated(state_factory):
    state = state_factory(
        hw_spec_id="calibrated_hw",
        calibration_factors={"calibrated_hw": 1.2345},
        calibration_provenance={
            "calibrated_hw": {
                "reference_energy_pj_per_mac": 0.047,
                "source": "Peng et al., NeuroSim V1.5",
                "note": "assumes 1 MAC == 1 op",
            }
        },
    )

    output = render_dashboard_html(state)

    assert '<p class="calib-ok">' in output
    assert "1.2345x" in output
    assert "Peng et al., NeuroSim V1.5" in output
    assert "assumes 1 MAC == 1 op" in output
    assert '<p class="calib-warning">' not in output


def test_calibration_section_ignores_factors_for_other_hw_spec_ids(state_factory):
    """A calibration_factors/calibration_provenance entry seeded for a
    *different* hw_spec_id (e.g. left over from a prior session on the same
    process) must not be mistaken for this run's own calibration status."""
    state = state_factory(
        hw_spec_id="this_run_hw",
        calibration_factors={"some_other_hw": 2.0},
        calibration_provenance={"some_other_hw": {"reference_energy_pj_per_mac": 0.1, "source": "x", "note": None}},
    )

    output = render_dashboard_html(state)

    assert "uncalibrated" in output


def test_dashboard_renders_from_a_real_stubbed_graph_run(state_factory, registered_bad_hw_config):
    """End-to-end: a real (stubbed) multi-iteration graph run's resulting
    state -- not hand-built synthetic state -- must render cleanly, proving
    planner_decisions/candidate_history/llm_usage actually line up the way
    the rest of this file assumes."""
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-dashboard-e2e-thread"}}
    init_state = state_factory(hw_spec_id=registered_bad_hw_config.hw_spec_id)

    result = compiled.invoke(init_state, config=config)

    output = render_dashboard_html(result)

    assert "<svg" in output
    assert len(result["candidate_history"]) == output.count("<circle")
    # every candidate_history iteration has a matching planner_decisions
    # entry, so its search_tag/rationale must show up in the rendered table
    for decision in result["planner_decisions"]:
        assert decision["search_tag"] in output
