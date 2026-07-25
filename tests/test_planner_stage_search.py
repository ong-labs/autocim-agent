"""@planner's real per-stage search space (nodes/planner.py + tools/search.py).

Every other planner test (tests/test_planner_llm.py) runs against
FAKE_LAYER_GROUPS' 2 stages (tests/conftest.py's autouse
`stub_planner_layer_groups`), which is too small to actually exercise the
"per-real-stage independent search" behavior this module adds -- with only
2 stages, `_warmup_count` never rises above its floor of 3, and a bug that
silently fanned one flat point out to every stage would be invisible. These
tests monkeypatch `get_layer_groups` again, locally, to a larger stage set.
"""

import nodes.planner as planner_module

FAKE_STAGES = {
    "stage_a": ["stage_a"],
    "stage_b": ["stage_b"],
    "stage_c": ["stage_c"],
    "stage_d": ["stage_d"],
    "stage_e": ["stage_e"],
}


def test_warmup_count_scales_with_real_stage_count(monkeypatch, state_factory, registered_hw_config):
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)

    stage_names = planner_module._real_stage_names(state["model_id"])
    assert len(stage_names) == 5
    assert planner_module._warmup_count(len(stage_names)) == 5  # floor(3) < 5 stages <= cap(12)


def test_warmup_count_is_capped_for_a_high_stage_count():
    assert planner_module._warmup_count(20) == planner_module._MAX_WARMUP_CANDIDATES


def test_propose_stage_points_returns_one_independent_point_per_stage(
    monkeypatch, state_factory, registered_hw_config
):
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    stage_names = planner_module._real_stage_names(state["model_id"])

    stage_points, search_tag = planner_module._propose_stage_points(state, stage_names)

    assert len(stage_points) == len(stage_names)
    assert "LHS warm-up" in search_tag
    # Real per-stage independence: not every stage collapsed to one shared
    # point, which the old fixed first/last-stage-bias expansion would do.
    assert len(set(stage_points)) > 1


def test_planner_node_produces_one_layer_config_per_real_stage_with_llm_fallback(
    monkeypatch, state_factory, registered_hw_config
):
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))

    def _raise():
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(planner_module, "get_planner_chat_model", _raise)

    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    update = planner_module.planner_node(state)

    assert {lc["layer_name"] for lc in update["planned_layer_configs"]} == set(FAKE_STAGES)
    assert len(update["planned_layer_configs"]) == 5


def test_surrogate_phase_uses_candidate_historys_layer_configs(monkeypatch, state_factory, registered_hw_config):
    """Once warm-up is exhausted, the surrogate must actually be able to
    read prior candidates back via candidate_history's `layer_configs`
    field (nodes/evaluator.py) -- not just the old across-stage averages --
    to place them in the real per-stage search space."""
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    stage_names = planner_module._real_stage_names(state["model_id"])
    n_warmup = planner_module._warmup_count(len(stage_names))

    good_layer_configs = [
        {"layer_name": name, "weight_bits": 8, "activation_bits": 8, "column_pruning_ratio": 0.0}
        for name in stage_names
    ]
    history = [
        {"layer_configs": good_layer_configs, "accuracy": 0.95}
        for _ in range(n_warmup)  # enough entries to exit the warm-up phase
    ]
    state = {**state, "candidate_history": history}

    stage_points, search_tag = planner_module._propose_stage_points(state, stage_names)

    assert "surrogate-model acquisition" in search_tag
    assert len(stage_points) == len(stage_names)
