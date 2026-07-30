"""@planner's real LLM call, forced through tool-calling.

`stub_planner_llm` (tests/conftest.py) is autouse, so every test here already
runs against `FakeToolCallingChatModel` -- no test in this suite makes a real
API call. These tests cover: the tool-calling contract itself (forced
tool_choice, rationale/rationale_ko/anomaly_note consumed from the LLM's
decision -- never layer_configs, which the search alone decides), the
LLM-failure fallback path, and that @tuner actually consumes
`planned_layer_configs` instead of its own hardcoded baseline.
"""

import pytest

import nodes.planner as planner_module
from nodes.planner import _SYSTEM_PROMPT, _TOOL_NAME, planner_node
from nodes.tuner import tuner_node
from schemas.planner import PlannerLayerDecision


def test_system_prompt_forbids_the_llm_from_proposing_its_own_numbers():
    """Regression test for a real production failure mode: a local LLM
    (qwen2.5:7b) read "accuracy below target" in recent_failure_history and
    reduced weight_bits "to ensure better performance" -- backwards, since
    less precision generally makes accuracy worse, not better, and silently
    produced a worse candidate than the search's own suggestion. Rather than
    trying to teach a small local model the correct direction for every
    failure type (fragile -- there's always another wrong direction it could
    guess), the fix removes its ability to propose numbers at all: this
    asserts the prompt tells it not to, and that PlannerLayerDecision (the
    forced tool schema) has no field for it to do so through even if it
    tried."""
    assert "do not choose or adjust weight_bits" in _SYSTEM_PROMPT.lower()
    assert "layer_configs" not in PlannerLayerDecision.model_fields


def test_planner_forces_tool_choice_to_decision_schema(state_factory, fake_planner_chat_model):
    state = state_factory()

    planner_node(state)

    assert fake_planner_chat_model.bind_tools_calls, "planner must call bind_tools on the chat model"
    assert fake_planner_chat_model.bind_tools_calls[-1]["tool_choice"] == _TOOL_NAME


def test_planner_stores_search_output_in_planned_layer_configs_ignoring_llm_numbers(state_factory):
    """planned_layer_configs must come from the search (stage_configs),
    never from the LLM's tool call -- PlannerLayerDecision has no
    layer_configs field to propose numbers through even if a real model
    tried (nodes/planner.py's module docstring)."""
    state = state_factory()
    stage_names = planner_module.real_stage_names(state["model_id"])
    stage_points, _ = planner_module._propose_stage_points(state, stage_names)
    expected_layer_configs = planner_module.points_to_stage_configs(stage_points, stage_names)

    update = planner_node(state)

    assert update["planned_layer_configs"] == [lc.model_dump(mode="json") for lc in expected_layer_configs]
    assert any("fake planner decision for testing" in m["content"] for m in update["messages"])


def test_planner_prompt_lists_real_stage_names(state_factory, fake_planner_chat_model):
    """Verifies @planner actually queries `tools.qat.get_layer_groups`
    (stubbed to FAKE_LAYER_GROUPS in conftest.py) and puts the real stage
    names in the LLM prompt, rather than only ever proposing a single flat
    "model_default" entry."""
    state = state_factory()

    planner_node(state)

    prompt_text = fake_planner_chat_model.invoke_calls[-1][-1].content  # last HumanMessage
    assert "'conv'" in prompt_text
    assert "'fc'" in prompt_text


def test_planner_falls_back_to_search_point_when_llm_call_raises(monkeypatch, state_factory, registered_hw_config):
    def _raise():
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(planner_module, "get_planner_chat_model", _raise)

    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    # Same search machinery planner_node itself calls -- verifies the
    # fallback actually uses the (deterministic) per-stage LHS/surrogate
    # search points, rather than a flat hardcoded default.
    stage_names = planner_module.real_stage_names(state["model_id"])
    stage_points, search_tag = planner_module._propose_stage_points(state, stage_names)
    expected_layer_configs = planner_module.points_to_stage_configs(stage_points, stage_names)

    update = planner_node(state)

    assert update["planned_layer_configs"] == [lc.model_dump(mode="json") for lc in expected_layer_configs]
    assert {lc["layer_name"] for lc in update["planned_layer_configs"]} == {"conv", "fc"}
    assert any("LLM proposal failed" in m["content"] for m in update["messages"])
    assert any(search_tag in m["content"] for m in update["messages"])


def test_planner_records_llm_usage_on_success(state_factory):
    state = state_factory()

    update = planner_node(state)

    assert len(update["llm_usage"]) == 1
    entry = update["llm_usage"][0]
    assert entry["node"] == "planner"
    assert entry["succeeded"] is True
    assert entry["attempts"] == 1
    # FakeToolCallingChatModel exposes no usage metadata -- must report
    # None rather than a fabricated token count.
    assert entry["input_tokens"] is None
    assert entry["estimated_cost_usd"] is None


def test_planner_records_llm_usage_on_retry_exhaustion(monkeypatch, state_factory):
    class _AlwaysRateLimited:
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def invoke(self, messages):
            raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setenv("AUTOCIM_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("AUTOCIM_LLM_BASE_DELAY_SECONDS", "0.001")
    monkeypatch.setenv("AUTOCIM_LLM_MAX_DELAY_SECONDS", "0.001")
    monkeypatch.setattr(planner_module, "get_planner_chat_model", lambda: _AlwaysRateLimited())

    state = state_factory()
    update = planner_node(state)

    assert len(update["llm_usage"]) == 1
    entry = update["llm_usage"][0]
    assert entry["succeeded"] is False
    assert entry["attempts"] == 2  # retried up to AUTOCIM_LLM_MAX_RETRIES, not just given up on the first failure
    assert "429" in entry["error"]
    assert any("LLM proposal failed after 2 attempt(s)" in m["content"] for m in update["messages"])
    # Still falls back to a valid planned config, exactly like the
    # non-retryable-failure path.
    assert update["planned_layer_configs"]


def test_planner_skips_llm_call_entirely_once_budget_is_exceeded(monkeypatch, state_factory, fake_planner_chat_model):
    monkeypatch.setenv("AUTOCIM_LLM_MAX_TOTAL_COST_USD", "1.00")
    state = state_factory(llm_usage=[{"estimated_cost_usd": 1.50, "total_tokens": 100}])

    update = planner_node(state)

    assert fake_planner_chat_model.invoke_calls == []  # never even attempted the call
    assert update["planned_layer_configs"]  # still falls back to a valid config
    assert update["llm_usage"][0]["attempts"] == 0
    assert update["llm_usage"][0]["succeeded"] is False
    assert "budget" in update["llm_usage"][0]["error"]
    assert update["planner_decisions"][0]["used_llm"] is False
    assert any("LLM call skipped" in m["content"] for m in update["messages"])


def test_planner_calls_llm_normally_when_budget_not_configured(state_factory, fake_planner_chat_model):
    state = state_factory(llm_usage=[{"estimated_cost_usd": 999.0, "total_tokens": 999999}])

    update = planner_node(state)

    assert fake_planner_chat_model.invoke_calls  # budget cap unset -> unbounded, same as before this feature
    assert update["planner_decisions"][0]["used_llm"] is True


def test_planner_falls_back_when_llm_returns_no_matching_tool_call(monkeypatch, state_factory):
    class NoToolCallModel:
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def invoke(self, messages):
            return type("EmptyResponse", (), {"tool_calls": []})()

    monkeypatch.setattr(planner_module, "get_planner_chat_model", lambda: NoToolCallModel())

    state = state_factory()
    update = planner_node(state)

    assert update["planned_layer_configs"], "must still fall back to a valid default, not crash"
    assert any("LLM proposal failed" in m["content"] for m in update["messages"])


def test_tuner_consumes_planner_planned_layer_configs_instead_of_its_own_default(state_factory, registered_hw_config):
    """planned_layer_configs is set directly (bypassing planner_node/the
    search entirely) -- this test's job is @tuner actually reading it, not
    what @planner's search happens to produce for a given seed."""
    state = state_factory(
        hw_spec_id=registered_hw_config.hw_spec_id,
        planned_layer_configs=[
            {"layer_name": "conv", "weight_bits": 6, "activation_bits": 6, "column_pruning_ratio": 0.1},
            {"layer_name": "fc", "weight_bits": 4, "activation_bits": 4, "column_pruning_ratio": 0.2},
        ],
    )

    tuner_result = tuner_node(state)["metrics_store"]["tuner"]

    assert tuner_result["status"] == "SUCCESS"
    # averaged across the two per-stage entries: (6+4)/2, (0.1+0.2)/2
    assert tuner_result["data"]["avg_weight_bits"] == 5.0
    assert tuner_result["data"]["avg_column_pruning_ratio"] == pytest.approx(0.15)


def test_tuner_applies_different_quantization_to_different_real_stages(state_factory, registered_hw_config):
    """End-to-end proof that per-layer granularity actually reaches the
    model, not just `layer_configs`' own data: two deliberately distinct
    per-stage configs ("conv" -> 6 bits, "fc" -> 4 bits, set directly --
    the search's own real output isn't guaranteed to differ per stage for
    every seed/hw combination, so this isolates @tuner's per-stage
    application from the search's variance) must land as genuinely
    different quantization grids on the fake backend's two real
    submodules."""
    import torch

    from tests.conftest import make_fake_qat_backend
    from tools.qat import apply_crossbar_quant_prune

    state = state_factory(
        hw_spec_id=registered_hw_config.hw_spec_id,
        planned_layer_configs=[
            {"layer_name": "conv", "weight_bits": 6, "activation_bits": 6, "column_pruning_ratio": 0.1},
            {"layer_name": "fc", "weight_bits": 4, "activation_bits": 4, "column_pruning_ratio": 0.2},
        ],
    )

    stage_configs = {
        lc["layer_name"]: (lc["weight_bits"], lc["column_pruning_ratio"]) for lc in state["planned_layer_configs"]
    }
    assert stage_configs["conv"][0] != stage_configs["fc"][0]  # sanity: deliberately distinct bits

    backend = make_fake_qat_backend()
    apply_crossbar_quant_prune(backend.base_model, stage_configs, default_config=(8, 0.0))

    # More bits -> more representable quantization levels, so the (many,
    # effectively continuous) randomly-initialized weights collapse onto
    # fewer distinct values at the lower bit-width.
    conv_unique = torch.unique(backend.base_model.conv.weight.detach()).numel()
    fc_unique = torch.unique(backend.base_model.fc.weight.detach()).numel()
    assert conv_unique > fc_unique
