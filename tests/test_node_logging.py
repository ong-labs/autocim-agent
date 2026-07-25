"""@planner and @evaluator's structured-logging integration
(observability.py) -- verifies the actual node functions emit real
candidate_proposed/candidate_evaluated events to their run's on-disk JSONL
log, not just that observability.py's primitives work in isolation
(tests/test_observability.py already covers those)."""

import json
import os
from pathlib import Path

from nodes.common import run_id_for
from nodes.evaluator import evaluator_node
from nodes.planner import planner_node
from observability import _safe_filename
from tests.test_evaluator_candidate_history import _metrics


def _log_lines(run_id: str):
    log_dir = Path(os.environ["AUTOCIM_LOG_DIR"])
    path = log_dir / f"{_safe_filename(run_id)}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_planner_node_logs_a_candidate_proposed_event(state_factory):
    state = state_factory()

    planner_node(state)

    lines = _log_lines(run_id_for(state))
    events = [line for line in lines if line["event"] == "candidate_proposed"]
    assert len(events) == 1
    assert events[0]["node"] == "planner"
    assert events[0]["iteration"] == 1
    assert "search_tag" in events[0]
    assert events[0]["used_llm"] is True  # FakeToolCallingChatModel succeeds by default


def test_evaluator_node_logs_a_candidate_evaluated_event_with_pareto_rank(state_factory):
    state = state_factory(metrics_store=_metrics(accuracy=0.8, energy_pj=5.0, noc_latency_ms=1.2), iteration_count=1)

    evaluator_node(state)

    lines = _log_lines(run_id_for(state))
    events = [line for line in lines if line["event"] == "candidate_evaluated"]
    assert len(events) == 1
    assert events[0]["node"] == "evaluator"
    assert events[0]["is_converged"] is True
    assert events[0]["accuracy"] == 0.8
    assert events[0]["pareto_rank"] == 1


def test_evaluator_node_logs_needs_hitl_and_reason_on_non_convergence(state_factory):
    state = state_factory(metrics_store=_metrics(is_converged=False), iteration_count=1, retry_count=2)

    evaluator_node(state)

    lines = _log_lines(run_id_for(state))
    events = [line for line in lines if line["event"] == "candidate_evaluated"]
    assert events[-1]["is_converged"] is False
    assert events[-1]["needs_hitl"] is True  # retry_count 2 -> new_retry_count 3 == MAX_RETRY_LIMIT
    assert "reason" in events[-1]
