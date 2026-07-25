"""observability.py: structured JSON-Lines event logging -- the separate
channel from state["messages"]' free text (nodes/planner.py) that a
researcher can grep/filter/machine-parse across a run.

`isolated_observability_log_dir` (tests/conftest.py, autouse) points
AUTOCIM_LOG_DIR at a per-test tmp_path, so these tests read real on-disk
log files without touching the real repo's .cache/logs/.
"""

import json

from observability import log_event


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_log_event_returns_the_record_it_logged():
    record = log_event("model_a::hw_a", node="planner", event="candidate_proposed", iteration=3, search_tag="LHS warm-up")

    assert record["run_id"] == "model_a::hw_a"
    assert record["node"] == "planner"
    assert record["event"] == "candidate_proposed"
    assert record["iteration"] == 3
    assert record["search_tag"] == "LHS warm-up"
    assert "ts" in record


def test_log_event_writes_one_json_line_per_call_to_the_runs_log_file(tmp_path, monkeypatch):
    import observability

    log_dir = tmp_path / "custom_logs"
    monkeypatch.setenv("AUTOCIM_LOG_DIR", str(log_dir))
    observability.reset_loggers()

    log_event("my_run", node="planner", event="candidate_proposed", iteration=1, foo="bar")
    log_event("my_run", node="evaluator", event="candidate_evaluated", iteration=1, accuracy=0.9)

    lines = _read_jsonl(log_dir / "my_run.jsonl")
    assert len(lines) == 2
    assert lines[0]["event"] == "candidate_proposed"
    assert lines[1]["event"] == "candidate_evaluated"
    assert lines[1]["accuracy"] == 0.9


def test_run_ids_with_reserved_filename_characters_are_sanitized_on_disk(tmp_path, monkeypatch):
    """`nodes.common.run_id_for` produces `<model_id>::<hw_spec_id>` -- `:`
    is a reserved Windows filename character, so this must not crash, and
    the raw run_id (with `::`) must still appear inside the JSON record."""
    import observability

    log_dir = tmp_path / "logs"
    monkeypatch.setenv("AUTOCIM_LOG_DIR", str(log_dir))
    observability.reset_loggers()

    log_event("resnet18::test_hw_good", node="planner", event="candidate_proposed", iteration=1)

    written_files = list(log_dir.glob("*.jsonl"))
    assert len(written_files) == 1
    records = _read_jsonl(written_files[0])
    assert records[0]["run_id"] == "resnet18::test_hw_good"


def test_reset_loggers_lets_a_new_log_dir_take_effect(tmp_path, monkeypatch):
    import observability

    first_dir = tmp_path / "first"
    monkeypatch.setenv("AUTOCIM_LOG_DIR", str(first_dir))
    observability.reset_loggers()
    log_event("same_run_id", node="planner", event="candidate_proposed", iteration=1)
    assert (first_dir / "same_run_id.jsonl").exists()

    second_dir = tmp_path / "second"
    monkeypatch.setenv("AUTOCIM_LOG_DIR", str(second_dir))
    observability.reset_loggers()
    log_event("same_run_id", node="planner", event="candidate_proposed", iteration=2)
    assert (second_dir / "same_run_id.jsonl").exists()
