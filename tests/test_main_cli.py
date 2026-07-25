"""main.py's --hw-config loading: a clean HWConfigError (caught in main()
and printed without a raw traceback) instead of letting a missing file,
malformed JSON, or schema-invalid HWConfig crash with a stack trace a
non-code-reading researcher can't act on.

Also main.py's --list-sessions (list_sessions/print_sessions): a
researcher shouldn't have to remember thread_ids manually to see what's
persisted in a checkpoint DB.
"""

import json

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from main import DEFAULT_HW_CONFIG, HWConfigError, list_sessions, load_hw_config, print_sessions, run_session


def test_load_hw_config_returns_default_when_path_is_none():
    assert load_hw_config(None) is DEFAULT_HW_CONFIG


def test_load_hw_config_raises_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(HWConfigError, match="not found"):
        load_hw_config(str(missing))


def test_load_hw_config_raises_for_invalid_json(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(HWConfigError, match="not valid JSON"):
        load_hw_config(str(bad_json))


def test_load_hw_config_raises_for_schema_violation(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"hw_spec_id": "missing_required_fields"}), encoding="utf-8")
    with pytest.raises(HWConfigError, match="HWConfig's schema"):
        load_hw_config(str(invalid))


@pytest.mark.parametrize(
    "example_file",
    ["cim_v1_128x128.json", "calibrated_neurosim_v1_5.json", "high_ir_drop_never_converges.json"],
)
def test_example_hw_configs_load_successfully(example_file):
    """The checked-in examples/hw_configs/ files must stay valid HWConfigs
    -- this would catch a schema change in schemas/config.py breaking them."""
    path = f"examples/hw_configs/{example_file}"
    hw = load_hw_config(path)
    assert hw.hw_spec_id


# --- --list-sessions ----------------------------------------------------------


def test_list_sessions_is_empty_for_a_fresh_checkpoint_db(tmp_path):
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        assert list_sessions(checkpointer) == []


def test_list_sessions_reports_a_converged_session(tmp_path, registered_hw_config):
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-a", checkpointer)  # good HW -> converges, no HITL
        sessions = list_sessions(checkpointer)

    assert len(sessions) == 1
    assert sessions[0]["thread_id"] == "thread-a"
    assert sessions[0]["model_id"] == "resnet18"
    assert sessions[0]["is_converged"] is True
    assert sessions[0]["paused"] is False


def test_list_sessions_reports_a_paused_session(monkeypatch, tmp_path, registered_bad_hw_config):
    monkeypatch.setattr("main.prompt_for_override", lambda payload: (_ for _ in ()).throw(EOFError()))
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_bad_hw_config, "thread-b", checkpointer)  # bad HW -> pauses at HITL
        sessions = list_sessions(checkpointer)

    assert len(sessions) == 1
    assert sessions[0]["paused"] is True
    assert sessions[0]["is_converged"] is False


def test_list_sessions_lists_multiple_distinct_threads(tmp_path, registered_hw_config):
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-x", checkpointer)
        run_session("resnet18", registered_hw_config, "thread-y", checkpointer)
        sessions = list_sessions(checkpointer)

    assert {s["thread_id"] for s in sessions} == {"thread-x", "thread-y"}


def test_print_sessions_reports_when_none_found(capsys):
    print_sessions([])
    assert "No sessions found" in capsys.readouterr().out


def test_print_sessions_shows_status_per_session(capsys):
    print_sessions(
        [
            {
                "thread_id": "t1",
                "model_id": "resnet18",
                "hw_spec_id": "hw1",
                "iteration_count": 3,
                "is_converged": True,
                "paused": False,
            },
            {
                "thread_id": "t2",
                "model_id": "resnet18",
                "hw_spec_id": "hw2",
                "iteration_count": 1,
                "is_converged": False,
                "paused": True,
            },
        ]
    )
    out = capsys.readouterr().out
    assert "t1" in out and "converged" in out
    assert "t2" in out and "paused (HITL)" in out
