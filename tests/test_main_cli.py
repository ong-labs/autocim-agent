"""main.py's --hw-config loading: a clean HWConfigError (caught in main()
and printed without a raw traceback) instead of letting a missing file,
malformed JSON, or schema-invalid HWConfig crash with a stack trace a
non-code-reading researcher can't act on.

Also main.py's --list-sessions (list_sessions/print_sessions): a
researcher shouldn't have to remember thread_ids manually to see what's
persisted in a checkpoint DB.

Also main.py's .env/.env.local auto-loading: a silent regression here
(e.g. someone drops the load_dotenv() calls) wouldn't be caught by any
other test, since it's the one piece of startup behavior that only
main() itself exercises.
"""

import json
import sys
from datetime import datetime

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

import main
from main import DEFAULT_HW_CONFIG, HWConfigError, list_sessions, load_hw_config, print_sessions, run_session
from tools.calibration import bootstrap_calibration_factors


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


_CALIBRATED_EXAMPLE_FILES = [
    "calibrated_neurosim_v1_5.json",
    "calibrated_correll_2025_reram_256x64.json",
    "calibrated_daram_2021_reram_64x32.json",
    "calibrated_yu_2020_sram_128x128_1bit.json",
    "calibrated_yu_2020_sram_128x128_5bit.json",
    "calibrated_dong_2020_sram_64x64.json",
    "calibrated_deaville_2022_mram_256x512.json",
    "calibrated_korea_univ_2022_sram_256x80.json",
]


@pytest.mark.parametrize(
    "example_file",
    ["default_cim_v1_128x128.json", "high_ir_drop_never_converges.json", *_CALIBRATED_EXAMPLE_FILES],
)
def test_example_hw_configs_load_successfully(example_file):
    """The checked-in examples/hw_configs/ files must stay valid HWConfigs
    -- this would catch a schema change in schemas/config.py breaking them."""
    path = f"examples/hw_configs/{example_file}"
    hw = load_hw_config(path)
    assert hw.hw_spec_id


@pytest.mark.parametrize("example_file", _CALIBRATED_EXAMPLE_FILES)
def test_calibrated_example_hw_configs_exactly_match_a_known_reference(example_file):
    """Each examples/hw_configs/calibrated_*.json file exists specifically to
    save a researcher from hand-writing crossbar_rows/cols/adc_bits that
    exactly match a tools.calibration.KNOWN_REFERENCES entry -- a typo here
    would silently turn a 'calibrated' example into an uncalibrated one, and
    only this test would catch it (main.py's own --hw-config loading only
    checks HWConfig schema validity, not calibration match)."""
    hw = load_hw_config(f"examples/hw_configs/{example_file}")
    assert bootstrap_calibration_factors(hw) != {}


# --- Interactive --hw-config picker --------------------------------------------


def _feed_inputs(monkeypatch, values):
    """Scripts builtins.input() to return `values` in order -- a call past
    the end raises StopIteration (an unhandled exception fails the test),
    which doubles as an assertion that the code under test doesn't prompt
    more times than expected."""
    iterator = iter(values)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(iterator))


def test_prompt_custom_hw_config_builds_from_scripted_input(monkeypatch):
    _feed_inputs(
        monkeypatch,
        ["my_custom_chip", "64", "64", "8", "4", "8", "ring", "20.0", "0.1", "0.02", "128.0"],
    )

    hw = main.prompt_custom_hw_config()

    assert hw.hw_spec_id == "my_custom_chip"
    assert hw.crossbar_rows == 64
    assert hw.crossbar_cols == 64
    assert hw.adc_bits == 4
    assert hw.noc_topology.value == "ring"
    assert hw.noc_link_bandwidth_gbps == 20.0


def test_prompt_custom_hw_config_uses_defaults_on_blank_input(monkeypatch):
    _feed_inputs(monkeypatch, [""] * 11)

    hw = main.prompt_custom_hw_config()

    assert hw.crossbar_rows == DEFAULT_HW_CONFIG.crossbar_rows
    assert hw.crossbar_cols == DEFAULT_HW_CONFIG.crossbar_cols
    assert hw.adc_bits == DEFAULT_HW_CONFIG.adc_bits
    assert hw.hw_spec_id.startswith("custom_")


def test_prompt_custom_hw_config_reprompts_on_invalid_noc_topology(monkeypatch, capsys):
    bad_pass = ["", "", "", "", "", "", "bogus", "", "", "", ""]  # 7th field is noc_topology
    good_pass = [""] * 11
    _feed_inputs(monkeypatch, bad_pass + good_pass)

    hw = main.prompt_custom_hw_config()

    assert hw.noc_topology == DEFAULT_HW_CONFIG.noc_topology
    assert "다시 확인해주세요" in capsys.readouterr().out


def test_prompt_custom_hw_config_reprompt_only_resets_the_offending_field(monkeypatch):
    """A validation failure must only reset the *bad* field to its original
    default -- every other already-entered (valid) value must survive the
    restart, so the researcher isn't forced to retype fields that were
    already correct."""
    bad_pass = ["my_chip", "77", "77", "16", "8", "8", "bogus", "10.0", "0.1", "0.02", "128.0"]
    fix_pass = ["", "", "", "", "", "", "", "", "", "", ""]  # blank noc_topology now falls back to a valid default
    _feed_inputs(monkeypatch, bad_pass + fix_pass)

    hw = main.prompt_custom_hw_config()

    assert hw.hw_spec_id == "my_chip"
    assert hw.crossbar_rows == 77
    assert hw.crossbar_cols == 77
    assert hw.sram_buffer_kb == 128.0
    assert hw.noc_topology == DEFAULT_HW_CONFIG.noc_topology  # the one field that got reset


def test_prompt_custom_hw_config_back_command_returns_to_the_previous_field(monkeypatch):
    # hw_spec_id="", crossbar_rows="999" (typo), then "b" at crossbar_cols
    # steps back to redo crossbar_rows as "64"; everything else blank.
    _feed_inputs(monkeypatch, ["", "999", "b", "64", ""] + [""] * 8)

    hw = main.prompt_custom_hw_config()

    assert hw.crossbar_rows == 64


def test_prompt_custom_hw_config_back_on_the_first_field_cancels(monkeypatch, capsys):
    # "b" on the very first field (hw_spec_id) has nowhere earlier to go
    # back to within this form -- it must cancel (return None) rather than
    # silently re-showing the identical prompt, which would look like
    # 'back' did nothing.
    _feed_inputs(monkeypatch, ["b"])

    result = main.prompt_custom_hw_config()

    assert result is None
    assert "최상위 필드로 돌아갑니다" in capsys.readouterr().out


def test_prompt_hw_config_from_examples_returns_the_selected_file(monkeypatch):
    files = sorted(main.EXAMPLE_HW_CONFIGS_DIR.glob("*.json"))
    _feed_inputs(monkeypatch, ["1"])

    hw = main.prompt_hw_config_from_examples()

    assert hw.hw_spec_id == main.load_hw_config(str(files[0])).hw_spec_id


def test_prompt_hw_config_from_examples_returns_none_on_blank_input(monkeypatch):
    _feed_inputs(monkeypatch, [""])
    assert main.prompt_hw_config_from_examples() is None


def test_prompt_hw_config_from_examples_returns_none_on_invalid_number(monkeypatch, capsys):
    _feed_inputs(monkeypatch, ["9999"])
    assert main.prompt_hw_config_from_examples() is None
    assert "잘못된 선택" in capsys.readouterr().out


def test_prompt_for_hw_config_default_choice_offers_approximate_when_unmatched(monkeypatch, capsys):
    # "3" -> menu choice 3 (default); DEFAULT_HW_CONFIG has no exact match
    # (its adc_bits=8 doesn't match the one 128x128 reference, adc_bits=7),
    # so a second input (the approximate-calibration y/N prompt) is consumed.
    _feed_inputs(monkeypatch, ["3", "n"])

    hw, allow_approximate = main.prompt_for_hw_config()

    assert hw.hw_spec_id == DEFAULT_HW_CONFIG.hw_spec_id
    assert allow_approximate is False
    assert "가장 가까운 레퍼런스" in capsys.readouterr().out


def test_prompt_for_hw_config_accepts_approximate_calibration_on_yes(monkeypatch):
    _feed_inputs(monkeypatch, ["3", "y"])
    _hw, allow_approximate = main.prompt_for_hw_config()
    assert allow_approximate is True


def test_prompt_for_hw_config_blank_choice_exits(monkeypatch, capsys):
    # Blank input now defaults to menu choice 4 (종료), not 3 (기본값) --
    # a researcher who doesn't know what to pick shouldn't be silently
    # dropped into a run using a hw_spec they never chose.
    _feed_inputs(monkeypatch, [""])

    with pytest.raises(SystemExit) as exc_info:
        main.prompt_for_hw_config()

    assert exc_info.value.code == 0
    assert "입력이 없습니다" in capsys.readouterr().out


def test_prompt_for_hw_config_choice_4_exits(monkeypatch, capsys):
    _feed_inputs(monkeypatch, ["4"])

    with pytest.raises(SystemExit) as exc_info:
        main.prompt_for_hw_config()

    assert exc_info.value.code == 0
    assert "종료합니다" in capsys.readouterr().out


def test_prompt_for_hw_config_exact_match_skips_the_approximate_prompt(monkeypatch, capsys):
    files = sorted(main.EXAMPLE_HW_CONFIGS_DIR.glob("*.json"))
    index = next(i for i, f in enumerate(files, 1) if f.name == "calibrated_neurosim_v1_5.json")
    # Only two inputs queued (menu choice "2", then the file number) -- a
    # third input() call (i.e. an unwanted approximate-calibration prompt
    # for an already-exact match) would raise StopIteration and fail the test.
    _feed_inputs(monkeypatch, ["2", str(index)])

    hw, allow_approximate = main.prompt_for_hw_config()

    assert hw.hw_spec_id == "calibrated_neurosim_v1_5"
    assert allow_approximate is False
    assert "exact match" in capsys.readouterr().out


def test_prompt_for_hw_config_reshows_menu_when_custom_entry_is_cancelled(monkeypatch):
    # "1" -> custom entry; "b" on hw_spec_id cancels back to the menu;
    # "3" -> default this time; "n" -> decline the approximate-calibration
    # offer (DEFAULT_HW_CONFIG has no exact match).
    _feed_inputs(monkeypatch, ["1", "b", "3", "n"])

    hw, allow_approximate = main.prompt_for_hw_config()

    assert hw.hw_spec_id == DEFAULT_HW_CONFIG.hw_spec_id
    assert allow_approximate is False


def test_prompt_for_hw_config_falls_back_to_default_on_eof(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))

    hw, allow_approximate = main.prompt_for_hw_config()

    assert hw is DEFAULT_HW_CONFIG
    assert allow_approximate is False
    assert "기본값을 사용" in capsys.readouterr().out


def test_prompt_for_dashboard_out_declines_by_default(monkeypatch):
    # "" -> [y/N] default is "no report"; a second input() call (the path
    # prompt) would raise StopIteration and fail the test.
    _feed_inputs(monkeypatch, [""])
    assert main.prompt_for_dashboard_out("cim_v1_128x128", "abcd1234-...") is None


def test_prompt_for_dashboard_out_yes_with_blank_path_uses_default_name(monkeypatch):
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 28, 15, 4, 5)

    monkeypatch.setattr(main, "datetime", _FixedDatetime)
    _feed_inputs(monkeypatch, ["y", ""])
    result = main.prompt_for_dashboard_out("cim_v1_128x128", "abcd1234-5678")
    assert result == str(main.REPORT_DIR / "report_cim_v1_128x128_abcd1234_20260728_150405.html")


def test_prompt_for_dashboard_out_yes_with_custom_path(monkeypatch, tmp_path):
    custom = str(tmp_path / "my_report.html")
    _feed_inputs(monkeypatch, ["y", custom])
    assert main.prompt_for_dashboard_out("cim_v1_128x128", "abcd1234-5678") == custom


def test_prompt_for_dashboard_out_falls_back_to_none_on_eof(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    assert main.prompt_for_dashboard_out("cim_v1_128x128", "abcd1234-5678") is None


def test_prompt_for_targets_blank_leaves_everything_ungated(monkeypatch):
    _feed_inputs(monkeypatch, ["", "", ""])
    assert main.prompt_for_targets(None, None, None) == (None, None, None)


def test_prompt_for_targets_parses_entered_numbers(monkeypatch):
    _feed_inputs(monkeypatch, ["0.7", "2000", "5.0"])
    assert main.prompt_for_targets(None, None, None) == (0.7, 2000.0, 5.0)


def test_prompt_for_targets_reprompts_on_unparseable_input(monkeypatch, capsys):
    _feed_inputs(monkeypatch, ["not-a-number", "0.7", "", ""])
    assert main.prompt_for_targets(None, None, None) == (0.7, None, None)
    assert "숫자를 입력하거나" in capsys.readouterr().out


def test_prompt_for_targets_skips_fields_already_given_on_the_cli(monkeypatch):
    # Only target_latency_ms is unset -- a single input() call is queued, so
    # a second call (e.g. re-asking for target_accuracy/target_energy_pj)
    # would raise StopIteration and fail the test.
    _feed_inputs(monkeypatch, ["5.0"])
    assert main.prompt_for_targets(0.7, 2000.0, None) == (0.7, 2000.0, 5.0)


def test_prompt_for_targets_all_given_skips_prompting_entirely(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(AssertionError("must not prompt")))
    assert main.prompt_for_targets(0.7, 2000.0, 5.0) == (0.7, 2000.0, 5.0)


def test_prompt_for_targets_eof_keeps_already_entered_values(monkeypatch):
    values = iter(["0.7"])

    def fake_input(prompt=""):
        try:
            return next(values)
        except StopIteration:
            raise EOFError()

    monkeypatch.setattr("builtins.input", fake_input)
    assert main.prompt_for_targets(None, None, None) == (0.7, None, None)


def test_main_invokes_interactive_prompt_when_hw_config_omitted_and_stdin_is_a_tty(monkeypatch, tmp_path):
    calls = {}

    def fake_prompt():
        calls["invoked"] = True
        return DEFAULT_HW_CONFIG, False

    monkeypatch.setattr(main, "prompt_for_hw_config", fake_prompt)
    monkeypatch.setattr(main, "prompt_for_targets", lambda a, e, l: (a, e, l))
    monkeypatch.setattr(main, "prompt_for_dashboard_out", lambda hw_spec_id, thread_id: None)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.argv", ["main.py", "--checkpoint-db", str(tmp_path / "checkpoints.sqlite")])

    main.main()

    assert calls.get("invoked") is True


def test_main_skips_interactive_prompt_when_hw_config_is_given(monkeypatch, tmp_path):
    def fail_if_called():
        raise AssertionError("prompt_for_hw_config must not be called when --hw-config is given")

    monkeypatch.setattr(main, "prompt_for_hw_config", fail_if_called)
    monkeypatch.setattr(main, "prompt_for_dashboard_out", lambda hw_spec_id, thread_id: None)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--hw-config",
            "examples/hw_configs/default_cim_v1_128x128.json",
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
        ],
    )

    main.main()  # must not raise (prompt_for_hw_config would if called)


def test_main_skips_interactive_prompt_when_stdin_is_not_a_tty(monkeypatch, tmp_path):
    def fail_if_called():
        raise AssertionError("prompt_for_hw_config must not be called when stdin isn't a tty")

    monkeypatch.setattr(main, "prompt_for_hw_config", fail_if_called)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.argv", ["main.py", "--checkpoint-db", str(tmp_path / "checkpoints.sqlite")])

    main.main()  # must not raise


def test_main_invokes_dashboard_prompt_when_dashboard_out_omitted_and_stdin_is_a_tty(monkeypatch, tmp_path):
    calls = {}

    def fake_prompt(hw_spec_id, thread_id):
        calls["invoked"] = True
        return None

    monkeypatch.setattr(main, "prompt_for_hw_config", lambda: (DEFAULT_HW_CONFIG, False))
    monkeypatch.setattr(main, "prompt_for_targets", lambda a, e, l: (a, e, l))
    monkeypatch.setattr(main, "prompt_for_dashboard_out", fake_prompt)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.argv", ["main.py", "--checkpoint-db", str(tmp_path / "checkpoints.sqlite")])

    main.main()

    assert calls.get("invoked") is True


def test_main_skips_dashboard_prompt_when_dashboard_out_is_given(monkeypatch, tmp_path):
    def fail_if_called(hw_spec_id, thread_id):
        raise AssertionError("prompt_for_dashboard_out must not be called when --dashboard-out is given")

    monkeypatch.setattr(main, "prompt_for_hw_config", lambda: (DEFAULT_HW_CONFIG, False))
    monkeypatch.setattr(main, "prompt_for_targets", lambda a, e, l: (a, e, l))
    monkeypatch.setattr(main, "prompt_for_dashboard_out", fail_if_called)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--dashboard-out",
            str(tmp_path / "report.html"),
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite"),
        ],
    )

    main.main()  # must not raise (prompt_for_dashboard_out would if called)


def test_main_skips_dashboard_prompt_when_stdin_is_not_a_tty(monkeypatch, tmp_path):
    def fail_if_called(hw_spec_id, thread_id):
        raise AssertionError("prompt_for_dashboard_out must not be called when stdin isn't a tty")

    monkeypatch.setattr(main, "prompt_for_dashboard_out", fail_if_called)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.argv", ["main.py", "--checkpoint-db", str(tmp_path / "checkpoints.sqlite")])

    main.main()  # must not raise


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


# --- --parallel-warmup-workers -------------------------------------------------


def test_run_session_seeds_candidate_history_from_parallel_warmup(tmp_path, registered_hw_config):
    from nodes.planner import real_stage_names, warmup_count
    from graph import build_graph

    n_stages = len(real_stage_names("resnet18"))
    db_path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "thread-warmup"}}
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-warmup", checkpointer, parallel_warmup_workers=2)
        final_state = build_graph(checkpointer=checkpointer).get_state(config).values

    # good_hw_config converges on the very first *sequential* iteration
    # (iteration_count == 1, unaffected by warm-up -- planner_node's own
    # counter), but candidate_history must contain every parallel warm-up
    # candidate *plus* that one real sequential candidate (the known
    # limitation documented in tools/batch_warmup.py: warm-up doesn't
    # short-circuit early even though one of its own candidates already
    # converged).
    assert final_state["is_converged"] is True
    assert final_state["iteration_count"] == 1
    assert len(final_state["candidate_history"]) == warmup_count(n_stages) + 1


def test_run_session_without_the_flag_keeps_todays_sequential_behavior(tmp_path, registered_hw_config):
    """Omitting --parallel-warmup-workers must not change anything --
    candidate_history should accumulate one entry per real graph iteration,
    exactly as before this feature existed."""
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-sequential", checkpointer)
        sessions = list_sessions(checkpointer)

    assert sessions[0]["is_converged"] is True
    assert sessions[0]["iteration_count"] == 1  # good_hw_config converges on the very first real iteration


# --- --allow-approximate-calibration -------------------------------------------
#
# registered_hw_config (good_hw_config: 128x128, adc_bits=8) has no exact
# match in tools/calibration.py's KNOWN_REFERENCES -- the one 128x128
# reference there is adc_bits=7, not 8 -- so it's a real unmatched HWConfig,
# not something that happens to already be exactly calibrated.


def test_build_initial_state_leaves_unmatched_hw_uncalibrated_by_default(registered_hw_config):
    state = main.build_initial_state("resnet18", registered_hw_config)
    assert state["calibration_factors"] == {}
    assert state["calibration_provenance"] == {}


def test_build_initial_state_with_allow_approximate_calibration_seeds_a_factor(registered_hw_config):
    state = main.build_initial_state("resnet18", registered_hw_config, allow_approximate_calibration=True)

    assert registered_hw_config.hw_spec_id in state["calibration_factors"]
    assert state["calibration_factors"][registered_hw_config.hw_spec_id] > 0
    assert state["calibration_provenance"][registered_hw_config.hw_spec_id]["approximate"] is True


def test_run_session_prints_uncalibrated_status_by_default(capsys, tmp_path, registered_hw_config):
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-calib-default", checkpointer)

    assert "[calibration] uncalibrated" in capsys.readouterr().out


def test_run_session_with_allow_approximate_calibration_prints_approximate_status(capsys, tmp_path, registered_hw_config):
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session(
            "resnet18",
            registered_hw_config,
            "thread-calib-approx",
            checkpointer,
            allow_approximate_calibration=True,
        )

    assert "[calibration] approximate" in capsys.readouterr().out


# --- .env/.env.local auto-loading ---------------------------------------------


def test_main_loads_dotenv_then_dotenv_local_with_override(monkeypatch, tmp_path):
    """main() must load plain .env first (no override -- real shell-exported
    vars still win) and then .env.local second with override=True (personal
    secrets take priority over shared .env defaults) -- not just call
    load_dotenv() once, and not in the opposite order."""
    calls = []
    monkeypatch.setattr(main, "load_dotenv", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        "sys.argv", ["main.py", "--list-sessions", "--checkpoint-db", str(tmp_path / "checkpoints.sqlite")]
    )

    main.main()

    assert len(calls) == 2
    first_args, first_kwargs = calls[0]
    second_args, second_kwargs = calls[1]
    assert first_args == () and not first_kwargs.get("override")  # plain .env, no override
    assert second_args == (".env.local",) and second_kwargs.get("override") is True
