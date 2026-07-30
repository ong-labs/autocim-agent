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

import argparse
import json
from datetime import datetime

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

import main
from main import DEFAULT_HW_CONFIG, HWConfigError, list_sessions, load_hw_config, print_sessions, run_session
from store import GlobalHWStore, ModelSpecificStore, SqliteLongTermStore
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


def _feed_inputs_capturing_prompts(monkeypatch, values):
    """Like _feed_inputs, but also returns the list of `prompt` strings
    input() was called with -- capsys can't see these (a mocked input()
    never writes its prompt argument to stdout), so this is the only way
    to assert on prompt *text* rather than just on print() output."""
    iterator = iter(values)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(iterator)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def _wizard_args(target_accuracy=None, target_energy_pj=None, target_latency_ms=None, dashboard_out=None):
    return argparse.Namespace(
        target_accuracy=target_accuracy,
        target_energy_pj=target_energy_pj,
        target_latency_ms=target_latency_ms,
        dashboard_out=dashboard_out,
    )


# DEFAULT_HW_CONFIG has no exact calibration match (adc_bits=8 doesn't match
# any KNOWN_REFERENCES entry), so every scripted-input sequence below that
# reaches choice "3" (or blank-defaulted custom fields, which produce the
# identical config) must also answer the approximate-calibration step.
_DECLINE_APPROX = "n"


def test_run_interactive_setup_custom_builds_from_scripted_input(monkeypatch):
    # crossbar_rows/cols/adc_bits deliberately don't match any
    # tools.calibration.KNOWN_REFERENCES entry (unlike a real 64x64/4-bit
    # config), so the approx-calibration step stays in the chain here.
    _feed_inputs(
        monkeypatch,
        ["1", "my_custom_chip", "64", "70", "8", "4", "8", "ring", "20.0", "0.1", "0.02", "128.0", _DECLINE_APPROX, "", "", "", "n", "", "y"],
    )

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")
    hw = result[0]

    assert hw.hw_spec_id == "my_custom_chip"
    assert hw.crossbar_rows == 64
    assert hw.crossbar_cols == 70
    assert hw.adc_bits == 4
    assert hw.noc_topology.value == "ring"
    assert hw.noc_link_bandwidth_gbps == 20.0
    assert result[1] is False  # approx calibration declined
    assert result[2:] == (None, None, None, None)  # targets ungated, no dashboard


def test_run_interactive_setup_custom_uses_defaults_on_blank_input(monkeypatch):
    _feed_inputs(monkeypatch, ["1"] + [""] * 11 + [_DECLINE_APPROX, "", "", "", "n", "", "y"])

    hw = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")[0]

    assert hw.crossbar_rows == DEFAULT_HW_CONFIG.crossbar_rows
    assert hw.crossbar_cols == DEFAULT_HW_CONFIG.crossbar_cols
    assert hw.adc_bits == DEFAULT_HW_CONFIG.adc_bits
    assert hw.hw_spec_id.startswith("custom_")


def test_run_interactive_setup_reprompts_on_invalid_noc_topology(monkeypatch, capsys):
    bad_pass = ["", "", "", "", "", "", "bogus", "", "", "", ""]  # 7th field is noc_topology
    good_pass = [""] * 11
    _feed_inputs(monkeypatch, ["1"] + bad_pass + good_pass + [_DECLINE_APPROX, "", "", "", "n", "", "y"])

    hw = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")[0]

    assert hw.noc_topology == DEFAULT_HW_CONFIG.noc_topology
    assert "다시 확인해주세요" in capsys.readouterr().out


def test_run_interactive_setup_validation_failure_resets_only_bad_field(monkeypatch):
    """A validation failure must only reset the *bad* field to its original
    default -- every other already-entered (valid) value must survive the
    restart, so the researcher isn't forced to retype fields that were
    already correct."""
    bad_pass = ["my_chip", "77", "77", "16", "8", "8", "bogus", "10.0", "0.1", "0.02", "128.0"]
    fix_pass = [""] * 11  # blank noc_topology now falls back to a valid default
    _feed_inputs(monkeypatch, ["1"] + bad_pass + fix_pass + [_DECLINE_APPROX, "", "", "", "n", "", "y"])

    hw = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")[0]

    assert hw.hw_spec_id == "my_chip"
    assert hw.crossbar_rows == 77
    assert hw.crossbar_cols == 77
    assert hw.sram_buffer_kb == 128.0
    assert hw.noc_topology == DEFAULT_HW_CONFIG.noc_topology  # the one field that got reset


def test_run_interactive_setup_back_returns_to_the_previous_field(monkeypatch):
    # hw_spec_id="", crossbar_rows="999" (typo), then "b" at crossbar_cols
    # steps back to redo crossbar_rows as "64"; everything else blank.
    _feed_inputs(
        monkeypatch, ["1", "", "999", "b", "64", ""] + [""] * 8 + [_DECLINE_APPROX, "", "", "", "n", "", "y"]
    )

    hw = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")[0]

    assert hw.crossbar_rows == 64


def test_run_interactive_setup_back_across_the_custom_target_boundary(monkeypatch):
    """'b' must reach backward across step-group boundaries -- from
    target_accuracy (outside the custom-fields group) back into
    sram_buffer_kb (the last custom field), not just within one group.
    Step order here is: ...sram_buffer_kb -> approx_calibration_yn ->
    target_accuracy -> ..., so reaching sram_buffer_kb from target_accuracy
    takes two 'b's (one back into approx_calibration_yn, one more back into
    sram_buffer_kb)."""
    _feed_inputs(
        monkeypatch,
        ["1"] + [""] * 11 + [_DECLINE_APPROX, "b", "b", "999.0", _DECLINE_APPROX, "", "", "", "n", "", "y"],
        # ^ approx-calib "n" -> target_accuracy "b" (-> approx_calibration_yn)
        #   -> "b" again (-> sram_buffer_kb) -> "999.0" (new sram_buffer_kb)
        #   -> approx-calib "n" again -> target_accuracy/energy/latency blank
        #   -> dashboard "n" -> confirm "y"
    )

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert result[0].sram_buffer_kb == 999.0
    assert result[2:] == (None, None, None, None)


def test_run_interactive_setup_back_on_the_first_step_exits(monkeypatch, capsys):
    # 'b' at the very first step (hw_mode) has nowhere earlier to go --
    # same as choice "4"/blank: exit rather than silently loop.
    _feed_inputs(monkeypatch, ["b"])

    with pytest.raises(SystemExit) as exc_info:
        main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert exc_info.value.code == 0
    assert "종료합니다" in capsys.readouterr().out


def test_run_interactive_setup_example_file_choice(monkeypatch):
    files = sorted(main.EXAMPLE_HW_CONFIGS_DIR.glob("*.json"))
    index = next(i for i, f in enumerate(files, 1) if f.name == "calibrated_neurosim_v1_5.json")
    # calibrated_neurosim_v1_5.json is an exact calibration match, so no
    # approx-calibration input is consumed here -- a third input() call
    # would raise StopIteration and fail the test.
    _feed_inputs(monkeypatch, ["2", str(index), "", "", "", "n", "y"])

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert result[0].hw_spec_id == "calibrated_neurosim_v1_5"
    assert result[1] is False


def test_run_interactive_setup_blank_choice_exits(monkeypatch, capsys):
    # Blank input at hw_mode now defaults to menu choice 4 (종료), not 3
    # (기본값) -- a researcher who doesn't know what to pick shouldn't be
    # silently dropped into a run using a hw_spec they never chose.
    _feed_inputs(monkeypatch, [""])

    with pytest.raises(SystemExit) as exc_info:
        main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert exc_info.value.code == 0
    assert "입력이 없습니다" in capsys.readouterr().out


def test_run_interactive_setup_choice_4_exits(monkeypatch, capsys):
    _feed_inputs(monkeypatch, ["4"])

    with pytest.raises(SystemExit) as exc_info:
        main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert exc_info.value.code == 0
    assert "종료합니다" in capsys.readouterr().out


def test_run_interactive_setup_offers_approximate_when_unmatched(monkeypatch, capsys):
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "", "", "", "n", "y"])

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert result[0].hw_spec_id == DEFAULT_HW_CONFIG.hw_spec_id
    assert result[1] is False
    assert "가장 가까운 레퍼런스" in capsys.readouterr().out


def test_run_interactive_setup_accepts_approximate_calibration_on_yes(monkeypatch):
    _feed_inputs(monkeypatch, ["3", "y", "", "", "", "n", "y"])
    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")
    assert result[1] is True


def test_run_interactive_setup_targets_skip_fields_given_on_the_cli(monkeypatch):
    # Only target_latency_ms is unset -- a single numeric input is queued
    # for it, so an extra unwanted target prompt would raise StopIteration.
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "5.0", "n", "y"])

    result = main.run_interactive_setup(_wizard_args(target_accuracy=0.7, target_energy_pj=2000.0), "abcd1234-5678")

    assert result[2:5] == (0.7, 2000.0, 5.0)


def test_run_interactive_setup_targets_all_given_skips_prompting_entirely(monkeypatch):
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "n", "y"])  # no target prompts queued
    result = main.run_interactive_setup(
        _wizard_args(target_accuracy=0.7, target_energy_pj=2000.0, target_latency_ms=5.0), "abcd1234-5678"
    )
    assert result[2:5] == (0.7, 2000.0, 5.0)


def test_run_interactive_setup_dashboard_yes_writes_under_report_dir(monkeypatch):
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 28, 15, 4, 5)

    monkeypatch.setattr(main, "datetime", _FixedDatetime)
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "", "", "", "y", "", "y"])

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert result[5] == str(main.REPORT_DIR / "report_cim_v1_128x128_abcd1234_20260728_150405.html")


def test_run_interactive_setup_confirm_back_returns_to_last_step(monkeypatch):
    # At confirm, "b" steps back into dashboard_yn ("n" -> "y" the second
    # time), rather than re-walking the whole chain from the top.
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "", "", "", "n", "b", "y", "", "y"])

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert result[5] is not None  # dashboard path was written after all


def test_run_interactive_setup_confirm_shows_session_info_before_asking_to_proceed(monkeypatch, capsys):
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)  # don't touch the real .env.local
    monkeypatch.delenv("AUTOCIM_PLANNER_MODEL", raising=False)
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "", "", "", "n", "y"])

    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    printed = capsys.readouterr().out
    session_line_pos = printed.find("[session] LLM=")
    header_pos = printed.find("=== 입력값 확인 ===")
    assert session_line_pos != -1 and header_pos != -1
    # Shown right after the confirm header, before the numbered field list
    # and well before the eventual [y/n] prompt (a separate input() call).
    assert header_pos < session_line_pos


def test_run_interactive_setup_confirm_refresh_rereads_env_without_losing_progress(monkeypatch, capsys):
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)  # don't touch the real .env.local
    # "r" must not consume/replay any wizard field -- only "y"/"n"/back/jump
    # inputs are queued after it, so an unwanted extra field prompt would
    # raise StopIteration.
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "0.7", "", "", "n", "r", "y"])

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    out = capsys.readouterr().out
    assert out.count("[session] LLM=") == 2  # once before refresh, once after re-render
    assert ".env/.env.local을 다시 읽었습니다" in out
    assert result[2] == 0.7  # target_accuracy survived the refresh round-trip


def test_run_interactive_setup_confirm_jump_to_field_edits_and_returns(monkeypatch):
    # From confirm, typing "3" (crossbar_rows's listed number for the
    # custom-entry path) jumps straight there instead of walking back one
    # step at a time -- and, unlike the sequential 'b' walk, must land
    # straight back at confirm afterward without replaying every field
    # after it (exactly ["3", "256", "y"] queued for the jump round-trip;
    # any extra unwanted input() call there would raise StopIteration).
    _feed_inputs(
        monkeypatch,
        ["1"]
        + [""] * 11  # hw_spec_id..sram_buffer_kb (11 fields)
        + [_DECLINE_APPROX, "0.7", "", "", "n", ""]  # approx-calib, 3 targets, dashboard_yn, save-config -> confirm
        + ["3", "256", "y"],  # confirm: jump to [3] crossbar_rows, enter 256, straight back to confirm, proceed
    )

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert result[0].crossbar_rows == 256
    assert result[2] == 0.7  # target_accuracy survived the jump-edit round-trip


def test_run_interactive_setup_confirm_jump_edit_prints_modified_confirmation(monkeypatch, capsys):
    _feed_inputs(
        monkeypatch,
        ["1"]
        + [""] * 11
        + [_DECLINE_APPROX, "", "", "", "n", ""]
        + ["3", "256", "y"],  # confirm: jump to [3] crossbar_rows, enter 256, straight back to confirm, proceed
    )

    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    out = capsys.readouterr().out
    assert "[3] 수정되었습니다." in out


def test_run_interactive_setup_confirm_lists_fields_before_session_info(monkeypatch, capsys):
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "", "", "", "n", "y"])

    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    out = capsys.readouterr().out
    list_pos = out.find("  [1] ")
    session_pos = out.find("[session] LLM=")
    assert list_pos != -1 and session_pos != -1
    assert list_pos < session_pos  # numbered field list comes before session info now


def test_run_interactive_setup_eof_proceeds_with_partial_answers(monkeypatch):
    values = iter(["3", _DECLINE_APPROX, "0.7"])

    def fake_input(prompt=""):
        try:
            return next(values)
        except StopIteration:
            raise EOFError()

    monkeypatch.setattr("builtins.input", fake_input)

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert result[0].hw_spec_id == DEFAULT_HW_CONFIG.hw_spec_id
    assert result[2] == 0.7  # entered before EOF -- not discarded
    assert result[5] is None  # never reached the dashboard step


def test_run_interactive_setup_keyboard_interrupt_cancels_instead_of_proceeding(monkeypatch, capsys):
    # Ctrl-C must cancel the whole setup (None), not silently start a real
    # session with whatever was entered so far -- unlike EOF, this is an
    # explicit "stop" signal from the researcher.
    values = iter(["3", _DECLINE_APPROX])

    def fake_input(prompt=""):
        try:
            return next(values)
        except StopIteration:
            raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", fake_input)

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert result is None
    assert "중단되었습니다" in capsys.readouterr().out


def test_run_interactive_setup_yn_prompts_show_y_slash_n(monkeypatch):
    # input()'s prompt text isn't captured by capsys once input() itself is
    # mocked, so this asserts on the actual prompt strings passed to it.
    prompts = _feed_inputs_capturing_prompts(monkeypatch, ["3", _DECLINE_APPROX, "", "", "", "n", "y"])
    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")
    yn_prompts = [p for p in prompts if "진행할까요" in p]
    assert yn_prompts, "expected at least one y/n prompt to have fired"
    assert all("[y/n]" in p for p in yn_prompts)
    assert not any("[Y]es" in p for p in prompts)


def test_run_interactive_setup_field_prompts_have_no_per_field_back_hint(monkeypatch):
    # The repeated "(뒤로: b)" suffix on every single field prompt was
    # explicitly asked to be removed in favor of one general explanation
    # up front (main()'s first-run banner) -- input()'s own prompt text
    # isn't visible via capsys once input() is mocked, so check it via the
    # prompt-capturing feeder instead of print() output.
    prompts = _feed_inputs_capturing_prompts(
        monkeypatch, ["1"] + [""] * 11 + [_DECLINE_APPROX, "", "", "", "n", "", "y"]
    )
    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")
    assert not any("뒤로: b" in p for p in prompts)


def test_run_interactive_setup_target_prompt_omits_blank_placeholder(monkeypatch):
    prompts = _feed_inputs_capturing_prompts(monkeypatch, ["3", _DECLINE_APPROX, "", "", "", "n", "y"])
    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")
    assert not any("[빈칸]" in p for p in prompts)


def test_run_interactive_setup_confirm_label_says_custom(monkeypatch, capsys):
    _feed_inputs(monkeypatch, ["1"] + [""] * 11 + [_DECLINE_APPROX, "", "", "", "n", "", "y"])
    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")
    out = capsys.readouterr().out
    assert "hw-config 선택 방식: custom" in out
    assert "직접 입력" not in out


def test_run_interactive_setup_jump_edit_to_invalid_custom_field_triggers_recovery(monkeypatch):
    # Jump-editing a *non-last* custom field (adc_bits) to an out-of-range
    # value must still trigger the same validation recovery the sequential
    # walk gets -- not silently fall back to DEFAULT_HW_CONFIG at the end.
    _feed_inputs(
        monkeypatch,
        ["1"]
        + [""] * 11
        + [_DECLINE_APPROX, "", "", "", "n"]  # -> confirm screen
        + ["6", "99"]  # jump to [6] adc_bits, invalid value 99
        + [""] * 11  # validation failure -> back to hw_spec_id, re-walk all 11 (adc_bits now reset)
        + [_DECLINE_APPROX, "", "", "", "n", "", "y"],
    )

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert result[0].adc_bits == DEFAULT_HW_CONFIG.adc_bits


def test_run_interactive_setup_saves_custom_hw_config_when_requested(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "EXAMPLE_HW_CONFIGS_DIR", tmp_path)
    _feed_inputs(
        monkeypatch,
        ["1", "saved_chip"] + [""] * 10 + [_DECLINE_APPROX, "", "", "", "n", "y", "y"],
        # ^ hw_spec_id="saved_chip", rest blank, approx-calib "n", targets
        #   blank, dashboard "n", save-config "y" -> confirm "y"
    )

    result = main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    saved_path = tmp_path / "saved_chip.json"
    assert saved_path.exists()
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["hw_spec_id"] == "saved_chip"
    assert saved["hw_spec_id"] == result[0].hw_spec_id


def test_run_interactive_setup_declines_saving_custom_hw_config_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "EXAMPLE_HW_CONFIGS_DIR", tmp_path)
    _feed_inputs(monkeypatch, ["1", "not_saved_chip"] + [""] * 10 + [_DECLINE_APPROX, "", "", "", "n", "", "y"])

    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert not (tmp_path / "not_saved_chip.json").exists()


def test_run_interactive_setup_refuses_to_overwrite_existing_hw_config_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "EXAMPLE_HW_CONFIGS_DIR", tmp_path)
    existing = tmp_path / "clash.json"
    existing.write_text('{"sentinel": true}', encoding="utf-8")

    _feed_inputs(monkeypatch, ["1", "clash"] + [""] * 10 + [_DECLINE_APPROX, "", "", "", "n", "y", "y"])

    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")

    assert json.loads(existing.read_text(encoding="utf-8")) == {"sentinel": True}  # untouched
    assert "이미 존재해서 덮어쓰지 않았습니다" in capsys.readouterr().out


def test_run_interactive_setup_save_step_not_offered_for_default_choice(monkeypatch, tmp_path):
    # hw_mode "3" (default) -- no save_custom_hw_config_yn input queued; an
    # unwanted extra input() call there would raise StopIteration.
    monkeypatch.setattr(main, "EXAMPLE_HW_CONFIGS_DIR", tmp_path)
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "", "", "", "n", "y"])
    main.run_interactive_setup(_wizard_args(), "abcd1234-5678")
    assert list(tmp_path.iterdir()) == []


def test_prompt_for_dashboard_out_declines_by_default(monkeypatch):
    # "" -> [y/n] default is "no report"; a second input() call would raise
    # StopIteration and fail the test (there's no path prompt anymore).
    _feed_inputs(monkeypatch, [""])
    assert main.prompt_for_dashboard_out("cim_v1_128x128", "abcd1234-...") is None


def test_prompt_for_dashboard_out_yes_always_uses_the_fixed_default_path(monkeypatch):
    # Only one input() call is queued -- the y/n answer. A second call
    # (e.g. an unwanted path prompt) would raise StopIteration; the path
    # itself is never something a researcher can type.
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 28, 15, 4, 5)

    monkeypatch.setattr(main, "datetime", _FixedDatetime)
    _feed_inputs(monkeypatch, ["y"])
    result = main.prompt_for_dashboard_out("cim_v1_128x128", "abcd1234-5678")
    assert result == str(main.REPORT_DIR / "report_cim_v1_128x128_abcd1234_20260728_150405.html")


def test_prompt_for_dashboard_out_falls_back_to_none_on_eof(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    assert main.prompt_for_dashboard_out("cim_v1_128x128", "abcd1234-5678") is None


# --- prompt_for_override (HITL [1]/[2]/[3] menu) ------------------------------


def _hitl_payload(
    reason="accuracy 0.3 below target 0.7",
    avg_bits=4,
    avg_pruning=0.2,
    hw_spec_id=None,
    target_accuracy=None,
    current_accuracy=None,
    target_energy_pj=None,
    current_energy_pj=None,
    target_latency_ms=None,
    current_latency_ms=None,
    search_bound_overrides=None,
):
    return {
        "reason": "Convergence stalled -- researcher review requested",
        "iteration_count": 3,
        "retry_count": 3,
        "failure_history": [{"iteration": 3, "reason": reason}],
        "latest_metrics": {
            "verifier": {"data": {"energy_pj": current_energy_pj, "noc_latency_ms": current_latency_ms}},
            "tuner": {
                "data": {
                    "avg_weight_bits": avg_bits,
                    "avg_column_pruning_ratio": avg_pruning,
                    "accuracy": current_accuracy,
                }
            },
        },
        "hw_spec_id": hw_spec_id,
        "target_accuracy": target_accuracy,
        "target_energy_pj": target_energy_pj,
        "target_latency_ms": target_latency_ms,
        "search_bound_overrides": search_bound_overrides,
    }


def test_prompt_for_override_choice_1_uses_the_suggestion(monkeypatch):
    _feed_inputs(monkeypatch, ["1"])
    result = main.prompt_for_override(_hitl_payload())
    assert result == {"new_bounds": main.suggest_override_bounds(_hitl_payload())}


def test_prompt_for_override_choice_1_unavailable_reprompts_then_stops(monkeypatch, capsys):
    # "verifier reported not converged" has no rule-based suggestion --
    # choosing [1] must not silently do nothing or crash, just reprompt.
    _feed_inputs(monkeypatch, ["1", "3"])
    result = main.prompt_for_override(_hitl_payload(reason="verifier reported not converged"))
    assert result is None
    assert "추천값이 없습니다" in capsys.readouterr().out or "다른 항목을 선택" in capsys.readouterr().out


def test_prompt_for_override_choice_2_accepts_valid_json(monkeypatch):
    _feed_inputs(monkeypatch, ["2", '{"weight_bits_min": 6}'])
    result = main.prompt_for_override(_hitl_payload())
    assert result == {"new_bounds": {"weight_bits_min": 6}}


def test_prompt_for_override_choice_2_invalid_json_reprompts_instead_of_crashing(monkeypatch, capsys):
    # This used to be an uncaught json.JSONDecodeError that killed the
    # whole run -- a typo here must never lose already-completed QAT work.
    _feed_inputs(monkeypatch, ["2", "{not valid json", '{"weight_bits_min": 6}'])
    result = main.prompt_for_override(_hitl_payload())
    assert result == {"new_bounds": {"weight_bits_min": 6}}
    assert "잘못된 JSON" in capsys.readouterr().out


def test_prompt_for_override_choice_2_warns_on_unrecognized_key_but_still_applies(monkeypatch, capsys):
    _feed_inputs(monkeypatch, ["2", '{"wieght_bits_min": 6}'])  # typo'd key name
    result = main.prompt_for_override(_hitl_payload())
    assert result == {"new_bounds": {"wieght_bits_min": 6}}
    assert "인식되는 키가 아니라" in capsys.readouterr().out


def test_prompt_for_override_choice_2_back_returns_to_the_menu(monkeypatch):
    _feed_inputs(monkeypatch, ["2", "b", "3"])
    result = main.prompt_for_override(_hitl_payload())
    assert result is None


def test_prompt_for_override_choice_2_blank_resumes_unchanged(monkeypatch):
    _feed_inputs(monkeypatch, ["2", ""])
    result = main.prompt_for_override(_hitl_payload())
    assert result == {"new_bounds": {}}


def test_prompt_for_override_choice_3_stops_the_run(monkeypatch):
    _feed_inputs(monkeypatch, ["3"])
    assert main.prompt_for_override(_hitl_payload()) is None


def test_prompt_for_override_invalid_menu_choice_reprompts(monkeypatch, capsys):
    _feed_inputs(monkeypatch, ["9", "3"])
    result = main.prompt_for_override(_hitl_payload())
    assert result is None
    assert "잘못된 선택입니다" in capsys.readouterr().out


# --- suggest_override_bounds gap-tier escalation ------------------------------


def test_suggest_override_bounds_small_gap_uses_tier_0_step(registered_hw_config):
    # (0.7 - 0.65) / 0.7 ~= 7% -- under the 15% tier-0 threshold. avg_bits=2
    # (not 4) so the +1 step lands at 3, safely under this hw_spec's own
    # ceiling (adc_bits=8, dac_bits=4 -> 4) -- not conflating "tier-0 step
    # applied" with "capped at the ceiling anyway".
    payload = _hitl_payload(
        avg_bits=2, hw_spec_id=registered_hw_config.hw_spec_id, target_accuracy=0.7, current_accuracy=0.65
    )
    assert main.suggest_override_bounds(payload)["weight_bits_min"] == 3  # 2 + step(1)


def test_suggest_override_bounds_medium_gap_uses_tier_1_step(registered_hw_config):
    # (0.7 - 0.5) / 0.7 ~= 29% -- in the 15-40% tier-1 band. Naive step
    # (2 + step(2) = 4) lands exactly on this hw_spec's ceiling (adc_bits=8,
    # dac_bits=4 -> 4) -- pulled back by _MIN_BITS_SEARCH_MARGIN(1) so
    # weight_bits_min..bits_ceiling isn't pinned to a single value.
    payload = _hitl_payload(
        avg_bits=2, hw_spec_id=registered_hw_config.hw_spec_id, target_accuracy=0.7, current_accuracy=0.5
    )
    assert main.suggest_override_bounds(payload)["weight_bits_min"] == 3  # 4, pulled back 1 from the ceiling


def test_suggest_override_bounds_large_gap_uses_tier_2_step_capped_at_hw_ceiling(registered_hw_config):
    # (0.7 - 0.1) / 0.7 ~= 86% -- well past the 40% tier-2 threshold, so the
    # naive suggestion (avg_bits 4 + step(3) = 7) would exceed this
    # hw_spec's real ceiling (adc_bits=8, dac_bits=4 -> min(8,8,4) = 4).
    # Capping it at the ceiling exactly would pin weight_bits to a single
    # value (weight_bits_max is untouched, still == ceiling) -- must land
    # one below it (_MIN_BITS_SEARCH_MARGIN) so there's still a real
    # weight_bits range left for the search to explore.
    payload = _hitl_payload(
        avg_bits=4, hw_spec_id=registered_hw_config.hw_spec_id, target_accuracy=0.7, current_accuracy=0.1
    )
    assert main.suggest_override_bounds(payload)["weight_bits_min"] == 3  # ceiling(4) - margin(1), not 7 or 4


def test_suggest_override_bounds_compression_direction_also_escalates_and_caps(registered_hw_config):
    # energy_pj 4000 vs target 2000 -- a 100% overshoot, tier-2 -> step(3)
    # for weight_bits_max, but never below nodes/planner.py's own search
    # floor (else the override would invert the range and get silently
    # ignored downstream) -- and, mirroring the wants_precision case, never
    # exactly AT that floor either, since weight_bits_min is untouched and
    # still == floor: that would pin weight_bits to a single value the same
    # way an unclamped tier-2 step pinned it at the ceiling in production.
    payload = _hitl_payload(
        reason="energy_pj 4000.0 above target 2000.0",
        avg_bits=3,
        hw_spec_id=registered_hw_config.hw_spec_id,
        target_energy_pj=2000.0,
        current_energy_pj=4000.0,
    )
    result = main.suggest_override_bounds(payload)
    assert result["weight_bits_max"] == 3  # floor(2) + margin(1), not 0 and not exactly the floor


def test_suggest_override_bounds_never_collapses_weight_bits_or_pruning_to_a_single_value(registered_hw_config):
    """Reproduces a real production stall: a >=40% accuracy miss (tier-2)
    on this hw_spec (dac_bits=4 -> weight_bits ceiling 4) suggested
    weight_bits_min=4 and pruning_ratio_max=0.0 -- both exactly pinning the
    untouched other end of their range (weight_bits_max was already 4,
    pruning_ratio_min was already 0.0). Every later surrogate-model
    proposal was then forced identical (nodes/planner.py has nothing left
    to search), and 3 more real QAT trials of the same config (varying only
    by training's own randomness) burned real time before HITL fired
    again. The suggestion must always leave a searchable range."""
    payload = _hitl_payload(
        avg_bits=3.3,
        avg_pruning=0.13083,
        hw_spec_id=registered_hw_config.hw_spec_id,
        target_accuracy=0.8,
        current_accuracy=0.436,
    )
    result = main.suggest_override_bounds(payload)
    bit_bounds, pruning_bounds = main.search_bounds(registered_hw_config.hw_spec_id)
    assert result["weight_bits_min"] < bit_bounds[1]  # strictly below the untouched weight_bits_max
    assert result["pruning_ratio_max"] > pruning_bounds[0]  # strictly above the untouched pruning_ratio_min


def test_suggest_override_bounds_does_not_regress_an_already_approved_override(registered_hw_config):
    """Regression: a real run approved weight_bits_min=4 at one HITL round
    (registered_hw_config's own ceiling: adc_bits=8, dac_bits=4 -> 4), then
    a LATER HITL round's suggestion silently proposed weight_bits_min=3 --
    looser than what was already approved -- because suggest_override_bounds
    computed fresh from the hw's raw ceiling every time, never seeing
    search_bound_overrides at all (nodes/hitl.py's interrupt payload didn't
    even carry it). The suggestion must never be looser than what a
    researcher has already approved earlier this run."""
    payload = _hitl_payload(
        avg_bits=3.0,
        hw_spec_id=registered_hw_config.hw_spec_id,
        target_accuracy=0.7,
        current_accuracy=0.3,
        search_bound_overrides={"weight_bits_min": 4},
    )
    result = main.suggest_override_bounds(payload)
    assert result["weight_bits_min"] >= 4


def test_suggest_override_bounds_missing_target_falls_back_to_tier_0(registered_hw_config):
    # No target_accuracy configured for this run -- can't compute a gap, so
    # this must behave exactly like the old fixed-step (always tier 0)
    # suggestion instead of erroring or guessing a tier.
    payload = _hitl_payload(avg_bits=2, hw_spec_id=registered_hw_config.hw_spec_id)
    assert main.suggest_override_bounds(payload)["weight_bits_min"] == 3  # 2 + step(1)


def test_run_interactive_setup_skips_dashboard_prompt_when_dashboard_out_given(monkeypatch):
    # --dashboard-out already given on the CLI -- the dashboard_yn/path
    # steps must not even be in the chain (a queued extra input would raise
    # StopIteration if they were).
    _feed_inputs(monkeypatch, ["3", _DECLINE_APPROX, "", "", "", "y"])

    result = main.run_interactive_setup(
        _wizard_args(dashboard_out="/tmp/explicit_report.html"), "abcd1234-5678"
    )

    assert result[5] == "/tmp/explicit_report.html"


def test_main_invokes_interactive_setup_when_hw_config_omitted_and_stdin_is_a_tty(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_setup(args, thread_id):
        calls["invoked"] = True
        return DEFAULT_HW_CONFIG, False, None, None, None, None

    monkeypatch.setattr(main, "run_interactive_setup", fake_setup)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.argv", ["main.py", "--checkpoint-db", str(tmp_path / "checkpoints.sqlite")])

    main.main()

    assert calls.get("invoked") is True
    # The wizard's own confirm screen already shows [session] LLM=... --
    # main() must not print a second, redundant copy after it returns.
    assert "[session] LLM=" not in capsys.readouterr().out


def test_main_banner_explains_enter_and_back_once_upfront(monkeypatch, tmp_path, capsys):
    # The repeated per-field "(뒤로: b)" hint was removed in favor of one
    # general explanation shown once before the wizard starts.
    monkeypatch.setattr(main, "run_interactive_setup", lambda args, thread_id: (DEFAULT_HW_CONFIG, False, None, None, None, None))
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.argv", ["main.py", "--checkpoint-db", str(tmp_path / "checkpoints.sqlite")])

    main.main()

    out = capsys.readouterr().out
    assert "Enter는 대괄호로 표시된 기본값을 그대로 쓰고" in out
    assert "'b'를 입력하면 이전 단계로 돌아갑니다" in out


def test_main_skips_interactive_setup_when_hw_config_is_given(monkeypatch, tmp_path):
    def fail_if_called(args, thread_id):
        raise AssertionError("run_interactive_setup must not be called when --hw-config is given")

    monkeypatch.setattr(main, "run_interactive_setup", fail_if_called)
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

    main.main()  # must not raise (run_interactive_setup would if called)


def test_main_skips_interactive_setup_when_stdin_is_not_a_tty(monkeypatch, tmp_path):
    def fail_if_called(args, thread_id):
        raise AssertionError("run_interactive_setup must not be called when stdin isn't a tty")

    monkeypatch.setattr(main, "run_interactive_setup", fail_if_called)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.argv", ["main.py", "--checkpoint-db", str(tmp_path / "checkpoints.sqlite")])

    main.main()  # must not raise


def test_main_prints_resolved_llm_and_device_before_running(monkeypatch, tmp_path, capsys):
    # Neutralize main()'s own load_dotenv(".env.local", override=True) --
    # otherwise a real .env.local (gitignored, may set AUTOCIM_PLANNER_MODEL)
    # would clobber the env vars this test sets below.
    monkeypatch.setattr(main, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.delenv("AUTOCIM_PLANNER_MODEL", raising=False)
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

    main.main()

    out = capsys.readouterr().out
    assert "[session] LLM=ollama:qwen2.5:7b, QAT device=" in out


def test_main_prints_llm_fallback_reason_when_api_key_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("AUTOCIM_PLANNER_MODEL", "anthropic:claude-sonnet-4-5-20250929")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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

    main.main()

    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" in out
    assert "[session] LLM=ollama:qwen2.5:7b" in out


def test_main_invokes_dashboard_prompt_when_hw_config_given_and_dashboard_out_omitted(monkeypatch, tmp_path):
    # --hw-config given (so run_interactive_setup is skipped entirely) but
    # --dashboard-out omitted with a real tty -- the standalone
    # prompt_for_dashboard_out path (unchanged) must still fire.
    calls = {}

    def fake_prompt(hw_spec_id, thread_id):
        calls["invoked"] = True
        return None

    monkeypatch.setattr(main, "prompt_for_dashboard_out", fake_prompt)
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

    main.main()

    assert calls.get("invoked") is True


def test_main_skips_dashboard_prompt_when_hw_config_and_dashboard_out_both_given(monkeypatch, tmp_path):
    def fail_if_called(hw_spec_id, thread_id):
        raise AssertionError("prompt_for_dashboard_out must not be called when --dashboard-out is given")

    monkeypatch.setattr(main, "prompt_for_dashboard_out", fail_if_called)
    monkeypatch.setattr(main, "run_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--hw-config",
            "examples/hw_configs/default_cim_v1_128x128.json",
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


def test_run_session_choice_3_writes_a_report_even_when_none_was_configured(
    monkeypatch, tmp_path, registered_bad_hw_config
):
    # prompt_for_override returning None simulates picking [3] 지금까지
    # 결과를 리포트에 기록 후 종료 -- run_session must not just silently do
    # nothing because --dashboard-out was never passed.
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(main, "REPORT_DIR", reports_dir)
    monkeypatch.setattr("main.prompt_for_override", lambda payload: None)
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_bad_hw_config, "thread-choice3", checkpointer)

    written = list(reports_dir.glob("report_*.html"))
    assert len(written) == 1


def test_run_session_keyboard_interrupt_mid_stream_stops_gracefully(monkeypatch, tmp_path, registered_hw_config, capsys):
    # A KeyboardInterrupt from inside graph.stream() (most likely real QAT
    # training in @tuner) used to propagate as a raw, uncaught traceback --
    # must now stop cleanly with the checkpoint left resumable.
    monkeypatch.setattr(main, "stream_until_interrupt", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-kbi", checkpointer)  # must not raise

    assert "중단되었습니다" in capsys.readouterr().out


def test_run_session_keyboard_interrupt_during_warmup_stops_without_crashing(
    monkeypatch, tmp_path, registered_hw_config, capsys
):
    monkeypatch.setattr(main, "run_parallel_warmup", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session(
            "resnet18", registered_hw_config, "thread-kbi-warmup", checkpointer, parallel_warmup_workers=2
        )  # must not raise

    assert "중단되었습니다" in capsys.readouterr().out


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

    # good_hw_config converges on the very first *sequential* (post-warm-up)
    # iteration, but candidate_history must contain every parallel warm-up
    # candidate *plus* that one real sequential candidate (the known
    # limitation documented in tools/batch_warmup.py: warm-up doesn't
    # short-circuit early even though one of its own candidates already
    # converged). iteration_count continues from the warm-up count (seeded
    # by run_session) rather than restarting at 1, so it doesn't collide
    # with a warm-up candidate's own iteration number.
    n_warmup = warmup_count(n_stages)
    assert final_state["is_converged"] is True
    assert final_state["iteration_count"] == n_warmup + 1
    assert len(final_state["candidate_history"]) == n_warmup + 1


# --- cross-session learning (store.py, wired through run_session) -----------


def _long_term_stores(tmp_path):
    backend = SqliteLongTermStore(tmp_path / "long_term_store.sqlite")
    return ModelSpecificStore(backend), GlobalHWStore(backend)


def test_run_session_persists_pareto_front_and_calibration_on_convergence(monkeypatch, tmp_path, registered_hw_config):
    lts_path = tmp_path / "long_term_store.sqlite"
    monkeypatch.setattr(main, "DEFAULT_LONG_TERM_STORE_DB", lts_path)
    db_path = str(tmp_path / "checkpoints.sqlite")

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-learn-a", checkpointer)

    model_store, hw_store = _long_term_stores(tmp_path)
    solutions = model_store.get_pareto_solutions(main._pareto_store_key("resnet18", registered_hw_config.hw_spec_id))
    assert solutions and len(solutions["candidates"]) >= 1
    assert hw_store.get_calibration_factors(registered_hw_config.hw_spec_id) == {
        # @precision_verifier's mock Stage-2 re-check (tools/precision_check.py)
        # runs on this run's converged candidate and updates the factor from
        # @profiler's own uncalibrated default (1.0) to the precise/raw ratio
        # -- 1.0 + registered_hw_config's device_noise_sigma (0.05) here.
        registered_hw_config.hw_spec_id: 1.05
    }


def test_run_session_seeds_candidate_history_from_a_previous_sessions_pareto_front(
    monkeypatch, tmp_path, registered_hw_config, capsys
):
    lts_path = tmp_path / "long_term_store.sqlite"
    monkeypatch.setattr(main, "DEFAULT_LONG_TERM_STORE_DB", lts_path)
    model_store, _ = _long_term_stores(tmp_path)
    stored_candidate = {
        "accuracy": 0.8,
        "energy_pj": 100.0,
        "noc_latency_ms": 1.0,
        "avg_weight_bits": 6,
        "avg_column_pruning_ratio": 0.1,
        "layer_configs": [],
        "is_converged": True,
        "pareto_rank": 1,
        "iteration": 1,
    }
    model_store.put_pareto_solutions(
        main._pareto_store_key("resnet18", registered_hw_config.hw_spec_id), {"candidates": [stored_candidate]}
    )

    from graph import build_graph

    db_path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "thread-learn-b"}}
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-learn-b", checkpointer)
        final_state = build_graph(checkpointer=checkpointer).get_state(config).values

    assert stored_candidate in final_state["candidate_history"]
    assert "이전 실행에서 찾은 Pareto 후보 1개를 불러왔습니다" in capsys.readouterr().out


def test_run_session_seeds_calibration_factor_from_a_previous_sessions_learning(
    monkeypatch, tmp_path, registered_hw_config, capsys
):
    lts_path = tmp_path / "long_term_store.sqlite"
    monkeypatch.setattr(main, "DEFAULT_LONG_TERM_STORE_DB", lts_path)
    _, hw_store = _long_term_stores(tmp_path)
    hw_store.put_calibration_factors(registered_hw_config.hw_spec_id, {registered_hw_config.hw_spec_id: 0.42})

    from graph import build_graph

    db_path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "thread-learn-c"}}
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-learn-c", checkpointer)
        build_graph(checkpointer=checkpointer).get_state(config).values

    assert "이전 실행에서 학습된 보정계수를 불러왔습니다" in capsys.readouterr().out


def test_persist_long_term_learnings_never_regresses_the_stored_pareto_front(tmp_path):
    model_store, hw_store = _long_term_stores(tmp_path)
    better = {"accuracy": 0.9, "energy_pj": 50.0, "noc_latency_ms": 0.5}
    model_store.put_pareto_solutions(main._pareto_store_key("resnet18", "some_hw"), {"candidates": [better]})

    worse = {"accuracy": 0.5, "energy_pj": 200.0, "noc_latency_ms": 2.0}  # dominated by `better` on every objective
    main._persist_long_term_learnings(
        model_store, hw_store, "resnet18", "some_hw", {"candidate_history": [worse], "calibration_factors": {}}
    )

    solutions = model_store.get_pareto_solutions(main._pareto_store_key("resnet18", "some_hw"))
    assert solutions["candidates"] == [better]  # `worse` never made it into the stored front


def test_persist_long_term_learnings_keeps_pareto_fronts_isolated_per_hw_spec_id(tmp_path):
    # The bug this guards against: a Pareto front measured on one hw_spec_id
    # (its energy_pj/noc_latency_ms come from that specific HWConfig's
    # physics) must never be read back as this model's front on a
    # *different* hw_spec_id -- doing so would mix incomparable objectives
    # into compute_pareto_rank and desync the new hw's own LHS warm-up index.
    model_store, hw_store = _long_term_stores(tmp_path)
    candidate = {"accuracy": 0.9, "energy_pj": 50.0, "noc_latency_ms": 0.5}
    main._persist_long_term_learnings(
        model_store, hw_store, "resnet18", "hw_a", {"candidate_history": [candidate], "calibration_factors": {}}
    )

    assert model_store.get_pareto_solutions(main._pareto_store_key("resnet18", "hw_b")) is None
    assert model_store.get_pareto_solutions(main._pareto_store_key("resnet18", "hw_a")) == {"candidates": [candidate]}


def test_run_session_without_the_flag_still_runs_warmup_batch_by_default(tmp_path, registered_hw_config):
    """Omitting --parallel-warmup-workers only omits the worker-count
    override -- the warm-up batch itself always runs (main.py's run_session
    docstring/comment above the warm-up block), so iteration_count still
    continues from warmup_count(n_stages), not from 1."""
    from nodes.planner import real_stage_names, warmup_count

    n_warmup = warmup_count(len(real_stage_names("resnet18")))
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-sequential", checkpointer)
        sessions = list_sessions(checkpointer)

    assert sessions[0]["is_converged"] is True
    # good_hw_config converges on the first post-warm-up iteration
    assert sessions[0]["iteration_count"] == n_warmup + 1


# --- approximate calibration (default) / --exact-calibration-only (opt-out) ---
#
# registered_hw_config (good_hw_config: 128x128, adc_bits=8) has no exact
# match in tools/calibration.py's KNOWN_REFERENCES -- the one 128x128
# reference there is adc_bits=7, not 8 -- so it's a real unmatched HWConfig,
# not something that happens to already be exactly calibrated.


def test_build_initial_state_defaults_to_approximate_calibration_for_unmatched_hw(registered_hw_config):
    state = main.build_initial_state("resnet18", registered_hw_config)

    assert registered_hw_config.hw_spec_id in state["calibration_factors"]
    assert state["calibration_factors"][registered_hw_config.hw_spec_id] > 0
    assert state["calibration_provenance"][registered_hw_config.hw_spec_id]["approximate"] is True


def test_build_initial_state_with_exact_calibration_only_leaves_unmatched_hw_uncalibrated(registered_hw_config):
    state = main.build_initial_state("resnet18", registered_hw_config, allow_approximate_calibration=False)
    assert state["calibration_factors"] == {}
    assert state["calibration_provenance"] == {}


def test_run_session_prints_approximate_status_by_default(capsys, tmp_path, registered_hw_config):
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, "thread-calib-default", checkpointer)

    assert "[calibration] approximate" in capsys.readouterr().out


def test_run_session_with_exact_calibration_only_prints_uncalibrated_status(capsys, tmp_path, registered_hw_config):
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session(
            "resnet18",
            registered_hw_config,
            "thread-calib-strict",
            checkpointer,
            allow_approximate_calibration=False,
        )

    assert "[calibration] uncalibrated" in capsys.readouterr().out


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
