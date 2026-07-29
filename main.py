"""AutoCIM-Agent CLI entrypoint.

Boots one AutoCIMState session, injects ExecutionContext/HWConfig at the
config-loading boundary (CLAUDE.md 5.C -- node/tool logic never hardcodes
these), streams node execution via `graph.stream()`, and on a dynamic
`interrupt()` (checklist item 1) prompts the researcher for an override and
resumes the same thread with `Command(resume={"new_bounds": ...})`.

Checkpointing is persistent (SqliteSaver, file-backed under `.cache/`), not
the in-process-only `MemorySaver` `graph.build_graph()` defaults to -- a
session paused at `hitl_human_approval` (or killed mid-run) survives the
process exiting. Re-invoking this CLI with the same `--thread-id` picks up
exactly where it left off: `run_session` checks `graph.get_state(config)`
before deciding whether to start a fresh run or resume a persisted one.

`main()` loads `.env` then `.env.local` (the latter overriding, if present
-- same convention as Next.js/CRA) before touching `argparse`, so
`AUTOCIM_PLANNER_MODEL`/API keys don't need re-exporting in every new
shell. Deliberately *not* done at module import time: `tests/test_main_cli.py`
imports this module directly, and a stray `.env.local` (real API keys,
gitignored) silently landing in `os.environ` during a test run would be a
surprising, hard-to-notice side effect of just importing a module -- this
way it only ever fires for an actual `python main.py` invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, Interrupt
from pydantic import ValidationError

from graph import build_graph
from nodes.planner import search_bounds
from middleware import register_hw_config
from schemas.config import HWConfig, NoCTopology
from state import AutoCIMState, merge_dicts
from store import GlobalHWStore, ModelSpecificStore, SqliteLongTermStore
from llm import resolve_planner_model
from tools.batch_warmup import run_parallel_warmup
from tools.calibration import (
    bootstrap_approximate_calibration_factors,
    bootstrap_approximate_calibration_provenance,
    bootstrap_calibration_factors,
    bootstrap_calibration_provenance,
    find_nearest_reference,
)
from tools.dashboard import render_dashboard_html
from tools.qat import _resolve_device
from tools.search import compute_pareto_rank

# Sample hardware spec, used when --hw-config isn't given. A real deployment
# points --hw-config at a JSON file describing the target chip instead of
# relying on this default.
DEFAULT_HW_CONFIG = HWConfig(
    hw_spec_id="cim_v1_128x128",
    crossbar_rows=128,
    crossbar_cols=128,
    num_tiles=16,
    adc_bits=8,
    dac_bits=4,
    noc_topology=NoCTopology.MESH,
    noc_link_bandwidth_gbps=10.0,
    wire_resistance_ohm_per_um=0.05,
    device_noise_sigma=0.05,
    sram_buffer_kb=64.0,
)

DEFAULT_CHECKPOINT_DB = Path(__file__).resolve().parent / ".cache" / "checkpoints.sqlite"
# store.py's cross-session learning (ModelSpecificStore/GlobalHWStore) --
# unlike checkpoints (per-thread, one run), this is a single shared file
# across every run for every model_id/hw_spec_id, so later runs benefit
# from what earlier ones already found. tests/conftest.py's autouse
# isolated_long_term_store fixture redirects this to a tmp_path for every
# test -- never the real file.
DEFAULT_LONG_TERM_STORE_DB = Path(__file__).resolve().parent / ".cache" / "long_term_store.sqlite"
EXAMPLE_HW_CONFIGS_DIR = Path(__file__).resolve().parent / "examples" / "hw_configs"
# Every interactively-generated report (both prompt_for_dashboard_out's and
# run_interactive_setup's) always lands here under a fixed, auto-generated
# filename -- neither the directory nor the filename is ever something a
# researcher types in, so there's no custom path to mistype/point at a
# missing parent directory. An explicit --dashboard-out PATH is unaffected
# (it's written exactly where given, as before).
REPORT_DIR = Path(__file__).resolve().parent / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AutoCIM-Agent optimization session.")
    parser.add_argument("--model-id", default="resnet18", help="Target model identifier.")
    parser.add_argument(
        "--hw-config",
        default=None,
        metavar="PATH",
        help="Path to a JSON file with HWConfig fields; defaults to a built-in sample spec.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help=(
            "LangGraph checkpoint thread id. Reusing the id of a session that "
            "was previously paused (HITL) or interrupted (Ctrl-C) resumes it "
            "from persisted state instead of starting over; defaults to a new "
            "random id (always a fresh session)."
        ),
    )
    parser.add_argument(
        "--checkpoint-db",
        default=None,
        metavar="PATH",
        help=(
            f"SQLite file for persistent session checkpoints; defaults to "
            f"{str(DEFAULT_CHECKPOINT_DB)!r}. Pass ':memory:' to opt out of "
            "persistence (a session then cannot survive the process exiting, "
            "matching the old MemorySaver-only behavior)."
        ),
    )
    parser.add_argument(
        "--dashboard-out",
        default=None,
        metavar="PATH",
        help=(
            "If set, write an HTML dashboard (tools/dashboard.py: per-iteration "
            "candidates, why each was tried, Pareto front movement, LLM cost) "
            "to this path whenever the session pauses or finishes."
        ),
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help=(
            "List every thread_id found in --checkpoint-db (status, iteration_count, "
            "model_id/hw_spec_id) and exit -- a researcher otherwise has to remember "
            "thread_ids manually to resume/inspect a past session."
        ),
    )
    parser.add_argument(
        "--parallel-warmup-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Opt-in: evaluate the LHS warm-up candidates (tools/batch_warmup.py) "
            "concurrently across N threads before the sequential graph loop starts, "
            "instead of one real QAT trial per graph iteration. Only applies to a "
            "brand-new session (not a resumed/paused one). Omit to keep today's "
            "fully-sequential behavior."
        ),
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Optional convergence gate (nodes/evaluator.py): a candidate must reach "
            "at least this accuracy (in addition to @verifier's own IR-drop/noise "
            "check) to count as converged, or it's treated as a non-convergent "
            "iteration (same MAX_RETRY_LIMIT -> HITL flow as any other failure). "
            "Omit to leave accuracy ungated, as before this option existed."
        ),
    )
    parser.add_argument(
        "--target-energy-pj",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Optional convergence gate: a candidate's energy_pj must be at or below this value. Omit to leave ungated.",
    )
    parser.add_argument(
        "--target-latency-ms",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Optional convergence gate: a candidate's noc_latency_ms must be at or below this value. Omit to leave ungated.",
    )
    parser.add_argument(
        "--auto-hitl",
        action="store_true",
        help=(
            "Opt-in: resolve HITL interrupts (nodes/hitl.py) without blocking on "
            "researcher input, via a rule-based override (main.py's "
            "auto_resolve_override). Only resolves target-accuracy/energy/latency "
            "misses that a weight_bits/pruning_ratio bounds adjustment could "
            "plausibly fix; anything else (physically-infeasible hw_spec_id per "
            "@verifier, a schema/validation error, or conflicting accuracy-vs-"
            "energy/latency misses in the same iteration) stops the run instead of "
            "guessing, since no bounds change can resolve those and blind retries "
            "would just burn real QAT trials. Omit to keep the default interactive "
            "input() prompt."
        ),
    )
    parser.add_argument(
        "--auto-hitl-max-rounds",
        type=int,
        default=2,
        metavar="N",
        help=(
            "With --auto-hitl: stop the run (instead of resolving further) after N "
            "auto-resolved HITL rounds in this invocation, so a search that keeps "
            "missing target even after bounds adjustments doesn't loop unattended "
            "forever. Ignored without --auto-hitl."
        ),
    )
    parser.add_argument(
        "--allow-approximate-calibration",
        action="store_true",
        help=(
            "Opt-in: when --hw-config doesn't exactly match a known "
            "calibration reference (tools/calibration.py's KNOWN_REFERENCES), "
            "fall back to the nearest reference's scaling-law-transferred "
            "correction factor instead of leaving @profiler uncalibrated "
            "(factor 1.0). The result is tagged 'approximate' in "
            "calibration_provenance, with the matched reference and distance "
            "printed at session start -- never presented as an exact "
            "citation. Omit to keep the default: an unmatched hw_config "
            "stays honestly uncalibrated."
        ),
    )
    return parser.parse_args()


class HWConfigError(RuntimeError):
    """Raised by `load_hw_config` on a missing/malformed/schema-invalid
    `--hw-config` file. Caught in `main()` to print a clean, actionable
    message instead of a raw traceback -- a researcher pointing this CLI
    at a hand-edited JSON file (see `examples/hw_configs/`) shouldn't need
    to read a Python stack trace to find a typo."""


def load_hw_config(path: Optional[str]) -> HWConfig:
    if path is None:
        return DEFAULT_HW_CONFIG
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise HWConfigError(f"--hw-config file not found: {path!r}") from None
    except json.JSONDecodeError as exc:
        raise HWConfigError(f"--hw-config file {path!r} is not valid JSON: {exc}") from None
    try:
        return HWConfig(**raw)
    except ValidationError as exc:
        raise HWConfigError(
            f"--hw-config file {path!r} doesn't match HWConfig's schema "
            f"(schemas/config.py; see examples/hw_configs/ for a working example):\n{exc}"
        ) from None


# =============================================================================
# Interactive --hw-config picker (main() only, when --hw-config is omitted
# and stdin is a real terminal -- a scripted/CI invocation, or one that
# already passes --hw-config, never hits any of this).
# =============================================================================


_BACK = object()  # sentinel: researcher typed a back-command at a field prompt
_BACK_COMMANDS = {"b", "back", "뒤로"}


def _read_step(label: str, default: str) -> Any:
    """One text-field prompt supporting the back-command -- returns `_BACK`
    if the researcher typed b/back/뒤로 instead of a value, so the caller
    can step to the previous field instead of accepting this one."""
    raw = input(f"{label} [{default}]: ").strip()
    if raw.lower() in _BACK_COMMANDS:
        return _BACK
    return raw if raw else default


def _parse_int_step(label: str, default: int) -> Any:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw.lower() in _BACK_COMMANDS:
            return _BACK
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  정수를 입력해주세요.")


def _parse_float_step(label: str, default: Optional[float]) -> Any:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw.lower() in _BACK_COMMANDS:
            return _BACK
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  숫자를 입력해주세요.")


# One (field_name, read_fn) pair per HWConfig field the wizard below
# (run_interactive_setup) asks for, in order -- read_fn takes that field's
# current default/prior answer and returns either the parsed value or `_BACK`.
_CUSTOM_HW_CONFIG_STEPS: List[Tuple[str, Callable[[Any], Any]]] = [
    ("hw_spec_id", lambda default: _read_step("hw_spec_id", default)),
    ("crossbar_rows", lambda default: _parse_int_step("crossbar_rows", default)),
    ("crossbar_cols", lambda default: _parse_int_step("crossbar_cols", default)),
    ("num_tiles", lambda default: _parse_int_step("num_tiles", default)),
    ("adc_bits", lambda default: _parse_int_step("adc_bits", default)),
    ("dac_bits", lambda default: _parse_int_step("dac_bits", default)),
    ("noc_topology", lambda default: _read_step("noc_topology (mesh/torus/ring/crossbar_bus)", default)),
    ("noc_link_bandwidth_gbps", lambda default: _parse_float_step("noc_link_bandwidth_gbps", default)),
    ("wire_resistance_ohm_per_um", lambda default: _parse_float_step("wire_resistance_ohm_per_um", default)),
    ("device_noise_sigma", lambda default: _parse_float_step("device_noise_sigma", default)),
    ("sram_buffer_kb", lambda default: _parse_float_step("sram_buffer_kb", default)),
]


class _WizardStep:
    """One step in the unified hw-config/target/dashboard wizard below.
    `prompt_fn(current_value, answers) -> new_value | _BACK` reads this
    step's value, using whatever's already stored for it (`current_value`)
    as the shown default -- re-visiting a step (via 'b' or a confirm-screen
    jump) never discards it, only Enter-through re-confirms it.
    `applicable(answers)` decides whether this step is even part of the
    chain given earlier answers (e.g. the 11 custom HWConfig fields only
    apply when hw_mode == "1"), recomputed on every visit so an earlier
    edit (e.g. switching hw_mode, or a crossbar edit that now exactly
    matches a calibration reference) can add/remove later steps."""

    __slots__ = ("key", "label", "prompt_fn", "applicable")

    def __init__(
        self,
        key: str,
        label: str,
        prompt_fn: Callable[[Any, Dict[str, Any]], Any],
        applicable: Callable[[Dict[str, Any]], bool] = lambda answers: True,
    ):
        self.key = key
        self.label = label
        self.prompt_fn = prompt_fn
        self.applicable = applicable


def _prompt_hw_mode(current: Any, answers: Dict[str, Any]) -> Any:
    print("--hw-config가 지정되지 않았습니다. 어떻게 진행할까요?\n")
    print("  [1] 직접 값 입력 (Custom HWConfig)")
    print("  [2] 저장된 예시 파일 사용")
    print("  [3] 기본값 사용")
    print("  [4] 종료")
    raw_choice = input("\n 선택: ").strip()
    choice = raw_choice or "4"
    # This is the very first step of the whole chain -- there's nothing
    # earlier to step 'b' back into, so treat it the same as choice 4
    # (matches this project's existing "nowhere to go back to -> exit
    # rather than silently re-showing the same prompt" convention).
    if choice == "4" or choice.lower() in _BACK_COMMANDS:
        print("\n입력이 없습니다.\n실행을 종료합니다." if not raw_choice else "\n종료합니다.")
        raise SystemExit(0)
    if choice not in {"1", "2", "3"}:
        choice = "3"
    return choice


_CUSTOM_HW_CONFIG_FIELD_NAMES = [name for name, _ in _CUSTOM_HW_CONFIG_STEPS]


def _make_custom_field_prompt(name: str, read_fn: Callable[[Any], Any], initial_defaults: Dict[str, Any]):
    def _prompt(current: Any, answers: Dict[str, Any]) -> Any:
        default = current if current is not None else initial_defaults[name]
        return read_fn(default)

    return _prompt


def _try_build_custom_hw_config(
    answers: Dict[str, Any], initial_defaults: Dict[str, Any]
) -> Tuple[Optional[HWConfig], List[str]]:
    """Validates the 11 custom fields once all are answered (called right
    after the last one). On failure, resets only the offending field(s) in
    `answers` to their *original* default (never the just-typed invalid
    value) and returns their names so the caller can jump back to the
    first custom field -- re-confirming already-good fields is then a
    quick Enter-through, not retyping everything from scratch."""
    values = {name: answers[name] for name in _CUSTOM_HW_CONFIG_FIELD_NAMES}
    try:
        return HWConfig(**values), []
    except ValidationError as exc:
        bad_fields = sorted({str(error["loc"][0]) for error in exc.errors() if error.get("loc")})
        print(
            f"입력값이 HWConfig 스키마에 맞지 않습니다 ({', '.join(bad_fields) or '알 수 없는 필드'}):\n{exc}\n"
            "해당 필드는 원래 기본값으로 되돌리고 나머지는 방금 입력한 값을 유지합니다 -- 다시 확인해주세요."
        )
        for field in bad_fields:
            if field in answers:
                answers[field] = initial_defaults[field]
        return None, bad_fields


def _prompt_example_choice(current: Any, answers: Dict[str, Any]) -> Any:
    files = sorted(EXAMPLE_HW_CONFIGS_DIR.glob("*.json"))
    print("\n=== 저장된 예시 --hw-config 파일 ===")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f.name}")
    label = f"번호 선택 [{current}]: " if current is not None else "번호 선택: "
    while True:
        raw = input(label).strip()
        if raw.lower() in _BACK_COMMANDS:
            return _BACK
        if not raw and current is not None:
            return current
        try:
            index = int(raw)
            if not (1 <= index <= len(files)):
                raise ValueError
            return index
        except ValueError:
            print("  잘못된 선택입니다.")


def _assemble_hw_config(answers: Dict[str, Any]) -> Optional[HWConfig]:
    """Builds the HWConfig the current answers imply, or None if not enough
    is known yet (hw_mode not chosen, an example file not picked yet, or
    the custom fields not all in) -- used for the confirm screen's display
    and to decide whether the approximate-calibration step applies."""
    mode = answers.get("hw_mode")
    if mode == "1":
        if any(answers.get(name) is None for name in _CUSTOM_HW_CONFIG_FIELD_NAMES):
            return None
        try:
            return HWConfig(**{name: answers[name] for name in _CUSTOM_HW_CONFIG_FIELD_NAMES})
        except ValidationError:
            return None
    if mode == "2":
        choice = answers.get("example_choice")
        if choice is None:
            return None
        files = sorted(EXAMPLE_HW_CONFIGS_DIR.glob("*.json"))
        try:
            return load_hw_config(str(files[choice - 1]))
        except (HWConfigError, IndexError):
            return None
    if mode == "3":
        return DEFAULT_HW_CONFIG
    return None


def _needs_approx_calibration_choice(answers: Dict[str, Any]) -> bool:
    hw = _assemble_hw_config(answers)
    if hw is None:
        return False
    return not bootstrap_calibration_factors(hw)


def _prompt_approx_calibration(current: Any, answers: Dict[str, Any]) -> Any:
    hw = _assemble_hw_config(answers)
    nearest = find_nearest_reference(hw) if hw is not None else None
    if nearest is None:
        print("[calibration] uncalibrated (no known reference)")
        return "n"
    reference, distance = nearest
    print(
        f"[calibration] 정확히 일치하는 레퍼런스가 없습니다. 가장 가까운 레퍼런스: "
        f"{reference.crossbar_rows}x{reference.crossbar_cols}/{reference.adc_bits}-bit ADC "
        f"(distance={distance:.2f})"
    )
    default = current if current in ("y", "n") else "n"
    raw = input("  이 레퍼런스로 근사 보정해서 진행할까요? [y/n]: ").strip().lower()
    if raw in _BACK_COMMANDS:
        return _BACK
    if not raw:
        return default
    return "y" if raw == "y" else "n"


def _prompt_target(label: str) -> Callable[[Any, Dict[str, Any]], Any]:
    def _prompt(current: Any, answers: Dict[str, Any]) -> Any:
        suffix = f" [{current}]" if current is not None else ""
        while True:
            raw = input(f"{label}{suffix}: ").strip()
            if raw.lower() in _BACK_COMMANDS:
                return _BACK
            if not raw:
                return current
            if raw in ("-", "빈칸"):
                return None
            try:
                return float(raw)
            except ValueError:
                print("  숫자, '-'(미설정), 또는 빈칸(현재 값 유지)으로 입력해주세요.")

    return _prompt


def _prompt_dashboard_yn(thread_id: str, report_timestamp: datetime) -> Callable[[Any, Dict[str, Any]], Any]:
    """Y/N only -- the path/filename is never something a researcher types
    (see REPORT_DIR's comment). Choosing "y" derives the fixed report path
    from the *current* hw_config right here and stashes it into
    answers['dashboard_path'] (no separate step/prompt for it) so the
    confirm screen can display it next to this line; choosing "n" clears
    any stale path from a previous "y" answer."""

    def _prompt(current: Any, answers: Dict[str, Any]) -> Any:
        default = current if current in ("y", "n") else "n"
        raw = input("\n리포트(HTML 대시보드)를 생성할까요? [y/n]: ").strip().lower()
        if raw in _BACK_COMMANDS:
            return _BACK
        result = default if not raw else ("y" if raw == "y" else "n")
        if result == "y":
            hw = _assemble_hw_config(answers) or DEFAULT_HW_CONFIG
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            answers["dashboard_path"] = str(
                REPORT_DIR / f"report_{hw.hw_spec_id}_{thread_id[:8]}_{report_timestamp:%Y%m%d_%H%M%S}.html"
            )
        else:
            answers.pop("dashboard_path", None)
        return result

    return _prompt


def _prompt_save_custom_hw_config(current: Any, answers: Dict[str, Any]) -> Any:
    default = current if current in ("y", "n") else "n"
    raw = input(
        "\n이 custom hw-config를 examples/hw_configs/에 저장할까요?"
        "\n(다음에 --hw-config로 재사용하거나 --thread-id로 재개할 때 필요합니다) [y/n]: "
    ).strip().lower()
    if raw in _BACK_COMMANDS:
        return _BACK
    if not raw:
        return default
    return "y" if raw == "y" else "n"


def _build_wizard_steps(
    args: argparse.Namespace, thread_id: str, initial_custom_defaults: Dict[str, Any], report_timestamp: datetime
) -> List[_WizardStep]:
    is_custom = lambda answers: answers.get("hw_mode") == "1"
    is_example = lambda answers: answers.get("hw_mode") == "2"

    steps = [_WizardStep("hw_mode", "hw-config 선택 방식", _prompt_hw_mode)]
    for name, read_fn in _CUSTOM_HW_CONFIG_STEPS:
        steps.append(_WizardStep(name, name, _make_custom_field_prompt(name, read_fn, initial_custom_defaults), is_custom))
    steps.append(_WizardStep("example_choice", "선택한 예시 파일", _prompt_example_choice, is_example))
    steps.append(
        _WizardStep("approx_calibration_yn", "근사 보정 적용", _prompt_approx_calibration, _needs_approx_calibration_choice)
    )
    if args.target_accuracy is None:
        steps.append(_WizardStep("target_accuracy", "target accuracy", _prompt_target("target accuracy (예: 0.7)")))
    if args.target_energy_pj is None:
        steps.append(_WizardStep("target_energy_pj", "target energy_pj", _prompt_target("target energy_pj (예: 2000)")))
    if args.target_latency_ms is None:
        steps.append(_WizardStep("target_latency_ms", "target latency_ms", _prompt_target("target latency_ms (예: 5.0)")))
    if args.dashboard_out is None:
        steps.append(_WizardStep("dashboard_yn", "리포트 생성", _prompt_dashboard_yn(thread_id, report_timestamp)))
    # Only meaningful for a hand-typed HWConfig -- an example-file/default
    # choice already exists as a file on disk (or isn't one at all).
    steps.append(_WizardStep("save_custom_hw_config_yn", "custom 설정 저장", _prompt_save_custom_hw_config, is_custom))
    return steps


def _index_of(steps: List[_WizardStep], key: str) -> int:
    return next(i for i, s in enumerate(steps) if s.key == key)


def _next_applicable(steps: List[_WizardStep], answers: Dict[str, Any], index: int) -> int:
    while 0 <= index < len(steps) and not steps[index].applicable(answers):
        index += 1
    return index


def _prev_applicable(steps: List[_WizardStep], answers: Dict[str, Any], index: int) -> int:
    while index >= 0 and not steps[index].applicable(answers):
        index -= 1
    return index


_WIZARD_FIELD_LABELS = {"hw_mode": "hw-config 선택 방식"}


def _format_wizard_answer(key: str, value: Any, answers: Dict[str, Any]) -> str:
    if key == "hw_mode":
        return {"1": "custom", "2": "예시 파일", "3": "기본값"}.get(value, str(value))
    if key == "example_choice":
        files = sorted(EXAMPLE_HW_CONFIGS_DIR.glob("*.json"))
        try:
            return files[value - 1].name
        except (IndexError, TypeError):
            return str(value)
    if key == "dashboard_yn":
        if value == "y":
            return f"예 ({answers.get('dashboard_path', '?')})"
        return "아니오"
    if key in ("approx_calibration_yn", "save_custom_hw_config_yn"):
        return "예" if value == "y" else "아니오"
    if value is None:
        return "(미설정)"
    return str(value)


def _show_confirm_screen(
    steps: List[_WizardStep],
    answers: Dict[str, Any],
    just_refreshed: bool = False,
    just_edited_display_no: Optional[int] = None,
) -> Any:
    """Prints every applicable, already-answered step, then the resolved
    LLM/QAT-device (recomputed on every call, so 'r' below actually
    refreshes it) -- list first so the numbered items a researcher might
    want to edit are together and stable, session/status info and the
    [y/n] question grouped right above the prompt itself. `just_refreshed`
    (set by the caller only on the render right after 'r' was chosen)
    prints the "다시 읽었습니다" confirmation there; `just_edited_display_no`
    (set only on the render right after a jump-edit completed) prints
    "[N] 수정되었습니다." the same way -- neither dangles after the
    *previous* render's already-answered prompt, which is where either
    would land if printed directly at the point the action happened
    instead of being carried into the next re-show.

    Returns "proceed", "back" (one step, into the last applicable step),
    "refresh" (re-read .env/.env.local and re-show with updated values --
    for "I forgot to set an API key", without restarting the whole
    wizard), or (steps_index, display_no) -- typing a listed item's number
    jumps straight there for a quick single-field edit (vs. walking back
    one step at a time); the caller re-asks only that one field and comes
    straight back here, it does not replay every field after it."""
    applicable_steps = [(i, s) for i, s in enumerate(steps) if s.applicable(answers)]
    print("\n=== 입력값 확인 ===")
    for display_no, (_, step) in enumerate(applicable_steps, 1):
        label = _WIZARD_FIELD_LABELS.get(step.key, step.label)
        print(f"  [{display_no}] {label}: {_format_wizard_answer(step.key, answers.get(step.key), answers)}")
    print()
    planner_model, planner_fallback_reason = resolve_planner_model()
    if planner_fallback_reason:
        print(f"[llm] {planner_fallback_reason}")
    print(f"[session] LLM={planner_model}, QAT device={_resolve_device()}")
    if just_refreshed:
        print("[session] .env/.env.local을 다시 읽었습니다.")
    if just_edited_display_no is not None:
        print(f"[{just_edited_display_no}] 수정되었습니다.")
    while True:
        raw = input("이대로 진행할까요? [y/n] (b=이전 단계, 숫자=해당 항목 수정, r=세션 정보 새로고침): ").strip().lower()
        if not raw or raw == "y":
            return "proceed"
        if raw == "n" or raw in _BACK_COMMANDS:
            return "back"
        if raw == "r":
            load_dotenv()
            load_dotenv(".env.local", override=True)
            return "refresh"
        try:
            display_no = int(raw)
            if not (1 <= display_no <= len(applicable_steps)):
                raise ValueError
            return applicable_steps[display_no - 1][0], display_no
        except ValueError:
            print("  잘못된 입력입니다.")


def _save_custom_hw_config(hw_config: HWConfig) -> None:
    """Writes hw_config to examples/hw_configs/<hw_spec_id>.json so it can
    be reused later via --hw-config -- including to resume a --thread-id
    session, since middleware._hw_config_registry only ever has whatever
    was actually re-registered *this* process, and this file is the only
    way an interactively-typed hw_spec_id survives past this process
    exiting. Refuses to overwrite an existing file (e.g. a real curated
    calibration example already at that name) -- warns and skips instead
    of silently clobbering it."""
    path = EXAMPLE_HW_CONFIGS_DIR / f"{hw_config.hw_spec_id}.json"
    if path.exists():
        print(f"[hw-config] {path}가 이미 존재해서 덮어쓰지 않았습니다 -- hw_spec_id를 바꿔서 다시 시도해주세요.")
        return
    path.write_text(json.dumps(hw_config.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[hw-config] 저장됨: {path}")


def run_interactive_setup(
    args: argparse.Namespace, thread_id: str
) -> Optional[Tuple[HWConfig, bool, Optional[float], Optional[float], Optional[float], Optional[str]]]:
    """Unified hw-config + target + dashboard wizard for main()'s
    `args.hw_config is None and sys.stdin.isatty()` branch -- one shared
    step chain and answers dict spans hw-config selection (custom fields,
    example-file pick, or the built-in default), the approximate-
    calibration choice, target_accuracy/energy_pj/latency_ms, and the
    dashboard Y/N + path, ending in a confirm screen. 'b' at any prompt (or
    'n'/'b' at the confirm screen) steps back one field at a time and can
    reach any earlier step without losing anything already entered --
    revisiting a step always shows its current answer as the default,
    so walking back and forward again is a quick Enter-through, not
    retyping. Typing a listed item's number at the confirm screen instead
    re-asks only that one field and returns straight back to confirm,
    without replaying everything after it. EOFError (e.g. piped stdin ran
    out) proceeds with whatever's been entered so far; KeyboardInterrupt
    (an explicit Ctrl-C) cancels the whole setup instead -- silently
    starting a real session after a researcher hit Ctrl-C to stop would be
    the wrong failure mode.
    """
    initial_custom_defaults: Dict[str, Any] = {
        "hw_spec_id": f"custom_{uuid.uuid4().hex[:8]}",
        "crossbar_rows": DEFAULT_HW_CONFIG.crossbar_rows,
        "crossbar_cols": DEFAULT_HW_CONFIG.crossbar_cols,
        "num_tiles": DEFAULT_HW_CONFIG.num_tiles,
        "adc_bits": DEFAULT_HW_CONFIG.adc_bits,
        "dac_bits": DEFAULT_HW_CONFIG.dac_bits,
        "noc_topology": DEFAULT_HW_CONFIG.noc_topology.value,
        "noc_link_bandwidth_gbps": DEFAULT_HW_CONFIG.noc_link_bandwidth_gbps,
        "wire_resistance_ohm_per_um": DEFAULT_HW_CONFIG.wire_resistance_ohm_per_um,
        "device_noise_sigma": DEFAULT_HW_CONFIG.device_noise_sigma,
        "sram_buffer_kb": DEFAULT_HW_CONFIG.sram_buffer_kb,
    }
    steps = _build_wizard_steps(args, thread_id, initial_custom_defaults, datetime.now())
    first_custom_key = _CUSTOM_HW_CONFIG_FIELD_NAMES[0]
    last_custom_key = _CUSTOM_HW_CONFIG_FIELD_NAMES[-1]

    answers: Dict[str, Any] = {}
    index = 0
    just_refreshed = False
    just_edited_display_no: Optional[int] = None
    try:
        while True:
            index = _next_applicable(steps, answers, index)
            if index >= len(steps):
                action = _show_confirm_screen(steps, answers, just_refreshed, just_edited_display_no)
                just_refreshed = False
                just_edited_display_no = None
                if action == "proceed":
                    break
                if action == "back":
                    index = _prev_applicable(steps, answers, len(steps) - 1)
                    continue
                if action == "refresh":
                    # index is already len(steps) -- loop straight back to
                    # the top and re-render confirm with the reloaded env;
                    # just_refreshed makes that one render also print the
                    # "다시 읽었습니다" confirmation, right after the
                    # session info instead of dangling after this prompt.
                    just_refreshed = True
                    continue
                # Jump-edit: re-ask only this one field, then go straight
                # back to confirm -- unlike the sequential 'b' walk, editing
                # one already-reviewed field shouldn't force re-confirming
                # every field after it too (everything else in `answers`
                # stays exactly as it was).
                jump_index, jump_display_no = action
                jump_step = steps[jump_index]
                value = jump_step.prompt_fn(answers.get(jump_step.key), answers)
                if value is not _BACK:
                    answers[jump_step.key] = value
                    if jump_step.key in _CUSTOM_HW_CONFIG_FIELD_NAMES:
                        _, bad_fields = _try_build_custom_hw_config(answers, initial_custom_defaults)
                        if bad_fields:
                            index = _index_of(steps, first_custom_key)
                            continue
                    # Only a clean edit (not cancelled via 'b', not a
                    # validation failure that redirected elsewhere) counts
                    # as "modified" for the next render's confirmation.
                    just_edited_display_no = jump_display_no
                index = len(steps)  # back to confirm (also covers a 'b' cancel)
                continue
            if index < 0:
                # Backed out past the very first step -- _prompt_hw_mode
                # itself exits before this can happen today, but this stays
                # as a safety net rather than silently looping forever.
                return None

            step = steps[index]
            value = step.prompt_fn(answers.get(step.key), answers)
            if value is _BACK:
                index = _prev_applicable(steps, answers, index - 1)
                continue
            answers[step.key] = value

            if step.key == last_custom_key:
                _, bad_fields = _try_build_custom_hw_config(answers, initial_custom_defaults)
                if bad_fields:
                    index = _index_of(steps, first_custom_key)
                    continue

            index += 1
    except EOFError:
        print("\n입력을 받지 못했습니다. 지금까지 입력한 값으로 진행합니다.")
    except KeyboardInterrupt:
        print("\n\n중단되었습니다.")
        return None

    hw_config = _assemble_hw_config(answers) or DEFAULT_HW_CONFIG
    if answers.get("hw_mode") == "1" and answers.get("save_custom_hw_config_yn") == "y":
        _save_custom_hw_config(hw_config)
    allow_approximate = answers.get("approx_calibration_yn") == "y"
    target_accuracy = answers.get("target_accuracy", args.target_accuracy)
    target_energy_pj = answers.get("target_energy_pj", args.target_energy_pj)
    target_latency_ms = answers.get("target_latency_ms", args.target_latency_ms)
    if args.dashboard_out is not None:
        dashboard_out = args.dashboard_out  # dashboard_yn/path steps weren't in the chain at all
    else:
        dashboard_out = answers.get("dashboard_path") if answers.get("dashboard_yn") == "y" else None
    return hw_config, allow_approximate, target_accuracy, target_energy_pj, target_latency_ms, dashboard_out


def build_initial_state(
    model_id: str,
    hw_config: HWConfig,
    target_accuracy: Optional[float] = None,
    target_energy_pj: Optional[float] = None,
    target_latency_ms: Optional[float] = None,
    allow_approximate_calibration: bool = False,
) -> AutoCIMState:
    if allow_approximate_calibration:
        # --allow-approximate-calibration: nearest-KNOWN_REFERENCES fallback
        # (tools/calibration.py) when hw_config has no exact match. When it
        # *does* have an exact match, this returns the identical factor
        # bootstrap_calibration_factors would (distance 0 to that same
        # reference) -- so opting in never changes an already-exact-matched
        # session, only rescues an otherwise-uncalibrated one.
        calibration_factors = bootstrap_approximate_calibration_factors(hw_config)
        calibration_provenance = bootstrap_approximate_calibration_provenance(hw_config)
    else:
        # tools/calibration.py: seeds a real, literature-derived correction
        # factor when hw_config exactly matches a known reference (e.g. the
        # NeuroSim-validated 128x128/7-bit-ADC config); otherwise {} --
        # @profiler stays uncalibrated (factor 1.0) rather than guessing.
        calibration_factors = bootstrap_calibration_factors(hw_config)
        calibration_provenance = bootstrap_calibration_provenance(hw_config)
    return {
        "messages": [],
        "failure_history": [],
        "candidate_history": [],
        "llm_usage": [],
        "planner_decisions": [],
        "metrics_store": {},
        # The citation/uncertainty behind calibration_factors (or {} if
        # uncalibrated) -- tools/dashboard.py surfaces this so a researcher
        # sees *which* published number backs a candidate's energy figure,
        # and whether it's an exact match or an approximate (see above)
        # scaling-law transfer.
        "calibration_factors": calibration_factors,
        "calibration_provenance": calibration_provenance,
        "human_overrides": {},
        "planned_layer_configs": [],
        "model_id": model_id,
        "hw_spec_id": hw_config.hw_spec_id,
        "target_accuracy": target_accuracy,
        "target_energy_pj": target_energy_pj,
        "target_latency_ms": target_latency_ms,
        "iteration_count": 0,
        "retry_count": 0,
        "is_converged": False,
        "needs_hitl": False,
    }


# Escalation tiers, indexed by _gap_tier()'s 0/1/2: how big a nudge to
# suggest scales with how far off target the candidate actually is, instead
# of always suggesting the same fixed step regardless of the miss size.
_AUTO_HITL_GAP_TIER_THRESHOLDS = (0.15, 0.4)  # <15% / 15-40% / >=40% relative miss
_AUTO_HITL_BITS_STEP_TIERS = (1, 2, 3)
_AUTO_HITL_PRUNING_STEP_TIERS = (0.1, 0.2, 0.3)


def _gap_tier(gap_ratio: float) -> int:
    """0/1/2 for the <15%/15-40%/>=40% escalation tiers above. `gap_ratio`
    is a relative miss (e.g. 0.3 == 30% short of / over target); the
    caller is responsible for clamping it to >= 0 first."""
    if gap_ratio < _AUTO_HITL_GAP_TIER_THRESHOLDS[0]:
        return 0
    if gap_ratio < _AUTO_HITL_GAP_TIER_THRESHOLDS[1]:
        return 1
    return 2


def suggest_override_bounds(interrupt_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rule-based `new_bounds` suggestion for a HITL interrupt -- the shared
    logic behind both `auto_resolve_override` (--auto-hitl, applies it
    unattended) and `prompt_for_override` (prints it as a suggestion a
    researcher can accept, tweak, or ignore). No side effects (no printing,
    doesn't apply anything) -- purely a function of `interrupt_payload`.

    Classifies the failure that triggered this interrupt from
    `failure_history`'s machine-readable reason strings
    (`nodes/evaluator.py`'s `check_targets` always prefixes each reason with
    "accuracy "/"energy_pj "/"noc_latency_ms "), and only proposes a
    weight_bits/pruning_ratio bounds override when the failure is actually
    something a layer_config search could plausibly fix. The suggested step
    size escalates with how far off target the candidate actually is
    (`_gap_tier`) -- computed from `nodes/hitl.py`'s target_accuracy/
    target_energy_pj/target_latency_ms and the candidate's own measured
    values in `latest_metrics`, not by parsing the reason string's embedded
    numbers -- and is capped at `nodes/planner.py`'s `search_bounds()`, the
    same hw_spec_id-derived (physical ADC/DAC ceiling) + policy bounds the
    real search itself respects, so a large miss can never suggest a value
    the search wouldn't even be allowed to try.

    Returns None whenever it can't safely suggest one:
      - reason == "verifier reported not converged" (nodes/evaluator.py):
        IR-drop/noise margin is a pure function of HWConfig
        (`tools/simulators.py`'s `verifier_tool`: wire_resistance_ohm_per_um /
        crossbar_rows / adc_bits / device_noise_sigma only), independent of
        any layer_config. No bounds override can ever fix this -- retrying
        would just burn real QAT trials against an hw_spec_id that's
        physically infeasible regardless of the model config.
      - a raw validation-error reason (unrecognized prefix): a schema/tool
        bug, not a search problem.
      - accuracy-too-low and energy_pj/noc_latency_ms-too-high are both
        reported in the same iteration: they pull the one shared knob
        (weight_bits / column_pruning_ratio) in opposite directions, so
        this is a genuine trade-off call for a human, not something one
        fixed rule should decide silently.
      - the last candidate's avg_weight_bits/avg_column_pruning_ratio
        (`nodes/evaluator.py`'s `build_candidate_entry`) aren't available to
        compute a step from.
    """
    failure_history = interrupt_payload.get("failure_history") or []
    if not failure_history:
        return None
    reason = failure_history[-1].get("reason", "")
    segments = [s.strip() for s in reason.split(";") if s.strip()]
    if not segments:
        return None

    wants_precision = False  # accuracy too low -> more weight_bits, less pruning
    wants_compression = False  # energy/latency too high -> fewer weight_bits, more pruning
    for seg in segments:
        if seg.startswith("accuracy "):
            wants_precision = True
        elif seg.startswith("energy_pj ") or seg.startswith("noc_latency_ms "):
            wants_compression = True
        else:
            return None  # "verifier reported not converged" or a validation error

    if wants_precision == wants_compression:
        # Both True (conflicting objectives) or both False (nothing
        # recognized) -- neither is safe to resolve with one fixed rule.
        return None

    latest_metrics = interrupt_payload.get("latest_metrics") or {}
    tuner_data = latest_metrics.get("tuner", {}).get("data") or {}
    verifier_data = latest_metrics.get("verifier", {}).get("data") or {}
    current_bits = tuner_data.get("avg_weight_bits")
    current_pruning = tuner_data.get("avg_column_pruning_ratio")
    if current_bits is None or current_pruning is None:
        return None

    # Relative miss per dimension that actually failed -- the worst
    # (largest) of them drives the escalation tier, so e.g. a barely-missed
    # accuracy alongside a badly-missed latency still escalates.
    gap_ratios: List[float] = []
    target_accuracy = interrupt_payload.get("target_accuracy")
    current_accuracy = tuner_data.get("accuracy")
    if wants_precision and target_accuracy and current_accuracy is not None:
        gap_ratios.append(max(0.0, (target_accuracy - current_accuracy) / target_accuracy))
    target_energy_pj = interrupt_payload.get("target_energy_pj")
    current_energy_pj = verifier_data.get("energy_pj")
    if wants_compression and target_energy_pj and current_energy_pj is not None:
        gap_ratios.append(max(0.0, (current_energy_pj - target_energy_pj) / target_energy_pj))
    target_latency_ms = interrupt_payload.get("target_latency_ms")
    current_latency_ms = verifier_data.get("noc_latency_ms")
    if wants_compression and target_latency_ms and current_latency_ms is not None:
        gap_ratios.append(max(0.0, (current_latency_ms - target_latency_ms) / target_latency_ms))
    tier = _gap_tier(max(gap_ratios)) if gap_ratios else 0
    bits_step = _AUTO_HITL_BITS_STEP_TIERS[tier]
    pruning_step = _AUTO_HITL_PRUNING_STEP_TIERS[tier]

    # Same bounds the real search is confined to (hw_spec_id's physical
    # ADC/DAC ceiling + this project's policy floor/ceiling) -- a large-tier
    # suggestion is clamped here rather than proposing something outside
    # what nodes/planner.py's search_bounds()/_clamp_layer_configs() would
    # even accept (silently ignored downstream if it inverted the range).
    (bits_floor, bits_ceiling), (pruning_floor, pruning_ceiling) = search_bounds(interrupt_payload.get("hw_spec_id") or "")

    if wants_precision:
        return {
            "weight_bits_min": min(bits_ceiling, round(current_bits) + bits_step),
            "pruning_ratio_max": max(pruning_floor, round(current_pruning - pruning_step, 4)),
        }
    return {
        "weight_bits_max": max(bits_floor, round(current_bits) - bits_step),
        "pruning_ratio_min": min(pruning_ceiling, round(current_pruning + pruning_step, 4)),
    }


_RECOGNIZED_OVERRIDE_KEYS = {"weight_bits_min", "weight_bits_max", "pruning_ratio_min", "pruning_ratio_max"}


def _prompt_manual_override(suggestion: Optional[Dict[str, Any]]) -> Any:
    """The [2] 직접 입력 sub-flow: loops on invalid JSON (a typo here used to
    propagate as an uncaught json.JSONDecodeError and kill the whole run --
    real QAT time already spent on this iteration, gone) instead of ever
    raising, and 'b'/blank both hand control back to the caller -- 'b' means
    "changed my mind, show the [1]/[2]/[3] menu again", blank means "resume
    unchanged" (`{}`), matching this prompt's pre-existing meaning. Returns
    `_BACK`, or a `new_bounds` dict (never raises on bad input)."""
    while True:
        raw = input(
            # Keys recognized by nodes/planner.py's search_bounds()/_clamp_layer_configs().
            "Enter new_bounds as JSON (recognized keys: weight_bits_min, weight_bits_max, "
            "pruning_ratio_min, pruning_ratio_max), 빈칸=변경 없이 재시도, b=메뉴로 돌아가기: "
        ).strip()
        if raw.lower() in _BACK_COMMANDS:
            return _BACK
        if not raw:
            return {}
        try:
            new_bounds = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"  잘못된 JSON입니다 ({exc.msg}) -- 다시 입력하거나 b로 메뉴로 돌아가세요.")
            continue
        if not isinstance(new_bounds, dict):
            print("  JSON 객체({...}) 형식으로 입력해주세요 (예: {\"weight_bits_min\": 4}).")
            continue
        unrecognized = sorted(set(new_bounds) - _RECOGNIZED_OVERRIDE_KEYS)
        if unrecognized:
            print(f"  참고: {unrecognized}는 인식되는 키가 아니라 적용되지 않습니다 (오타 확인).")
        return new_bounds


def prompt_for_override(interrupt_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Blocks on researcher input and shapes it into the
    `Command(resume={"new_bounds": ...})` contract `hitl_node` expects, or
    returns None to stop the run instead (run_session's existing "None ->
    write_dashboard + return" handling covers this the same as any other
    stop path)."""
    print("\n=== HITL interrupt: researcher review requested ===")
    print(f"reason         : {interrupt_payload.get('reason')}")
    print(f"iteration_count: {interrupt_payload.get('iteration_count')}")
    print(f"retry_count    : {interrupt_payload.get('retry_count')}")
    for entry in interrupt_payload.get("failure_history") or []:
        print(f"  - iter {entry.get('iteration')}: {entry.get('reason')}")
    latest = (interrupt_payload.get("latest_metrics") or {}).get("verifier", {}).get("data") or {}
    if latest:
        print(f"latest verifier: {latest}")

    suggestion = suggest_override_bounds(interrupt_payload)
    if suggestion is not None:
        print(f"suggested new_bounds (heuristic): {json.dumps(suggestion)}")

    while True:
        print("\n  [1] 추천 값 사용" + ("" if suggestion is not None else " (추천값 없음)"))
        print("  [2] 직접 입력")
        print("  [3] 지금까지 결과를 리포트에 기록 후 종료")
        choice = input("선택: ").strip()

        if choice == "1":
            if suggestion is None:
                print("  이번 실패는 규칙 기반 추천값이 없습니다 (예: 물리적 HW 검증 실패는 bounds로 못 고칩니다) -- 다른 항목을 선택해주세요.")
                continue
            return {"new_bounds": suggestion}
        if choice == "2":
            result = _prompt_manual_override(suggestion)
            if result is _BACK:
                continue
            return {"new_bounds": result}
        if choice == "3":
            return None
        print("  잘못된 선택입니다 -- 1, 2, 3 중 하나를 입력해주세요.")


def auto_resolve_override(interrupt_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rule-based, unattended stand-in for `prompt_for_override` (--auto-hitl)
    -- applies `suggest_override_bounds`'s suggestion directly instead of
    just printing it, or returns None (caller must stop the run rather than
    keep retrying) when there's nothing safe to suggest."""
    new_bounds = suggest_override_bounds(interrupt_payload)
    if new_bounds is None:
        return None
    reason = (interrupt_payload.get("failure_history") or [{}])[-1].get("reason", "")
    print(f"\n[auto-hitl] resolved without researcher input: reason={reason!r} -> new_bounds={new_bounds}")
    return {"new_bounds": new_bounds}


def prompt_for_dashboard_out(hw_spec_id: str, thread_id: str) -> Optional[str]:
    """Interactive --dashboard-out picker for the case run_interactive_setup()
    doesn't cover: --hw-config was given explicitly (so there's no hw-config/
    target wizard to fold this into) but --dashboard-out was omitted and
    stdin is a real terminal. The path is never something a researcher
    types -- always REPORT_DIR + a filename derived from hw_spec_id +
    thread_id (already unique per fresh session), so back-to-back
    interactive runs never clobber each other's report and a custom path
    can't point at a missing parent directory. EOFError/KeyboardInterrupt
    falls back to None (no report) rather than crashing.
    """
    try:
        answer = input("\n리포트(HTML 대시보드)를 생성할까요? [y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if answer != "y":
        return None
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return str(REPORT_DIR / f"report_{hw_spec_id}_{thread_id[:8]}_{datetime.now():%Y%m%d_%H%M%S}.html")


def _truncate(text: Optional[str], max_chars: int = 80) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _format_update(iteration: int, node_name: str, update: Dict[str, Any]) -> str:
    """One human-readable line per graph node update -- the raw update
    dict (full layer_configs, messages, etc.) is real data but far too
    dense to watch live; this pulls out just enough to follow progress at
    a glance. Full detail is still in candidate_history/--dashboard-out/
    observability.py's JSONL log (AUTOCIM_LOG_STDOUT=1 to also mirror that
    to stdout) if it's ever needed."""
    prefix = f"[{iteration}] {node_name:<9}:"
    data = ((update.get("metrics_store") or {}).get(node_name) or {}).get("data") or {}

    if node_name == "planner":
        decisions = update.get("planner_decisions") or []
        if not decisions:
            return f"{prefix} (no decision recorded)"
        d = decisions[-1]
        llm_state = "llm=ok" if d.get("used_llm") else "llm=fallback"
        return f"{prefix} {d.get('search_tag')} ({llm_state}) -- {_truncate(d.get('rationale'))}"

    if node_name == "tuner":
        return f"{prefix} accuracy={data.get('accuracy')} device={data.get('device')} epochs_run={data.get('epochs_run')}"

    if node_name == "mapper":
        return f"{prefix} noc_latency_ms={data.get('noc_latency_ms')} tiles_needed={data.get('tiles_needed')}"

    if node_name == "profiler":
        return f"{prefix} energy_pj={data.get('energy_pj')}"

    if node_name == "verifier":
        return (
            f"{prefix} hw_converged={data.get('is_converged')} "
            f"ir_drop_error_pct={data.get('ir_drop_error_pct')} noise_margin_db={data.get('noise_margin_db')}"
        )

    if node_name == "evaluator":
        if update.get("is_converged"):
            return f"{prefix} CONVERGED"
        failure = (update.get("failure_history") or [{}])[0]
        status = "HITL" if update.get("needs_hitl") else "not converged, retrying"
        return f"{prefix} {status} (retry {update.get('retry_count')}) -- {failure.get('reason')}"

    return f"{prefix} {update}"  # fallback for any future node this doesn't special-case yet


def stream_until_interrupt(graph, resumable_input: Any, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Streams node-by-node state deltas to stdout as one readable summary
    line per node (`_format_update`); returns the interrupt payload if the
    run paused mid-graph, or None if it ran to completion. `iteration`
    tracks the current graph iteration across nodes -- only `planner`'s own
    update carries `iteration_count` directly, so it's captured there and
    reused for the rest of that iteration's node lines."""
    iteration = 0
    for chunk in graph.stream(resumable_input, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name == "__interrupt__":
                interrupts: tuple[Interrupt, ...] = update
                return interrupts[0].value
            if node_name == "planner":
                iteration = update.get("iteration_count", iteration)
            print(_format_update(iteration, node_name, update))
    return None


def get_pending_interrupt(graph, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The payload of `thread_id`'s pending interrupt, if `graph.get_state`
    shows it currently paused there -- this graph has exactly one dynamic
    `interrupt()` site (hitl_node), so there is at most one to find. `None`
    if the thread has no persisted state, or isn't currently paused."""
    snapshot = graph.get_state(config)
    if not snapshot.next:
        return None
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def print_final_state(state: Dict[str, Any]) -> None:
    print(f"is_converged   : {state.get('is_converged')}")
    print(f"iteration_count: {state.get('iteration_count')}")
    history = state.get("candidate_history") or []
    if history:
        latest = history[-1]
        print(
            f"latest candidate: accuracy={latest.get('accuracy')} energy_pj={latest.get('energy_pj')} "
            f"noc_latency_ms={latest.get('noc_latency_ms')} pareto_rank={latest.get('pareto_rank')}"
        )
    print("(full detail: --dashboard-out report, or candidate_history/llm_usage in the checkpoint DB)")


def list_sessions(checkpointer) -> List[Dict[str, Any]]:
    """One row per distinct `thread_id` found in `checkpointer`'s DB,
    newest-first (`SqliteSaver.list`'s own order). `checkpointer.list(None)`
    -- no `config`, hence no `thread_id` filter -- returns every checkpoint
    across every thread; each thread has many (one per completed node), so
    this keeps only the first (newest) one seen per thread_id. Reads each
    thread's actual current state via `graph.get_state()` (the same public
    API `write_dashboard`/`get_pending_interrupt` already use) rather than
    parsing raw checkpoint internals."""
    graph = build_graph(checkpointer=checkpointer)
    # Materialize fully before calling graph.get_state() on any of them:
    # SqliteSaver.list() is a generator holding its own cursor/transaction
    # open on `checkpointer`'s connection for as long as it's being
    # iterated -- calling get_state() (which needs the same connection)
    # *during* that iteration hangs. Draining it into a plain list first
    # closes that cursor before any nested query runs.
    checkpoint_tuples = list(checkpointer.list(None))

    sessions: List[Dict[str, Any]] = []
    seen_thread_ids = set()
    for checkpoint_tuple in checkpoint_tuples:
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        if thread_id in seen_thread_ids:
            continue
        seen_thread_ids.add(thread_id)

        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        values = snapshot.values
        sessions.append(
            {
                "thread_id": thread_id,
                "model_id": values.get("model_id"),
                "hw_spec_id": values.get("hw_spec_id"),
                "iteration_count": values.get("iteration_count"),
                "is_converged": values.get("is_converged"),
                "paused": bool(snapshot.next),
            }
        )
    return sessions


def print_sessions(sessions: List[Dict[str, Any]]) -> None:
    if not sessions:
        print("No sessions found in this checkpoint DB.")
        return
    print(f"{'thread_id':<38} {'model_id':<14} {'hw_spec_id':<24} {'iter':>4}  status")
    for s in sessions:
        if s["paused"]:
            status = "paused (HITL)"
        elif s["is_converged"]:
            status = "converged"
        else:
            status = "in-progress"
        print(
            f"{s['thread_id']:<38} {str(s['model_id']):<14} {str(s['hw_spec_id']):<24} "
            f"{str(s['iteration_count']):>4}  {status}"
        )


def write_dashboard(graph, config: Dict[str, Any], dashboard_out: Optional[str]) -> None:
    """Renders tools/dashboard.py's report for `thread_id`'s current
    persisted state and writes it to `dashboard_out` -- a no-op if that
    wasn't requested (--dashboard-out). Called at every `run_session` exit
    point (paused or finished) so a researcher checking mid-run gets a
    report reflecting whatever iterations have actually completed, not only
    a fully-converged run."""
    if not dashboard_out:
        return
    state_values = graph.get_state(config).values
    Path(dashboard_out).write_text(render_dashboard_html(state_values), encoding="utf-8")
    print(f"[dashboard] wrote {dashboard_out}")


def _persist_long_term_learnings(
    model_store: ModelSpecificStore,
    hw_store: GlobalHWStore,
    model_id: str,
    hw_spec_id: str,
    state_values: Dict[str, Any],
) -> None:
    """Called at every run_session exit point (converged, paused, or
    explicitly stopped) via `_finish()` so a *later* run on this same
    model_id/hw_spec_id -- a different --thread-id, possibly a different
    process entirely -- starts from what this one already found instead of
    from scratch every time. Merges rather than overwrites: a worse run's
    candidate_history can only ever add candidates to the stored Pareto
    front, never regress it, since `compute_pareto_rank` is recomputed over
    the union of what's already stored plus what's new -- a candidate that
    used to be non-dominated but got dominated by something new (or
    already stored) correctly drops out."""
    calibration_factors = state_values.get("calibration_factors") or {}
    factor = calibration_factors.get(hw_spec_id)
    if factor is not None:
        hw_store.put_calibration_factors(hw_spec_id, {hw_spec_id: factor})

    candidate_history = state_values.get("candidate_history") or []
    if not candidate_history:
        return
    existing = (model_store.get_pareto_solutions(model_id) or {}).get("candidates", [])
    union = existing + candidate_history
    front = [c for c in union if compute_pareto_rank(c, union) == 1]
    model_store.put_pareto_solutions(model_id, {"candidates": front})


def _describe_calibration_status(calibration_provenance: Dict[str, Any], hw_spec_id: str) -> str:
    """One-line summary of how (or whether) `hw_spec_id` got calibrated --
    printed at fresh-session start so a researcher never has to dig into
    `calibration_provenance` just to learn whether an energy number is an
    exact citation, a --allow-approximate-calibration extrapolation, or
    uncalibrated (factor 1.0)."""
    described = calibration_provenance.get(hw_spec_id)
    if described is None:
        return "uncalibrated (no exact match; pass --allow-approximate-calibration for a best-effort factor)"
    if described.get("approximate"):
        return (
            f"approximate (nearest reference {described['matched_crossbar_rows']}x"
            f"{described['matched_crossbar_cols']}/{described['matched_adc_bits']}-bit ADC, "
            f"distance={described['distance']})"
        )
    return "exact match"


def run_session(
    model_id: str,
    hw_config: HWConfig,
    thread_id: str,
    checkpointer,
    dashboard_out: Optional[str] = None,
    parallel_warmup_workers: Optional[int] = None,
    target_accuracy: Optional[float] = None,
    target_energy_pj: Optional[float] = None,
    target_latency_ms: Optional[float] = None,
    allow_approximate_calibration: bool = False,
    auto_hitl: bool = False,
    auto_hitl_max_rounds: int = 2,
) -> None:
    register_hw_config(hw_config)
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    long_term_backend = SqliteLongTermStore(DEFAULT_LONG_TERM_STORE_DB)
    model_store = ModelSpecificStore(long_term_backend)
    hw_store = GlobalHWStore(long_term_backend)

    def _finish() -> None:
        """Every run_session exit point (converged, paused, or explicitly
        stopped) calls this instead of write_dashboard() directly -- same
        report-writing as before, plus persisting whatever this run
        learned (calibration_factors, candidate_history) so a later run on
        this model_id/hw_spec_id doesn't start from scratch."""
        write_dashboard(graph, config, dashboard_out)
        _persist_long_term_learnings(model_store, hw_store, model_id, hw_config.hw_spec_id, graph.get_state(config).values)

    auto_hitl_rounds_used = 0

    def resolve_override(interrupt_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """None means "stop the run instead" -- either the researcher chose
        [3] 지금까지 결과를 리포트에 기록 후 종료 in prompt_for_override, or
        --auto-hitl declined to guess (auto_resolve_override) / its round
        cap is spent."""
        nonlocal dashboard_out
        if not auto_hitl:
            override = prompt_for_override(interrupt_payload)
            if override is None and not dashboard_out:
                # [3] was chosen specifically to not lose this run's
                # progress -- write a report even though none was ever
                # requested, instead of silently doing nothing.
                REPORT_DIR.mkdir(parents=True, exist_ok=True)
                dashboard_out = str(
                    REPORT_DIR / f"report_{hw_config.hw_spec_id}_{thread_id[:8]}_{datetime.now():%Y%m%d_%H%M%S}.html"
                )
                print(f"[dashboard] --dashboard-out가 지정되지 않아서 여기에 기록합니다: {dashboard_out}")
            return override
        nonlocal auto_hitl_rounds_used
        if auto_hitl_rounds_used >= auto_hitl_max_rounds:
            print(
                f"\n[auto-hitl] round cap reached ({auto_hitl_max_rounds}); stopping instead of resolving "
                f"further without researcher review. Resume with --thread-id {thread_id} (without --auto-hitl) "
                "to take over manually."
            )
            return None
        resolved = auto_resolve_override(interrupt_payload)
        if resolved is None:
            print(
                "\n[auto-hitl] could not resolve this failure automatically (not a search-fixable target miss); "
                f"stopping. Resume with --thread-id {thread_id} (without --auto-hitl) to review manually."
            )
            return None
        auto_hitl_rounds_used += 1
        return resolved

    snapshot = graph.get_state(config)
    if snapshot.values and not snapshot.next:
        print(f"\nSession thread_id={thread_id!r} already ran to completion -- nothing to resume.")
        print_final_state(snapshot.values)
        _finish()
        return

    if snapshot.next:
        # Persisted + paused (e.g. at hitl_human_approval) from a previous
        # process invocation -- this is what --thread-id resuming across
        # restarts actually means now that `checkpointer` is persistent.
        interrupt_payload = get_pending_interrupt(graph, config)
        if interrupt_payload is None:
            # This graph's only pause point is hitl_node's interrupt(), so
            # "paused but no interrupt recorded" shouldn't happen -- but
            # don't guess a resume value if it somehow does.
            print(f"\nSession thread_id={thread_id!r} is paused but has no pending researcher interrupt; " "nothing safe to resume automatically.")
            return
        print(
            f"\nResuming persisted session thread_id={thread_id!r} "
            f"(iteration_count={snapshot.values.get('iteration_count')})"
        )
        try:
            override = resolve_override(interrupt_payload)
        except (EOFError, KeyboardInterrupt):
            print(f"\nNo researcher input received; session remains paused. Resume later with --thread-id {thread_id}.")
            _finish()
            return
        if override is None:
            _finish()
            return
        resumable_input: Any = Command(resume=override)
    else:
        print(
            f"\nStarting new AutoCIM-Agent session: model_id={model_id!r} "
            f"hw_spec_id={hw_config.hw_spec_id!r} thread_id={thread_id!r}"
        )
        resumable_input = build_initial_state(
            model_id, hw_config, target_accuracy, target_energy_pj, target_latency_ms, allow_approximate_calibration
        )

        # Cross-session learning (store.py): a hw_spec_id this project has
        # already calibrated against real profiler runs before takes that
        # refined factor over the static literature bootstrap above; a
        # model_id already optimized before seeds candidate_history with
        # its stored Pareto front instead of starting cold, so warm-up
        # needs fewer (or zero) new LHS samples before the surrogate phase.
        learned_calibration = hw_store.get_calibration_factors(hw_config.hw_spec_id)
        if learned_calibration:
            resumable_input["calibration_factors"] = merge_dicts(resumable_input["calibration_factors"], learned_calibration)
            print(f"[long-term] 이전 실행에서 학습된 보정계수를 불러왔습니다: hw_spec_id={hw_config.hw_spec_id!r}")
        learned_solutions = model_store.get_pareto_solutions(model_id)
        if learned_solutions and learned_solutions.get("candidates"):
            resumable_input["candidate_history"] = list(learned_solutions["candidates"])
            print(
                f"[long-term] 이전 실행에서 찾은 Pareto 후보 {len(learned_solutions['candidates'])}개를 "
                f"불러왔습니다: model_id={model_id!r}"
            )

        print(
            f"[calibration] {_describe_calibration_status(resumable_input['calibration_provenance'], hw_config.hw_spec_id)}"
        )

        if parallel_warmup_workers is not None:
            # Fresh session only -- a resumed/paused session already has
            # whatever candidate_history it persisted, and this step's
            # whole point is seeding history *before* the first real
            # iteration (tools/batch_warmup.py's module docstring).
            print(f"Evaluating LHS warm-up candidates across up to {parallel_warmup_workers} worker(s)...")
            try:
                warmup_candidates, warmup_failures = run_parallel_warmup(
                    model_id,
                    hw_config,
                    calibration_factors=resumable_input["calibration_factors"],
                    max_workers=parallel_warmup_workers,
                )
            except KeyboardInterrupt:
                # Nothing persisted yet at this point (warm-up runs before
                # the graph's first checkpointed node) -- there is no
                # --thread-id state to resume, this run is simply gone.
                print("\n\n중단되었습니다 (warm-up 단계라 저장된 진행 상황이 없습니다).")
                return
            resumable_input["candidate_history"] = warmup_candidates
            resumable_input["failure_history"] = warmup_failures
            print(
                f"Parallel warm-up done: {len(warmup_candidates)} candidate(s) recorded, "
                f"{len(warmup_failures)} failure(s)."
            )

    while True:
        try:
            interrupt_payload = stream_until_interrupt(graph, resumable_input, config)
        except KeyboardInterrupt:
            # A node mid-run (most likely @tuner's real PyTorch training)
            # doesn't return control to Python between individual C-level
            # tensor ops, so Ctrl-C here isn't instant -- it takes effect
            # once the current node yields back (e.g. after the in-flight
            # batch finishes), same as any Python program blocked inside a
            # synchronous C extension call. LangGraph only checkpoints
            # *between* nodes, so whatever the last fully-completed node
            # was is what --thread-id resumes from; nothing from a node
            # that was interrupted mid-way is a lost, replayable retry
            # (the eventual candidate_history is exactly as if it were
            # still running when this session was resumed later).
            print(f"\n\n중단되었습니다. 마지막으로 완료된 지점까지는 저장되어 있습니다 -- --thread-id {thread_id}로 재개할 수 있습니다.")
            _finish()
            return
        if interrupt_payload is None:
            break
        try:
            override = resolve_override(interrupt_payload)
        except (EOFError, KeyboardInterrupt):
            # stdin closed / researcher aborted mid-prompt: exit cleanly.
            # The checkpoint is already persisted at this interrupt (it was
            # written before hitl_node's interrupt() call returned control
            # here), so this is a soft pause, not a lost session.
            print(f"\nNo researcher input received; session paused. Resume later with --thread-id {thread_id}.")
            _finish()
            return
        if override is None:
            _finish()
            return
        # Safe resume: only the dynamic interrupt()'s return value carries
        # this back into hitl_node -- no graph.update_state() call here that
        # could duplicate it (checklist item 1 / CLAUDE.md 5.B).
        resumable_input = Command(resume=override)

    print("\n=== Session finished ===")
    print_final_state(graph.get_state(config).values)
    _finish()


def main() -> None:
    load_dotenv()  # .env: shared/base defaults, if present
    load_dotenv(".env.local", override=True)  # .env.local: personal overrides (API keys), gitignored

    args = parse_args()
    checkpoint_db = args.checkpoint_db or str(DEFAULT_CHECKPOINT_DB)
    if checkpoint_db != ":memory:":
        Path(checkpoint_db).parent.mkdir(parents=True, exist_ok=True)

    if args.list_sessions:
        # No --hw-config needed to just inspect what's persisted -- a
        # researcher checking session status shouldn't need a valid
        # HWConfig file on hand for that.
        with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
            print_sessions(list_sessions(checkpointer))
        return

    thread_id = args.thread_id or str(uuid.uuid4())
    allow_approximate_from_prompt = False
    try:
        if args.hw_config is None and sys.stdin.isatty():
            # No --hw-config given and a human is actually at the terminal
            # (not a script/CI pipe) -- offer the interactive wizard instead
            # of silently falling back to DEFAULT_HW_CONFIG. Passing
            # --hw-config explicitly (any value) always skips this, so
            # existing scripts/automation are unaffected -- this banner
            # only ever reaches someone who ran `python main.py` with no
            # flags and has no idea yet what any of this does.
            print(
                "\n[AutoCIM-Agent] --hw-config가 없어 하드웨어 스펙을 대화형으로 고릅니다.\n\n"
                "  이후 나오는 각 입력창에서 Enter는 대괄호로 표시된 기본값을 그대로 쓰고,\n"
                "  'b'를 입력하면 이전 단계로 돌아갑니다.\n"
                "  이후 실제 QAT 학습을 포함한 최적화 세션이 시작되며, 반복마다 수 분 이상 걸릴 수 있습니다.\n\n"
                "  세션 진행 중 Ctrl+C는 즉시 멈추지 않고 진행 중인 학습 배치가 끝난 뒤에 반영됩니다.\n"
                "  중단 시점까지는 저장되어 --thread-id로 재개할 수 있습니다.\n\n"
                "  --target-accuracy / --target-energy-pj / --target-latency-ms\n"
                "  위 입력을 생략했을 때 정확도/에너지/지연시간이 좋지 않아도\n"
                "  물리적 HW 검증만으로 '수렴' 처리됩니다.\n\n"
                "  전체 옵션은 --help\n"
                "  예시: python main.py --model-id resnet18 "
                "--hw-config examples/hw_configs/default_cim_v1_128x128.json "
                "--target-accuracy 0.7\n"
            )
            used_wizard = True
            setup = run_interactive_setup(args, thread_id)
            if setup is None:
                raise SystemExit(0)
            hw_config, allow_approximate_from_prompt, target_accuracy, target_energy_pj, target_latency_ms, dashboard_out = (
                setup
            )
        else:
            used_wizard = False
            hw_config = load_hw_config(args.hw_config)
            target_accuracy = args.target_accuracy
            target_energy_pj = args.target_energy_pj
            target_latency_ms = args.target_latency_ms
            dashboard_out = args.dashboard_out
            if dashboard_out is None and sys.stdin.isatty():
                dashboard_out = prompt_for_dashboard_out(hw_config.hw_spec_id, thread_id)
    except HWConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    if not used_wizard:
        # The wizard's own confirm screen already showed this (and let the
        # researcher refresh it with 'r') right before asking "이대로
        # 진행할까요?" -- printing it again here would just be a second,
        # redundant copy with nothing new in it.
        planner_model, planner_fallback_reason = resolve_planner_model()
        if planner_fallback_reason:
            print(f"[llm] {planner_fallback_reason}")
        print(f"[session] LLM={planner_model}, QAT device={_resolve_device()}")

    with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
        run_session(
            args.model_id,
            hw_config,
            thread_id,
            checkpointer,
            dashboard_out=dashboard_out,
            parallel_warmup_workers=args.parallel_warmup_workers,
            target_accuracy=target_accuracy,
            target_energy_pj=target_energy_pj,
            target_latency_ms=target_latency_ms,
            allow_approximate_calibration=args.allow_approximate_calibration or allow_approximate_from_prompt,
            auto_hitl=args.auto_hitl,
            auto_hitl_max_rounds=args.auto_hitl_max_rounds,
        )


if __name__ == "__main__":
    main()
