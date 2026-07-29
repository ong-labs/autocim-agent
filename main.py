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
from middleware import register_hw_config
from schemas.config import HWConfig, NoCTopology
from state import AutoCIMState
from tools.batch_warmup import run_parallel_warmup
from tools.calibration import (
    bootstrap_approximate_calibration_factors,
    bootstrap_approximate_calibration_provenance,
    bootstrap_calibration_factors,
    bootstrap_calibration_provenance,
    find_nearest_reference,
)
from tools.dashboard import render_dashboard_html

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
EXAMPLE_HW_CONFIGS_DIR = Path(__file__).resolve().parent / "examples" / "hw_configs"
# prompt_for_dashboard_out's auto-generated report filenames land here
# instead of the repo root, so back-to-back interactive runs don't scatter
# report_*.html files next to main.py -- an explicit --dashboard-out PATH is
# unaffected (it's written exactly where given, as before).
REPORT_DIR = Path(__file__).resolve().parent / "report"


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
    raw = input(f"{label} [{default}] (뒤로: b): ").strip()
    if raw.lower() in _BACK_COMMANDS:
        return _BACK
    return raw if raw else default


def _parse_int_step(label: str, default: int) -> Any:
    while True:
        raw = input(f"{label} [{default}] (뒤로: b): ").strip()
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
        raw = input(f"{label} [{default}] (뒤로: b): ").strip()
        if raw.lower() in _BACK_COMMANDS:
            return _BACK
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  숫자를 입력해주세요.")


# One (field_name, read_fn) pair per HWConfig field prompt_custom_hw_config
# asks for, in order -- read_fn takes that field's current default/prior
# answer and returns either the parsed value or `_BACK`.
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


def prompt_custom_hw_config() -> Optional[HWConfig]:
    """Field-by-field HWConfig entry (Enter accepts the shown default; the
    first pass through defaults to DEFAULT_HW_CONFIG's own values, so a
    researcher who only cares about e.g. crossbar_rows/cols/adc_bits
    doesn't have to know every field). Typing 'b'/'back'/'뒤로' at any
    prompt steps back to the previous field instead of accepting the
    current one -- no need to restart the whole 11-field sequence over one
    typo. Typing it at the very *first* field (hw_spec_id) has nowhere
    earlier to go back to within this form, so it cancels out of custom
    entry entirely and returns `None` -- the caller (`prompt_for_hw_config`)
    treats that as "show the [1]/[2]/[3] menu again", which is what a
    researcher backing out of the first field of a wizard actually expects
    (re-seeing the same first-field prompt would look like 'back' did
    nothing at all).

    Individual field prompts only catch type errors (e.g. a non-numeric
    adc_bits); HWConfig's own range/enum rules (adc_bits<=16, a
    noc_topology string it doesn't recognize, etc.) are only checked once
    every field is in. On that failure, the offending field name(s) --
    read straight from pydantic's own `ValidationError.errors()` -- are
    printed and the whole step sequence restarts with every just-entered
    value (including the invalid one, so it's visible to fix) as the new
    default: re-confirming already-good fields is then a quick Enter-through,
    not retyping everything from scratch.
    """
    original_defaults: Dict[str, Any] = {
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
    values: Dict[str, Any] = dict(original_defaults)

    while True:
        print("\n=== 커스텀 HWConfig 입력 (Enter로 기본값 유지, 'b'로 이전 필드) ===")
        index = 0
        while index < len(_CUSTOM_HW_CONFIG_STEPS):
            name, read_fn = _CUSTOM_HW_CONFIG_STEPS[index]
            result = read_fn(values[name])
            if result is _BACK:
                if index == 0:
                    print("\n최상위 필드로 돌아갑니다.")
                    return None
                index -= 1
                continue
            values[name] = result
            index += 1

        try:
            return HWConfig(**values)
        except ValidationError as exc:
            bad_fields = sorted({str(error["loc"][0]) for error in exc.errors() if error.get("loc")})
            print(
                f"입력값이 HWConfig 스키마에 맞지 않습니다 ({', '.join(bad_fields) or '알 수 없는 필드'}):\n{exc}\n"
                "해당 필드는 원래 기본값으로 되돌리고 나머지는 방금 입력한 값을 유지합니다 -- 다시 확인해주세요."
            )
            # Reset only the offending field(s) to their *original* default
            # (never the just-typed invalid value) -- otherwise hitting
            # Enter on a bad field re-submits the same invalid value and
            # loops forever. Every other field keeps what was already
            # entered, so re-confirming them is a quick Enter-through.
            for field in bad_fields:
                if field in values:
                    values[field] = original_defaults[field]


def prompt_hw_config_from_examples() -> Optional[HWConfig]:
    """Lets a researcher pick one of the checked-in examples/hw_configs/
    files by number instead of typing its path -- `None` (caller falls back
    to DEFAULT_HW_CONFIG) if the directory is empty or the researcher
    cancels (blank input) or mistypes the number."""
    files = sorted(EXAMPLE_HW_CONFIGS_DIR.glob("*.json"))
    if not files:
        return None
    print("\n=== 저장된 예시 --hw-config 파일 ===")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f.name}")
    raw = input("번호 선택 (Enter로 취소): ").strip()
    if not raw:
        return None
    try:
        index = int(raw)
        if not (1 <= index <= len(files)):
            raise ValueError
    except ValueError:
        print("잘못된 선택입니다 -- 기본값을 사용합니다.")
        return None
    return load_hw_config(str(files[index - 1]))


def prompt_for_hw_config() -> Tuple[HWConfig, bool]:
    """Interactive --hw-config picker -- main() calls this only when
    --hw-config was omitted and stdin is a real terminal, so a scripted/CI
    invocation (or one that already passes --hw-config) never sees it and
    behaves exactly as before this existed.

    Offers: type a custom HWConfig directly (prompt_custom_hw_config), pick
    a saved examples/hw_configs/ file (prompt_hw_config_from_examples), or
    the built-in default. Immediately reports whether the resulting
    hw_config has an exact tools/calibration.py match and, if not, offers
    approximate calibration for just this run (the returned bool) --
    without requiring the researcher to already know
    --allow-approximate-calibration exists. EOFError/KeyboardInterrupt at
    any point falls back to (DEFAULT_HW_CONFIG, False) rather than crashing
    -- same "soft, non-destructive fallback" spirit as prompt_for_override's
    HITL handling.
    """
    try:
        hw_config: Optional[HWConfig] = None
        while hw_config is None:
            print("--hw-config가 지정되지 않았습니다. 어떻게 진행할까요?\n")
            print("  [1] 직접 값 입력 (Custom HWConfig)")
            print("  [2] 저장된 예시 파일 사용")
            print("  [3] 기본값 사용")
            print("  [4] 종료")
            raw_choice = input("\n 선택: ").strip()
            choice = raw_choice or "4"

            if choice == "1":
                hw_config = prompt_custom_hw_config()  # None if cancelled via 'back' on the first field -> loop, re-show menu
            elif choice == "2":
                hw_config = prompt_hw_config_from_examples() or DEFAULT_HW_CONFIG
            elif choice == "3":
                hw_config = DEFAULT_HW_CONFIG
            elif choice == "4":
                print("\n입력이 없습니다.\n실행을 종료합니다." if not raw_choice else "\n종료합니다.")
                raise SystemExit(0)
            else:
                hw_config = DEFAULT_HW_CONFIG
    except (EOFError, KeyboardInterrupt):
        print("\n입력을 받지 못해 기본값을 사용합니다.")
        return DEFAULT_HW_CONFIG, False

    if bootstrap_calibration_factors(hw_config):
        print("[calibration] exact match")
        return hw_config, False

    nearest = find_nearest_reference(hw_config)
    if nearest is None:
        print("[calibration] uncalibrated (no known reference)")
        return hw_config, False

    reference, distance = nearest
    print(
        f"[calibration] 정확히 일치하는 레퍼런스가 없습니다. 가장 가까운 레퍼런스: "
        f"{reference.crossbar_rows}x{reference.crossbar_cols}/{reference.adc_bits}-bit ADC "
        f"(distance={distance:.2f})"
    )
    try:
        answer = input("  이 레퍼런스로 근사 보정해서 진행할까요? [Y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    return hw_config, answer == "y"


def _read_optional_float(label: str) -> Optional[float]:
    """One target-metric prompt: blank -> None (ungated), a number -> that
    target. Loops on unparseable non-blank input instead of silently
    discarding a typo as "ungated"."""
    while True:
        raw = input(f"{label} (빈칸=미설정): ").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            print("  숫자를 입력하거나 빈칸으로 두세요.")


def prompt_for_targets(
    target_accuracy: Optional[float],
    target_energy_pj: Optional[float],
    target_latency_ms: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Interactive target-accuracy/energy/latency picker -- main() calls this
    right after prompt_for_hw_config() (any of its [1]/[2]/[3] branches), so
    it's not skippable by picking one hw_spec source over another: all three
    leave every target ungated by default, and nodes/evaluator.py's
    is_converged only checks a configured target -- with none configured, a
    candidate that merely clears @verifier's physical IR-drop/noise check
    (accuracy/energy/latency notwithstanding) still gets reported
    "Converged". Only prompts for whichever of the three wasn't already
    passed on the command line; a value already given via
    --target-accuracy/--target-energy-pj/--target-latency-ms is never
    overridden here. EOFError/KeyboardInterrupt leaves whatever hasn't been
    answered yet as ungated (None), same non-destructive-fallback spirit as
    prompt_for_hw_config/prompt_for_override -- it does not discard targets
    already entered before the interrupt."""
    if target_accuracy is not None and target_energy_pj is not None and target_latency_ms is not None:
        return target_accuracy, target_energy_pj, target_latency_ms

    print(
        "\n--target-accuracy/--target-energy-pj/--target-latency-ms가 지정되지 않았습니다.\n"
        "값을 입력하면 그 기준을 만족해야 '수렴'으로 인정되고, 빈칸으로 두면 물리적 HW 검증\n"
        "(IR-drop/노이즈)만으로 수렴 판정됩니다."
    )
    try:
        if target_accuracy is None:
            target_accuracy = _read_optional_float("target accuracy (예: 0.7)")
        if target_energy_pj is None:
            target_energy_pj = _read_optional_float("target energy_pj (예: 2000)")
        if target_latency_ms is None:
            target_latency_ms = _read_optional_float("target latency_ms (예: 5.0)")
    except (EOFError, KeyboardInterrupt):
        print("\n입력을 받지 못해 나머지 target은 미설정(빈칸)으로 둡니다.")
    return target_accuracy, target_energy_pj, target_latency_ms


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


_AUTO_HITL_BITS_STEP = 1
_AUTO_HITL_PRUNING_STEP = 0.1


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
    something a layer_config search could plausibly fix.

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

    tuner_data = (interrupt_payload.get("latest_metrics") or {}).get("tuner", {}).get("data") or {}
    current_bits = tuner_data.get("avg_weight_bits")
    current_pruning = tuner_data.get("avg_column_pruning_ratio")
    if current_bits is None or current_pruning is None:
        return None

    if wants_precision:
        return {
            "weight_bits_min": round(current_bits) + _AUTO_HITL_BITS_STEP,
            "pruning_ratio_max": max(0.0, round(current_pruning - _AUTO_HITL_PRUNING_STEP, 4)),
        }
    return {
        "weight_bits_max": max(1, round(current_bits) - _AUTO_HITL_BITS_STEP),
        "pruning_ratio_min": min(0.9, round(current_pruning + _AUTO_HITL_PRUNING_STEP, 4)),
    }


def prompt_for_override(interrupt_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Blocks on researcher input and shapes it into the
    `Command(resume={"new_bounds": ...})` contract `hitl_node` expects."""
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
        print(f"suggested new_bounds (heuristic, not applied automatically -- paste as-is or edit): {json.dumps(suggestion)}")

    raw = input(
        # Keys recognized by nodes/planner.py's search_bounds()/_clamp_layer_configs() --
        # any other key is accepted as valid JSON but silently has no effect.
        "Enter new_bounds as JSON (recognized keys: weight_bits_min, weight_bits_max, "
        "pruning_ratio_min, pruning_ratio_max), or leave blank to retry unchanged: "
    ).strip()
    new_bounds = json.loads(raw) if raw else {}
    return {"new_bounds": new_bounds}


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
    """Interactive --dashboard-out picker -- main() calls this only when
    --dashboard-out was omitted and stdin is a real terminal, same gating as
    prompt_for_hw_config so a scripted/CI invocation never sees it. Default
    filename is derived from hw_spec_id + thread_id (already unique per
    fresh session) so back-to-back interactive runs never clobber each
    other's report, and lands under REPORT_DIR instead of the repo root.
    EOFError/KeyboardInterrupt falls back to None (no report), matching
    prompt_for_hw_config's non-destructive fallback.
    """
    try:
        answer = input("\n리포트(HTML 대시보드)를 생성할까요? [Y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if answer != "y":
        return None
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    default_path = REPORT_DIR / f"report_{hw_spec_id}_{thread_id[:8]}_{datetime.now():%Y%m%d_%H%M%S}.html"
    try:
        raw = input(f"저장 경로 [{default_path}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""
    return raw or str(default_path)


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

    auto_hitl_rounds_used = 0

    def resolve_override(interrupt_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """None means "stop the run instead" -- either --auto-hitl declined
        to guess (auto_resolve_override) or its round cap is spent."""
        if not auto_hitl:
            return prompt_for_override(interrupt_payload)
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
        write_dashboard(graph, config, dashboard_out)
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
            write_dashboard(graph, config, dashboard_out)
            return
        if override is None:
            write_dashboard(graph, config, dashboard_out)
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
        print(
            f"[calibration] {_describe_calibration_status(resumable_input['calibration_provenance'], hw_config.hw_spec_id)}"
        )

        if parallel_warmup_workers is not None:
            # Fresh session only -- a resumed/paused session already has
            # whatever candidate_history it persisted, and this step's
            # whole point is seeding history *before* the first real
            # iteration (tools/batch_warmup.py's module docstring).
            print(f"Evaluating LHS warm-up candidates across up to {parallel_warmup_workers} worker(s)...")
            warmup_candidates, warmup_failures = run_parallel_warmup(
                model_id,
                hw_config,
                calibration_factors=resumable_input["calibration_factors"],
                max_workers=parallel_warmup_workers,
            )
            resumable_input["candidate_history"] = warmup_candidates
            resumable_input["failure_history"] = warmup_failures
            print(
                f"Parallel warm-up done: {len(warmup_candidates)} candidate(s) recorded, "
                f"{len(warmup_failures)} failure(s)."
            )

    while True:
        interrupt_payload = stream_until_interrupt(graph, resumable_input, config)
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
            write_dashboard(graph, config, dashboard_out)
            return
        if override is None:
            write_dashboard(graph, config, dashboard_out)
            return
        # Safe resume: only the dynamic interrupt()'s return value carries
        # this back into hitl_node -- no graph.update_state() call here that
        # could duplicate it (checklist item 1 / CLAUDE.md 5.B).
        resumable_input = Command(resume=override)

    print("\n=== Session finished ===")
    print_final_state(graph.get_state(config).values)
    write_dashboard(graph, config, dashboard_out)


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

    allow_approximate_from_prompt = False
    try:
        if args.hw_config is None and sys.stdin.isatty():
            # No --hw-config given and a human is actually at the terminal
            # (not a script/CI pipe) -- offer the interactive picker instead
            # of silently falling back to DEFAULT_HW_CONFIG. Passing
            # --hw-config explicitly (any value) always skips this, so
            # existing scripts/automation are unaffected -- this banner
            # only ever reaches someone who ran `python main.py` with no
            # flags and has no idea yet what any of this does.
            print(
                "\n[AutoCIM-Agent] --hw-config가 없어 하드웨어 스펙을 대화형으로 고릅니다.\n\n"
                "  이후 실제 QAT 학습을 포함한 최적화 세션이 시작되며, 반복마다 수 분 이상 걸릴 수 있습니다.\n"
                "  --target-accuracy / --target-energy-pj / --target-latency-ms "
                "  입력을 생략했을 때 정확도/에너지/지연시간이 좋지 않아도\n"
                "  물리적 HW 검증만으로 '수렴' 처리됩니다.\n"
                "  전체 옵션은 --help.\n"
                "  예시: python main.py --model-id resnet18 "
                "--hw-config examples/hw_configs/default_cim_v1_128x128.json "
                "--target-accuracy 0.7\n"
            )
            hw_config, allow_approximate_from_prompt = prompt_for_hw_config()
            # Same interactive-only gate as the hw_config picker above, and
            # reached regardless of which of its [1]/[2]/[3] branches was
            # taken -- none of custom/example/default asks about targets on
            # its own.
            target_accuracy, target_energy_pj, target_latency_ms = prompt_for_targets(
                args.target_accuracy, args.target_energy_pj, args.target_latency_ms
            )
        else:
            hw_config = load_hw_config(args.hw_config)
            target_accuracy = args.target_accuracy
            target_energy_pj = args.target_energy_pj
            target_latency_ms = args.target_latency_ms
    except HWConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    thread_id = args.thread_id or str(uuid.uuid4())

    dashboard_out = args.dashboard_out
    if dashboard_out is None and sys.stdin.isatty():
        dashboard_out = prompt_for_dashboard_out(hw_config.hw_spec_id, thread_id)

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
