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
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, Interrupt
from pydantic import ValidationError

from graph import build_graph
from middleware import register_hw_config
from schemas.config import HWConfig, NoCTopology
from state import AutoCIMState
from tools.calibration import bootstrap_calibration_factors, bootstrap_calibration_provenance
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


def build_initial_state(model_id: str, hw_config: HWConfig) -> AutoCIMState:
    return {
        "messages": [],
        "failure_history": [],
        "candidate_history": [],
        "llm_usage": [],
        "planner_decisions": [],
        "metrics_store": {},
        # tools/calibration.py: seeds a real, literature-derived correction
        # factor when hw_config exactly matches a known reference (e.g. the
        # NeuroSim-validated 128x128/7-bit-ADC config); otherwise {} --
        # @profiler stays uncalibrated (factor 1.0) rather than guessing.
        "calibration_factors": bootstrap_calibration_factors(hw_config),
        # The citation/uncertainty behind that factor (or {} if
        # uncalibrated) -- tools/dashboard.py surfaces this so a researcher
        # sees *which* published number backs a candidate's energy figure.
        "calibration_provenance": bootstrap_calibration_provenance(hw_config),
        "human_overrides": {},
        "planned_layer_configs": [],
        "model_id": model_id,
        "hw_spec_id": hw_config.hw_spec_id,
        "iteration_count": 0,
        "retry_count": 0,
        "is_converged": False,
        "needs_hitl": False,
    }


def prompt_for_override(interrupt_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Blocks on researcher input and shapes it into the
    `Command(resume={"new_bounds": ...})` contract `hitl_node` expects."""
    print("\n=== HITL interrupt: researcher review requested ===")
    print(f"reason         : {interrupt_payload.get('reason')}")
    print(f"iteration_count: {interrupt_payload.get('iteration_count')}")
    print(f"retry_count    : {interrupt_payload.get('retry_count')}")
    print(f"failure_history: {interrupt_payload.get('failure_history')}")
    print(f"latest_metrics : {interrupt_payload.get('latest_metrics')}")

    raw = input('Enter new_bounds as JSON (e.g. {"weight_bits_min": 2}), or leave blank to retry unchanged: ').strip()
    new_bounds = json.loads(raw) if raw else {}
    return {"new_bounds": new_bounds}


def stream_until_interrupt(graph, resumable_input: Any, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Streams node-by-node state deltas to stdout; returns the interrupt
    payload if the run paused mid-graph, or None if it ran to completion."""
    for chunk in graph.stream(resumable_input, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name == "__interrupt__":
                interrupts: tuple[Interrupt, ...] = update
                return interrupts[0].value
            print(f"[{node_name}] {update}")
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
    print(f"is_converged  : {state.get('is_converged')}")
    print(f"iteration_count: {state.get('iteration_count')}")
    print(f"metrics_store : {state.get('metrics_store')}")


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


def run_session(
    model_id: str, hw_config: HWConfig, thread_id: str, checkpointer, dashboard_out: Optional[str] = None
) -> None:
    register_hw_config(hw_config)
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

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
            resumable_input: Any = Command(resume=prompt_for_override(interrupt_payload))
        except (EOFError, KeyboardInterrupt):
            print(f"\nNo researcher input received; session remains paused. Resume later with --thread-id {thread_id}.")
            write_dashboard(graph, config, dashboard_out)
            return
    else:
        print(
            f"\nStarting new AutoCIM-Agent session: model_id={model_id!r} "
            f"hw_spec_id={hw_config.hw_spec_id!r} thread_id={thread_id!r}"
        )
        resumable_input = build_initial_state(model_id, hw_config)

    while True:
        interrupt_payload = stream_until_interrupt(graph, resumable_input, config)
        if interrupt_payload is None:
            break
        try:
            override = prompt_for_override(interrupt_payload)
        except (EOFError, KeyboardInterrupt):
            # stdin closed / researcher aborted mid-prompt: exit cleanly.
            # The checkpoint is already persisted at this interrupt (it was
            # written before hitl_node's interrupt() call returned control
            # here), so this is a soft pause, not a lost session.
            print(f"\nNo researcher input received; session paused. Resume later with --thread-id {thread_id}.")
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

    try:
        hw_config = load_hw_config(args.hw_config)
    except HWConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    thread_id = args.thread_id or str(uuid.uuid4())

    with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
        run_session(args.model_id, hw_config, thread_id, checkpointer, dashboard_out=args.dashboard_out)


if __name__ == "__main__":
    main()
