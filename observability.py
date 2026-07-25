"""Structured, per-run event logging for AutoCIM-Agent.

`state["messages"]` (state.py) is free-text meant for the LLM's own context
window (`middleware.py`'s context-pruning operates on it) -- not something a
researcher can grep, filter by iteration, or machine-parse across a run
shared with teammates. This module is the separate structured channel real
multi-researcher use needs: one JSON object per notable event
(`log_event`), written as JSON Lines to `<AUTOCIM_LOG_DIR>/<run_id>.jsonl`
(one file per run -- `nodes.common.run_id_for` -- so concurrent
researchers' sessions never interleave in the same file) and mirrored to
stdout via the stdlib `logging` module -- no new dependency, `tail -f`-able,
and machine-parseable line by line.

Callers (`nodes/planner.py`'s `candidate_proposed`, `nodes/evaluator.py`'s
`candidate_evaluated`) log events that echo already-structured
`AutoCIMState` fields (`planner_decisions`, `candidate_history`) -- this
file is the narrative/audit trail of *when* and *in what order* things
happened, not a second source of truth; `tools/dashboard.py` reads state
directly, not this log.

Every record also carries `code_version` (`_get_code_version`): the search
algorithm and simulators here are still actively changing, so "which code
produced this candidate_history/Pareto front" is real information a
researcher needs across a run that might span several code changes
(a session can be paused at HITL for days) -- `AutoCIMState` itself has no
natural place for this (it's a property of the process that ran a step,
not of the optimization data itself), so it lives in this audit log
instead, computed fresh (and cheaply cached) from `git`, not persisted
into state.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_LOG_DIR_ENV = "AUTOCIM_LOG_DIR"
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / ".cache" / "logs"
_REPO_ROOT = Path(__file__).resolve().parent

_loggers: Dict[str, logging.Logger] = {}
_code_version_cache: Optional[str] = None


def _get_code_version() -> str:
    """Short git commit hash, `-dirty` suffixed if the working tree has
    uncommitted changes, or `"unknown"` if `git`/a repo isn't available
    (e.g. a packaged install with no `.git` directory) -- never raises,
    since a missing version string shouldn't break logging. Cached after
    the first real call: this shells out to `git`, and the answer can't
    change within one process's lifetime."""
    global _code_version_cache
    if _code_version_cache is not None:
        return _code_version_cache

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode != 0:
            _code_version_cache = "unknown"
            return _code_version_cache

        version = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0 and status.stdout.strip():
            version += "-dirty"
        _code_version_cache = version
    except Exception:  # noqa: BLE001 -- git missing/not a repo/timeout: never fatal here
        _code_version_cache = "unknown"
    return _code_version_cache


def _log_dir() -> Path:
    configured = os.environ.get(_LOG_DIR_ENV)
    return Path(configured) if configured else _DEFAULT_LOG_DIR


def _safe_filename(run_id: str) -> str:
    """`run_id` (`nodes.common.run_id_for`) is `<model_id>::<hw_spec_id>` --
    `:` is a reserved character in Windows filenames, so it (and anything
    else non-filename-safe) gets swapped for `_` here. The raw `run_id`
    still appears verbatim inside each JSON record (`log_event`); only the
    on-disk filename is sanitized."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


class _JsonLineFormatter(logging.Formatter):
    """Ignores the record's own message/args -- every call site here logs
    via `log_event`'s `extra={"event_payload": ...}`, never a plain
    printf-style message, so the JSON payload is always the entire line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "event_payload", None)
        if payload is None:
            return super().format(record)
        return json.dumps(payload, default=str)


def _get_run_logger(run_id: str) -> logging.Logger:
    if run_id in _loggers:
        return _loggers[run_id]

    logger = logging.getLogger(f"autocim.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # never also feed the root logger's handlers

    formatter = _JsonLineFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{_safe_filename(run_id)}.jsonl", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _loggers[run_id] = logger
    return logger


def log_event(run_id: str, node: str, event: str, iteration: int, **fields: Any) -> Dict[str, Any]:
    """Emits one structured JSON-Lines record (stdout + on-disk log file)
    and returns the same dict, so callers/tests can assert on exactly what
    was logged without re-parsing stdout or the log file."""
    record = {
        "ts": time.time(),
        "run_id": run_id,
        "iteration": iteration,
        "node": node,
        "event": event,
        "code_version": _get_code_version(),
        **fields,
    }
    logger = _get_run_logger(run_id)
    logger.info("", extra={"event_payload": record})
    return record


def reset_loggers() -> None:
    """Closes and forgets every cached per-run logger/handler. Real callers
    never need this (loggers live for the process's lifetime) -- it exists
    so tests get a fresh logger (and thus a fresh log file, honoring a
    just-changed AUTOCIM_LOG_DIR) instead of a stale cached one left over
    from an earlier test's tmp_path.

    Deliberately does *not* clear `_code_version_cache`: unlike the log
    dir, the code version is a real process-level constant (git HEAD can't
    change mid-test-run), so recomputing it (a `git` subprocess call) on
    every single test via the autouse fixture that calls this would be
    pure overhead. The one test exercising the git-unavailable fallback
    resets `_code_version_cache` itself."""
    for logger in _loggers.values():
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    _loggers.clear()
