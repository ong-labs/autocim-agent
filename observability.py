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
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

_LOG_DIR_ENV = "AUTOCIM_LOG_DIR"
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / ".cache" / "logs"

_loggers: Dict[str, logging.Logger] = {}


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
    from an earlier test's tmp_path."""
    for logger in _loggers.values():
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    _loggers.clear()
