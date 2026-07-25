"""Tool interceptor and context-management middleware for AutoCIM-Agent.

Two independent concerns live here (CLAUDE.md 5.D):

1. `wrap_tool_call` -- the single choke point every backend tool (PyTorch /
   NeuroSim / CIM-Loop wrapper) passes through. It validates LLM-supplied
   arguments against a Pydantic schema, overwrites the run's
   `ExecutionContext`/`HWConfig` onto the call (never trusting LLM-supplied
   values for these -- this is what Research_Plan.md 3 means by preventing
   "prompt pollution and hallucination"), and converts any exception
   (Python or a bound C-extension's) into a structured `ToolResult` instead
   of propagating and breaking the agent loop.

2. Context Pruning / Summarization -- bounds how much history is handed to
   the LLM on each turn, to prevent token overflow.
"""

from __future__ import annotations

import contextvars
import functools
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from schemas.config import ExecutionContext, HWConfig
from schemas.tools import ToolResult, ToolStatus

InputT = TypeVar("InputT", bound=BaseModel)

# =============================================================================
# 1. wrap_tool_call: run-scoped context + validation + exception interceptor
# =============================================================================

_execution_context_var: contextvars.ContextVar[Optional[ExecutionContext]] = contextvars.ContextVar(
    "autocim_execution_context", default=None
)
_hw_config_registry: Dict[str, HWConfig] = {}


def set_execution_context(context: ExecutionContext) -> contextvars.Token:
    """Called once per graph run/step (e.g. by @planner) before any tool
    call, so every tool call in that run is auto-injected with the same
    identifiers -- the LLM never has to (and is not trusted to) supply them."""
    return _execution_context_var.set(context)


def reset_execution_context(token: contextvars.Token) -> None:
    _execution_context_var.reset(token)


def get_execution_context() -> ExecutionContext:
    context = _execution_context_var.get()
    if context is None:
        raise RuntimeError("No ExecutionContext set for this run; call set_execution_context() first.")
    return context


def register_hw_config(hw_config: HWConfig) -> None:
    """Register a resolved HWConfig so tools can look it up by hw_spec_id
    instead of hardcoding hardware numbers (CLAUDE.md 5.C)."""
    _hw_config_registry[hw_config.hw_spec_id] = hw_config


def get_hw_config(hw_spec_id: str) -> HWConfig:
    try:
        return _hw_config_registry[hw_spec_id]
    except KeyError as exc:
        raise RuntimeError(f"No HWConfig registered for hw_spec_id={hw_spec_id!r}") from exc


def wrap_tool_call(
    input_schema: Type[InputT], node_name: str
) -> Callable[[Callable[[InputT], ToolResult]], Callable[..., Dict[str, Any]]]:
    """Decorator factory for backend tools.

    The wrapped function must accept a single validated `input_schema`
    instance and return a `ToolResult` (or subclass) instance. The
    decorated callable instead accepts raw kwargs (as an LLM tool-calling
    layer would supply) and always returns `{node_name: ToolResult_dict}`,
    ready to merge into `AutoCIMState.metrics_store` via
    `namespaced_metrics_reducer` (state.py).
    """

    def decorator(tool_fn: Callable[[InputT], ToolResult]) -> Callable[..., Dict[str, Any]]:
        @functools.wraps(tool_fn)
        def wrapper(**raw_kwargs: Any) -> Dict[str, Any]:
            raw_kwargs = dict(raw_kwargs)

            # Context injection + validation share one try block: a failure
            # to resolve ExecutionContext/HWConfig must be contained exactly
            # like a validation error, not propagate and crash the caller.
            try:
                # Auto-inject run-scoped metadata, discarding any LLM-supplied
                # values for these two keys.
                execution_context = get_execution_context()
                raw_kwargs["execution_context"] = execution_context
                if "hw_config" in input_schema.model_fields:
                    raw_kwargs["hw_config"] = get_hw_config(execution_context.hw_spec_id)

                validated_input = input_schema(**raw_kwargs)
            except ValidationError as exc:
                result = ToolResult(status=ToolStatus.FAILED, data=None, error=f"ValidationError: {exc}")
                return {node_name: result.model_dump(mode="json")}
            except Exception as exc:  # noqa: BLE001 -- e.g. unresolved hw_spec_id/session setup
                result = ToolResult(
                    status=ToolStatus.FAILED,
                    data=None,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
                return {node_name: result.model_dump(mode="json")}

            try:
                result = tool_fn(validated_input)
            except Exception as exc:  # noqa: BLE001 -- intentionally broad: contain any
                # Python or C-extension (NeuroSim/CIM-Loop) exception here so
                # a single failed tool call never crashes the agent loop.
                result = ToolResult(
                    status=ToolStatus.FAILED,
                    data=None,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )

            # mode="json" so enums (e.g. ToolStatus) land in metrics_store as
            # plain strings -- AutoCIMState gets checkpointed (MemorySaver
            # etc.), and a live Enum instance in a Pydantic-model dump is not
            # msgpack/JSON-safe by default.
            return {node_name: result.model_dump(mode="json")}

        wrapper.__wrapped_tool__ = tool_fn  # raw function, for direct unit-testing
        return wrapper

    return decorator


# =============================================================================
# 2. Context Pruning / Summarization -- bound what's fed to the LLM
# =============================================================================
#
# NOTE: AutoCIMState["messages"] uses an append-only `operator.add` reducer
# (state.py), so a graph node cannot shrink it by returning a smaller list --
# that would only ever grow it. These helpers therefore operate as a
# *read-time* view: called right before building the next LLM prompt, not
# persisted back into state.


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", message))
    return str(message)


def default_summarizer(messages: List[Any], max_chars: int) -> str:
    """Mock-First (CLAUDE.md 5.D) summarizer: truncated concatenation, no
    LLM call. Swap for a real LLM-backed summarizer later -- callers only
    depend on the `Callable[[List[Any], int], str]` signature."""
    joined = " | ".join(_message_text(m) for m in messages)
    return truncate_text(joined, max_chars)


@dataclass
class ContextPruningConfig:
    max_recent_messages: int = 20
    """Most-recent messages kept verbatim before pruning kicks in."""
    summarize_before_drop: bool = True
    """If True, messages beyond the window are folded into one summary
    message instead of being silently dropped."""
    summary_max_chars: int = 1000
    summarizer_fn: Callable[[List[Any], int], str] = field(default=default_summarizer)


def get_pruned_context(messages: List[Any], config: Optional[ContextPruningConfig] = None) -> List[Any]:
    """Bounded view of `messages` for the next LLM call."""
    config = config or ContextPruningConfig()

    if len(messages) <= config.max_recent_messages:
        return list(messages)

    cutoff = len(messages) - config.max_recent_messages
    dropped, kept = messages[:cutoff], messages[cutoff:]

    if not config.summarize_before_drop:
        return kept

    summary_text = config.summarizer_fn(dropped, config.summary_max_chars)
    summary_message = {
        "role": "system",
        "content": f"[context summary of {len(dropped)} earlier messages] {summary_text}",
    }
    return [summary_message, *kept]


def summarize_tool_result_for_messages(
    node_name: str, result: Dict[str, Any], max_chars: int = 500
) -> Dict[str, Any]:
    """Compact, token-bounded version of a tool result for appending to
    `messages`. `AutoCIMState.metrics_store` keeps the full-precision
    result separately for downstream numeric logic (@evaluator etc.) --
    this pruning only affects what the LLM sees."""
    error = result.get("error")
    content = {
        "status": result.get("status"),
        "data": result.get("data"),
        "error": truncate_text(error, max_chars) if error else None,
    }
    return {"role": "tool", "name": node_name, "content": truncate_text(str(content), max_chars)}
