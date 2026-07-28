"""Chat-model factory + call-operations for AutoCIM-Agent's LLM-backed nodes.

Provider/model choice is config-driven, mirroring CLAUDE.md 5.C's "load
hardware specs via HWConfig, never hardcode" principle at the LLM boundary:
defaults to a local Ollama model (_DEFAULT_LOCAL_MODEL) so a fresh checkout
never has to send hardware specs or candidate configs to a cloud API just to
run -- IP-sensitive deployments (chip vendors) get that off-network guarantee
for free. Set AUTOCIM_PLANNER_MODEL to any other "<provider>:<model>" string
`langchain.chat_models.init_chat_model` accepts (e.g.
"anthropic:claude-sonnet-4-5-20250929" or "openai:gpt-4.1") to opt into a
cloud provider instead -- a provider swap is still just an env var change,
not a code change.

`invoke_with_retry` is the operational layer around that raw chat model,
the LLM-call analogue of `middleware.wrap_tool_call`'s exception
containment for backend tools: @planner used to call `.invoke()` directly
and fall back to its search-point suggestion on *any* failure, including a
transient rate-limit/network blip that a real production loop should just
retry. It:

- retries only errors that look transient (rate limit / timeout /
  5xx-style / connection errors -- `_is_retryable`), with exponential
  backoff + jitter, honoring a provider-reported Retry-After delay when one
  is present. A malformed-response error (e.g. the model returned no tool
  call) is not retried here -- that is a deterministic failure retrying the
  same input will not fix; `nodes/planner.py` still falls back to the
  search-point suggestion for that case, unchanged from before.
- returns an `LLMCallRecord` alongside the response (or raises
  `LLMCallFailed`, which carries one) recording attempts, latency, and --
  when the underlying provider's response exposes it -- real token counts,
  so `AutoCIMState.llm_usage` (state.py) can accumulate real per-iteration
  LLM cost/attempt data across a run instead of that information only
  existing in provider-side logs.

`check_budget` is the enforcement side of that same accumulated
`llm_usage`: without it, retry/backoff alone can make a rate-limited or
degenerate run keep calling the LLM (and accumulating real cost)
indefinitely across `@evaluator`'s re-plan loop. It never talks to a
provider itself -- `nodes/planner.py` calls it *before* attempting an LLM
call each iteration and skips the call entirely (falling back to the
search-point suggestion, same as any other LLM failure) once a configured
`AUTOCIM_LLM_MAX_TOTAL_COST_USD`/`AUTOCIM_LLM_MAX_TOTAL_TOKENS` budget for
the run is spent.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

_MODEL_ENV_VAR = "AUTOCIM_PLANNER_MODEL"
# Local, tool-calling-verified default -- keeps hardware specs/candidate
# configs off any network by default. Requires `ollama serve` running with
# this model pulled (`ollama pull qwen2.5:7b`); set AUTOCIM_PLANNER_MODEL to
# override with a cloud provider spec instead.
_DEFAULT_LOCAL_MODEL = "ollama:qwen2.5:7b"


def get_planner_chat_model() -> BaseChatModel:
    model = os.environ.get(_MODEL_ENV_VAR) or _DEFAULT_LOCAL_MODEL
    return init_chat_model(model, temperature=0)


# =============================================================================
# Retry/backoff + usage tracking around real LLM calls
# =============================================================================

# Operational policy constants, not hardware specs (CLAUDE.md 5.C only
# prohibits hardcoding hardware numbers) -- still env-overridable so a
# deployment can tune retry/backoff without a code change, same spirit as
# AUTOCIM_PLANNER_MODEL above.
_MAX_RETRIES_ENV = "AUTOCIM_LLM_MAX_RETRIES"
_BASE_DELAY_ENV = "AUTOCIM_LLM_BASE_DELAY_SECONDS"
_MAX_DELAY_ENV = "AUTOCIM_LLM_MAX_DELAY_SECONDS"
# $/1K-token rates: no default -- provider pricing changes and varies per
# model, so guessing one here would silently report a fabricated cost.
# Real token *counts* (see LLMCallRecord) are always reported when the
# provider exposes them; cost is only ever computed from these if a
# deployment explicitly configures its own current rate.
_COST_PER_1K_INPUT_ENV = "AUTOCIM_LLM_COST_PER_1K_INPUT_USD"
_COST_PER_1K_OUTPUT_ENV = "AUTOCIM_LLM_COST_PER_1K_OUTPUT_USD"
# Per-run budget caps (see `check_budget` below) -- unset by default (no
# cap), since a hardcoded default here would silently throttle a
# deployment that never asked for one.
_MAX_TOTAL_COST_ENV = "AUTOCIM_LLM_MAX_TOTAL_COST_USD"
_MAX_TOTAL_TOKENS_ENV = "AUTOCIM_LLM_MAX_TOTAL_TOKENS"

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY_SECONDS = 1.0
_DEFAULT_MAX_DELAY_SECONDS = 20.0

# Substring markers for provider/transport errors that don't expose a
# structured status code (langchain wraps some provider SDK exceptions in
# plain RuntimeError/ConnectionError with only the message preserved) --
# matched case-insensitively against `str(exc)` as a fallback after
# `_status_code_of` finds nothing.
_RETRYABLE_MESSAGE_MARKERS = (
    "rate limit",
    "ratelimit",
    "429",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "service unavailable",
    "502",
    "503",
    "504",
)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class LLMCallRecord:
    """One real LLM call's operational stats -- appended to
    `AutoCIMState.llm_usage` (state.py) so cost/attempts/latency accumulate
    across a run instead of only existing in provider-side logs."""

    node: str
    iteration: int
    attempts: int
    succeeded: bool
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "iteration": self.iteration,
            "attempts": self.attempts,
            "succeeded": self.succeeded,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "error": self.error,
        }


class LLMCallFailed(RuntimeError):
    """Raised by `invoke_with_retry` once retries are exhausted (or the
    error wasn't retryable). Carries the `LLMCallRecord` so the caller can
    still log real attempts/latency for a failed call, not just a bare
    exception, before falling back."""

    def __init__(self, record: LLMCallRecord):
        super().__init__(record.error or "LLM call failed")
        self.record = record


def _status_code_of(exc: Exception) -> Optional[int]:
    """Best-effort status code from whatever shape the underlying provider
    SDK / langchain wrapper used -- checked defensively (no hard dependency
    on any one provider's exception classes) since this project supports
    swapping AUTOCIM_PLANNER_MODEL across providers (llm.py's own
    docstring)."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def _is_retryable(exc: Exception) -> bool:
    status_code = _status_code_of(exc)
    if status_code is not None:
        return status_code in _RETRYABLE_STATUS_CODES
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MESSAGE_MARKERS)


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """A provider-reported Retry-After delay, if one is present anywhere
    findable without a hard provider-SDK dependency -- preferred over our
    own backoff schedule when available, since it's the provider's own
    signal for when capacity will free up."""
    for attr in ("retry_after", "retry_after_seconds"):
        value = getattr(exc, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except AttributeError:
            value = None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _backoff_delay_seconds(exc: Exception, attempt: int, base_delay: float, max_delay: float) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(max(retry_after, 0.0), max_delay)
    # Exponential backoff (attempt 1 -> base_delay, attempt 2 -> 2x, ...)
    # with +/-50% jitter so concurrent callers don't retry in lockstep.
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return delay * (0.5 + random.random())


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _extract_token_usage(response: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Real token counts from whatever usage metadata `response` exposes --
    `usage_metadata` is langchain-core's normalized attribute across
    providers; `response_metadata`'s raw provider payload is the fallback
    for chat model integrations that don't populate it. Returns
    (None, None, None) rather than guessing when neither is present (e.g.
    `FakeToolCallingChatModel` in tests, which mimics neither)."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        usage = response_metadata.get("usage") or response_metadata.get("token_usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
            total_tokens = usage.get("total_tokens")
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            return input_tokens, output_tokens, total_tokens

    return None, None, None


def _estimate_cost_usd(input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    if input_tokens is None or output_tokens is None:
        return None
    input_rate = os.environ.get(_COST_PER_1K_INPUT_ENV)
    output_rate = os.environ.get(_COST_PER_1K_OUTPUT_ENV)
    if input_rate is None or output_rate is None:
        return None
    try:
        return round(input_tokens / 1000 * float(input_rate) + output_tokens / 1000 * float(output_rate), 6)
    except ValueError:
        return None


@dataclass
class BudgetStatus:
    exceeded: bool
    reason: Optional[str] = None


def check_budget(llm_usage: Sequence[Dict[str, Any]]) -> BudgetStatus:
    """Sums `llm_usage` (`AutoCIMState.llm_usage`, state.py -- one entry per
    past LLM call this run) against whichever of `AUTOCIM_LLM_MAX_TOTAL_COST_USD`/
    `AUTOCIM_LLM_MAX_TOTAL_TOKENS` is configured. Cost enforcement only works
    when `AUTOCIM_LLM_COST_PER_1K_*` is also configured (`_estimate_cost_usd`
    -- without a rate, `estimated_cost_usd` is always None, so a cost cap
    alone can never trip); the token cap works regardless, since real token
    counts come straight from the provider's response whenever it exposes
    them. Neither configured (the default) -> never exceeded, i.e. today's
    unbounded behavior."""
    max_cost_raw = os.environ.get(_MAX_TOTAL_COST_ENV)
    max_tokens_raw = os.environ.get(_MAX_TOTAL_TOKENS_ENV)
    if not max_cost_raw and not max_tokens_raw:
        return BudgetStatus(exceeded=False)

    total_cost = sum(u["estimated_cost_usd"] for u in llm_usage if u.get("estimated_cost_usd") is not None)
    total_tokens = sum(u["total_tokens"] for u in llm_usage if u.get("total_tokens") is not None)

    if max_cost_raw:
        try:
            max_cost = float(max_cost_raw)
        except ValueError:
            max_cost = None
        if max_cost is not None and total_cost >= max_cost:
            return BudgetStatus(
                exceeded=True, reason=f"cumulative estimated LLM cost ${total_cost:.4f} >= budget ${max_cost:.4f}"
            )

    if max_tokens_raw:
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError:
            max_tokens = None
        if max_tokens is not None and total_tokens >= max_tokens:
            return BudgetStatus(exceeded=True, reason=f"cumulative LLM tokens {total_tokens} >= budget {max_tokens}")

    return BudgetStatus(exceeded=False)


def invoke_with_retry(
    chat_model: Any,
    messages: List[Any],
    *,
    node: str,
    iteration: int,
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Tuple[Any, LLMCallRecord]:
    """`chat_model.invoke(messages)`, retried with backoff on transient
    errors (`_is_retryable`), returning `(response, LLMCallRecord)` on
    success. Raises `LLMCallFailed` (carrying the record) once retries are
    exhausted or the error isn't retryable -- the caller (`nodes/planner.py`)
    still falls back to the search-point suggestion in that case, same as
    before this wrapper existed; this only adds real retry/backoff for the
    transient-error case that used to fall back on the very first blip.

    `max_retries`/`base_delay`/`max_delay` default to the
    `AUTOCIM_LLM_*` env vars (see module docstring) when not passed
    explicitly -- `sleep_fn` is the only pure-test seam (tests inject a
    no-op so retry tests don't actually wait out a real backoff)."""
    max_retries = max_retries if max_retries is not None else _int_env(_MAX_RETRIES_ENV, _DEFAULT_MAX_RETRIES)
    base_delay = base_delay if base_delay is not None else _float_env(_BASE_DELAY_ENV, _DEFAULT_BASE_DELAY_SECONDS)
    max_delay = max_delay if max_delay is not None else _float_env(_MAX_DELAY_ENV, _DEFAULT_MAX_DELAY_SECONDS)

    start = time.monotonic()
    last_exc: Optional[Exception] = None
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            response = chat_model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 -- provider/transport errors, contained like wrap_tool_call
            last_exc = exc
            if attempt >= max_retries or not _is_retryable(exc):
                break
            sleep_fn(_backoff_delay_seconds(exc, attempt, base_delay, max_delay))
            continue

        input_tokens, output_tokens, total_tokens = _extract_token_usage(response)
        record = LLMCallRecord(
            node=node,
            iteration=iteration,
            attempts=attempt,
            succeeded=True,
            latency_ms=(time.monotonic() - start) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=_estimate_cost_usd(input_tokens, output_tokens),
        )
        return response, record

    record = LLMCallRecord(
        node=node,
        iteration=iteration,
        attempts=attempt,
        succeeded=False,
        latency_ms=(time.monotonic() - start) * 1000,
        error=f"{type(last_exc).__name__}: {last_exc}" if last_exc is not None else "unknown error",
    )
    raise LLMCallFailed(record) from last_exc
