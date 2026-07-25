"""llm.py's `invoke_with_retry`: retry/backoff on transient LLM-call errors,
real token/cost tracking -- the operational layer @planner needs for actual
production use (real rate limits, real per-iteration LLM cost) instead of
falling back to its search-point suggestion on the very first blip.

No test here makes a real network call: `chat_model` is always a tiny fake
exposing only `.invoke(messages)`, and `sleep_fn` is always a no-op so
"retries actually happen" tests don't wait out a real backoff.
"""

import pytest

from llm import LLMCallFailed, check_budget, invoke_with_retry


class _FailNTimesThenSucceed:
    def __init__(self, exceptions, response):
        self._exceptions = list(exceptions)
        self._response = response
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self._exceptions:
            raise self._exceptions.pop(0)
        return self._response


class _AlwaysRaises:
    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise self._exc


class _FakeResponseWithUsage:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


def _no_sleep(seconds):
    pass


# --- Retry on transient errors ----------------------------------------------


def test_retries_on_rate_limit_style_error_then_succeeds():
    model = _FailNTimesThenSucceed([RuntimeError("429 rate limit exceeded")], _FakeResponseWithUsage())

    response, record = invoke_with_retry(
        model, ["hi"], node="planner", iteration=1, max_retries=3, base_delay=0.01, max_delay=0.01, sleep_fn=_no_sleep
    )

    assert model.calls == 2
    assert record.attempts == 2
    assert record.succeeded is True


def test_does_not_retry_a_non_retryable_error():
    model = _AlwaysRaises(ValueError("invalid request: missing required field"))

    with pytest.raises(LLMCallFailed) as exc_info:
        invoke_with_retry(model, ["hi"], node="planner", iteration=1, max_retries=5, base_delay=0.01, sleep_fn=_no_sleep)

    assert model.calls == 1
    assert exc_info.value.record.attempts == 1
    assert exc_info.value.record.succeeded is False


def test_raises_llm_call_failed_after_exhausting_retries_on_persistent_transient_error():
    model = _AlwaysRaises(RuntimeError("503 service unavailable"))

    with pytest.raises(LLMCallFailed) as exc_info:
        invoke_with_retry(model, ["hi"], node="planner", iteration=1, max_retries=3, base_delay=0.01, sleep_fn=_no_sleep)

    assert model.calls == 3
    assert exc_info.value.record.attempts == 3
    assert exc_info.value.record.succeeded is False
    assert "503" in exc_info.value.record.error


def test_retryable_status_code_attribute_is_detected_without_message_markers():
    class _StatusCodeError(Exception):
        def __init__(self):
            super().__init__("something broke")
            self.status_code = 429

    model = _FailNTimesThenSucceed([_StatusCodeError()], _FakeResponseWithUsage())

    _, record = invoke_with_retry(
        model, ["hi"], node="planner", iteration=1, max_retries=3, base_delay=0.01, sleep_fn=_no_sleep
    )
    assert record.attempts == 2


def test_retry_after_header_overrides_backoff_schedule():
    class _Response:
        headers = {"retry-after": "5"}

    class _RateLimitError(Exception):
        def __init__(self):
            super().__init__("429 rate limit")
            self.response = _Response()

    model = _FailNTimesThenSucceed([_RateLimitError()], _FakeResponseWithUsage())
    waited = []

    _, record = invoke_with_retry(
        model, ["hi"], node="planner", iteration=1, max_retries=3, base_delay=0.01, max_delay=10.0, sleep_fn=waited.append
    )
    assert waited == [5.0]


# --- Real token/cost tracking -------------------------------------------------


def test_extracts_token_usage_from_usage_metadata():
    model = _FailNTimesThenSucceed(
        [], _FakeResponseWithUsage(usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    )

    _, record = invoke_with_retry(model, ["hi"], node="planner", iteration=1, max_retries=1, sleep_fn=_no_sleep)

    assert record.input_tokens == 100
    assert record.output_tokens == 50
    assert record.total_tokens == 150


def test_no_usage_metadata_reports_none_rather_than_guessing():
    class _BareResponse:
        pass

    model = _FailNTimesThenSucceed([], _BareResponse())
    _, record = invoke_with_retry(model, ["hi"], node="planner", iteration=1, max_retries=1, sleep_fn=_no_sleep)

    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.estimated_cost_usd is None


def test_estimated_cost_uses_configured_rates_when_present(monkeypatch):
    monkeypatch.setenv("AUTOCIM_LLM_COST_PER_1K_INPUT_USD", "0.003")
    monkeypatch.setenv("AUTOCIM_LLM_COST_PER_1K_OUTPUT_USD", "0.015")
    model = _FailNTimesThenSucceed(
        [], _FakeResponseWithUsage(usage_metadata={"input_tokens": 1000, "output_tokens": 1000, "total_tokens": 2000})
    )

    _, record = invoke_with_retry(model, ["hi"], node="planner", iteration=1, max_retries=1, sleep_fn=_no_sleep)

    assert record.estimated_cost_usd == pytest.approx(0.018)


def test_estimated_cost_is_none_without_configured_rates(monkeypatch):
    monkeypatch.delenv("AUTOCIM_LLM_COST_PER_1K_INPUT_USD", raising=False)
    monkeypatch.delenv("AUTOCIM_LLM_COST_PER_1K_OUTPUT_USD", raising=False)
    model = _FailNTimesThenSucceed(
        [], _FakeResponseWithUsage(usage_metadata={"input_tokens": 1000, "output_tokens": 1000, "total_tokens": 2000})
    )

    _, record = invoke_with_retry(model, ["hi"], node="planner", iteration=1, max_retries=1, sleep_fn=_no_sleep)

    assert record.estimated_cost_usd is None


# --- Budget cap / kill-switch -------------------------------------------------


def test_check_budget_never_exceeded_when_unconfigured(monkeypatch):
    monkeypatch.delenv("AUTOCIM_LLM_MAX_TOTAL_COST_USD", raising=False)
    monkeypatch.delenv("AUTOCIM_LLM_MAX_TOTAL_TOKENS", raising=False)

    status = check_budget([{"estimated_cost_usd": 999.0, "total_tokens": 999999}])

    assert status.exceeded is False


def test_check_budget_trips_on_cumulative_cost(monkeypatch):
    monkeypatch.setenv("AUTOCIM_LLM_MAX_TOTAL_COST_USD", "1.00")
    usage = [{"estimated_cost_usd": 0.6}, {"estimated_cost_usd": 0.5}]

    status = check_budget(usage)

    assert status.exceeded is True
    assert "1.1000" in status.reason


def test_check_budget_trips_on_cumulative_tokens(monkeypatch):
    monkeypatch.delenv("AUTOCIM_LLM_MAX_TOTAL_COST_USD", raising=False)
    monkeypatch.setenv("AUTOCIM_LLM_MAX_TOTAL_TOKENS", "1000")
    usage = [{"total_tokens": 600}, {"total_tokens": 500}]

    status = check_budget(usage)

    assert status.exceeded is True
    assert "1100" in status.reason


def test_check_budget_not_exceeded_below_either_cap(monkeypatch):
    monkeypatch.setenv("AUTOCIM_LLM_MAX_TOTAL_COST_USD", "10.00")
    monkeypatch.setenv("AUTOCIM_LLM_MAX_TOTAL_TOKENS", "100000")
    usage = [{"estimated_cost_usd": 0.1, "total_tokens": 500}]

    status = check_budget(usage)

    assert status.exceeded is False


def test_check_budget_ignores_entries_with_no_cost_data(monkeypatch):
    """A cost cap can't be enforced from entries that never got a real
    estimated_cost_usd (e.g. AUTOCIM_LLM_COST_PER_1K_* was never
    configured) -- those entries must not be silently treated as $0 and
    diluted into a false sense of being under budget forever, but they also
    must not crash the sum."""
    monkeypatch.setenv("AUTOCIM_LLM_MAX_TOTAL_COST_USD", "0.01")
    usage = [{"estimated_cost_usd": None, "total_tokens": 100}, {"estimated_cost_usd": 0.02, "total_tokens": 50}]

    status = check_budget(usage)

    assert status.exceeded is True  # the one entry with real cost data alone already exceeds 0.01
