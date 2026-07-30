"""@planner LLM decision schema.

`PlannerLayerDecision` is registered as a forced tool call for the
@planner LLM. Deliberately does NOT include the numeric layer_configs
proposal Research_Plan.md 3 ("LLM은 오직 파라미터 판단 및 의사결정
(tool_calls)만 수행합니다") originally assigned the LLM -- a real
production run (this session) caught a local tool-calling model reasoning
backwards about which direction to adjust weight_bits for an accuracy
shortfall (reducing weight_bits "to improve accuracy", which does the
opposite), silently producing a worse candidate than the surrogate's own
suggestion. `tools/search.py`'s LHS/NSGA-II search already decides the
actual (weight_bits, column_pruning_ratio) per stage before @planner's LLM
is ever called (nodes/planner.py's `stage_configs`); the LLM's role is
narrowed to explaining that already-decided candidate and flagging
anything concerning enough for a researcher to review immediately, not
judging or adjusting the numbers themselves.
"""

from typing import Optional

from pydantic import BaseModel, Field


class PlannerLayerDecision(BaseModel):
    """@planner's per-iteration explanation of (not a numeric alternative
    to) the search-decided candidate."""

    rationale: str = Field(
        ..., min_length=1, description="Brief justification for the given per_stage_suggested_candidate, referencing hw bounds and failure_history"
    )
    # Optional (not required) so older recorded tool calls / fakes that
    # predate this field still validate -- tools/dashboard.py falls back to
    # `rationale` (English) when this is absent.
    rationale_ko: Optional[str] = Field(
        default=None, description="The same justification as `rationale`, translated into Korean"
    )
    anomaly_note: Optional[str] = Field(
        default=None,
        description=(
            "Set ONLY if recent_failure_history or the current candidate shows something genuinely "
            "concerning enough to warrant immediate researcher review -- e.g. the exact same candidate "
            "failing repeatedly with no change, or physically implausible metrics. Leave null otherwise; "
            "this is not for routine commentary."
        ),
    )
