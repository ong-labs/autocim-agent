"""@planner LLM decision schema.

`PlannerLayerDecision` is registered as a forced tool call for the
@planner LLM (Research_Plan.md 3: "LLM은 오직 파라미터 판단 및 의사결정
(tool_calls)만 수행합니다"). It reuses `LayerBitConfig` from schemas/tools.py
so the LLM's output validates directly against @tuner's `TunerToolInput`
contract, with no separate translation step between planner's decision and
tuner's input.
"""

from typing import List

from pydantic import BaseModel, Field

from schemas.tools import LayerBitConfig


class PlannerLayerDecision(BaseModel):
    """Next iteration's layer-wise quantization/pruning proposal."""

    layer_configs: List[LayerBitConfig] = Field(..., min_length=1)
    rationale: str = Field(
        ..., min_length=1, description="Brief justification referencing hw bounds and failure_history"
    )
