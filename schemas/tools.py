"""Tool input/output validation schemas for AutoCIM-Agent.

Every backend tool (@tuner, @mapper, @profiler, @verifier, @search_paper)
is validated through `wrap_tool_call` using the models below. Inputs carry
an `ExecutionContext` + `HWConfig` pair instead of hardcoded model/hardware
values, and outputs share a common envelope so they merge cleanly into
`AutoCIMState.metrics_store` (see CLAUDE.md 5.A/5.C).
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from schemas.config import ExecutionContext, HWConfig


class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ToolResult(BaseModel):
    """Common output envelope, matching the metrics_store[node] shape."""

    status: ToolStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# --- @tuner: PyTorch QAT / crossbar-aligned pruning tool --------------------


class LayerBitConfig(BaseModel):
    layer_name: str
    weight_bits: int = Field(..., gt=0, le=32)
    activation_bits: int = Field(..., gt=0, le=32)
    column_pruning_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)


class TunerToolInput(BaseModel):
    execution_context: ExecutionContext
    hw_config: HWConfig
    layer_configs: List[LayerBitConfig] = Field(..., min_length=1)
    max_epochs: int = Field(..., gt=0)
    early_stopping_patience: Optional[int] = Field(default=None, gt=0)


class TunerToolOutput(ToolResult):
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "e.g. {'accuracy': float, 'epochs_run': int, 'model_ref': str, "
            "'layer_configs': list[dict]} -- accuracy is real measured "
            "post-QAT accuracy (tools/qat.py); layer_configs is the "
            "validated input echoed back so @mapper/@profiler simulate "
            "against the exact configuration @tuner actually used. No "
            "pareto_rank here -- a single QAT trial has no basis to rank "
            "itself; @evaluator computes that against `candidate_history` "
            "(tools/search.py) once this iteration's full multi-objective "
            "result is known."
        ),
    )


# --- @mapper: Multi-Tile / NoC mapping simulator ----------------------------


class MapperToolInput(BaseModel):
    execution_context: ExecutionContext
    hw_config: HWConfig
    model_ref: str = Field(..., description="Reference to the tuned model artifact produced by @tuner")
    layer_configs: List[LayerBitConfig] = Field(
        ..., min_length=1, description="The layer_configs @tuner actually used, echoed via TunerToolOutput.data"
    )
    tiling_strategy: Optional[str] = None


class MapperToolOutput(ToolResult):
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "e.g. {'noc_latency_ms': float, 'buffer_delay_ms': float, "
            "'tile_utilization': float, 'tiles_needed': int}"
        ),
    )


# --- @profiler: NeuroSim / CIM-Loop profiling tool --------------------------


class ProfilerToolInput(BaseModel):
    execution_context: ExecutionContext
    hw_config: HWConfig
    model_ref: str
    layer_configs: List[LayerBitConfig] = Field(
        ..., min_length=1, description="The layer_configs @tuner actually used, echoed via TunerToolOutput.data"
    )
    calibration_factors: Optional[Dict[str, float]] = None


class ProfilerToolOutput(ToolResult):
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="e.g. {'energy_pj': float, 'latency_ns': float, 'calibration_factors': dict}"
    )


# --- @verifier: fast-approximation / IR-drop & noise verification ----------


class VerifierToolInput(BaseModel):
    execution_context: ExecutionContext
    hw_config: HWConfig
    mapper_result: MapperToolOutput
    profiler_result: ProfilerToolOutput
    input_distribution_ref: Optional[str] = None


class VerifierToolOutput(ToolResult):
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="e.g. {'ir_drop_error_pct': float, 'noise_margin_db': float, 'is_converged': bool}"
    )


# --- @search_paper: external literature search tool -------------------------


class SearchPaperToolInput(BaseModel):
    execution_context: ExecutionContext
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, gt=0, le=50)


class SearchPaperResult(BaseModel):
    title: str
    url: Optional[str] = None
    summary: Optional[str] = None


class SearchPaperToolOutput(ToolResult):
    data: Optional[Dict[str, List[SearchPaperResult]]] = None
