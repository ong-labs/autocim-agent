"""Configuration schemas for AutoCIM-Agent.

Hardware constraints and run-scoped identifiers live here. Node/tool logic
must load these values through `HWConfig`/`ExecutionContext` instances
rather than hardcoding hardware numbers or model architecture details
(see CLAUDE.md 5.C).
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NoCTopology(str, Enum):
    MESH = "mesh"
    TORUS = "torus"
    RING = "ring"
    CROSSBAR_BUS = "crossbar_bus"


class HWConfig(BaseModel):
    """Hardware constraint specification for a target CIM chip.

    An instance is resolved (e.g. from a hardware spec store) using
    `ExecutionContext.hw_spec_id` and passed into tools explicitly --
    values are never hardcoded in node/tool logic.
    """

    hw_spec_id: str = Field(..., description="Unique identifier for this hardware spec, e.g. 'cim_v1_128x128'")

    # Crossbar array
    crossbar_rows: int = Field(..., gt=0, description="Rows per crossbar tile")
    crossbar_cols: int = Field(..., gt=0, description="Columns per crossbar tile")
    num_tiles: int = Field(..., gt=0, description="Number of crossbar tiles available on-chip")

    # Converters
    adc_bits: int = Field(..., gt=0, le=16, description="ADC resolution in bits")
    dac_bits: int = Field(..., gt=0, le=16, description="DAC resolution in bits")

    # Interconnect
    noc_topology: NoCTopology = Field(..., description="Network-on-Chip topology")
    noc_link_bandwidth_gbps: float = Field(..., gt=0, description="Per-link NoC bandwidth in Gbps")

    # Device non-idealities (optional: not every hardware spec models these)
    wire_resistance_ohm_per_um: Optional[float] = Field(
        default=None, ge=0, description="Wire resistance used for IR-drop estimation"
    )
    device_noise_sigma: Optional[float] = Field(
        default=None, ge=0, description="Std-dev of device-level conductance noise"
    )
    sram_buffer_kb: Optional[float] = Field(default=None, gt=0, description="Per-tile SRAM buffer size in KB")

    model_config = {"frozen": True}


class ExecutionContext(BaseModel):
    """Run-scoped metadata injected into tools via ToolRuntime.

    Carries identifiers only. Tools resolve `model_id` -> the actual
    nn.Module/ONNX graph and `hw_spec_id` -> an `HWConfig` instance via
    their own registries/stores -- this model never carries the resolved
    objects itself, keeping tool logic model-/hardware-agnostic.
    """

    session_id: str = Field(..., description="Unique id for this LangGraph run/thread")
    model_id: str = Field(..., description="Identifier for the target model, resolved externally")
    hw_spec_id: str = Field(..., description="Identifier for the target HWConfig, resolved externally")
    iteration_count: int = Field(default=0, ge=0)
    user_id: Optional[str] = Field(default=None, description="Requesting researcher/user id, if applicable")
