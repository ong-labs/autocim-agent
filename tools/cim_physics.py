"""Physically-grounded analytical models for @mapper / @profiler.

Replaces the earlier placeholder arithmetic in tools/simulators.py with real
device-physics and NoC-architecture scaling relationships:

- NoC routing: real topology hop-count graph distances (mesh Manhattan
  distance, torus wraparound, ring shortest-arc, bus single-hop) and
  bandwidth-limited serialization time -- no arbitrary latency constant
  beyond a standard NoC flit header size.
- CIM profiling (NeuroSim/ISAAC-literature scaling laws): ADC energy scales
  exponentially with resolution (flash-ADC dominated), DAC energy scales
  linearly with resolution, crossbar read/MAC energy scales with array size
  and weight bit-width (bit-serial CIM), and per-layer compute latency is
  bounded by ADC/DAC conversion time -- exploiting CIM's defining property
  that one crossbar performs a full row x col MAC in parallel per cycle.

`CIM_TECH_PARAMS` are representative order-of-magnitude reference values,
not silicon-measured numbers for any specific process node -- this is
exactly why `AutoCIMState.calibration_factors` exists (Research_Plan.md 3):
these are a *fast approximation*, meant to be scaled by a per-hw_spec_id
factor learned against precise simulation over time (applied in
tools/simulators.py's profiler_tool), not treated as ground truth on their
own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from schemas.config import HWConfig, NoCTopology
from schemas.tools import LayerBitConfig


@dataclass(frozen=True)
class CIMTechParams:
    """Reference (uncalibrated) technology constants.

    Order-of-magnitude placeholders, not values fit or validated against any
    specific real chip -- there is no sensitivity analysis here showing
    downstream Pareto rankings are robust to a different-but-plausible set
    of these numbers. `calibration_factors`/`calibration_provenance`
    (tools/calibration.py) is this project's actual honesty mechanism for
    that gap: any `hw_spec_id` without a real published-reference match
    stays visibly "uncalibrated" (tools/dashboard.py) rather than silently
    trusting these constants' absolute scale."""

    adc_ref_energy_fj: float = 0.2  # energy of a 1-bit-equivalent flash-ADC conversion step
    dac_ref_energy_fj: float = 0.05  # energy per DAC bit driven
    crossbar_ref_energy_fj: float = 0.01  # energy per (cell * weight-bit) MAC
    adc_cycle_time_ns: float = 0.1  # time per SAR-ADC bit-cycle
    dac_settle_time_ns: float = 0.05  # DAC settling time per bit
    router_flit_header_bits: int = 32  # standard NoC flit header overhead per hop


CIM_TECH_PARAMS = CIMTechParams()


# ---------------------------------------------------------------------------
# NoC topology: real hop-count graph distance formulas
# ---------------------------------------------------------------------------


def _tile_coordinates(index: int, grid_dim: int) -> Tuple[int, int]:
    return index // grid_dim, index % grid_dim


def hop_distance(topology: NoCTopology, i: int, j: int, ring_or_grid_size: int) -> int:
    """Real graph hop-count between tile `i` and tile `j` under `topology`,
    where `ring_or_grid_size` is the physical tile count the NoC is built
    for (`HWConfig.num_tiles`), independent of how many tiles a given
    workload actually uses."""
    if i == j:
        return 0
    if topology == NoCTopology.CROSSBAR_BUS:
        return 1  # shared bus: every node is one hop from every other
    if topology == NoCTopology.RING:
        direct = abs(i - j)
        return min(direct, ring_or_grid_size - direct)

    grid_dim = max(1, math.ceil(math.sqrt(ring_or_grid_size)))
    xi, yi = _tile_coordinates(i, grid_dim)
    xj, yj = _tile_coordinates(j, grid_dim)
    dx, dy = abs(xi - xj), abs(yi - yj)
    if topology == NoCTopology.TORUS:
        dx = min(dx, grid_dim - dx)
        dy = min(dy, grid_dim - dy)
    return dx + dy  # MESH, and TORUS after wraparound folding


# ---------------------------------------------------------------------------
# @mapper: multi-tile placement + pipelined NoC transfer latency
# ---------------------------------------------------------------------------


def simulate_noc_mapping(
    hw: HWConfig, layer_configs: List[LayerBitConfig], params: CIMTechParams = CIM_TECH_PARAMS
) -> Dict[str, Any]:
    """Assumption: one crossbar tile per `layer_configs` entry (a tile's
    `crossbar_rows` x `crossbar_cols` array is assumed large enough for one
    layer's pruned/quantized weight matrix), placed in `layer_configs` order
    and pipelined tile-to-tile -- the standard layer-pipelined CIM dataflow.
    Real per-hop topology distance (`hop_distance`) and bandwidth-limited
    transfer time; `noc_link_bandwidth_gbps` (Gb/s) converts to bits/ns
    directly (1 Gb/s == 1 bit/ns), so no unit-conversion constant is needed
    beyond the documented flit header size.
    """
    tiles_needed = len(layer_configs)
    bandwidth_bits_per_ns = hw.noc_link_bandwidth_gbps

    total_latency_ns = 0.0
    for idx in range(1, tiles_needed):
        hops = hop_distance(hw.noc_topology, idx - 1, idx, hw.num_tiles)
        payload_bits = hw.crossbar_cols * layer_configs[idx].activation_bits
        serialization_ns = payload_bits / bandwidth_bits_per_ns
        header_ns = hops * params.router_flit_header_bits / bandwidth_bits_per_ns
        total_latency_ns += serialization_ns + header_ns

    buffer_bits = (hw.sram_buffer_kb or 0.0) * 8 * 1024
    buffer_delay_ns = buffer_bits / bandwidth_bits_per_ns if buffer_bits else 0.0

    return {
        "noc_latency_ms": round(total_latency_ns / 1e6, 6),
        "buffer_delay_ms": round(buffer_delay_ns / 1e6, 6),
        "tile_utilization": round(min(1.0, tiles_needed / hw.num_tiles), 4),
        "tiles_needed": tiles_needed,
    }


# ---------------------------------------------------------------------------
# @profiler: CIM device-physics energy/latency model
# ---------------------------------------------------------------------------


def simulate_cim_profile(
    hw: HWConfig, layer_configs: List[LayerBitConfig], params: CIMTechParams = CIM_TECH_PARAMS
) -> Dict[str, Any]:
    """See module docstring for the scaling relationships. `energy_pj` sums
    per-layer ADC + DAC + crossbar-MAC energy across all `layer_configs`
    entries; `latency_ns` sums per-layer ADC/DAC conversion time, since a
    CIM crossbar's row x col MAC itself completes in parallel per cycle
    (the array-size term does not appear in the latency model, only in
    energy)."""
    total_energy_fj = 0.0
    total_latency_ns = 0.0

    for lc in layer_configs:
        adc_energy_fj = params.adc_ref_energy_fj * (2**hw.adc_bits) * hw.crossbar_cols
        dac_energy_fj = params.dac_ref_energy_fj * hw.dac_bits * hw.crossbar_rows
        crossbar_energy_fj = params.crossbar_ref_energy_fj * hw.crossbar_rows * hw.crossbar_cols * lc.weight_bits
        total_energy_fj += adc_energy_fj + dac_energy_fj + crossbar_energy_fj
        total_latency_ns += hw.adc_bits * params.adc_cycle_time_ns + hw.dac_bits * params.dac_settle_time_ns

    return {
        "energy_pj": round(total_energy_fj / 1000.0, 6),
        "latency_ns": round(total_latency_ns, 6),
    }
