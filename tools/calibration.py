"""Calibrates tools/cim_physics.py's analytical energy model against real
published CIM accelerator data.

`CIM_TECH_PARAMS` (tools/cim_physics.py) are representative order-of-
magnitude reference constants, not silicon-measured for any specific
process node -- Research_Plan.md 3's `calibration_factors` mechanism exists
precisely to correct that fast-approximation against precise/real numbers
over time. This module is the first real feed into that loop: `KNOWN_REFERENCES`
holds actual published data points; `bootstrap_calibration_factors` computes
a per-hw_spec_id correction factor (real reference energy / this model's
predicted energy for the matching config) and seeds
`AutoCIMState.calibration_factors` with it (main.py) so @profiler's output
is corrected against real data from the first iteration, for any HWConfig
whose (crossbar_rows, crossbar_cols, adc_bits) exactly matches a known
reference.

Deliberately exact-match only -- no interpolation/extrapolation across array
sizes or ADC resolutions. ADC energy scales exponentially with resolution in
this model, so applying a calibration factor computed at one adc_bits to a
different adc_bits would compound two different error sources (analytical-
model error and interpolation error) with no way to separate them. A
HWConfig that doesn't exactly match a known reference stays uncalibrated
(factor 1.0, same as before this module existed) rather than silently
getting a questionable correction.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from schemas.calibration import CalibrationReference
from schemas.config import HWConfig
from schemas.tools import LayerBitConfig
from tools.cim_physics import simulate_cim_profile

# Real, published data points -- not invented to flatter the analytical
# model. Add more entries as more real/precise-sim data becomes available;
# each addition only calibrates HWConfigs that exactly match it.
KNOWN_REFERENCES: List[CalibrationReference] = [
    CalibrationReference(
        crossbar_rows=128,
        crossbar_cols=128,
        adc_bits=7,
        weight_bits=8,
        # NeuroSim V1.5's default configuration (128x128 array, 7-bit ADC,
        # 8b weight / 8b input, 22nm RRAM, ResNet-18/CIFAR-100) reports
        # 11.6 TOPS at 21.3 TOPS/W. 21.3 TOPS/W = 21.3e12 ops/J ->
        # 1/21.3e12 J/op ~= 0.047 pJ/op.
        reference_energy_pj_per_mac=0.047,
        source=(
            "Peng et al., 'NeuroSim V1.5: Improved Software Backbone for "
            "Benchmarking Compute-in-Memory Accelerators with Device and "
            "Circuit-level Non-idealities' (arXiv:2505.02314), Table II "
            "default configuration"
        ),
        note=(
            "Energy/MAC derived from the paper's reported 21.3 TOPS/W; "
            "assumes 1 MAC == 1 'op' as the paper counts operations. If the "
            "paper's TOPS convention counts a MAC as 2 ops instead, true "
            "energy/MAC is ~2x this value -- stated here rather than "
            "silently absorbed into a falsely precise-looking number."
        ),
    ),
]


def _find_matching_reference(hw: HWConfig) -> Optional[CalibrationReference]:
    for reference in KNOWN_REFERENCES:
        if (
            reference.crossbar_rows == hw.crossbar_rows
            and reference.crossbar_cols == hw.crossbar_cols
            and reference.adc_bits == hw.adc_bits
        ):
            return reference
    return None


def compute_calibration_factor(hw: HWConfig, reference: CalibrationReference) -> float:
    """`reference.reference_energy_pj_per_mac` / this model's predicted
    energy-per-MAC for a single layer at the reference's own `weight_bits`
    -- the correction `profiler_tool` applies as a flat multiplier
    (tools/simulators.py)."""
    probe_layer = LayerBitConfig(
        layer_name="_calibration_probe",
        weight_bits=reference.weight_bits,
        activation_bits=reference.weight_bits,
        column_pruning_ratio=0.0,
    )
    predicted = simulate_cim_profile(hw, [probe_layer])
    num_macs = hw.crossbar_rows * hw.crossbar_cols
    predicted_energy_pj_per_mac = predicted["energy_pj"] / num_macs if num_macs else 0.0
    if predicted_energy_pj_per_mac <= 0:
        return 1.0
    return reference.reference_energy_pj_per_mac / predicted_energy_pj_per_mac


def bootstrap_calibration_factors(hw: HWConfig) -> Dict[str, float]:
    """`{}` if no known reference exactly matches `hw` -- the honest
    default (stays uncalibrated) rather than guessing. Otherwise
    `{hw.hw_spec_id: factor}`, ready to seed
    `AutoCIMState.calibration_factors` (main.py's `build_initial_state`)."""
    reference = _find_matching_reference(hw)
    if reference is None:
        return {}
    return {hw.hw_spec_id: compute_calibration_factor(hw, reference)}
