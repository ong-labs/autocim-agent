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

`bootstrap_calibration_factors`/`bootstrap_calibration_provenance` (main.py's
default path) are deliberately exact-match only -- no interpolation/
extrapolation across array sizes or ADC resolutions. ADC energy scales
exponentially with resolution in this model, so applying a calibration
factor computed at one adc_bits to a different adc_bits would compound two
different error sources (analytical-model error and interpolation error)
with no way to separate them. A HWConfig that doesn't exactly match a known
reference stays uncalibrated (factor 1.0, same as before this module
existed) rather than silently getting a questionable correction.

For callers that explicitly want a best-effort number anyway (e.g.
exploring a custom HWConfig with no published exact match), the
`bootstrap_approximate_calibration_factors`/`_provenance` family below is an
opt-in scaling-law fallback: it picks KNOWN_REFERENCES' nearest entry (by
`_reference_distance`) and transfers *that reference's own* correction
ratio to the query HWConfig, on the assumption the analytical model's
relative error is roughly stable across nearby configs. This is a real
extrapolation, not a citation -- every value it returns is tagged
`"approximate": True` with the matched reference and the distance to it, and
it is never called from `bootstrap_calibration_factors` itself, so the
default (main.py) path's behavior is unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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
    CalibrationReference(
        crossbar_rows=256,
        crossbar_cols=64,
        adc_bits=8,
        weight_bits=3,
        # Correll et al. 2025 (JSSC): silicon-measured (65nm, foundry MLC
        # ReRAM), not simulated. "Measured raw peak efficiency" at full
        # 256x64 array utilization is 20.7 TOPS/W (Sec. V.G / Table VII) --
        # the paper's own throughput equation explicitly states "System
        # throughput is scaled by a factor of two, as both multiply and
        # accumulate occur in a single step," i.e. 1 MAC == 2 ops in its
        # TOPS convention (unlike the NeuroSim reference above, no
        # convention assumption needed here): 2 / 20.7e12 J/MAC ~= 0.0966
        # pJ/MAC.
        reference_energy_pj_per_mac=0.0966,
        source=(
            "Correll et al., 'An 8-bit 20.7 TOPS/W Multilevel Cell ReRAM "
            "Macro With ADC-Assisted Bit-Serial Processing' (IEEE J. "
            "Solid-State Circuits, vol. 60, no. 8, pp. 2995-3007, Aug. "
            "2025), Sec. II.A (256x64 array, 8-bit ADCs/DACs) and Sec. V.G "
            "(20.7 TOPS/W measured raw peak efficiency)"
        ),
        note=(
            "weight_bits=3 is the native per-cell MLC ReRAM precision "
            "(write-verify programs 3-bit weights, Fig. 20/Sec. V.C) -- the "
            "paper's dual-array technique composes two 3-bit cells into an "
            "effective ~4-bit signed weight for its LeNet1 demo, but that is "
            "a system-level composition on top of this crossbar, not its "
            "per-cell weight_bits. '8-bit' in the title refers to the "
            "input/ADC precision (256 DACs drive 8-bit inputs; 32 8-bit "
            "ADCs read out), not the weight precision. The paper also "
            "reports a 662 TOPS/W 'normalized' figure (scaled to 1b x 1b "
            "MAC) -- deliberately not used here in favor of the raw, "
            "as-measured 20.7 TOPS/W at this exact array/ADC/weight config."
        ),
    ),
    CalibrationReference(
        crossbar_rows=64,
        crossbar_cols=32,
        adc_bits=5,
        weight_bits=4,
        # Chen, Chen & Gu 2021 (ISSCC 15.3): silicon-measured, 65nm 3T
        # dynamic-analog-RAM (DARAM) CIM macro, a genuine 5-bit SAR ADC (not
        # a custom VSA/CSA scheme like the ISSCC digest papers this project
        # rejected as calibration sources -- see git history) -- matches
        # this codebase's flash/SAR-ADC energy-scaling assumption
        # (tools/cim_physics.py) cleanly. "This work achieves a macro
        # efficiency of 217TOPS/W at 4b" is the CIM-macro-only figure
        # (analog array + ADC/DAC), distinct from the paper's 44.7TOPS/W
        # *system* efficiency (includes the surrounding digital ASIC) --
        # the macro-only number is used here since simulate_cim_profile
        # models the crossbar/ADC/DAC macro alone, not a host ASIC.
        # 1/217e12 J/op ~= 0.0046 pJ/op.
        reference_energy_pj_per_mac=0.0046,
        source=(
            "Chen, Chen & Gu, 'A 65nm 3T Dynamic Analog RAM-Based "
            "Computing-in-Memory Macro and CNN Accelerator with Retention "
            "Enhancement, Adaptive Analog Sparsity and 44TOPS/W System "
            "Energy Efficiency' (ISSCC 2021, Session 15.3, pp. 236-238) -- "
            "'CIM macro contains a 64x32 DARAM array... A 5b SAR ADC and a "
            "4b current DAC are implemented at each column' and 'this work "
            "achieves a macro efficiency of 217TOPS/W at 4b'"
        ),
        note=(
            "weight_bits=4 is the DARAM cell's native analog-stored weight "
            "precision ('A 4b weight is stored as an analog voltage on the "
            "internal MEM node'); the design natively supports 4b/4b "
            "input-weight and can combine two DARAM cells for 8b/8b, which "
            "is a system-level composition, not this array's per-cell "
            "weight_bits. As with the NeuroSim reference above, the paper "
            "does not state whether its TOPS convention counts a MAC as 1 "
            "or 2 ops; assumed 1 MAC == 1 op here, so true energy/MAC could "
            "be ~2x this value -- stated rather than silently absorbed."
        ),
    ),
    CalibrationReference(
        crossbar_rows=128,
        crossbar_cols=128,
        adc_bits=1,
        weight_bits=1,
        # Yu, Yoo, Kim, Chai & Kim 2020 (CICC): silicon-measured, 65nm 8T
        # SRAM CIM macro with a genuine reconfigurable column ADC (1-31
        # sense cycles for N=1..5 bits), binary (-1/+1) weight. Unlike every
        # reference above, this paper states its energy number directly as
        # "Energy/OP 2.04fJ/OP (1bit)" and its body text defines "OP" as
        # "the bitwise multiply and accumulate operation" -- i.e. 1 MAC ==
        # 1 "OP" by the paper's own definition, so no 1-op-vs-2-op
        # conversion assumption is needed here at all (unlike the NeuroSim
        # and DARAM references above). 2.04 fJ/MAC = 0.00204 pJ/MAC;
        # cross-checks exactly against the abstract's "490...TOPS/W at
        # 1...5bit": 1 / 2.04fJ = 490 TOPS/W.
        reference_energy_pj_per_mac=0.00204,
        source=(
            "Yu, Yoo, Kim, Chai & Kim, 'A 16K Current-Based 8T SRAM "
            "Compute-In-Memory Macro with Decoupled Read/Write and 1-5bit "
            "Column ADC' (IEEE Custom Integrated Circuits Conference, "
            "2020) -- 'Array Size 128 x 128 (16K)' and 'Energy/OP "
            "2.04fJ/OP (1bit)' (Fig. 10 chip summary), 1-bit ADC mode"
        ),
        note=(
            "weight_bits=1 is this design's native precision: weights are "
            "binary (-1/+1) SRAM-cell states, not a quantized multi-bit "
            "value (Table I: 'Weight Bit# = 1'). This is the 1-bit-ADC "
            "operating point of the same physical 128x128 macro as the "
            "adc_bits=5 reference below -- the pair lets a caller sanity-"
            "check tools/cim_physics.py's 2**adc_bits ADC-energy scaling "
            "against two real measured points on identical hardware, not "
            "just one."
        ),
    ),
    CalibrationReference(
        crossbar_rows=128,
        crossbar_cols=128,
        adc_bits=5,
        weight_bits=1,
        # Same macro/paper as the adc_bits=1 entry above, reconfigured to
        # its 5-bit column-ADC mode: "energy efficiency of the 1bit
        # operation is 490-to-15.8TOPS/W at 1-5bit ADC mode" (abstract) --
        # 15.8 TOPS/W is the 5-bit endpoint. Same "OP" == 1 MAC definition
        # as the 1-bit entry (same paper, same accounting): 1/15.8e12 J/MAC
        # ~= 0.0633 pJ/MAC.
        reference_energy_pj_per_mac=0.0633,
        source=(
            "Yu, Yoo, Kim, Chai & Kim, 'A 16K Current-Based 8T SRAM "
            "Compute-In-Memory Macro with Decoupled Read/Write and 1-5bit "
            "Column ADC' (IEEE Custom Integrated Circuits Conference, "
            "2020) -- 'Array Size 128 x 128 (16K)' and abstract's "
            "'...15.8TOPS/W at 1-5bit ADC mode', 5-bit ADC mode endpoint"
        ),
        note=(
            "weight_bits=1 (binary, see the adc_bits=1 entry above's note). "
            "The paper reports the 490->15.8 TOPS/W range as its two stated "
            "endpoints (1-bit and 5-bit); the 2/3/4-bit interior points "
            "were only shown in a bar chart in the source PDF, not stated "
            "as numbers in the text, so they are deliberately not "
            "interpolated into additional entries here."
        ),
    ),
    CalibrationReference(
        crossbar_rows=64,
        crossbar_cols=64,
        adc_bits=4,
        weight_bits=4,
        # Dong, Sinangil et al. 2020 (ISSCC 15.3, TSMC): silicon-measured,
        # 7nm FinFET, a genuine "4b Flash ADC" (explicit in the text --
        # "column-wise Flash ADCs", "Each 4b Flash ADC consists of 15 SAs")
        # -- matches this codebase's flash-ADC energy assumption more
        # directly than any reference above. "351 TOPS/W" is the paper's
        # headline efficiency; cross-checked its own MAC-to-op convention
        # via throughput/cycle-time rather than assuming: 372.4 GOPS x 5.5ns
        # cycle time = 2048 ops/cycle, and the macro does 1024 MAVs/cycle
        # (64 inputs x 16 weights) -- so this paper also counts 1 MAC == 2
        # ops (same convention as the Correll 2025 reference above, verified
        # independently here rather than assumed). 2 / 351e12 J/MAC ~=
        # 0.0057 pJ/MAC. (The paper separately reports a 7.8pJ *maximum*
        # energy per cycle -- deliberately not used, since that is the
        # worst-case all-1s pattern, not what the 351TOPS/W average-case
        # headline efficiency reflects.)
        reference_energy_pj_per_mac=0.0057,
        source=(
            "Dong, Sinangil, Erbagci, Sun, Khwa, Liao, Wang & Chang, 'A "
            "351TOPS/W and 372.4GOPS Compute-in-Memory SRAM Macro in 7nm "
            "FinFET CMOS for Machine-Learning Applications' (ISSCC 2020, "
            "Session 15.3, pp. 242-243) -- '64x64 8T macro is fabricated "
            "in 7nm FinFET technology... Energy efficiency is 351 TOPS/W "
            "and throughput is 372.4 GOPS for 1024 (64x16) 4x4b MAV "
            "operations'"
        ),
        note=(
            "weight_bits=4 ('4b weight is realized by charge sharing among "
            "binary-weighted computation caps'), input is also 4b (via RWL "
            "pulse count). MAV = 'multiply-and-average', this paper's own "
            "term for its accumulation operation across 64 inputs -- treated "
            "as equivalent to a MAC here, same as every other reference in "
            "this list. Originally sought for a 512x512/8-bit-ADC 'large-"
            "scale server' target hardware profile; no silicon-measured "
            "match at that array size/ADC resolution was found, so this "
            "64x64/4-bit point was substituted as the closest real, "
            "flash-ADC-based alternative -- a real repositioning, not a "
            "stand-in for the original spec."
        ),
    ),
    CalibrationReference(
        crossbar_rows=256,
        crossbar_cols=512,
        adc_bits=6,
        weight_bits=1,
        # Deaville, Zhang & Verma 2022 (Princeton, VLSI Symposium): silicon-
        # measured, 22nm FD-SOI, a genuine 6-bit SAR ADC -- "a 6-b current-
        # to-digital converter (IDC), based on a SAR architecture". Array is
        # explicitly "a 256(row)x512(col.) MRAM array of 1T1R cells".
        # Originally sought for a 256x256/6-bit-ADC/22nm ReRAM 'mainstream'
        # target; this is MRAM, not ReRAM -- a different NVM device
        # technology, not just a different array shape (unlike the Chip
        # A/C repositionings above) -- and the array is 256x512, not
        # square. Accepted anyway (22nm and 6-bit ADC, the two most
        # load-bearing specs, match exactly) after explicit confirmation.
        # weight_bits=1: the chip's *native* atomic operation is a 1b x 1b
        # XNOR MAC ("signed binarized multiplication... implementing an
        # XNOR operation"); its 4-bit weight mode composes four of these
        # digitally ("mapping bits to parallel columns, where four columns
        # feed one readout channel"). The paper's only reported efficiency,
        # "41.6 1b-TOPS/W", is already stated in this same 1-bit-normalized
        # unit -- calibrating at weight_bits=1 uses that number directly,
        # rather than assuming a specific bits-to-TOPS/W normalization rule
        # to back out a 4-bit-weight number that was never directly
        # measured. 1/41.6e12 J/MAC ~= 0.0240 pJ/MAC (as with the NeuroSim
        # and DARAM references above, the paper does not state whether its
        # TOPS convention counts 1 MAC as 1 or 2 ops; assumed 1 MAC == 1 op
        # here).
        reference_energy_pj_per_mac=0.0240,
        source=(
            "Deaville, Zhang & Verma, 'A 22nm 128-kb MRAM Row/Column-"
            "Parallel In-Memory Computing Macro with Memory-Resistance "
            "Boosting and Multi-Column ADC Readout' (2022 IEEE Symposium "
            "on VLSI Technology and Circuits, pp. 268-269) -- 'a 256"
            "(row)x512(col.) MRAM array... a 6-b current-to-digital "
            "converter (IDC), based on a SAR architecture' and 'high "
            "energy efficiency (41.6 1b-TOPS/W)'"
        ),
        note=(
            "MRAM, not ReRAM -- flagged explicitly since every other "
            "reference in this list up to here is ReRAM or SRAM; the "
            "underlying device physics differ, though both are foundry-"
            "integrated eNVM crossbars read out through a conventional "
            "multi-bit ADC, which is what tools/cim_physics.py's model "
            "actually depends on. Array is 256x512, not the square 256x256 "
            "originally sought -- the crossbar_cols=512 side is twice the "
            "requested size. Substituted for the original 256x256/6-bit/"
            "22nm ReRAM 'mainstream' target after no silicon-measured "
            "exact match at that array size/technology was found -- the "
            "adc_bits=6 and 22nm process, arguably the two specs a "
            "calibration factor is most sensitive to, are exact matches."
        ),
    ),
    CalibrationReference(
        crossbar_rows=256,
        crossbar_cols=80,
        adc_bits=4,
        weight_bits=8,
        # Korea Univ. (arXiv:2211.16008 -- open-access preprint of the
        # paywalled Lee/Kim/Park IEEE JSSC 2024 journal version of the same
        # design): silicon-measured, 28nm CMOS, a genuine "4-bit Coarse-Fine
        # Flash ADC" (Table II). Originally sought for a 128x128/4-bit-ADC/
        # 4-bit-weight/28nm target hardware profile; this is 256x80 (20480
        # cells, vs. the requested 16384 -- closer in scale than either of
        # the two substitutions above) and 8-bit weight, not 4-bit. adc_bits
        # and the 28nm process, the two specs this exact-match mechanism
        # keys on plus the one the analytical model is most sensitive to,
        # match exactly. The weight_bits=8 mismatch against a real target's
        # 4-bit layers is not a new source of error introduced by this
        # substitution specifically -- compute_calibration_factor never
        # matches on weight_bits for *any* reference in this list (not even
        # an "exact" one): it always applies the reference's own
        # weight_bits-derived correction ratio as a flat multiplier to
        # whatever weight_bits a real candidate layer actually uses.
        # 50.07 TOPS/W (Table II, 16 rows activated, 0.6V) is this work's
        # reported energy efficiency at this exact (4-bit input, 8-bit
        # weight) config; no statement in the paper resolves whether its
        # TOPS convention counts 1 MAC as 1 or 2 ops (unlike the Correll/
        # Dong references above) -- assumed 1 MAC == 1 op here, same as the
        # NeuroSim and DARAM references. 1/50.07e12 J/MAC ~= 0.0200 pJ/MAC.
        reference_energy_pj_per_mac=0.0200,
        source=(
            "Anonymous authors (Korea University), 'A Charge Domain P-8T "
            "SRAM Compute-In-Memory with Low-Cost DAC/ADC Operation for "
            "4-bit Input Processing' (arXiv:2211.16008; open-access "
            "preprint of Lee, Kim & Park, IEEE JSSC vol. 59 no. 6, "
            "pp. 1926-1937, June 2024) -- Table II: 'SRAM Array Size "
            "256x80 (16x5 AMUs)', 'ADC Scheme 4-bit Coarse-Fine Flash', "
            "'Input / Weight 4 bit / 8 bit', 'Energy Efficiency (16 Rows) "
            "50.07 (0.6V)'"
        ),
        note=(
            "Originally sought for a 128x128/4-bit-ADC/4-bit-weight/28nm "
            "'edge' target; substituted after the exact JSSC 2024 target "
            "(same design) was confirmed unobtainable (IEEE Xplore "
            "paywall) -- this arXiv preprint of the same work reports the "
            "identical headline number (50.07 TOPS/W, vs. the journal "
            "version's rounded 50.1 TOPS/W) with full Table I/II detail "
            "the published abstract alone did not expose. See crossbar "
            "size and weight_bits caveats above."
        ),
    ),
]


def find_matching_reference(hw: HWConfig) -> Optional[CalibrationReference]:
    """Public so callers besides `bootstrap_calibration_factors` (e.g.
    `tools/dashboard.py`, indirectly via `describe_calibration`) can learn
    *which* reference calibrated a given `HWConfig`, not just the resulting
    factor -- an opaque multiplier alone doesn't tell a researcher whether
    to trust it, or which paper's stated caveats (`CalibrationReference.note`)
    apply to it."""
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
    reference = find_matching_reference(hw)
    if reference is None:
        return {}
    return {hw.hw_spec_id: compute_calibration_factor(hw, reference)}


def describe_calibration(hw: HWConfig) -> Optional[Dict[str, Any]]:
    """The matching reference's citation/uncertainty, as a plain dict --
    `None` if `hw` is uncalibrated. Kept separate from
    `bootstrap_calibration_factors` (rather than folding this into that
    function's return value) so `AutoCIMState.calibration_factors` stays
    the simple `{hw_spec_id: float}` shape `tools/simulators.py`'s
    `profiler_tool` already multiplies by -- this is purely additive
    metadata for `tools/dashboard.py` to display, not part of the
    correction math."""
    reference = find_matching_reference(hw)
    if reference is None:
        return None
    return {
        "reference_energy_pj_per_mac": reference.reference_energy_pj_per_mac,
        "source": reference.source,
        "note": reference.note,
    }


def bootstrap_calibration_provenance(hw: HWConfig) -> Dict[str, Dict[str, Any]]:
    """`{}` if uncalibrated, else `{hw.hw_spec_id: describe_calibration(hw)}`
    -- the provenance analogue of `bootstrap_calibration_factors`, seeded
    into `AutoCIMState.calibration_provenance` by the same caller
    (main.py's `build_initial_state`)."""
    described = describe_calibration(hw)
    if described is None:
        return {}
    return {hw.hw_spec_id: described}


# =============================================================================
# Opt-in scaling-law (nearest-reference) approximate calibration
#
# Everything above this line is the exact-match path main.py uses by
# default. Everything below is a separate, explicitly-named fallback for
# callers that want a best-effort correction for a HWConfig with no exact
# match in KNOWN_REFERENCES -- never wired into the functions above, so
# adding entries here changes nothing for existing exact-match callers.
# =============================================================================


def _reference_distance(hw: HWConfig, reference: CalibrationReference) -> float:
    """Nearest-neighbor ranking distance between `hw` and `reference`, zero
    iff `hw` exactly matches `reference` on `find_matching_reference`'s three
    fields. `adc_bits` is compared directly (in bits): the analytical
    model's dominant nonlinearity, `2**adc_bits` in tools/cim_physics.py,
    already lives on this exact axis, so this distance only needs to *rank*
    candidates, not reproduce that curve itself. Total array size
    (`crossbar_rows * crossbar_cols`) is compared in log2 space -- i.e. by
    how many array-size *doublings* apart two configs are, not their raw
    cell-count gap -- so e.g. 128x128 -> 256x256 (one doubling) ranks closer
    than 128x128 -> 64x32 (a smaller raw cell-count gap, but a differently-
    shaped array reached by mixed halving/quartering, not one clean step)."""
    size_distance = abs(
        math.log2(hw.crossbar_rows * hw.crossbar_cols) - math.log2(reference.crossbar_rows * reference.crossbar_cols)
    )
    adc_distance = abs(hw.adc_bits - reference.adc_bits)
    return size_distance + adc_distance


def find_nearest_reference(hw: HWConfig) -> Optional[Tuple[CalibrationReference, float]]:
    """`KNOWN_REFERENCES`' closest entry to `hw` by `_reference_distance`,
    paired with that distance -- `None` only if `KNOWN_REFERENCES` is empty.
    Unlike `find_matching_reference`, always returns *something* (there is
    always a "nearest" reference as long as the list is non-empty), which is
    exactly why callers must treat this as an approximation and surface the
    distance, never present it the way an exact match is presented."""
    if not KNOWN_REFERENCES:
        return None
    nearest = min(KNOWN_REFERENCES, key=lambda reference: _reference_distance(hw, reference))
    return nearest, _reference_distance(hw, nearest)


def compute_approximate_calibration_factor(hw: HWConfig, reference: CalibrationReference) -> float:
    """Like `compute_calibration_factor`, but evaluates the analytical
    model's prediction *at the reference's own* (crossbar_rows,
    crossbar_cols, adc_bits) rather than `hw`'s -- `hw` may not equal the
    reference's config at all here, so predicting at `hw`'s own config would
    silently compare the reference's real chip energy against the model's
    prediction for a *different* array/ADC size, an apples-to-oranges ratio.
    Evaluating at the reference's own point instead extracts just that
    reference's dimensionless correction ratio (how wrong the analytical
    model tends to run relative to that one real/cited data point); the
    caller (`bootstrap_approximate_calibration_factors`) then applies that
    ratio to `hw` by seeding it into `calibration_factors[hw.hw_spec_id]`,
    which `tools/simulators.py`'s `profiler_tool` multiplies against energy
    it computes at `hw`'s *own* config -- so the scaling-law transfer
    happens there, once, not twice."""
    reference_shaped_hw = hw.model_copy(
        update={
            "crossbar_rows": reference.crossbar_rows,
            "crossbar_cols": reference.crossbar_cols,
            "adc_bits": reference.adc_bits,
        }
    )
    return compute_calibration_factor(reference_shaped_hw, reference)


def bootstrap_approximate_calibration_factors(hw: HWConfig) -> Dict[str, float]:
    """Scaling-law fallback for `bootstrap_calibration_factors`: `{}` only
    if `KNOWN_REFERENCES` is empty, else `{hw.hw_spec_id: factor}` from the
    nearest reference's own correction ratio -- computed and returned
    regardless of whether `hw` has an exact match (if it does, the nearest
    reference *is* the exact match and this returns the same factor
    `bootstrap_calibration_factors` would). Deliberately not called from
    `bootstrap_calibration_factors` itself; a caller (e.g. a future opt-in
    main.py CLI flag) must choose this path explicitly."""
    nearest = find_nearest_reference(hw)
    if nearest is None:
        return {}
    reference, _distance = nearest
    return {hw.hw_spec_id: compute_approximate_calibration_factor(hw, reference)}


def describe_approximate_calibration(hw: HWConfig) -> Optional[Dict[str, Any]]:
    """Provenance analogue of `describe_calibration` for the approximate
    path: `None` only if `KNOWN_REFERENCES` is empty, else the nearest
    reference's citation plus which (crossbar_rows, crossbar_cols, adc_bits)
    it actually describes and `_reference_distance` to it -- always tagged
    `"approximate": True` so a consumer (`tools/dashboard.py`) can never
    render this the way an exact `describe_calibration` hit is rendered."""
    nearest = find_nearest_reference(hw)
    if nearest is None:
        return None
    reference, distance = nearest
    return {
        "reference_energy_pj_per_mac": reference.reference_energy_pj_per_mac,
        "source": reference.source,
        "note": reference.note,
        "approximate": True,
        "matched_crossbar_rows": reference.crossbar_rows,
        "matched_crossbar_cols": reference.crossbar_cols,
        "matched_adc_bits": reference.adc_bits,
        "distance": round(distance, 4),
    }


def bootstrap_approximate_calibration_provenance(hw: HWConfig) -> Dict[str, Dict[str, Any]]:
    """`{}` if `KNOWN_REFERENCES` is empty, else
    `{hw.hw_spec_id: describe_approximate_calibration(hw)}` -- the
    provenance analogue of `bootstrap_approximate_calibration_factors`,
    mirroring how `bootstrap_calibration_provenance` pairs with
    `bootstrap_calibration_factors`."""
    described = describe_approximate_calibration(hw)
    if described is None:
        return {}
    return {hw.hw_spec_id: described}
