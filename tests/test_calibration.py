"""tools/calibration.py: seeding @profiler's `calibration_factors` from a
real, cited literature reference (not an invented number), and staying
honestly uncalibrated (factor 1.0) for any HWConfig that doesn't exactly
match a known reference.
"""

from schemas.config import HWConfig, NoCTopology
from schemas.tools import LayerBitConfig
from tools.calibration import KNOWN_REFERENCES, bootstrap_calibration_factors, compute_calibration_factor
from tools.cim_physics import simulate_cim_profile

_REFERENCE = KNOWN_REFERENCES[0]  # the NeuroSim V1.5 128x128/7-bit-ADC point


def _hw(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw",
        crossbar_rows=_REFERENCE.crossbar_rows,
        crossbar_cols=_REFERENCE.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_bootstrap_returns_factor_for_exact_matching_hw_config():
    factors = bootstrap_calibration_factors(_hw())
    assert list(factors.keys()) == ["calib_test_hw"]
    assert factors["calib_test_hw"] > 0


def test_bootstrap_is_empty_when_adc_bits_differs():
    assert bootstrap_calibration_factors(_hw(adc_bits=_REFERENCE.adc_bits + 1)) == {}


def test_bootstrap_is_empty_when_crossbar_size_differs():
    assert bootstrap_calibration_factors(_hw(crossbar_rows=64)) == {}
    assert bootstrap_calibration_factors(_hw(crossbar_cols=64)) == {}


def test_compute_calibration_factor_equals_reference_over_predicted_per_mac():
    hw = _hw()
    probe = LayerBitConfig(
        layer_name="_calibration_probe",
        weight_bits=_REFERENCE.weight_bits,
        activation_bits=_REFERENCE.weight_bits,
        column_pruning_ratio=0.0,
    )
    predicted_energy_pj_per_mac = simulate_cim_profile(hw, [probe])["energy_pj"] / (hw.crossbar_rows * hw.crossbar_cols)

    factor = compute_calibration_factor(hw, _REFERENCE)

    assert factor == _REFERENCE.reference_energy_pj_per_mac / predicted_energy_pj_per_mac


def test_known_reference_has_a_real_citation():
    # Guards against a future reference entry being added without a source
    # -- this module's entire premise is "grounded in real data, not
    # invented," so an uncited entry would be a regression of that premise.
    for reference in KNOWN_REFERENCES:
        assert reference.source.strip()
