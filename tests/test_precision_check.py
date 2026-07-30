"""tools/precision_check.py's MOCK Stage-2 high-fidelity re-verification
(CLAUDE.md 5.D Mock First) -- see that module's docstring for why a
placeholder computation is intentional here."""

from schemas.config import HWConfig, NoCTopology
from schemas.tools import LayerBitConfig
from tools.cim_physics import simulate_cim_profile
from tools.precision_check import compute_updated_calibration_factor, simulate_precise_energy

_LAYER_CONFIGS = [LayerBitConfig(layer_name="conv", weight_bits=6, activation_bits=6, column_pruning_ratio=0.1)]


def _hw(**overrides) -> HWConfig:
    defaults = dict(
        hw_spec_id="test_hw",
        crossbar_rows=128,
        crossbar_cols=128,
        num_tiles=16,
        adc_bits=8,
        dac_bits=4,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.001,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    defaults.update(overrides)
    return HWConfig(**defaults)


def test_simulate_precise_energy_perturbs_the_raw_analytical_prediction_by_device_noise_sigma():
    hw = _hw(device_noise_sigma=0.2)
    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)

    result = simulate_precise_energy(hw, _LAYER_CONFIGS)

    assert result["raw_energy_pj"] == raw_profile["energy_pj"]
    assert result["precise_energy_pj"] == raw_profile["energy_pj"] * 1.2


def test_simulate_precise_energy_falls_back_to_default_margin_when_device_noise_sigma_unset():
    hw = _hw(device_noise_sigma=None)
    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)

    result = simulate_precise_energy(hw, _LAYER_CONFIGS)

    assert result["precise_energy_pj"] == raw_profile["energy_pj"] * 1.1


def test_compute_updated_calibration_factor_is_the_precise_over_raw_ratio():
    assert compute_updated_calibration_factor(100.0, 120.0) == 1.2


def test_compute_updated_calibration_factor_guards_against_zero_or_negative_raw_energy():
    assert compute_updated_calibration_factor(0.0, 120.0) == 1.0
    assert compute_updated_calibration_factor(-5.0, 120.0) == 1.0
