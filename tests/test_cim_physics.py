"""tools/cim_physics.py: real NoC hop-count + CIM device-physics scaling.

These are pure-function unit tests (no LangGraph, no tool wrapping) directly
against the topology-distance formulas and the mapper/profiler analytical
models, isolating physics-correctness from the @wrap_tool_call plumbing
covered elsewhere.
"""

from schemas.config import NoCTopology
from schemas.tools import LayerBitConfig
from tools.cim_physics import hop_distance, simulate_cim_profile, simulate_noc_mapping


def _layer(weight_bits=6, activation_bits=6, pruning=0.0, name="l0"):
    return LayerBitConfig(
        layer_name=name, weight_bits=weight_bits, activation_bits=activation_bits, column_pruning_ratio=pruning
    )


# --- hop_distance: real topology graph-distance formulas --------------------


def test_hop_distance_same_tile_is_zero():
    assert hop_distance(NoCTopology.MESH, 3, 3, 16) == 0


def test_hop_distance_mesh_is_manhattan():
    # 4x4 grid: tile0=(0,0), tile1=(0,1) -> 1 hop; tile5=(1,1) -> 2 hops
    assert hop_distance(NoCTopology.MESH, 0, 1, 16) == 1
    assert hop_distance(NoCTopology.MESH, 0, 5, 16) == 2


def test_hop_distance_torus_wraps_around():
    # 4x4 grid: tile0=(0,0), tile3=(0,3) direct dx=3, but torus wraps to 1
    assert hop_distance(NoCTopology.TORUS, 0, 3, 16) == 1
    assert hop_distance(NoCTopology.MESH, 0, 3, 16) == 3  # mesh: no wraparound


def test_hop_distance_ring_takes_shorter_arc():
    assert hop_distance(NoCTopology.RING, 0, 15, 16) == 1
    assert hop_distance(NoCTopology.RING, 0, 8, 16) == 8


def test_hop_distance_bus_is_always_one_hop():
    assert hop_distance(NoCTopology.CROSSBAR_BUS, 0, 15, 16) == 1
    assert hop_distance(NoCTopology.CROSSBAR_BUS, 2, 9, 16) == 1


# --- simulate_noc_mapping: pipelined transfer over real topology distance ---


def test_single_layer_needs_no_noc_transfer(good_hw_config):
    result = simulate_noc_mapping(good_hw_config, [_layer()])

    assert result["tiles_needed"] == 1
    assert result["noc_latency_ms"] == 0.0  # nothing to route to


def test_more_layers_increase_noc_latency(good_hw_config):
    one_layer = simulate_noc_mapping(good_hw_config, [_layer()])
    three_layers = simulate_noc_mapping(good_hw_config, [_layer(), _layer(), _layer()])

    assert three_layers["tiles_needed"] == 3
    assert three_layers["noc_latency_ms"] > one_layer["noc_latency_ms"]


def test_tile_utilization_scales_with_layer_count_and_num_tiles(good_hw_config):
    result = simulate_noc_mapping(good_hw_config, [_layer(), _layer()])
    assert result["tile_utilization"] == round(2 / good_hw_config.num_tiles, 4)


# --- simulate_cim_profile: real ADC/DAC/crossbar energy scaling laws -------


def test_higher_adc_bits_increase_energy_exponentially(good_hw_config):
    low = simulate_cim_profile(good_hw_config, [_layer()])

    hotter_hw = good_hw_config.model_copy(update={"adc_bits": good_hw_config.adc_bits + 2})
    high = simulate_cim_profile(hotter_hw, [_layer()])

    assert high["energy_pj"] > low["energy_pj"] * 2  # exponential, not linear, growth


def test_higher_weight_bits_increase_energy_linearly_via_crossbar_term(good_hw_config):
    low = simulate_cim_profile(good_hw_config, [_layer(weight_bits=2)])
    high = simulate_cim_profile(good_hw_config, [_layer(weight_bits=8)])

    assert high["energy_pj"] > low["energy_pj"]


def test_more_layers_increase_both_energy_and_latency(good_hw_config):
    one_layer = simulate_cim_profile(good_hw_config, [_layer()])
    two_layers = simulate_cim_profile(good_hw_config, [_layer(), _layer()])

    assert two_layers["energy_pj"] == round(one_layer["energy_pj"] * 2, 6)
    assert two_layers["latency_ns"] == round(one_layer["latency_ns"] * 2, 6)
