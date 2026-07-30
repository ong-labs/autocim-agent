"""MOCK high-fidelity CIM verification (CLAUDE.md 5.D Mock First).

`nodes/precision_verifier.py` calls this only for a candidate @evaluator's
fast-approximation already judged converged (graph.py routes evaluator's
"Converged (Done)" edge to @precision_verifier instead of straight to END)
-- never every candidate a run searches. A real precise simulator call
(NeuroSim/CIM-Loop) is far more expensive than tools/cim_physics.py's
analytical formulas; re-checking only the (rare) candidates a run is about
to stop on keeps that cost bounded regardless of how many candidates the
search tries along the way.

This module defines that real interface's data shape now, with a clearly-
labeled placeholder computation standing in for it, so the graph routing/
state-flow around it can be built and tested before a real NeuroSim/CIM-Loop
binary is actually wired in behind it -- swap `simulate_precise_energy`'s
body for a real call later; every caller here only depends on its returned
dict shape.

Also doubles as the feed for tools/calibration.py's calibration_factors:
one precision check both (a) judges this one candidate against the real
targets using a more-trustworthy energy number, and (b) derives an updated
calibration_factor from the gap between the analytical prediction and that
"precise" number -- one expensive call serving both purposes, not two
separate NeuroSim calls per candidate.
"""

from typing import Dict, List

from schemas.config import HWConfig
from schemas.tools import LayerBitConfig
from tools.cim_physics import simulate_cim_profile

# Used only when hw.device_noise_sigma isn't set (schemas/config.py: an
# optional field -- "not every hardware spec models these") -- a fixed
# placeholder uncertainty margin, not a measured value.
_DEFAULT_UNCERTAINTY_MARGIN = 0.1


def simulate_precise_energy(hw: HWConfig, layer_configs: List[LayerBitConfig]) -> Dict[str, float]:
    """MOCK precise-simulator energy prediction: deterministically perturbs
    `tools.cim_physics.simulate_cim_profile`'s own analytical prediction by
    `hw.device_noise_sigma` (or the placeholder margin above) rather than
    returning it unchanged, so a caller can actually exercise "the precise
    number differs from the fast one" without a real circuit simulator.
    NOT a real physical model -- there is no claim here that real silicon
    specifically runs hotter than the analytical estimate; this only needs
    to be *some* deterministic, hw-derived (never a bare literal, CLAUDE.md
    5.C) perturbation so the rest of the pipeline (calibration_factor
    derivation, re-checking targets) has a real number to work with while
    this module waits for an actual NeuroSim/CIM-Loop integration."""
    raw_profile = simulate_cim_profile(hw, layer_configs)
    margin = 1.0 + (hw.device_noise_sigma if hw.device_noise_sigma is not None else _DEFAULT_UNCERTAINTY_MARGIN)
    raw_energy_pj = raw_profile["energy_pj"]
    return {"raw_energy_pj": raw_energy_pj, "precise_energy_pj": raw_energy_pj * margin}


def compute_updated_calibration_factor(raw_energy_pj: float, precise_energy_pj: float) -> float:
    """Same ratio `tools.calibration.compute_calibration_factor` computes
    from a published literature reference -- here from this run's own
    precision check instead of a citation. `raw_energy_pj` must be the
    *pre*-calibration analytical prediction (`simulate_precise_energy`'s
    own fresh `simulate_cim_profile` call), never the already-calibrated
    number @profiler reported (`ProfilerToolOutput.data['energy_pj']`) --
    that already has an old factor baked in, and dividing by it would
    apply that factor twice."""
    if raw_energy_pj <= 0:
        return 1.0
    return precise_energy_pj / raw_energy_pj
