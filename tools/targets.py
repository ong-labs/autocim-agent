"""Target-achievement checking, shared between @evaluator's fast-approximation
gate (nodes/evaluator.py) and @precision_verifier's re-check against a more
trustworthy energy number (nodes/precision_verifier.py, tools/simulators.py's
precision_check_tool).

Lives in tools/, not nodes/evaluator.py where it originated, specifically so
tools/simulators.py can import it without a tools-depends-on-nodes edge:
nodes/__init__.py's own import chain already pulls tools.simulators in via
nodes.mapper/profiler/tuner/verifier, so a tools/simulators.py -> nodes.*
import back would risk a real circular-import failure depending on which
module a caller imports first (nodes/evaluator.py still exposes `check_targets`
itself, via `from tools.targets import check_targets`, so no existing caller
importing it from there needs to change).
"""

from typing import List, Optional


def check_targets(
    accuracy: Optional[float],
    energy_pj: Optional[float],
    noc_latency_ms: Optional[float],
    target_accuracy: Optional[float],
    target_energy_pj: Optional[float],
    target_latency_ms: Optional[float],
) -> List[str]:
    """Human-readable reasons any *configured* target wasn't met -- empty
    if every configured target is satisfied, or none are configured at
    all (the default: this dimension simply isn't gated). Accuracy is
    higher-is-better (must be >= target); energy/latency are
    lower-is-better (must be <= target). A missing metric value (`None`)
    can never be judged as meeting a configured target -- same "incomplete
    data is never silently fine" rule `validate_metrics` already applies,
    just for target-achievement instead of schema validity."""
    reasons = []
    if target_accuracy is not None and (accuracy is None or accuracy < target_accuracy):
        reasons.append(f"accuracy {accuracy} below target {target_accuracy}")
    if target_energy_pj is not None and (energy_pj is None or energy_pj > target_energy_pj):
        reasons.append(f"energy_pj {energy_pj} above target {target_energy_pj}")
    if target_latency_ms is not None and (noc_latency_ms is None or noc_latency_ms > target_latency_ms):
        reasons.append(f"noc_latency_ms {noc_latency_ms} above target {target_latency_ms}")
    return reasons
