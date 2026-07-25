"""Long-term memory stores for AutoCIM-Agent.

Implements the two long-term tiers of the 3-tier memory architecture from
Research_Plan.md section 3 (short-term state + reducers live in state.py):

- ModelSpecificStore: per-model sensitivity landscape / Pareto solutions,
  with indexed search to transfer know-how from structurally similar models.
- GlobalHWStore: per-hw_spec_id calibration factors and NoC/buffer delay
  rule-base, shared across all models run on that hardware.

`InMemoryLongTermStore` is a Mock-First (CLAUDE.md 5.D) backend for
development; swap in a persistent/vector-backed `BaseLongTermStore`
implementation later without changing `ModelSpecificStore`/`GlobalHWStore`
or any node/tool code that depends on them.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from state import merge_dicts

Namespace = Tuple[str, ...]


@dataclass
class SearchResult:
    namespace: Namespace
    key: str
    value: Dict[str, Any]
    score: Optional[float] = None


class BaseLongTermStore(abc.ABC):
    """Namespace-scoped long-term store: raw key lookup + indexed search."""

    @abc.abstractmethod
    def get(self, namespace: Namespace, key: str) -> Optional[Dict[str, Any]]:
        """Raw lookup of a single key within an exact namespace."""

    @abc.abstractmethod
    def put(self, namespace: Namespace, key: str, value: Dict[str, Any]) -> None:
        """Write/overwrite a single key within an exact namespace."""

    @abc.abstractmethod
    def search(self, namespace_prefix: Namespace, query: str, limit: int = 5) -> List[SearchResult]:
        """Indexed search over all entries whose namespace starts with `namespace_prefix`."""


class InMemoryLongTermStore(BaseLongTermStore):
    """Mock, process-local implementation for development/testing.

    `search` ranks by token overlap rather than real embeddings -- enough
    to exercise the interface end-to-end before wiring a vector DB.
    """

    def __init__(self) -> None:
        self._data: Dict[Namespace, Dict[str, Dict[str, Any]]] = {}

    def get(self, namespace: Namespace, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(namespace, {}).get(key)

    def put(self, namespace: Namespace, key: str, value: Dict[str, Any]) -> None:
        self._data.setdefault(namespace, {})[key] = value

    def search(self, namespace_prefix: Namespace, query: str, limit: int = 5) -> List[SearchResult]:
        query_tokens = set(query.lower().split())
        results: List[SearchResult] = []

        for namespace, bucket in self._data.items():
            if namespace[: len(namespace_prefix)] != tuple(namespace_prefix):
                continue
            for key, value in bucket.items():
                value_tokens = set(" ".join(str(v) for v in value.values()).lower().split())
                overlap = len(query_tokens & value_tokens)
                if overlap == 0:
                    continue
                score = overlap / max(len(query_tokens), 1)
                results.append(SearchResult(namespace=namespace, key=key, value=value, score=score))

        results.sort(key=lambda r: r.score or 0.0, reverse=True)
        return results[:limit]


class ModelSpecificStore:
    """Target-model long-term memory (Raw get/put + Indexed search).

    Known models load their sensitivity profile directly via `get`. A
    newly onboarded `model_id` has no profile yet, so `search_similar_models`
    performs an indexed/semantic search across all stored models to transfer
    optimization know-how from structurally similar ones.
    """

    NAMESPACE_PREFIX: Namespace = ("autocim", "model")

    def __init__(self, backend: BaseLongTermStore):
        self._backend = backend

    def _namespace(self, model_id: str) -> Namespace:
        return (*self.NAMESPACE_PREFIX, model_id)

    def get_sensitivity_profile(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._backend.get(self._namespace(model_id), "sensitivity_profile")

    def put_sensitivity_profile(self, model_id: str, profile: Dict[str, Any]) -> None:
        self._backend.put(self._namespace(model_id), "sensitivity_profile", profile)

    def get_pareto_solutions(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._backend.get(self._namespace(model_id), "pareto_solutions")

    def put_pareto_solutions(self, model_id: str, solutions: Dict[str, Any]) -> None:
        self._backend.put(self._namespace(model_id), "pareto_solutions", solutions)

    def search_similar_models(self, architecture_summary: str, limit: int = 5) -> List[SearchResult]:
        return self._backend.search(self.NAMESPACE_PREFIX, architecture_summary, limit=limit)


class GlobalHWStore:
    """Global HW-aware long-term memory, keyed by `hw_spec_id`.

    Accumulates calibration factors (fast-approximation vs. precise-sim
    error correction) and NoC/buffer-delay rule-base entries per crossbar
    spec, so repeat runs on the same hardware skip redundant precise
    simulation overhead.
    """

    NAMESPACE_PREFIX: Namespace = ("autocim", "hw")

    def __init__(self, backend: BaseLongTermStore):
        self._backend = backend

    def _namespace(self, hw_spec_id: str) -> Namespace:
        return (*self.NAMESPACE_PREFIX, hw_spec_id)

    def get_calibration_factors(self, hw_spec_id: str) -> Optional[Dict[str, float]]:
        return self._backend.get(self._namespace(hw_spec_id), "calibration_factors")

    def put_calibration_factors(self, hw_spec_id: str, factors: Dict[str, float]) -> None:
        existing = self.get_calibration_factors(hw_spec_id) or {}
        self._backend.put(self._namespace(hw_spec_id), "calibration_factors", merge_dicts(existing, factors))

    def get_noc_delay_rulebase(self, hw_spec_id: str) -> Optional[Dict[str, Any]]:
        return self._backend.get(self._namespace(hw_spec_id), "noc_delay_rulebase")

    def put_noc_delay_rulebase(self, hw_spec_id: str, rulebase: Dict[str, Any]) -> None:
        self._backend.put(self._namespace(hw_spec_id), "noc_delay_rulebase", rulebase)
