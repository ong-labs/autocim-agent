"""store.py's long-term memory: SqliteLongTermStore (the persistent backend
main.py's run_session actually uses) plus ModelSpecificStore/GlobalHWStore
(the namespaced wrappers around it) -- none of this had any test coverage
before, since nothing in the codebase called it yet.
"""

from store import GlobalHWStore, ModelSpecificStore, SqliteLongTermStore


def test_sqlite_long_term_store_get_returns_none_for_missing_key(tmp_path):
    backend = SqliteLongTermStore(tmp_path / "lts.sqlite")
    assert backend.get(("autocim", "model", "resnet18"), "pareto_solutions") is None


def test_sqlite_long_term_store_put_then_get_round_trips(tmp_path):
    backend = SqliteLongTermStore(tmp_path / "lts.sqlite")
    value = {"candidates": [{"accuracy": 0.7, "energy_pj": 100.0}]}
    backend.put(("autocim", "model", "resnet18"), "pareto_solutions", value)
    assert backend.get(("autocim", "model", "resnet18"), "pareto_solutions") == value


def test_sqlite_long_term_store_put_overwrites_existing_value(tmp_path):
    backend = SqliteLongTermStore(tmp_path / "lts.sqlite")
    backend.put(("autocim", "hw", "chip_a"), "calibration_factors", {"chip_a": 1.0})
    backend.put(("autocim", "hw", "chip_a"), "calibration_factors", {"chip_a": 2.0})
    assert backend.get(("autocim", "hw", "chip_a"), "calibration_factors") == {"chip_a": 2.0}


def test_sqlite_long_term_store_persists_across_separate_instances(tmp_path):
    # The entire point of this backend over InMemoryLongTermStore: a second
    # SqliteLongTermStore pointed at the same file (== a later `python
    # main.py` process) must see what an earlier one wrote.
    db_path = tmp_path / "lts.sqlite"
    SqliteLongTermStore(db_path).put(("autocim", "model", "resnet18"), "pareto_solutions", {"candidates": []})

    reopened = SqliteLongTermStore(db_path)
    assert reopened.get(("autocim", "model", "resnet18"), "pareto_solutions") == {"candidates": []}


def test_sqlite_long_term_store_search_matches_by_namespace_prefix_and_token_overlap(tmp_path):
    backend = SqliteLongTermStore(tmp_path / "lts.sqlite")
    backend.put(("autocim", "model", "resnet18"), "sensitivity_profile", {"note": "resnet18 conv layers sensitive"})
    backend.put(("autocim", "model", "mobilenet_v2"), "sensitivity_profile", {"note": "mobilenet depthwise robust"})
    backend.put(("autocim", "hw", "chip_a"), "calibration_factors", {"chip_a": 1.0})  # different prefix, must not match

    results = backend.search(("autocim", "model"), "resnet18 conv")

    assert len(results) == 1
    assert results[0].namespace == ("autocim", "model", "resnet18")


def test_model_specific_store_pareto_solutions_round_trip(tmp_path):
    store = ModelSpecificStore(SqliteLongTermStore(tmp_path / "lts.sqlite"))
    assert store.get_pareto_solutions("resnet18") is None
    store.put_pareto_solutions("resnet18", {"candidates": [{"accuracy": 0.7}]})
    assert store.get_pareto_solutions("resnet18") == {"candidates": [{"accuracy": 0.7}]}


def test_global_hw_store_calibration_factors_merge_not_overwrite(tmp_path):
    store = GlobalHWStore(SqliteLongTermStore(tmp_path / "lts.sqlite"))
    store.put_calibration_factors("chip_a", {"chip_a": 1.5})
    # A later put for the same hw_spec_id with a *different* key must merge
    # into, not replace, what's already stored there.
    store.put_calibration_factors("chip_a", {"chip_a_alt_config": 1.8})
    assert store.get_calibration_factors("chip_a") == {"chip_a": 1.5, "chip_a_alt_config": 1.8}
