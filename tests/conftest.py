"""Shared pytest fixtures for AutoCIM-Agent's test suite."""

from typing import Any, Callable, Dict, List, Optional

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import main
import nodes.planner as planner_module
import observability
import tools.simulators as simulators_module
from middleware import register_hw_config
from schemas.config import HWConfig, NoCTopology
from state import AutoCIMState
from tools.qat import QATBackend

GOOD_HW_SPEC_ID = "test_hw_good"
BAD_HW_SPEC_ID = "test_hw_bad"


@pytest.fixture(autouse=True)
def isolated_observability_log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Every test gets its own AUTOCIM_LOG_DIR (observability.py) -- without
    this, structured-logging events from every test in this suite would
    write into the real repo's `.cache/logs/`, and `observability.py`'s
    per-run-id logger cache would keep a stale `FileHandler` pointing at a
    previous test's tmp_path once one exists. `reset_loggers()` clears that
    cache both before (fresh dir takes effect) and after (no leaked open
    file handles into the next test)."""
    monkeypatch.setenv("AUTOCIM_LOG_DIR", str(tmp_path / "logs"))
    observability.reset_loggers()
    yield
    observability.reset_loggers()


@pytest.fixture(autouse=True)
def isolated_long_term_store(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Every test gets its own long-term store DB (store.py's
    SqliteLongTermStore, which main.py's run_session reads/writes for
    cross-session learning) -- without this, every test that calls
    run_session() would read/write the real repo's
    `.cache/long_term_store.sqlite`, leaking test data into real runs and
    letting one test's stored Pareto front/calibration factors bleed into
    another test."""
    monkeypatch.setattr(main, "DEFAULT_LONG_TERM_STORE_DB", tmp_path / "long_term_store.sqlite")


def make_state(**overrides: Any) -> AutoCIMState:
    """Build a minimal, valid AutoCIMState, overriding any field by keyword."""
    state: AutoCIMState = {
        "messages": [],
        "failure_history": [],
        "candidate_history": [],
        "llm_usage": [],
        "planner_decisions": [],
        "metrics_store": {},
        "calibration_factors": {},
        "calibration_provenance": {},
        "human_overrides": {},
        "search_bound_overrides": {},
        "planned_layer_configs": [],
        "planner_flagged_anomaly": None,
        "model_id": "resnet18",
        "hw_spec_id": GOOD_HW_SPEC_ID,
        "target_accuracy": None,
        "target_energy_pj": None,
        "target_latency_ms": None,
        "iteration_count": 0,
        "retry_count": 0,
        "is_converged": False,
        "needs_hitl": False,
    }
    state.update(overrides)
    return state


@pytest.fixture
def state_factory() -> Callable[..., AutoCIMState]:
    return make_state


@pytest.fixture
def good_hw_config() -> HWConfig:
    """Converges immediately under verifier_tool's mock IR-drop/noise check
    (small wire resistance -> low ir_drop_error_pct)."""
    return HWConfig(
        hw_spec_id=GOOD_HW_SPEC_ID,
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


@pytest.fixture
def bad_hw_config() -> HWConfig:
    """Never converges under verifier_tool's mock IR-drop check
    (large wire resistance -> ir_drop_error_pct stays >= 5%)."""
    return HWConfig(
        hw_spec_id=BAD_HW_SPEC_ID,
        crossbar_rows=128,
        crossbar_cols=128,
        num_tiles=16,
        adc_bits=8,
        dac_bits=4,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=5.0,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )


@pytest.fixture
def registered_hw_config(good_hw_config: HWConfig) -> HWConfig:
    register_hw_config(good_hw_config)
    return good_hw_config


@pytest.fixture
def registered_bad_hw_config(bad_hw_config: HWConfig) -> HWConfig:
    register_hw_config(bad_hw_config)
    return bad_hw_config


class _FakeToolResponse:
    """Duck-types the subset of AIMessage that `nodes/planner.py` reads."""

    def __init__(self, tool_calls: List[Dict[str, Any]]):
        self.tool_calls = tool_calls


class FakeToolCallingChatModel:
    """Stand-in for `llm.get_planner_chat_model()`'s return value.

    No test in this suite may make a real LLM API call (no key, no network
    in CI) -- this fake mimics only the `.bind_tools(...).invoke(...)`
    surface `nodes/planner.py` uses and always returns a canned, valid
    `PlannerLayerDecision` tool call.
    """

    def __init__(self, tool_call_args: Optional[Dict[str, Any]] = None):
        # PlannerLayerDecision no longer has a layer_configs field (nodes/
        # planner.py's module docstring: the search decides stage_configs,
        # the LLM only explains it) -- this canned response only needs
        # rationale/rationale_ko/anomaly_note.
        self._tool_call_args = tool_call_args or {
            "rationale": "fake planner decision for testing",
            "rationale_ko": "테스트용 가짜 planner 결정",
            "anomaly_note": None,
        }
        self.bind_tools_calls: List[Dict[str, Any]] = []
        self.invoke_calls: List[Any] = []

    def bind_tools(self, tools: Any, *, tool_choice: Optional[str] = None, **kwargs: Any) -> "FakeToolCallingChatModel":
        self.bind_tools_calls.append({"tools": tools, "tool_choice": tool_choice})
        return self

    def invoke(self, messages: Any) -> _FakeToolResponse:
        self.invoke_calls.append(messages)
        tool_name = self.bind_tools_calls[-1]["tool_choice"] if self.bind_tools_calls else "PlannerLayerDecision"
        return _FakeToolResponse([{"name": tool_name, "args": self._tool_call_args, "id": "fake-call-1"}])


@pytest.fixture
def fake_planner_chat_model() -> FakeToolCallingChatModel:
    return FakeToolCallingChatModel()


@pytest.fixture(autouse=True)
def stub_planner_llm(monkeypatch: pytest.MonkeyPatch, fake_planner_chat_model: FakeToolCallingChatModel) -> None:
    """Every test gets a stubbed @planner LLM by default, so no test --
    including full-graph e2e tests that exercise @planner indirectly --
    accidentally makes a real network call. Tests that want to exercise the
    LLM-failure fallback path monkeypatch `nodes.planner.get_planner_chat_model`
    again themselves, after this fixture has already run."""
    monkeypatch.setattr(planner_module, "get_planner_chat_model", lambda: fake_planner_chat_model)


class _TinyConvNet(nn.Module):
    """Smallest model that exercises real Conv2d + Linear fake-quant/pruning
    parametrization -- no test in this suite downloads CIFAR10 or a
    pretrained checkpoint. Two named top-level submodules ("conv", "fc") so
    `tools.qat.group_quantizable_layers` finds two real stages, matching
    FAKE_LAYER_GROUPS below and letting tests exercise genuine per-stage
    (not just uniform) quantization/pruning end to end."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


FAKE_LAYER_GROUPS: Dict[str, List[str]] = {"conv": ["conv"], "fc": ["fc"]}


def make_fake_qat_backend(
    num_classes: int = 4,
    train_size: int = 16,
    val_size: int = 8,
    test_size: int = 8,
    image_size: int = 8,
    seed: int = 0,
) -> QATBackend:
    """Stand-in for `tools.qat.get_qat_backend()`'s return value: same
    `QATBackend` contract (real `nn.Module` + real `DataLoader`s, three
    disjoint splits), so `run_qat_tuning`'s fake-quant/pruning/training-loop
    code runs for real against this -- just on a tiny synthetic model and
    synthetic random data instead of resnet18 + CIFAR10, so tests run in
    well under a second with no download."""
    generator = torch.Generator().manual_seed(seed)

    def _split(size: int) -> DataLoader:
        images = torch.randn(size, 3, image_size, image_size, generator=generator)
        labels = torch.randint(0, num_classes, (size,), generator=generator)
        return DataLoader(TensorDataset(images, labels), batch_size=4)

    return QATBackend(
        base_model=_TinyConvNet(num_classes),
        train_loader=_split(train_size),
        val_loader=_split(val_size),
        test_loader=_split(test_size),
        num_classes=num_classes,
        layer_groups=dict(FAKE_LAYER_GROUPS),
    )


@pytest.fixture
def fake_qat_backend() -> QATBackend:
    return make_fake_qat_backend()


@pytest.fixture(autouse=True)
def stub_tuner_qat_backend(monkeypatch: pytest.MonkeyPatch, fake_qat_backend: QATBackend) -> None:
    """Every test gets a stubbed @tuner QAT backend by default -- no test in
    this suite downloads CIFAR10/a pretrained checkpoint or spends real
    training time. Mirrors `stub_planner_llm` above."""
    monkeypatch.setattr(simulators_module, "get_qat_backend", lambda model_id: fake_qat_backend)


@pytest.fixture(autouse=True)
def stub_planner_layer_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets @planner's view of "real" layer stages stubbed to
    FAKE_LAYER_GROUPS -- without this, `nodes.planner.real_stage_names`
    would call the real `tools.qat.get_layer_groups`, which builds the real
    resnet18 (a checkpoint download) just to read its module names."""
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_LAYER_GROUPS))
