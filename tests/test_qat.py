"""tools/qat.py: real fake-quantization + structured pruning + fine-tuning.

`stub_tuner_qat_backend` (tests/conftest.py) is autouse, so `tuner_tool`
itself (exercised elsewhere via tuner_node in test_reducers.py etc.) never
downloads CIFAR10/a checkpoint or spends real training time. These tests
call `tools.qat`'s functions directly against the tiny synthetic backend to
verify the actual PyTorch mechanics: fake-quant lands weights on the
correct grid, pruning zeroes the correct output channels, gradients flow
through the straight-through estimator, and `run_qat_tuning` never mutates
the shared cached base model.
"""

import copy

import pytest
import torch
import torch.nn as nn

from tools.qat import (
    _MODEL_REGISTRY,
    _resolve_int_env,
    _resolve_size_env,
    _resolve_split_sizes,
    apply_crossbar_quant_prune,
    group_quantizable_layers,
    run_qat_tuning,
)


def test_model_registry_covers_resnet18_mobilenet_v2_and_vit_tiny():
    # No network call here -- just confirms the registry entries exist and
    # are callable; the builders themselves (real checkpoint downloads) are
    # exercised manually, not in this no-network test suite.
    assert set(_MODEL_REGISTRY.keys()) == {"resnet18", "mobilenet_v2", "vit_tiny"}
    assert all(callable(builder) for builder in _MODEL_REGISTRY.values())


class _TwoStageModel(nn.Module):
    """Two distinct real top-level stages, neither one a numeric-indexed
    container child -- exercises `group_quantizable_layers`'s basic
    dotted-name grouping without touching resnet18."""

    def __init__(self):
        super().__init__()
        self.stage_a = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.stage_b = nn.Linear(4, 2)

    def forward(self, x):
        return self.stage_b(self.stage_a(x).mean(dim=(2, 3)))


class _RepeatingBlockModel(nn.Module):
    """Mirrors vit_tiny/mobilenet_v2's structure: every block nested under
    one container attribute, indexed numerically (e.g. "blocks.0",
    "blocks.1", ...) -- exercises `_stage_of`'s digit-aware grouping, which
    is what keeps each block its own real stage instead of every block
    collapsing into a single "blocks" group under a naive "first segment
    only" rule."""

    def __init__(self, num_blocks: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_blocks)])
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        x = self.stem(x).mean(dim=(2, 3))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_group_quantizable_layers_groups_by_top_level_stage():
    groups = group_quantizable_layers(_TwoStageModel())
    assert groups == {"stage_a": ["stage_a"], "stage_b": ["stage_b"]}


def test_group_quantizable_layers_gives_each_repeating_block_its_own_stage():
    groups = group_quantizable_layers(_RepeatingBlockModel(num_blocks=3))
    assert groups == {
        "stem": ["stem"],
        "blocks.0": ["blocks.0"],
        "blocks.1": ["blocks.1"],
        "blocks.2": ["blocks.2"],
        "head": ["head"],
    }


def test_apply_crossbar_quant_prune_applies_different_config_per_real_stage():
    model = _TwoStageModel()
    apply_crossbar_quant_prune(model, stage_configs={"stage_a": (6, 0.0), "stage_b": (2, 0.0)}, default_config=(8, 0.0))

    stage_a_unique = torch.unique(model.stage_a.weight.detach()).numel()
    stage_b_unique = torch.unique(model.stage_b.weight.detach()).numel()
    # 6-bit grid has far more representable levels than 2-bit -> more
    # distinct quantized values survive from the randomly-initialized weights.
    assert stage_a_unique > stage_b_unique


def test_apply_crossbar_quant_prune_applies_different_config_per_repeating_block():
    model = _RepeatingBlockModel(num_blocks=3)
    apply_crossbar_quant_prune(
        model,
        stage_configs={"blocks.0": (8, 0.0), "blocks.1": (2, 0.0)},
        default_config=(8, 0.0),
    )

    block0_unique = torch.unique(model.blocks[0].weight.detach()).numel()
    block1_unique = torch.unique(model.blocks[1].weight.detach()).numel()
    block2_unique = torch.unique(model.blocks[2].weight.detach()).numel()  # falls back to default (8 bits)
    assert block0_unique > block1_unique
    assert block2_unique > block1_unique


def test_apply_crossbar_quant_prune_falls_back_to_default_for_unnamed_stage():
    model = _TwoStageModel()
    # stage_configs only covers "stage_a" -- "stage_b" must fall back to
    # default_config rather than erroring or silently going unquantized.
    apply_crossbar_quant_prune(model, stage_configs={"stage_a": (6, 0.0)}, default_config=(2, 0.0))

    stage_a_unique = torch.unique(model.stage_a.weight.detach()).numel()
    stage_b_unique = torch.unique(model.stage_b.weight.detach()).numel()
    assert stage_a_unique > stage_b_unique  # stage_b used the 2-bit default


def test_apply_crossbar_quant_prune_zeroes_expected_fraction_of_output_channels():
    conv = nn.Conv2d(3, 8, kernel_size=3)
    apply_crossbar_quant_prune(conv, stage_configs={}, default_config=(4, 0.5))

    weight = conv.weight  # triggers the registered parametrization
    zeroed_channels = (weight.reshape(8, -1).abs().sum(dim=1) == 0).sum().item()

    assert zeroed_channels == 4  # 50% of 8 output channels


def test_apply_crossbar_quant_prune_snaps_weights_to_the_symmetric_grid():
    linear = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        # max |x| = 1.0 -> scale = 1.0 / qmax; every element must land on
        # an exact multiple of that scale (the quantization grid), matching
        # the STE forward formula round(x / scale) * scale directly.
        linear.weight.copy_(torch.tensor([[1.0, 0.6, 0.2, -1.0]]))

    apply_crossbar_quant_prune(linear, stage_configs={}, default_config=(3, 0.0))

    qmax = 2 ** (3 - 1) - 1  # 3
    scale = 1.0 / qmax
    expected = torch.round(torch.tensor([1.0, 0.6, 0.2, -1.0]) / scale) * scale

    assert torch.allclose(linear.weight.detach().flatten(), expected, atol=1e-6)


def test_gradient_flows_through_fake_quant_via_straight_through_estimator():
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    apply_crossbar_quant_prune(conv, stage_configs={}, default_config=(4, 0.0))

    x = torch.randn(2, 3, 8, 8)
    loss = conv(x).sum()
    loss.backward()

    assert conv.parametrizations.weight.original.grad is not None
    assert torch.any(conv.parametrizations.weight.original.grad != 0)


def test_run_qat_tuning_returns_measured_accuracy_without_mutating_base_model(fake_qat_backend):
    base_model_before = copy.deepcopy(fake_qat_backend.base_model.state_dict())

    result = run_qat_tuning(fake_qat_backend, stage_configs={}, default_config=(4, 0.25), max_epochs=2)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 1 <= result["epochs_run"] <= 2
    # the cached base_model itself must be untouched -- run_qat_tuning must
    # deep-copy before quantizing/pruning/fine-tuning, so a second call
    # (e.g. the next @planner-proposed candidate) starts from the same
    # pretrained weights, not a previous candidate's fine-tuned ones.
    assert not hasattr(fake_qat_backend.base_model, "parametrizations")
    for key, value in base_model_before.items():
        assert torch.equal(value, fake_qat_backend.base_model.state_dict()[key])


def test_run_qat_tuning_respects_max_epochs_cap(fake_qat_backend):
    result = run_qat_tuning(fake_qat_backend, stage_configs={}, default_config=(8, 0.0), max_epochs=1)
    assert result["epochs_run"] == 1


# --- Configurable QAT dataset scope (AUTOCIM_QAT_*) -------------------------


def test_resolve_size_env_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("AUTOCIM_QAT_TRAIN_SIZE", raising=False)
    assert _resolve_size_env("AUTOCIM_QAT_TRAIN_SIZE", 512, allow_full=True) == 512


def test_resolve_size_env_parses_a_configured_integer(monkeypatch):
    monkeypatch.setenv("AUTOCIM_QAT_TRAIN_SIZE", "2000")
    assert _resolve_size_env("AUTOCIM_QAT_TRAIN_SIZE", 512, allow_full=True) == 2000


def test_resolve_size_env_recognizes_full_sentinel_case_insensitively(monkeypatch):
    monkeypatch.setenv("AUTOCIM_QAT_TRAIN_SIZE", "FULL")
    assert _resolve_size_env("AUTOCIM_QAT_TRAIN_SIZE", 512, allow_full=True) == "full"


def test_resolve_size_env_falls_back_to_default_on_garbage_value(monkeypatch):
    monkeypatch.setenv("AUTOCIM_QAT_TRAIN_SIZE", "not-a-number")
    assert _resolve_size_env("AUTOCIM_QAT_TRAIN_SIZE", 512, allow_full=True) == 512


def test_resolve_size_env_falls_back_to_default_on_non_positive_value(monkeypatch):
    monkeypatch.setenv("AUTOCIM_QAT_TRAIN_SIZE", "0")
    assert _resolve_size_env("AUTOCIM_QAT_TRAIN_SIZE", 512, allow_full=True) == 512


def test_resolve_int_env_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("AUTOCIM_QAT_BATCH_SIZE", raising=False)
    assert _resolve_int_env("AUTOCIM_QAT_BATCH_SIZE", 32) == 32


def test_resolve_int_env_parses_a_configured_value(monkeypatch):
    monkeypatch.setenv("AUTOCIM_QAT_BATCH_SIZE", "64")
    assert _resolve_int_env("AUTOCIM_QAT_BATCH_SIZE", 32) == 64


def test_resolve_split_sizes_passes_through_explicit_sizes_within_bounds():
    train, test = _resolve_split_sizes(512, 128, 1000, train_available=50000, test_available=10000)
    assert (train, test) == (512, 1000)


def test_resolve_split_sizes_full_train_reserves_val_size_first():
    train, test = _resolve_split_sizes("full", 128, 1000, train_available=50000, test_available=10000)
    assert train == 50000 - 128


def test_resolve_split_sizes_full_test_uses_the_whole_test_split():
    train, test = _resolve_split_sizes(512, 128, "full", train_available=50000, test_available=10000)
    assert test == 10000


def test_resolve_split_sizes_raises_when_train_plus_val_exceeds_available():
    with pytest.raises(ValueError, match="exceeds CIFAR10's 100-image train split"):
        _resolve_split_sizes(90, 20, 10, train_available=100, test_available=100)


def test_resolve_split_sizes_raises_when_test_exceeds_available():
    with pytest.raises(ValueError, match="exceeds CIFAR10's 100-image test split"):
        _resolve_split_sizes(50, 20, 200, train_available=100, test_available=100)
