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
    _resolve_device,
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


# --- group_quantizable_layers against the three real registered architectures ---
#
# The tests above only ever exercise `_stage_of`'s digit-aware grouping
# against small hand-built stand-ins (_TwoStageModel/_RepeatingBlockModel).
# `_MODEL_REGISTRY`'s actual builders (`_build_mobilenet_v2`/`_build_vit_tiny`)
# always request pretrained ImageNet weights, so nothing in this no-network
# suite ever built these two real architectures at all -- a real breakage in
# `_stage_of` against mobilenet_v2/vit_tiny's real module tree (e.g. a
# torchvision/timm version bump changing attribute names) could go
# undetected. `weights=None`/`pretrained=False` construct the identical
# module tree with random init and no download, so these stay real
# architecture tests without adding any network dependency to the suite.


def test_group_quantizable_layers_against_real_resnet18_architecture():
    import torchvision.models as tv_models

    groups = group_quantizable_layers(tv_models.resnet18(weights=None))

    # 4 stages x 2 BasicBlocks each (resnet18's [2, 2, 2, 2] layout) plus the
    # stem conv and final fc -- exactly what get_qat_backend's docstring
    # claims ("conv1/layer1..layer4/fc"), not collapsed or split differently.
    assert set(groups.keys()) == {
        "conv1",
        "layer1.0", "layer1.1",
        "layer2.0", "layer2.1",
        "layer3.0", "layer3.1",
        "layer4.0", "layer4.1",
        "fc",
    }


def test_group_quantizable_layers_against_real_mobilenet_v2_architecture():
    import torchvision.models as tv_models

    groups = group_quantizable_layers(tv_models.mobilenet_v2(weights=None))

    # mobilenet_v2 nests every block under one "features" container
    # (unlike resnet18's per-stage top-level attributes) -- this is the
    # exact case tools/qat.py's module docstring calls out as needing
    # _stage_of's digit-aware rule: a plain "first segment only" grouping
    # would collapse all 19 feature blocks into a single "features" group.
    assert "features" not in groups
    feature_groups = {key for key in groups if key.startswith("features.")}
    assert len(feature_groups) == 19  # features.0 .. features.18
    assert "classifier.1" in groups


def test_group_quantizable_layers_against_real_vit_tiny_architecture():
    import timm

    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    groups = group_quantizable_layers(model)

    # Same digit-aware concern as mobilenet_v2, for vit_tiny's transformer
    # blocks: 12 real blocks (vit_tiny_patch16_224's depth), not one
    # collapsed "blocks" group.
    assert "blocks" not in groups
    block_groups = {key for key in groups if key.startswith("blocks.")}
    assert len(block_groups) == 12  # blocks.0 .. blocks.11
    assert "head" in groups


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
    # A single output channel (out_features=1) -- per-column scale and
    # per-tensor scale coincide here, so this only exercises the grid
    # formula itself, not column independence (see the per-column test
    # below for that).
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


def test_fake_quant_scales_each_output_channel_independently():
    """Column-wise quantization: each output channel ("crossbar column")
    gets its own scale, not one scale shared across the whole layer.
    Channel 0's weights are 100x larger than channel 1's; a single shared
    per-tensor scale (sized off channel 0's max) would round channel 1's
    much smaller values all the way down to 0, losing that column's
    weights entirely."""
    linear = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[10.0, -10.0], [0.1, -0.1]]))

    apply_crossbar_quant_prune(linear, stage_configs={}, default_config=(4, 0.0))

    quantized = linear.weight.detach()
    # Channel 1 (small magnitudes) must still be distinguishable from zero
    # and from each other -- a shared per-tensor scale derived from
    # channel 0's max=10.0 would collapse both to exactly 0.0.
    assert quantized[1, 0] != 0.0
    assert quantized[1, 0] != quantized[1, 1]


def test_gradient_flows_through_fake_quant_via_straight_through_estimator():
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    apply_crossbar_quant_prune(conv, stage_configs={}, default_config=(4, 0.0))

    x = torch.randn(2, 3, 8, 8)
    loss = conv(x).sum()
    loss.backward()

    assert conv.parametrizations.weight.original.grad is not None
    assert torch.any(conv.parametrizations.weight.original.grad != 0)


# --- Column-wise partial-sum quantization (adc_bits) -------------------------


def test_apply_crossbar_quant_prune_skips_partial_sum_quantization_by_default():
    """adc_bits=None (the default) must leave the module's output
    untouched -- backward compatible with every caller that predates this
    parameter (e.g. tests above, which never pass it)."""
    linear = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.ones(2, 4))
    apply_crossbar_quant_prune(linear, stage_configs={}, default_config=(8, 0.0))

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    output = linear(x)
    expected = x @ linear.weight.detach().t()  # weight already reflects its own quantization
    assert torch.allclose(output, expected, atol=1e-5)


def test_partial_sum_quant_scales_each_output_channel_independently():
    """Column-wise partial-sum quantization: two output channels with very
    different accumulated magnitudes must each get their own ADC-precision
    scale -- a scale shared across the whole output (sized off the larger
    channel) would round the smaller channel's partial sum to 0."""
    linear = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[10.0], [0.1]]))

    apply_crossbar_quant_prune(linear, stage_configs={}, default_config=(8, 0.0), adc_bits=4)

    output = linear(torch.tensor([[1.0]]))
    assert output[0, 1].item() != 0.0


def test_partial_sum_quant_snaps_output_to_the_adc_grid():
    linear = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0]]))
    apply_crossbar_quant_prune(linear, stage_configs={}, default_config=(8, 0.0), adc_bits=3)

    # Batch of 2 so the one output channel sees two different values --
    # scale is derived from the batch max (1.0), so 0.3 actually lands on
    # a grid point other than itself, proving real quantization happened
    # (not just an identity pass-through).
    output = linear(torch.tensor([[1.0], [0.3]]))

    qmax = 2 ** (3 - 1) - 1  # 3
    scale = 1.0 / qmax
    expected = torch.round(torch.tensor([1.0, 0.3]) / scale) * scale
    assert torch.allclose(output.detach().flatten(), expected, atol=1e-6)
    assert output.detach()[1, 0].item() != pytest.approx(0.3)


def test_partial_sum_quant_uses_adc_bits_not_the_per_stage_weight_bits():
    """adc_bits is a single hw-wide value (HWConfig.adc_bits), applied the
    same way regardless of a stage's own weight_bits -- two stages with
    very different weight_bits must still see identically-precise
    partial-sum quantization."""
    model = nn.ModuleDict({"a": nn.Linear(1, 1, bias=False), "b": nn.Linear(1, 1, bias=False)})
    with torch.no_grad():
        model["a"].weight.copy_(torch.tensor([[1.0]]))
        model["b"].weight.copy_(torch.tensor([[1.0]]))
    apply_crossbar_quant_prune(
        model, stage_configs={"a": (2, 0.0), "b": (8, 0.0)}, default_config=(8, 0.0), adc_bits=3
    )

    out_a = model["a"](torch.tensor([[1.0], [0.3]])).detach()
    out_b = model["b"](torch.tensor([[1.0], [0.3]])).detach()
    # Both stages' weight (1.0) is lossless at either 2 or 8 bits, so any
    # difference here comes only from partial-sum quantization -- which
    # must be identical since both share the same adc_bits.
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_gradient_flows_through_partial_sum_quant():
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    apply_crossbar_quant_prune(conv, stage_configs={}, default_config=(8, 0.0), adc_bits=6)

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


# --- Device selection (AUTOCIM_QAT_DEVICE) -----------------------------------


def test_resolve_device_explicit_argument_wins_over_everything():
    assert _resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_env_var_used_when_no_explicit_argument(monkeypatch):
    monkeypatch.setenv("AUTOCIM_QAT_DEVICE", "cpu")
    assert _resolve_device(None) == torch.device("cpu")


def test_resolve_device_explicit_argument_beats_env_var(monkeypatch):
    monkeypatch.setenv("AUTOCIM_QAT_DEVICE", "cpu")
    assert _resolve_device("meta") == torch.device("meta")


def test_resolve_device_auto_detects_a_real_device_when_unconfigured(monkeypatch):
    monkeypatch.delenv("AUTOCIM_QAT_DEVICE", raising=False)
    resolved = _resolve_device(None)
    # Whatever this machine actually has (GPU or not), _resolve_device must
    # agree with torch's own capability checks -- not silently assume one
    # or the other.
    if torch.cuda.is_available():
        assert resolved == torch.device("cuda")
    else:
        assert resolved.type in ("cpu", "mps")


def test_run_qat_tuning_reports_the_device_it_actually_used(fake_qat_backend):
    result = run_qat_tuning(fake_qat_backend, stage_configs={}, default_config=(4, 0.25), max_epochs=1, device="cpu")
    assert result["device"] == "cpu"
