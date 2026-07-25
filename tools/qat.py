"""Real PyTorch QAT engine for @tuner.

Replaces the earlier accuracy-by-formula mock with genuine PyTorch
mechanics: real crossbar-aligned fake-quantization (straight-through
estimator) and real magnitude-ranked structured column pruning, applied to
a real pretrained model, fine-tuned with real gradient descent, and
evaluated for real measured accuracy.

Per-layer granularity: `get_layer_groups` exposes the target model's real
top-level module names (e.g. resnet18's `conv1`/`layer1`/`layer2`/`layer3`/
`layer4`/`fc`), grouping every Conv2d/Linear submodule under whichever real
stage it belongs to. @planner (nodes/planner.py) reads this to propose a
`LayerBitConfig` per real stage instead of one flat entry for the whole
model; `apply_crossbar_quant_prune` then applies each stage's own
(weight_bits, column_pruning_ratio) to only that stage's layers, falling
back to `default_config` (the request's averaged config) for any stage
@planner didn't cover -- so an LLM response that only names a subset of
stages, or names none at all (e.g. `@tuner`'s own single-entry fallback,
nodes/common.py's `default_layer_configs`), still degrades gracefully to
uniform quantization rather than erroring.

Scope, stated plainly (CPU-only, no GPU in this environment):
- `_MODEL_REGISTRY` has three entries: "resnet18" (torchvision, CNN),
  "mobilenet_v2" (torchvision, depthwise-separable CNN), and "vit_tiny"
  (timm's `vit_tiny_patch16_224`, transformer) -- all ImageNet-pretrained,
  final classifier head replaced for CIFAR10's 10 classes. Extending to
  further `model_id`s is adding a registry entry, not touching call sites
  (CLAUDE.md 5.C "Model Agnostic").
- Hardware-mapping honesty: a CIM crossbar is fundamentally a dense
  matrix-vector multiply engine. ViT/DeiT's attention (qkv/proj) and MLP
  (fc1/fc2) layers are *all* dense `Linear` ops -- a natural fit. MobileNetV2's
  depthwise-separable convolutions include per-channel depthwise convs where
  each output channel only touches one input channel, so they don't exercise
  a crossbar's parallel-MAC advantage the way a dense/pointwise layer does.
  `apply_crossbar_quant_prune` still quantizes/prunes every Conv2d/Linear
  uniformly regardless of this distinction (a reasonable simplification for
  first-cut multi-architecture coverage) -- `tools/cim_physics.py`'s
  mapper/profiler modeling depthwise vs. dense MACs differently would be a
  real follow-up, not attempted here.
- Fine-tuning/eval default to a fixed CIFAR10 subset (not the full 50k/10k
  images), so a QAT trial completes in CPU time by default -- an explicit,
  approved scope decision, not a hidden shortcut, and one a deployment can
  widen (up to the full split) via `AUTOCIM_QAT_TRAIN_SIZE`/`_VAL_SIZE`/
  `_TEST_SIZE`/`_BATCH_SIZE` without a code change (CLAUDE.md 5.C). Whatever
  the configured sizes, `_DEFAULT_TRAIN_SIZE`/`_VAL_SIZE`/`_TEST_SIZE` are a
  genuine three-way split (see `QATBackend`'s docstring): training and
  early-stopping never touch the test set that `run_qat_tuning`
  reports accuracy on, so that number isn't leaked into the decisions that
  produced it. Still a subset, not the full benchmark -- comparing candidate
  configs against each other is sound; treating the absolute number as a
  publishable benchmark result is not.
- Grouping (`_stage_of`) is by top-level attribute name, extended through
  the first purely-numeric path segment if there is one -- e.g. resnet18's
  "layer2.0.conv1" groups as "layer2.0" (one real stage per residual block),
  and vit_tiny's "blocks.7.attn.qkv" groups as "blocks.7" (one real stage
  per transformer block), not everything under "blocks" collapsing into a
  single stage. Real architectural granularity, tractable for both the LLM
  prompt and tools/search.py's search space, rather than either a single
  flat model-wide entry or a much higher-dimensional per-individual-layer
  search.

`get_qat_backend`/`get_layer_groups` are the swappable seams tests replace
(tests/conftest.py) with a tiny synthetic model + synthetic data, exactly
like `llm.py`'s `get_planner_chat_model` -- no test in this suite downloads
a dataset/checkpoint or trains on real data.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import DataLoader

_CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache"

StageConfig = Tuple[int, float]  # (weight_bits, column_pruning_ratio)


@dataclass
class QATBackend:
    """A model + its fine-tuning/eval data, bundled so `run_qat_tuning` is
    agnostic to whether it's driving the real resnet18/CIFAR10 backend or a
    test double.

    Three disjoint loaders, not two: `val_loader` is what `run_qat_tuning`
    monitors for early stopping, and `test_loader` is only ever touched once,
    after training/early-stopping decisions are final, for the accuracy
    actually reported. Using the same set for both (an earlier version of
    this module did) leaks the "test" set into the training-decision loop --
    the reported accuracy would be mildly optimistic, biased toward whatever
    that set happens to look like, rather than a genuinely held-out number.
    """

    base_model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    layer_groups: Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Real per-stage layer grouping
# ---------------------------------------------------------------------------


def _stage_of(module_name: str) -> str:
    """Groups by the dotted name up through the first purely-numeric path
    segment, if any -- that segment is always a `ModuleList`/`Sequential`
    child index, i.e. one repeating architectural block. Architectures that
    give each stage its own top-level attribute (resnet18's `layer1`..`layer4`)
    still resolve to per-residual-block groups ("layer2.0", "layer2.1", ...),
    and architectures that nest *all* their blocks under one container
    attribute (mobilenet_v2's `features`, vit_tiny's `blocks`) resolve to one
    group per real block ("features.3", "blocks.7", ...) instead of every
    block collapsing into a single "features"/"blocks" group -- which a plain
    "just take the first segment" rule would do."""
    parts = module_name.split(".")
    for index, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[: index + 1])
    return parts[0]


def group_quantizable_layers(model: nn.Module) -> Dict[str, List[str]]:
    """Every Conv2d/Linear submodule's dotted name, grouped by real stage.
    Pure function of `model` (no model_id/registry lookup) so it's directly
    unit-testable against an arbitrary small module -- no network/dataset
    involved."""
    groups: Dict[str, List[str]] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            groups.setdefault(_stage_of(name), []).append(name)
    return groups


# ---------------------------------------------------------------------------
# Real crossbar-aligned fake-quantization + structured column pruning
# ---------------------------------------------------------------------------


class _FakeQuantSTE(torch.autograd.Function):
    """Symmetric per-tensor fake quantization with a straight-through
    gradient estimator -- standard QAT technique: forward simulates the
    quantization grid, backward passes the gradient through unchanged so
    the optimizer can still update the underlying full-precision weight."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int) -> torch.Tensor:
        qmax = max(2 ** (bits - 1) - 1, 0)
        qmin = -(2 ** (bits - 1))
        scale = x.detach().abs().max().clamp(min=1e-8) / max(qmax, 1)
        return torch.clamp(torch.round(x / scale), qmin, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _CrossbarQuantPrune(nn.Module):
    """`torch.nn.utils.parametrize` transform applied to a layer's `weight`:
    fake-quantize to `bits`, then zero the `pruning_ratio` fraction of
    output channels ("crossbar columns", dim 0 in PyTorch's
    [out_channels, in_channels, ...] weight layout) with the smallest L1
    magnitude -- a standard structured-pruning ranking criterion.
    """

    def __init__(self, bits: int, pruning_ratio: float):
        super().__init__()
        self.bits = bits
        self.pruning_ratio = pruning_ratio

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        w = _FakeQuantSTE.apply(weight, self.bits)
        out_channels = w.shape[0]
        k = int(round(out_channels * self.pruning_ratio))
        if k > 0:
            l1_per_channel = w.detach().reshape(out_channels, -1).abs().sum(dim=1)
            prune_idx = torch.argsort(l1_per_channel)[:k]
            mask = torch.ones(out_channels, dtype=w.dtype, device=w.device)
            mask[prune_idx] = 0.0
            w = w * mask.view(-1, *([1] * (w.dim() - 1)))
        return w


def apply_crossbar_quant_prune(
    model: nn.Module, stage_configs: Dict[str, StageConfig], default_config: StageConfig
) -> None:
    """Mutates `model` in place: registers `_CrossbarQuantPrune` on every
    `Conv2d`/`Linear` submodule's `weight`, using that module's real stage's
    entry in `stage_configs` (keyed by `LayerBitConfig.layer_name` ==
    `group_quantizable_layers`'s stage names) if present, else
    `default_config`. Callers must pass a model they own (e.g. a fresh
    `copy.deepcopy` of a cached base model), never the shared cached
    instance -- see `run_qat_tuning`."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            bits, pruning_ratio = stage_configs.get(_stage_of(name), default_config)
            parametrize.register_parametrization(module, "weight", _CrossbarQuantPrune(bits, pruning_ratio))


# ---------------------------------------------------------------------------
# Fine-tuning + evaluation loop
# ---------------------------------------------------------------------------


def _train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion) -> None:
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion) -> Dict[str, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        outputs = model(images)
        loss_sum += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return {"accuracy": correct / total if total else 0.0, "loss": loss_sum / total if total else 0.0}


def run_qat_tuning(
    backend: QATBackend,
    stage_configs: Dict[str, StageConfig],
    default_config: StageConfig,
    max_epochs: int,
    early_stopping_patience: Optional[int] = None,
) -> Dict[str, float]:
    """Real QAT trial for one candidate config: deep-copies `backend`'s base
    model (so each candidate starts from the same pretrained weights, never
    a previous candidate's fine-tuned weights), applies each real stage's
    own crossbar-aligned fake-quant + structured pruning (falling back to
    `default_config` for any stage `stage_configs` doesn't cover),
    fine-tunes for up to `max_epochs` real gradient-descent epochs against
    `backend.train_loader` (monitoring `backend.val_loader` for early
    stopping), and returns real measured accuracy on `backend.test_loader`
    -- touched exactly once, after every training/early-stopping decision is
    already final, so the reported number isn't leaked into those decisions.
    """
    model = copy.deepcopy(backend.base_model)
    apply_crossbar_quant_prune(model, stage_configs, default_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        _train_one_epoch(model, backend.train_loader, optimizer, criterion)
        epochs_run = epoch + 1
        val_metrics = _evaluate(model, backend.val_loader, criterion)
        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            break

    final_metrics = _evaluate(model, backend.test_loader, criterion)
    return {"accuracy": final_metrics["accuracy"], "epochs_run": epochs_run}


# ---------------------------------------------------------------------------
# Default backend: real resnet18 (ImageNet-pretrained) + real CIFAR10 subset
# ---------------------------------------------------------------------------


def _build_resnet18(num_classes: int) -> nn.Module:
    import torchvision.models as tv_models

    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mobilenet_v2(num_classes: int) -> nn.Module:
    import torchvision.models as tv_models

    model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _build_vit_tiny(num_classes: int) -> nn.Module:
    import timm

    # timm's `num_classes=` kwarg replaces the pretrained head itself
    # (unlike torchvision's models, there's no separate manual-surgery step).
    return timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=num_classes)


_MODEL_REGISTRY: Dict[str, Callable[[int], nn.Module]] = {
    "resnet18": _build_resnet18,
    "mobilenet_v2": _build_mobilenet_v2,
    "vit_tiny": _build_vit_tiny,
}

_NUM_CLASSES = 10  # CIFAR10

_model_cache: Dict[str, nn.Module] = {}


def _get_base_model(model_id: str) -> nn.Module:
    """Just the model architecture (no dataset) -- cached separately from
    `get_qat_backend` so `get_layer_groups` (which @planner calls to learn
    real stage names) never has to pay CIFAR10's download/load cost."""
    if model_id not in _MODEL_REGISTRY:
        raise ValueError(f"No QAT backend registered for model_id={model_id!r}; supported: {list(_MODEL_REGISTRY)}")
    if model_id not in _model_cache:
        _model_cache[model_id] = _MODEL_REGISTRY[model_id](_NUM_CLASSES)
    return _model_cache[model_id]


def get_layer_groups(model_id: str) -> Dict[str, List[str]]:
    """Real per-stage layer names for `model_id`, for @planner to propose
    per-stage `layer_configs` against (nodes/planner.py)."""
    return group_quantizable_layers(_get_base_model(model_id))


# Fine-tuning set (backward pass -- the per-QAT-trial cost driver), the
# early-stopping monitor (forward-only, evaluated once per epoch), and the
# final-report set (forward-only, evaluated exactly once per trial no matter
# how many epochs run). TEST_SIZE can be much larger than TRAIN_SIZE for
# comparatively little extra wall-clock time precisely because it's only
# ever touched once: 1000 images is ~8x CIFAR10's old 128-image test subset,
# which cuts the accuracy figure's binomial-proportion 95% CI from roughly
# +/-8pp down to roughly +/-3pp -- the earlier size was too small to treat
# as more than a rough directional signal between candidates.
#
# These were hardcoded module constants (CLAUDE.md 5.C: config-driven, not
# hardcoded) -- a deployment validating against its *real* target dataset
# rather than this demo subset had no way to widen scope without editing
# this file. Now env-var overridable (same pattern as llm.py's
# AUTOCIM_LLM_* / AUTOCIM_PLANNER_MODEL), defaulting to the values above so
# nothing changes for a caller that doesn't opt in. `_FULL_SPLIT` lets
# AUTOCIM_QAT_TRAIN_SIZE/AUTOCIM_QAT_TEST_SIZE request "every image CIFAR10's
# own split has" without hand-computing 50000/10000 -- resolved against the
# actual loaded dataset length in `_build_cifar10_loaders`, not guessed here.
# Widening these multiplies real per-trial wall-clock (CPU-only, no GPU --
# see this module's docstring); that tradeoff is the caller's to make, not
# silently defaulted for them.
_TRAIN_SIZE_ENV = "AUTOCIM_QAT_TRAIN_SIZE"
_VAL_SIZE_ENV = "AUTOCIM_QAT_VAL_SIZE"
_TEST_SIZE_ENV = "AUTOCIM_QAT_TEST_SIZE"
_BATCH_SIZE_ENV = "AUTOCIM_QAT_BATCH_SIZE"
_SEED_ENV = "AUTOCIM_QAT_SEED"

_DEFAULT_TRAIN_SIZE = 512
_DEFAULT_VAL_SIZE = 128
_DEFAULT_TEST_SIZE = 1000
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_DATASET_SEED = 0

_FULL_SPLIT = "full"


def _resolve_size_env(env_var: str, default: int, *, allow_full: bool) -> int | str:
    raw = os.environ.get(env_var)
    if not raw:
        return default
    if allow_full and raw.strip().lower() == _FULL_SPLIT:
        return _FULL_SPLIT
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _resolve_int_env(env_var: str, default: int) -> int:
    raw = os.environ.get(env_var)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _resolve_split_sizes(
    train_size: int | str, val_size: int, test_size: int | str, train_available: int, test_available: int
) -> Tuple[int, int]:
    """Resolves `_FULL_SPLIT` sentinels against the actual split sizes and
    validates the result fits -- pulled out of `_build_cifar10_loaders` so
    it's unit-testable without a real CIFAR10 download (`train_available`/
    `test_available` are just plain ints here, not `len(some_dataset)`).
    Raises `ValueError` (not a silent clamp) on a misconfigured explicit
    size -- a silently truncated split would produce a real but misleading
    accuracy number rather than a loud, immediate failure."""
    resolved_train_size = train_available - val_size if train_size == _FULL_SPLIT else train_size
    resolved_test_size = test_available if test_size == _FULL_SPLIT else test_size

    if resolved_train_size + val_size > train_available:
        raise ValueError(
            f"{_TRAIN_SIZE_ENV} ({resolved_train_size}) + {_VAL_SIZE_ENV} ({val_size}) exceeds CIFAR10's "
            f"{train_available}-image train split -- reduce one of them or set {_TRAIN_SIZE_ENV}={_FULL_SPLIT!r}."
        )
    if resolved_test_size > test_available:
        raise ValueError(
            f"{_TEST_SIZE_ENV} ({resolved_test_size}) exceeds CIFAR10's {test_available}-image test split."
        )
    return resolved_train_size, resolved_test_size


def _build_cifar10_loaders(
    train_size: int | str, val_size: int, test_size: int | str, batch_size: int, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Three *disjoint* loaders: `train`/`val` are non-overlapping slices of
    CIFAR10's own 50k-image train split (so validation-set early stopping
    never sees a training example), and `test` is CIFAR10's separate
    10k-image test split, entirely untouched until the one final-accuracy
    call in `run_qat_tuning`. `train_size`/`test_size` may be the literal
    string `"full"` (`_FULL_SPLIT`) to use every image in that split;
    `train_size="full"` reserves `val_size` images for validation first and
    uses the remainder for training, since both can't literally claim the
    whole 50k-image train split at once."""
    import torchvision.transforms as T
    from torch.utils.data import Subset
    from torchvision.datasets import CIFAR10

    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            # ImageNet channel stats -- matches the pretrained resnet18's
            # expected input normalization.
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_full = CIFAR10(root=str(_CACHE_ROOT / "cifar10"), train=True, download=True, transform=transform)
    test_full = CIFAR10(root=str(_CACHE_ROOT / "cifar10"), train=False, download=True, transform=transform)

    resolved_train_size, resolved_test_size = _resolve_split_sizes(
        train_size, val_size, test_size, len(train_full), len(test_full)
    )

    generator = torch.Generator().manual_seed(seed)
    train_val_idx = torch.randperm(len(train_full), generator=generator).tolist()
    train_idx = train_val_idx[:resolved_train_size]
    val_idx = train_val_idx[resolved_train_size : resolved_train_size + val_size]
    test_idx = torch.randperm(len(test_full), generator=generator)[:resolved_test_size].tolist()

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(train_full, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(test_full, test_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


_backend_cache: Dict[str, QATBackend] = {}


def get_qat_backend(model_id: str) -> QATBackend:
    """Real backend, cached per `model_id` for the life of the process (the
    ~44MB pretrained-weight download and ~170MB CIFAR10 download only
    happen once; both cache to disk under `.cache/` after that). Each
    `run_qat_tuning` call still deep-copies `base_model` before mutating,
    so the cached instance itself is never fine-tuned in place. Dataset
    scope (train/val/test/batch size) is read from `AUTOCIM_QAT_*` env vars
    at call time (see the block above), not baked in at import time, so
    tests that set them via `monkeypatch.setenv` before the first call for
    a given `model_id` take effect."""
    if model_id not in _backend_cache:
        base_model = _get_base_model(model_id)
        train_loader, val_loader, test_loader = _build_cifar10_loaders(
            train_size=_resolve_size_env(_TRAIN_SIZE_ENV, _DEFAULT_TRAIN_SIZE, allow_full=True),
            val_size=_resolve_int_env(_VAL_SIZE_ENV, _DEFAULT_VAL_SIZE),
            test_size=_resolve_size_env(_TEST_SIZE_ENV, _DEFAULT_TEST_SIZE, allow_full=True),
            batch_size=_resolve_int_env(_BATCH_SIZE_ENV, _DEFAULT_BATCH_SIZE),
            seed=_resolve_int_env(_SEED_ENV, _DEFAULT_DATASET_SEED),
        )
        _backend_cache[model_id] = QATBackend(
            base_model=base_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=_NUM_CLASSES,
            layer_groups=group_quantizable_layers(base_model),
        )
    return _backend_cache[model_id]
