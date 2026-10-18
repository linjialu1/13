"""ACBM interface (v2).

Fixes over v1:
  * ``freeze_representation`` freezes the ENTIRE representation path -- every parameter
    of the model, including the CfC encoder AND the temporal attention. The trainable
    part is a NEWLY built sparse head. Per-parameter names are all listed; the check no
    longer relies on the ``encoder.*`` prefix alone.
  * ``representation`` performs REAL batching and counts model invocations with a
    forward hook, instead of estimating ``ceil(n / EVAL_BATCH)``. The model is called
    once per batch, and the returned count comes from the hook.
  * every array carries a ``row_id``; the joint builder asserts row-wise consistency.

Interface only. This module NEVER trains, never calls backward, never steps an optimizer.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from trunk.descriptors import NonFiniteError, assert_all_finite  # noqa: E402

PIPELINE_SOURCE = ROOT / "experiments/archive/2026-10-13/cfc_acbm_t3_component_ablation_3seeds/pipeline.py"
CICBM_SOURCE = ROOT / "experiments/archive/stage4/refined/benchmark.py"
BENCH_ROOT = ROOT / "src/benchmark.py"
BENCH_LEGACY = ROOT / "src/legacy/benchmark.py"
HIDDEN = 48
DESCRIPTOR_DIM = 8
JOINT_DIM = HIDDEN + DESCRIPTOR_DIM          # 56
CONCEPT_BOTTLENECK_ENABLED = False

_pipe = None
_cicbm = None


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pipeline_module():
    global _pipe
    if _pipe is None:
        _pipe = _load("trunk_v2_pipeline_impl", PIPELINE_SOURCE)
    return _pipe


def cicbm_module():
    global _cicbm
    if _cicbm is None:
        _cicbm = _load("trunk_v2_cicbm_iface", CICBM_SOURCE)
    return _cicbm


def resolved_module_paths() -> dict:
    return {
        "data/benchmark (SrcBenchmark)": str(BENCH_ROOT),
        "cil/benchmark (LegacyBenchmark)": str(BENCH_LEGACY),
        "shared_pipeline": str(PIPELINE_SOURCE),
        "cicbm_implementation": str(CICBM_SOURCE),
        "sys_modules['benchmark']": getattr(sys.modules.get("benchmark"), "__file__", None),
        "sys_modules['pipeline']": getattr(sys.modules.get("pipeline"), "__file__", None),
        "note": "两个同名 benchmark 模块职责不同；裸 import 会静默取错。",
    }


# ------------------------------------------------------------------ model
def build_model(input_channels: int, n_classes: int, seed: int = 11):
    """Randomly initialised model for SHAPE CHECKING ONLY. Outputs are not performance."""
    torch.manual_seed(seed)
    model = pipeline_module().TemporalAttentionSequenceClassifier("CfC", input_channels, HIDDEN, n_classes)
    model.eval()
    return model


def freeze_representation(model) -> dict:
    """Freeze the WHOLE representation path: CfC encoder AND temporal attention.

    Every parameter is listed explicitly by name; the assertion is over the full
    parameter dict, not over the ``encoder.*`` prefix.
    """
    for _name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
    per_param = {name: bool(p.requires_grad) for name, p in model.named_parameters()}
    encoder = [n for n in per_param if n.startswith("encoder.")]
    attention = [n for n in per_param if n.startswith("temporal_attention.")]
    head = [n for n in per_param if n.startswith("head.")]
    return {"per_parameter_requires_grad": per_param,
            "encoder_parameter_names": encoder, "attention_parameter_names": attention,
            "head_parameter_names": head,
            "any_parameter_trainable_on_model": any(per_param.values()),
            "encoder_all_frozen": all(not per_param[n] for n in encoder),
            "attention_all_frozen": all(not per_param[n] for n in attention),
            "head_all_frozen": all(not per_param[n] for n in head),
            "trainable_names": [n for n, v in per_param.items() if v],
            "check_method": "listed every parameter by name (not an encoder.* prefix shortcut)"}


def build_sparse_head(in_dim: int, n_classes: int) -> torch.nn.Linear:
    """Same shape as the archived L1 sparse head; freshly built, therefore trainable. NOT trained."""
    from torch import nn
    return nn.Linear(in_dim, n_classes)


@dataclass
class ForwardResult:
    latent: np.ndarray
    row_ids: np.ndarray
    encoder_hook_calls: int        # torch forward hook on model.encoder (one per batch)
    method_call_count: int         # direct invocations of model.representation_and_logits
    model_call_hook_calls: int     # hook on the TOP-LEVEL module; expected 0 here
    batches: int
    batch_size: int


def representation(model, x_std: np.ndarray, row_ids, device, batch_size: int | None = None) -> ForwardResult:
    """REAL batching. Invocations are counted by a torch forward hook, not estimated.

    Counting note (this is a correction, see ERRATA_interface_check_v2.md E2): the
    representation is obtained through a *direct method call*
    ``model.representation_and_logits(x)``. That bypasses ``nn.Module.__call__``, so a
    hook registered on the top-level module never fires -- hooking the model itself
    would silently report 0. The genuine torch forward hook is therefore registered on
    ``model.encoder``, which ``representation_and_logits`` always reaches through
    ``self.encoder(x)`` (a real ``__call__``) exactly once per batch. Two independent
    counters are returned and the caller asserts they agree:
      * ``encoder_hook_calls``   -- torch forward hook
      * ``method_call_count``    -- wrapper around the bound method
      * ``model_call_hook_calls``-- hook on the top-level module (0 by construction;
                                    returned so the absence is on the record, not implied)
    Ordering is preserved exactly; the caller supplies the row_ids.
    """
    x = assert_all_finite(np.asarray(x_std, dtype=np.float32), "representation input")
    ids = np.asarray(row_ids, dtype=np.int64)
    if len(ids) != len(x):
        raise ValueError(f"row_ids length {len(ids)} != rows {len(x)}")
    bs = int(batch_size or pipeline_module().EVAL_BATCH)
    calls = {"encoder_hook": 0, "method": 0, "model_hook": 0}

    def _enc_hook(_module, _inputs, _output):
        calls["encoder_hook"] += 1

    def _model_hook(_module, _inputs, _output):
        calls["model_hook"] += 1

    enc_handle = model.encoder.register_forward_hook(_enc_hook)
    model_handle = model.register_forward_hook(_model_hook)

    original = model.representation_and_logits
    had = "representation_and_logits" in model.__dict__
    previous = model.__dict__.get("representation_and_logits")

    def _counting(*a, **kw):
        calls["method"] += 1
        return original(*a, **kw)

    chunks, n_batches = [], 0
    try:
        model.representation_and_logits = _counting
        with torch.inference_mode():
            for start in range(0, len(x), bs):
                part = x[start:start + bs]
                t = torch.from_numpy(np.ascontiguousarray(part)).to(device)
                rep, _logits = model.representation_and_logits(t)
                arr = rep.detach().cpu().numpy().astype(np.float32)
                assert_all_finite(arr, f"latent batch {n_batches}")
                chunks.append(arr)
                n_batches += 1
    finally:
        enc_handle.remove()
        model_handle.remove()
        if had:
            model.__dict__["representation_and_logits"] = previous
        else:
            model.__dict__.pop("representation_and_logits", None)
    latent = np.concatenate(chunks, axis=0) if chunks else np.empty((0, HIDDEN), dtype=np.float32)
    if latent.shape[0] != len(ids):
        raise ValueError(f"latent rows {latent.shape[0]} != requested rows {len(ids)}")
    return ForwardResult(latent=latent, row_ids=ids,
                         encoder_hook_calls=calls["encoder_hook"],
                         method_call_count=calls["method"],
                         model_call_hook_calls=calls["model_hook"],
                         batches=n_batches, batch_size=bs)


@dataclass
class JointBundle:
    row_ids: np.ndarray
    labels: np.ndarray
    latent: np.ndarray
    desc_raw: np.ndarray
    desc_std: np.ndarray
    joint: np.ndarray


def build_joint(row_ids, labels, latent: np.ndarray, desc_raw: np.ndarray, desc_std: np.ndarray,
                latent_row_ids=None, desc_row_ids=None) -> JointBundle:
    """Assert row-wise consistency of latent / descriptor / label, then concatenate.

    The first HIDDEN columns come from the latent and the last DESCRIPTOR_DIM from the
    standardized descriptor; both are asserted to belong to the same row_id ordering.
    """
    ids = np.asarray(row_ids, dtype=np.int64)
    lab = np.asarray(labels, dtype=np.int64)
    latent = assert_all_finite(np.asarray(latent, dtype=np.float32), "joint latent")
    desc_raw = assert_all_finite(np.asarray(desc_raw, dtype=np.float64), "joint raw descriptor")
    desc_std = assert_all_finite(np.asarray(desc_std, dtype=np.float32), "joint standardized descriptor")
    n = len(ids)
    if not (len(lab) == len(latent) == len(desc_raw) == len(desc_std) == n):
        raise ValueError(f"row count mismatch: ids={n} labels={len(lab)} latent={len(latent)} "
                         f"desc_raw={len(desc_raw)} desc_std={len(desc_std)}")
    if latent_row_ids is not None:
        if not np.array_equal(np.asarray(latent_row_ids, dtype=np.int64), ids):
            raise ValueError("latent row_ids do not match the requested row order")
    if desc_row_ids is not None:
        if not np.array_equal(np.asarray(desc_row_ids, dtype=np.int64), ids):
            raise ValueError("descriptor row_ids do not match the requested row order")
    if latent.shape[1] != HIDDEN or desc_std.shape[1] != DESCRIPTOR_DIM:
        raise ValueError(f"unexpected dims: latent {latent.shape}, descriptor {desc_std.shape}")
    joint = np.concatenate([latent, desc_std], axis=1).astype(np.float32)
    assert_all_finite(joint, "joint output")
    return JointBundle(row_ids=ids, labels=lab, latent=latent, desc_raw=desc_raw,
                       desc_std=desc_std, joint=joint)


def local_lookup(y, seen) -> np.ndarray:
    lookup = {int(c): i for i, c in enumerate(seen)}
    return np.asarray([lookup[int(v)] for v in np.asarray(y)], dtype=np.int64)


def standardize_input(x_raw: np.ndarray, mean, denom) -> np.ndarray:
    """Apply the ACCEPTED Task-1 frozen input statistics. No re-standardization inside the model."""
    x = assert_all_finite(np.asarray(x_raw, dtype=np.float64), "standardize_input raw input")
    m = np.asarray(mean, dtype=np.float64)
    d = np.asarray(denom, dtype=np.float64)
    assert_all_finite(m, "standardize_input mean")
    assert_all_finite(d, "standardize_input denominator")
    if not (d > 0).all():
        raise NonFiniteError("non-positive input standardization denominator")
    out = ((x - m) / d).astype(np.float32)
    return assert_all_finite(out, "standardize_input output")


def params_signature(model) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def pooled_shift_interface_check() -> dict:
    src = PIPELINE_SOURCE.read_text(encoding="utf-8")
    return {"pseudo_generation_default": "pooled_shift",
            "formula_present": "pseudo_z = source - current_means[nearest] + old_mean" in src,
            "executed_this_round": False,
            "note": "本轮只核对接口存在与公式文本，不生成任何伪样本"}
