"""S3 production adapters — FIX6.

Interface corrections from admin/REVIEW_s3_fix2_20261017.md:

  1. `freeze_all` uses `acbm_interface.freeze_representation` (not rb1), zeroes grads and
     verifies the whole parameter+buffer signature plus grad=None.
  2. `make_joint` returns a 4-tuple — `extract_joint` unpacks it and returns the joint only;
     every caller (train / val / test) uses the same contract.
  3. counters are initialised with EVERY key the accepted trainer/head fitter increments,
     and encoder/sparse-head forward hooks are attached so forwards are measured.
  4. device is explicit and propagated (CUDA required by protocol; CPU only for the
     labelled engineering path).
  5. per-boundary evidence is persisted: state package (float32 centroids + counts +
     mapping), ORIGINAL row ids / labels, train+val joint, unrounded val logits and
     confusion, freeze verification, and a per-boundary budget ledger.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
S1 = ROOT / "experiments/baselines/trunk_baseline_4datasets"
S2 = ROOT / "experiments/sensitivity/trunk_readout_sensitivity_4datasets"
EVAL_BATCH = 256
JOINT_DIM = 56
TASK_COUNTS = {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}
ALL_TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

COUNTER_KEYS = ["encoder_forwards", "encoder_backward", "encoder_optimizer_steps",
                "head_forwards", "head_backward", "head_optimizer_steps",
                "original_head_forwards", "sparse_head_fits", "head_epochs",
                "gradient_updates", "test_forwards"]


def new_counters() -> dict:
    return {k: 0 for k in COUNTER_KEYS}


def sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def resolve_device(allow_cpu: bool = False):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA unavailable: protocol requires GPU (no silent CPU fallback)")


def set_seed_all(seed: int) -> None:
    """Baseline-identical seeding (python/numpy/torch/cuda/cudnn)."""
    from trunk import run_baseline_fix2 as rb1
    rb1.set_seed(seed)


# ---------------------------------------------------------------- context
class TrainCtx:
    """Train/val-only context; positional indices map back to ORIGINAL row ids."""

    def __init__(self, dataset: str):
        from trunk import readout_sensitivity as rs
        base = rs.load_train_context(dataset)
        self.name = dataset
        self.keep = np.asarray(base["keep"], dtype=np.int64)
        self.x_std = base["x_std_keep"]
        self.x_raw = base["x_raw"][base["keep"]]
        self.y = np.asarray(base["y"][base["keep"]])
        self.pos = {int(o): i for i, o in enumerate(self.keep.tolist())}
        self.d_mean = base["d_mean"]
        self.d_std = base["d_std"]
        self.input_channels = int(base["input_channels"])
        self.n_classes = int(base["n_classes"])
        self.train_all = [self.pos[int(i)] for i in base["train_all"] if int(i) in self.pos]
        self.val_all = [self.pos[int(i)] for i in base["val_all"] if int(i) in self.pos]

    def rows(self, classes, which: str) -> np.ndarray:
        base = self.train_all if which == "train" else self.val_all
        return np.asarray(sorted(i for i in base if int(self.y[i]) in classes), dtype=np.int64)

    def orig_ids(self, pos_ids) -> np.ndarray:
        return np.asarray([int(self.keep[int(i)]) for i in pos_ids], dtype=np.int64)


def extract_joint(model, ctx: TrainCtx, pos_ids) -> np.ndarray:
    """Returns ONLY the 56-dim joint; `make_joint`'s 4-tuple is unpacked here."""
    from trunk import train_smoke as ts
    out = ts.make_joint(model, ctx.x_std, ctx.x_raw, pos_ids,
                        ctx.d_mean, ctx.d_std, ctx.name, _device(model))
    joint, latent, desc_raw, desc_std = out          # explicit 4-value contract
    if joint.shape[1] != JOINT_DIM:
        raise RuntimeError(f"joint dim {joint.shape[1]} != {JOINT_DIM}")
    return joint


def _device(model):
    return next(model.parameters()).device


# ---------------------------------------------------------------- build / train / freeze
def build_candidate(candidate: str, ctx: TrainCtx, seed: int, device):
    from trunk import s3_candidates_torch_fix1 as ct
    set_seed_all(seed)
    model = ct.build_wired_candidate(candidate, ctx.input_channels, ctx.n_classes)
    return model.to(device)


def attach_hooks(model, counters: dict) -> list:
    """Install model-level hooks ONCE (encoder + original head); return removable handles.

    Sparse-head forwards are counted by the accepted `fit_sparse_head` itself, so no head
    hook is registered here — that would double count.
    """
    handles = []
    if hasattr(model, "encoder"):
        handles.append(model.encoder.register_forward_hook(
            lambda *a: counters.__setitem__("encoder_forwards",
                                            counters["encoder_forwards"] + 1)))
    head = getattr(model, "head", None)
    if head is not None:
        handles.append(head.register_forward_hook(
            lambda *a: counters.__setitem__("original_head_forwards",
                                            counters["original_head_forwards"] + 1)))
    return handles


def detach_hooks(handles) -> None:
    for h in handles or []:
        try:
            h.remove()
        except Exception:
            pass


class _CountingAdam(torch.optim.Adam):
    """Counts real optimizer steps (gradient_updates) instead of inferring them."""

    def __init__(self, params, counters, **kw):
        super().__init__(params, **kw)
        self._counters = counters

    def step(self, closure=None):
        out = super().step(closure)
        self._counters["gradient_updates"] = self._counters.get("gradient_updates", 0) + 1
        return out


def train_task1(model, ctx: TrainCtx, seed: int, counters: dict, device) -> dict:
    from trunk import run_baseline_fix2 as rb1
    mini = {"x_std": ctx.x_std, "y": ctx.y}
    t1_train = ctx.rows([0, 1], "train")
    t1_val = ctx.rows([0, 1], "val_cum")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = _CountingAdam(params, counters, lr=rb1.ENCODER_LR)
    info = rb1.train_task1_encoder(model, opt, mini, t1_train, t1_val,
                                   ctx.n_classes, device, counters, seed)
    return info


def freeze_all(model, counters: dict | None = None) -> dict:
    """Freeze whole representation, clear grads, verify signature and grad state."""
    from trunk import acbm_interface as ai
    from trunk import run_baseline_fix2 as rb1
    sig_before = rb1.full_signature(model)
    report = ai.freeze_representation(model)          # correct module
    model.zero_grad(set_to_none=True)
    sig_after = rb1.full_signature(model)
    grads_none = all(p.grad is None for p in model.parameters())
    frozen = all(not p.requires_grad for p in model.parameters())
    if sig_before != sig_after:
        raise RuntimeError("freeze changed the parameter/buffer signature")
    if not grads_none or not frozen:
        raise RuntimeError("freeze did not clear grads / fully freeze")
    return {"signature_sha256": sig_after, "signature_unchanged": True,
            "grads_all_none": grads_none, "all_frozen": frozen,
            "per_parameter": report.get("per_parameter_requires_grad", {})}


def common_param_snapshot(candidate: str, input_channels: int, n_classes: int,
                          seed: int) -> dict:
    """Initial values of parameters shared by LN / POOL / C1 at the SAME seed.

    Generated before any training; a mismatch means the candidates no longer differ only
    in their intended structure and the run must stop.
    """
    import hashlib
    from trunk import s3_candidates_torch_fix1 as ct
    snaps = {}
    for cand in ("LN_layernorm_injection", "POOL_mean_pooling", "C1_backbone_freeze"):
        set_seed_all(seed)          # reseed per candidate so RNG state is identical
        m = ct.build_wired_candidate(cand, input_channels, n_classes)
        shared = {n: p.detach().cpu().numpy().tobytes() for n, p in m.named_parameters()
                  if n.startswith(("encoder.rnn_cell.backbone", "encoder.rnn_cell.ff",
                                   "encoder.rnn_cell.time_", "temporal_attention"))}
        snaps[cand] = {n: hashlib.sha256(b).hexdigest() for n, b in shared.items()}
    keys = set(snaps["LN_layernorm_injection"])
    for cand in ("POOL_mean_pooling", "C1_backbone_freeze"):
        keys &= set(snaps[cand])
    mismatched = [n for n in sorted(keys)
                  if len({snaps[c][n] for c in snaps}) != 1]
    return {"shared_param_count": len(keys), "mismatched": mismatched,
            "all_shared_initial_values_equal": not mismatched,
            "per_candidate_shared_sha256": {c: {n: snaps[c][n] for n in sorted(keys)}
                                            for c in snaps}}


# ---------------------------------------------------------------- boundary heads
def run_boundaries(model, ctx: TrainCtx, *, seed: int, out_dir: Path, counters: dict,
                   candidate: str, device, frozen_baseline: str | None = None) -> dict:
    import torch.nn.functional as F
    from trunk import run_baseline_fix2 as rb1
    from trunk import descriptors as ds
    tasks = ALL_TASKS[:TASK_COUNTS[ctx.name]]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    centroids, counts, task_of = {}, {}, {}
    per_task, freeze_ev = {}, []
    for task, new_classes in enumerate(tasks, start=1):
        seen = list(range(2 * task))
        pos = ctx.rows(new_classes, "train")
        z = extract_joint(model, ctx, pos)
        y_t = ctx.y[pos].astype(np.int64)
        current = {c: z[y_t == c].mean(0).astype(np.float32) for c in new_classes}
        old = [c for c in seen if c not in new_classes]
        parts, plabels, manifest = ([], [], []) if not old else \
            rb1.pooled_shift_pseudo(centroids, current, z, y_t, pos, old, new_classes)
        for c in new_classes:
            centroids[c] = current[c]
            counts[c] = int((y_t == c).sum())
            task_of[c] = task
        # evidence: original row ids (not positional) in the pseudo manifest
        for e in manifest:
            e["source_row_ids_original"] = ctx.orig_ids(e["source_row_ids"]).tolist()
        X = np.concatenate([z.astype(np.float32)] + parts, axis=0).astype(np.float32)
        Y = np.concatenate([y_t] + plabels, axis=0).astype(np.int64)
        torch.manual_seed(seed + 10000 * task)
        head = nn.Linear(JOINT_DIM, 2 * task).to(device)
        head_fits_before = counters["sparse_head_fits"]
        losses, _ = rb1.fit_sparse_head(head, X, Y, device,
                                        f"{candidate}:{ctx.name}:s{seed}:t{task}", counters)
        counters["sparse_head_fits"] = head_fits_before + 1
        counters["head_epochs"] += len(losses)
        torch.save(head.state_dict(), out_dir / f"task{task}_head.pt")
        vpos = ctx.rows(seen, "val_cum")
        vz = extract_joint(model, ctx, vpos)
        # boundary validation: count this head call with a one-shot hook (measured, not inferred)
        _h = head.register_forward_hook(
            lambda *a: counters.__setitem__("head_forwards", counters["head_forwards"] + 1))
        try:
            with torch.inference_mode():
                logits = head(torch.from_numpy(vz).to(device)).cpu().numpy()
        finally:
            _h.remove()
        pred = logits.argmax(1)
        accs = [float((pred[ctx.y[vpos] == c] == c).mean()) for c in seen]
        cm = np.zeros((2 * task, 2 * task), dtype=np.int64)
        yv = ctx.y[vpos].astype(np.int64)
        for i in range(len(yv)):
            cm[yv[i], int(pred[i])] += 1
        # persisted per-boundary evidence
        np.savez_compressed(out_dir / f"boundary_task{task}.npz",
                            train_pos=pos, train_orig_ids=ctx.orig_ids(pos),
                            train_labels=y_t, train_joint=z,
                            val_pos=vpos, val_orig_ids=ctx.orig_ids(vpos),
                            val_labels=yv, val_joint=vz,
                            val_logits=logits.astype(np.float64),
                            confusion=cm)
        np.savez_compressed(out_dir / f"state_package_task{task}.npz",
                            seen_classes=np.asarray(seen, dtype=np.int64),
                            class_counts=np.asarray([counts[c] for c in seen], dtype=np.int64),
                            task_ids=np.asarray([task_of[c] for c in seen], dtype=np.int64),
                            **{f"centroid_{c}": centroids[c] for c in seen})
        sig_now = rb1.full_signature(model)
        if frozen_baseline is not None and sig_now != frozen_baseline:
            raise RuntimeError(f"frozen representation changed at task{task}")
        freeze_ev.append({"task": task, "frozen_signature_sha256": sig_now,
                          "matches_task1_baseline": (frozen_baseline is None or sig_now == frozen_baseline),
                          "grads_none": all(p.grad is None for p in model.parameters())})
        per_task[f"task{task}"] = {
            "val_macro": float(np.mean(accs)), "n_train": int(len(pos)),
            "n_val_cum": int(len(vpos)), "n_pseudo": int(sum(len(p) for p in parts)),
            "head_path": str(out_dir / f"task{task}_head.pt"),
            "head_sha256": sha(out_dir / f"task{task}_head.pt"),
            "boundary_npz_sha256": sha(out_dir / f"boundary_task{task}.npz"),
            "state_pkg_sha256": sha(out_dir / f"state_package_task{task}.npz"),
            "sources": manifest}
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "candidate": candidate, "dataset": ctx.name, "seed": seed},
               out_dir / "model.pt")
    return {"per_task": per_task, "model_path": str(out_dir / "model.pt"),
            "model_sha256": sha(out_dir / "model.pt"),
            "freeze_evidence": freeze_ev,
            "frozen_signature_sha256": rb1.full_signature(model)}


def final_val_macro(per_task: dict, dataset: str) -> float:
    """In-memory validation for the final task — no file read."""
    return float(per_task[f"task{TASK_COUNTS[dataset]}"]["val_macro"])


# ---------------------------------------------------------------- readouts (0 fits)
def readout_val_macro(candidate: str, dataset: str, seed: int) -> dict:
    from trunk import s3_candidates_numpy_fix1 as cn
    task = TASK_COUNTS[dataset]
    data = np.load(S2 / dataset / f"seed{seed}" / f"task{task}" / "features.npz")
    asset = cn.load_s2_readout_asset(S2, dataset, seed, task)
    X = data["val_joint"]
    seen = list(range(2 * task))
    if candidate == "R1_analytic_head":
        _, pred = cn.ridge_predict(asset["D1_W"], asset["D1_b"], X.astype(np.float64), seen)
    else:
        Z = cn.apply_pca_after_pseudo(X, asset["D3_mean"], asset["D3_components"])
        _, pred = cn.ridge_predict(asset["D3_W"], asset["D3_b"], Z, seen)
    return {"val_macro": cn.macro_accuracy(pred, data["val_labels"], seen),
            "n_val_cum": int(len(X)), "asset_refs": asset["source_hashes"]}


def s1_baseline_val_macro(dataset: str, seed: int, task: int, device) -> float:
    data = np.load(S2 / dataset / f"seed{seed}" / f"task{task}" / "features.npz")
    X = torch.from_numpy(data["val_joint"]).to(device)      # float32 on the protocol device
    y = data["val_labels"]
    sd = torch.load(S1 / dataset / f"seed{seed}" / f"task{task}_head.pt",
                    map_location="cpu", weights_only=False)
    head = nn.Linear(JOINT_DIM, 2 * task).to(device)
    head.load_state_dict(sd)
    head.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(X), EVAL_BATCH):
            preds.append(head(X[i:i + EVAL_BATCH]).argmax(1))
    pred = torch.cat(preds).cpu().numpy()
    return float(np.mean([float((pred[y == c] == c).mean()) for c in range(2 * task)]))


# ---------------------------------------------------------------- test (sealed only)
def test_macro_from_seed_result(candidate: str, dataset: str, seed: int,
                                seed_result: dict, device, *, out_dir: Path | None = None,
                                counters: dict | None = None) -> float:
    """`seed_result` is the verified per-seed record (has model_path/per_task)."""
    from trunk import run_baseline_fix2 as rb1
    from trunk import train_smoke as ts
    task = TASK_COUNTS[dataset]
    seen = list(range(2 * task))
    ctx = rb1.load_context(dataset)                     # test view only here
    tst = ctx["task_rows"](seen, "test")
    labels = ctx["test_view"]["y"][tst].astype(np.int64)
    if candidate in ("R1_analytic_head", "R2_pca16_readout"):
        from trunk import s3_candidates_numpy_fix1 as cn
        asset = cn.load_s2_readout_asset(S2, dataset, seed, task)
        model = rb1.build_model(ctx["input_channels"], ctx["n_classes"], 0).to(device)
        ck = torch.load(S1 / dataset / f"seed{seed}" / "task1_encoder_best.pt",
                        map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        from trunk import acbm_interface as ai
        ai.freeze_representation(model)          # correct module (was rb1.*)
        model.to(device).eval()
        joint = extract_joint_test(model, ctx, tst, dataset, device)
        if candidate == "R1_analytic_head":
            logits, pred = cn.ridge_predict(asset["D1_W"], asset["D1_b"],
                                            joint.astype(np.float64), seen)
        else:
            Z = cn.apply_pca_after_pseudo(joint, asset["D3_mean"], asset["D3_components"])
            logits, pred = cn.ridge_predict(asset["D3_W"], asset["D3_b"], Z, seen)
        if counters is not None:
            counters["test_forwards"] = counters.get("test_forwards", 0) + 1
        if out_dir is not None:
            _save_test_evidence(Path(out_dir), task, tst, labels, joint, logits, pred, seen,
                                "float64 (ridge readout)")
        return cn.macro_accuracy(pred, labels, seen)

    ck = torch.load(seed_result["model_path"], map_location="cpu", weights_only=False)
    from trunk import s3_candidates_torch_fix1 as ct
    model = ct.build_wired_candidate(candidate, ctx["input_channels"], ctx["n_classes"])
    model.load_state_dict(ck["model_state_dict"], strict=True)
    from trunk import acbm_interface as ai
    ai.freeze_representation(model)              # correct module
    model.to(device).eval()
    joint = extract_joint_test(model, ctx, tst, dataset, device)
    head = nn.Linear(JOINT_DIM, 2 * task).to(device)
    head.load_state_dict(torch.load(seed_result["per_task"][f"task{task}"]["head_path"],
                                    map_location="cpu", weights_only=False))
    head.eval()
    with torch.inference_mode():
        logits_t = head(torch.from_numpy(joint).to(device))
        pred = logits_t.argmax(1).cpu().numpy()
        scores = logits_t.cpu().numpy()          # float32 head scores
    if counters is not None:
        counters["test_forwards"] = counters.get("test_forwards", 0) + 1
    if out_dir is not None:
        _save_test_evidence(Path(out_dir), task, tst, labels, joint, scores, pred, seen,
                            "float32 (trained sparse head)")
    return float(np.mean([float((pred[labels == c] == c).mean()) for c in seen]))


def _save_test_evidence(out_dir: Path, task: int, tst_idx, labels, joint, scores, pred,
                        seen, compute_dtype_note: str, overwrite: bool = False):
    """Persist REAL score matrices (not class ids) plus per-row identity and confusion.

    Never overwrites: a second save to the same path is an error, so a partially written
    evaluation can not be silently replaced.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = Path(out_dir) / f"test_task{task}.npz"
    if target.exists() and not overwrite:
        raise RuntimeError(f"test evidence already exists (no overwrite): {target}")
    scores = np.asarray(scores)
    pred = np.asarray(pred, dtype=np.int64)
    n, k = scores.shape
    if (n, k) != (len(labels), len(seen)):
        raise RuntimeError(f"score matrix {scores.shape} != ({len(labels)},{len(seen)})")
    if not np.isfinite(scores.astype(np.float64)).all():
        raise RuntimeError("non-finite test scores")
    if list(np.asarray(seen).tolist()) != sorted(int(c) for c in seen):
        raise RuntimeError("seen columns are not in canonical ascending order")
    if not np.array_equal(scores.argmax(axis=1).astype(np.int64), pred):
        raise RuntimeError("argmax(scores) != prediction")
    cm = np.zeros((2 * task, 2 * task), dtype=np.int64)
    for i in range(len(labels)):
        cm[labels[i], int(pred[i])] += 1
    np.savez_compressed(target,
                        row_ids=np.asarray(tst_idx, dtype=np.int64),
                        labels=np.asarray(labels, dtype=np.int64),
                        joint=np.asarray(joint, dtype=np.float32),
                        logits=scores, prediction=pred, confusion=cm,
                        seen=np.asarray(seen, dtype=np.int64),
                        compute_dtype=compute_dtype_note,
                        note="logits = real class scores [n_rows, n_seen]; "
                             "compute_dtype records the inference precision")


def extract_joint_test(model, ctx, tst_idx, dataset, device) -> np.ndarray:
    """device is MANDATORY (no default) so a CUDA model can never feed CPU inputs."""
    from trunk import train_smoke as ts
    out = ts.make_joint(model, ctx["x_std_test"], ctx["test_view"]["x_raw"], tst_idx,
                        ctx["d_mean"], ctx["d_std"], dataset, device)
    joint = out[0]
    return joint
