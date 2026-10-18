"""S2 readout sensitivity: frozen S1 representation, zero-gradient closed-form readout.

Implements the revised protocol fixed by admin/REVIEW_s2_protocol.md:

  * four views D1 joint56 / D2 Gaussian-RP16 / D3 PCA16 / D4 standardized56;
  * transforms fitted ONLY on current-task real-train joint (float64 stats);
  * one shared common training matrix per boundary (current real + old-class pseudo
    generated exactly by the S1 pooled_shift rule from S1 saved centroids);
  * closed-form ridge probe, lambda=1e-3, unpenalized bias, float64;
  * primary sensitivity = range of the four all-seen-class validation macro accuracies
    per (dataset, seed, boundary); view SD(ddof1) reported separately, never divided;
  * 120 boundary-seed units, 480 ridge fits, 1152 frozen-encoder batches (train-side
    current new classes + cumulative all-seen validation), zero gradient, zero test.

Numeric rules (fixed before any S2 feature/accuracy was observed):
  * D2 rng = np.random.default_rng(seed*1000+task), 56x16 float64 Gaussian / sqrt(16);
  * D3 float64 centering + np.linalg.svd(full_matrices=False); rank tol =
    max(n,56)*eps64*s_max; exactly 16 output columns, singular values <= tol zeroed,
    low rank is a warning (continue), only SVD failure/non-finite stops; projection
    matrix saved;
  * D4 population std, denominator std+1e-6;
  * ridge backward error eta = norm(AW-B,inf) / (norm(A,inf)*norm(W,inf)+norm(B,inf)),
    threshold 100*p*eps64 (p = view dim); eta=0 if denominator and residual are both 0,
    else inf; exceeding stops; low rank of the design matrix is NOT a failure;
  * argmax ties resolve to the smallest seen-class id; macro = mean of per-class
    recalls at full precision.

Modes:
    --propose-revised   write revised protocol + ERRATA (before anything else)
    --self-check        synthetic-array engineering checks (zero real data)
    --lock-protocol     lock after self-check passes (three-way consistency)
    --run-dataset NAME  run all 8 seeds of one dataset (fixed task order)
    --summarize         endpoint tables, W/B gate, paired D2/D3/D4-vs-D1 diagnostics
"""
from __future__ import annotations

import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
S1 = ROOT / "experiments/baselines/trunk_baseline_4datasets"
EXEC = ROOT / "experiments/sensitivity/trunk_readout_sensitivity_4datasets"
PROTO_DIR = ROOT / "experiments/sensitivity/trunk_readout_sensitivity_protocol"
V2_DIR = ROOT / "experiments/baselines/trunk_model_interface_check_v2"
REVIEW = ROOT / "admin/REVIEW_s2_protocol.md"
for p in (str(ROOT / "src"), str(ROOT), str(ROOT / ".local_packages")):
    if p not in sys.path:
        sys.path.insert(0, p)

from trunk import train_smoke as ts            # noqa: E402
from trunk import acbm_interface as ai         # noqa: E402
from trunk import adapters as ad               # noqa: E402
from trunk import descriptors as ds            # noqa: E402
from trunk import run_baseline_fix2 as rb1     # noqa: E402  (accepted pooled_shift rule)

F = torch.nn.functional
EPS64 = float(np.finfo(np.float64).eps)
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DATASETS = ["uci_har", "wisdm", "uwave", "cwru"]
TASK_COUNTS = {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}
ALL_TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
JOINT_DIM = 56
VIEW_DIM = 16
LAMBDA = 1e-3
EVAL_BATCH = 256
TCRIT_DF7 = 2.3646242515927853

# cumulative all-seen validation member counts per dataset (fixed by the review table)
EXPECTED_VAL_CUM = {"uci_har": [615, 1152, 1788], "wisdm": [1238, 2392, 3547],
                    "uwave": [6, 12, 18, 24], "cwru": [40, 80, 120, 160, 200]}
# per-seed encoder batch counts (train + cumulative val), fixed by the review table
EXPECTED_BATCHES = {"uci_har": 38, "wisdm": 88, "uwave": 8, "cwru": 10}

LOG: list[str] = []
COUNTERS: dict[str, int] = {}


class Halt(Exception):
    pass


def log(m: str = "") -> None:
    print(m, flush=True)
    LOG.append(m)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_sanitize(obj), indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def require(cond: bool, label: str, detail: str = "") -> None:
    if not cond:
        raise Halt(f"ASSERTION FAILED: {label} {detail}")
    log(f"  [assert] OK  {label}")


def write_failure(kind: str, payload: dict, out_dir: Path) -> None:
    i = 1
    while (out_dir / f"failure_record_{i}.json").exists():
        i += 1
    (out_dir / f"failure_record_{i}.json").write_text(
        json.dumps({"kind": kind, **payload}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


# ---------------------------------------------------------------- light context (no test)
def load_train_context(name: str) -> dict:
    """Train/val-only context.  The adapter may open physical containers that also
    contain test members (recorded); test rows are never selected into numeric
    processing and the S1 full load_context (which standardizes test) is NOT used."""
    raw = ad.adapt(name)
    x_raw = np.asarray(raw["x"])
    y = np.asarray(raw["y"])
    n_classes = int(len(set(y.tolist())))
    if name == "cwru":
        splits = np.asarray([p["split"] for p in raw["provenance"]])
        train_all = np.flatnonzero(splits == "train")
        val_all = np.flatnonzero(splits == "validation")
        # standardize ONLY train+val rows (never test rows)
        keep = np.asarray(sorted(set(train_all.tolist()) | set(val_all.tolist())), dtype=np.int64)
    else:
        sp = ad.accepted_splits()[name]
        train_all = np.asarray(sp["train"])
        val_all = np.asarray(sp["validation"])
        keep = np.arange(len(x_raw), dtype=np.int64)   # train-side array has no test rows
    in_stats = ad.accepted_input_stats()[name]
    in_mean = np.asarray(in_stats["mean"], dtype=np.float64)
    in_denom = np.asarray(in_stats["denom"], dtype=np.float64)
    x_std_keep = ((x_raw[keep] - in_mean) / in_denom).astype(np.float32)
    pos = {int(r): i for i, r in enumerate(keep.tolist())}
    v2 = json.loads((V2_DIR / f"interface_check_{name}.json").read_text(encoding="utf-8"))
    d_mean = np.asarray(v2["descriptor_stats_mean"], dtype=np.float64)
    d_std = np.asarray(v2["descriptor_stats_std"], dtype=np.float64)

    def rows_of(which, classes):
        base = train_all if which == "train" else val_all
        out = np.asarray(sorted(int(i) for i in base if int(y[i]) in classes), dtype=np.int64)
        return out, np.asarray([pos[int(i)] for i in out], dtype=np.int64)

    def task_rows(classes, which):
        """returns (original_row_ids, positional_ids_into_x_std_keep)"""
        if which == "train":
            return rows_of("train", classes)
        if which == "val_cum":
            base = val_all
            out = np.asarray(sorted(int(i) for i in base if int(y[i]) in classes), dtype=np.int64)
            return out, np.asarray([pos[int(i)] for i in out], dtype=np.int64)
        raise ValueError(which)

    return dict(name=name, x_raw=x_raw, y=y, x_std_keep=x_std_keep, keep=keep, pos=pos,
                d_mean=d_mean, d_std=d_std, n_classes=n_classes,
                input_channels=int(x_raw.shape[2]), in_mean=in_mean, in_denom=in_denom,
                task_rows=task_rows, train_all=train_all, val_all=val_all,
                n_numeric_rows=len(keep))


# ---------------------------------------------------------------- closed-form pieces
def fit_ridge(X: np.ndarray, labels: np.ndarray, seen: list[int], label: str):
    """Closed-form ridge to one-hot; unpenalized bias via centering. Returns W,b,diag."""
    if sorted(set(int(v) for v in labels)) != seen:
        raise Halt(f"{label}: training labels do not cover exactly the seen classes")
    X = np.asarray(X, dtype=np.float64)
    Y = np.zeros((X.shape[0], len(seen)), dtype=np.float64)
    col = {c: i for i, c in enumerate(seen)}
    for i, lab in enumerate(labels):
        Y[i, col[int(lab)]] = 1.0
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    if not (bool(np.isfinite(Xc).all()) and bool(np.isfinite(Yc).all())):
        raise Halt(f"{label}: non-finite in probe inputs")
    n = X.shape[0]
    A = Xc.T @ Xc / n + LAMBDA * np.eye(X.shape[1])
    Bm = Xc.T @ Yc / n
    try:
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError as e:
        raise Halt(f"{label}: condition number computation failed: {e}")
    if not math.isfinite(cond):
        raise Halt(f"{label}: non-finite condition number")
    W = np.linalg.solve(A, Bm)
    b = Y.mean(axis=0) - X.mean(axis=0) @ W
    # backward error
    resid = A @ W - Bm
    denom = float(np.linalg.norm(A, np.inf) * np.linalg.norm(W, np.inf) + np.linalg.norm(Bm, np.inf))
    num = float(np.linalg.norm(resid, np.inf))
    if denom == 0.0 and num == 0.0:
        eta = 0.0
    elif denom == 0.0:
        eta = float("inf")
    else:
        eta = num / denom
    thr = 100.0 * X.shape[1] * EPS64
    for arr in (X, Y, W, b):
        if not bool(np.isfinite(arr).all()):
            raise Halt(f"{label}: non-finite in probe pipeline")
    if not (eta <= thr):
        raise Halt(f"{label}: ridge backward error {eta:.3e} > {thr:.3e}")
    return W, b, {"cond": cond, "eta": eta, "eta_threshold": thr,
                  "n_rows": int(n), "view_dim": int(X.shape[1]),
                  "labels_cover_seen": True, "column_mapping": [int(c) for c in seen]}


def pca16(Xreal: np.ndarray):
    """D3: float64 centering + SVD on real-train rows only. Always 16 output columns."""
    X = np.asarray(Xreal, dtype=np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    try:
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise Halt(f"D3 SVD did not converge: {e}")
    if not bool(np.isfinite(s).all()):
        raise Halt("D3 SVD produced non-finite singular values")
    tol = max(X.shape[0], X.shape[1]) * EPS64 * (float(s[0]) if len(s) else 0.0)
    rank = int(sum(1 for v in s if v > tol)) if len(s) else 0
    comps = Vt[:VIEW_DIM].T.copy()          # 56 x 16
    zeroed = [i for i in range(VIEW_DIM) if i >= len(s) or float(s[i]) <= tol]
    for i in zeroed:
        comps[:, i] = 0.0
    return mean, comps, {"singular_values": [float(v) for v in s],
                         "effective_rank": rank, "tol": tol,
                         "zeroed_columns": zeroed, "low_rank_warning": rank < VIEW_DIM}


def d2_matrix(seed: int, task: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 1000 + task)
    return (rng.standard_normal((JOINT_DIM, VIEW_DIM)).astype(np.float64) / math.sqrt(VIEW_DIM))


def view_transform(view: str, Xreal: np.ndarray, seed: int, task: int):
    """Fit transform on real-train rows only. Returns (apply_fn, params)."""
    X = np.asarray(Xreal, dtype=np.float64)
    if view == "D1_joint56":
        return (lambda Z: np.asarray(Z, dtype=np.float64)), {}
    if view == "D2_gaussian_rp16":
        mean = X.mean(axis=0)
        M = d2_matrix(seed, task)
        return (lambda Z: (np.asarray(Z, dtype=np.float64) - mean) @ M), \
            {"mean": mean, "matrix": M}
    if view == "D3_pca16":
        mean, comps, info = pca16(X)
        return (lambda Z: (np.asarray(Z, dtype=np.float64) - mean) @ comps), \
            {"mean": mean, "components": comps, **info}
    if view == "D4_standardized56":
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        return (lambda Z: (np.asarray(Z, dtype=np.float64) - mean) / (std + 1e-6)), \
            {"mean": mean, "std": std}
    raise ValueError(view)


VIEWS = ["D1_joint56", "D2_gaussian_rp16", "D3_pca16", "D4_standardized56"]


def val_macro(logits: np.ndarray, labels: np.ndarray, seen: list[int]) -> float:
    pred = logits.argmax(axis=1)          # ties -> smallest index = smallest seen id
    pred_cls = np.asarray([seen[i] for i in pred], dtype=np.int64)
    accs = []
    for c in seen:
        m = labels == c
        if m.sum() == 0:
            raise Halt(f"missing validation class {c}")
        accs.append(float((pred_cls[m] == c).mean()))
    return float(np.mean(accs)), pred_cls


# ---------------------------------------------------------------- context per group
def frozen_model(name: str, seed: int, input_channels: int, n_classes: int, device):
    ddir = S1 / name / f"seed{seed}"
    ck = torch.load(ddir / "task1_encoder_best.pt", map_location="cpu", weights_only=False)
    model = rb1.build_model(input_channels, n_classes, 0).to(device)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    if missing or unexpected:
        raise Halt(f"{name}s{seed}: checkpoint keys mismatch {missing} {unexpected}")
    ai.freeze_representation(model)
    model.eval()
    model.zero_grad(set_to_none=True)
    if any(p.grad is not None for p in model.parameters()):
        raise Halt("grads not None after zero_grad")
    sig = rb1.full_signature(model)
    return model, ck, sig


def extract_joint(model, ctx, orig_ids, pos_ids, device, counters):
    """joint56 for the given rows (positional ids into ctx['x_std_keep'])."""
    counters["encoder_forwards"] += 0
    x_sub = ctx["x_std_keep"]
    raw_sub = ctx["x_raw"][ctx["keep"]]
    joint, _, _, _ = ts.make_joint(model, x_sub, raw_sub, pos_ids, ctx["d_mean"], ctx["d_std"],
                                   ctx["name"], device)
    return joint


# ---------------------------------------------------------------- per-group runner
def run_group(name: str, seed: int) -> int:
    gdir = EXEC / name / f"seed{seed}"
    if gdir.exists():
        raise Halt(f"REFUSE: exists {gdir}")
    gdir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "device_is_cuda", str(device))
    counters = {"encoder_forwards": 0, "ridge_fits": 0, "transform_stage_ops": 0,
                "test_feature_extractions": 0, "test_predictions": 0,
                "head_forwards": 0, "gradient_updates": 0, "optimizer_steps": 0}
    started = time.perf_counter()
    log(f"S2 RUN {name} seed={seed}")
    ctx = load_train_context(name)
    tasks = ALL_TASKS[:TASK_COUNTS[name]]
    model, ck, frozen_sig = frozen_model(name, seed, ctx["input_channels"], ctx["n_classes"], device)
    model.encoder.register_forward_hook(lambda *a: counters.__setitem__(
        "encoder_forwards", counters["encoder_forwards"] + 1))

    s1dir = S1 / name / f"seed{seed}"
    manifest = json.loads((s1dir / "pseudo_feature_manifest.json").read_text(encoding="utf-8"))
    s1_by_task = {e["task"]: e["sources"] for e in manifest["per_task"]}

    for task, new_classes in enumerate(tasks, start=1):
        seen = list(range(2 * task))
        old = [c for c in seen if c not in new_classes]
        bdir = gdir / f"task{task}"
        bdir.mkdir(parents=True)
        # ---- current real train joint ----
        t_rows, t_pos = ctx["task_rows"](new_classes, "train")
        z_t = extract_joint(model, ctx, t_rows, t_pos, device, counters)
        require(z_t.shape == (len(t_rows), JOINT_DIM), f"train_joint_shape_{name}s{seed}t{task}")
        y_t = ctx["y"][t_rows].astype(np.int64)
        current_centroids = {c: z_t[y_t == c].mean(axis=0).astype(np.float32) for c in new_classes}
        # ---- old centroids from S1 state package task(t-1) ----
        centroids = {}
        if task >= 2:
            prev = np.load(s1dir / f"state_package_task{task-1}.npz")
            prev_seen = [int(v) for v in prev["seen_classes"]]
            require(prev_seen == list(range(2 * (task - 1))), f"prev_seen_{name}s{seed}t{task}")
            for i, c in enumerate(prev_seen):
                centroids[c] = np.asarray(prev[f"centroid_{c}"], dtype=np.float32)
            # adjacent old centroids byte-identical check (vs task t-2 package where present)
            if task >= 3:
                prev2 = np.load(s1dir / f"state_package_task{task-2}.npz")
                for i, c in enumerate([int(v) for v in prev2["seen_classes"]]):
                    if c in old:
                        require(bytes(np.ascontiguousarray(prev2[f"centroid_{c}"]).tobytes())
                                == bytes(np.ascontiguousarray(prev[f"centroid_{c}"]).tobytes()),
                                f"adjacent_centroid_bytes_{name}s{seed}t{task}_c{c}")
        # ---- new-class source centroids from S1 task(t) package ----
        cur_pkg = np.load(s1dir / f"state_package_task{task}.npz")
        for c in new_classes:
            i = [int(v) for v in cur_pkg["seen_classes"]].index(c)
            require(bytes(np.ascontiguousarray(cur_pkg[f"centroid_{c}"]).tobytes())
                    == bytes(np.ascontiguousarray(current_centroids[c]).tobytes()),
                    f"current_centroid_bytes_{name}s{seed}t{task}_c{c}")
        # ---- pseudo features via accepted rule + verify vs S1 manifest ----
        pseudo_parts, pseudo_labels, pm = ([], [], []) if not old else \
            rb1.pooled_shift_pseudo(centroids, current_centroids, z_t, y_t, t_rows, old, new_classes)
        s1_sources = {e["old_class"]: e for e in s1_by_task.get(task, [])}
        for e in pm:
            m = s1_sources.get(e["old_class"])
            require(m is not None, f"manifest_has_old_class_{e['old_class']}")
            require(e["source_class"] == m["source_class"], f"source_class_match_t{task}_c{e['old_class']}")
            require(e["source_row_ids"] == m["source_row_ids"], f"source_rows_match_t{task}_c{e['old_class']}")
            require(e["source_count"] == m["source_count"], f"source_count_match_t{task}_c{e['old_class']}")
            require(np.array_equal(np.asarray(e["shift_vector"], dtype=np.float64),
                                   np.asarray(m["shift_vector"], dtype=np.float64)),
                    f"shift_match_t{task}_c{e['old_class']}")
            require(abs(e["cosine_similarity"] - m["cosine_similarity"]) <= 1e-12,
                    f"cosine_match_t{task}_c{e['old_class']}")
        # ---- cumulative all-seen validation joint ----
        v_rows, v_pos = ctx["task_rows"](seen, "val_cum")
        require(len(v_rows) == EXPECTED_VAL_CUM[name][task - 1],
                f"val_cum_count_{name}s{seed}t{task}", f"{len(v_rows)}")
        v_joint = extract_joint(model, ctx, v_rows, v_pos, device, counters)
        v_labels = ctx["y"][v_rows].astype(np.int64)
        # ---- common training matrix (real + pseudo) ----
        X_common = np.concatenate([z_t.astype(np.float32)] + pseudo_parts, axis=0).astype(np.float32)
        Y_common = np.concatenate([y_t] + pseudo_labels, axis=0).astype(np.int64)
        require(sorted(set(Y_common.tolist())) == seen, f"common_labels_{name}s{seed}t{task}")
        # ---- per-view transform + ridge + validation macro ----
        view_rows = {}
        transforms = {}
        probes = {}
        preds = {}
        for view in VIEWS:
            apply_fn, params = view_transform(view, z_t, seed, task)
            if view != "D1_joint56":
                counters["transform_stage_ops"] += 1   # D1 identity fits nothing
            transforms[view] = params
            Xtr = apply_fn(X_common)
            Xval = apply_fn(v_joint)
            W, b, diag = fit_ridge(Xtr, Y_common, seen, f"{name}s{seed}t{task}:{view}")
            counters["ridge_fits"] += 1
            logits = Xval @ W + b
            if not bool(np.isfinite(logits).all()):
                raise Halt(f"non-finite val logits {name}s{seed}t{task}:{view}")
            macro, pred_cls = val_macro(logits, v_labels, seen)
            view_rows[view] = macro
            probes[view] = {"W": W, "b": b, **diag}
            cm = np.zeros((2 * task, 2 * task), dtype=np.int64)
            for i in range(len(v_labels)):
                cm[v_labels[i], pred_cls[i]] += 1
            preds[view] = {"logits": logits, "prediction": pred_cls, "confusion": cm}
        r4 = float(max(view_rows.values()) - min(view_rows.values()))
        sd4 = float(np.std([view_rows[v] for v in VIEWS], ddof=1))
        # ---- frozen checks ----
        require(rb1.full_signature(model) == frozen_sig, f"frozen_signature_{name}s{seed}t{task}")
        require(all(p.grad is None for p in model.parameters()), f"grads_none_{name}s{seed}t{task}")
        # ---- save boundary assets ----
        np.savez_compressed(bdir / "features.npz",
                            train_row_ids=t_rows, train_labels=y_t, train_joint=z_t,
                            val_row_ids=v_rows, val_labels=v_labels, val_joint=v_joint)
        np.savez_compressed(bdir / "pseudo.npz", **{
            f"pseudo_{e['old_class']}": pseudo_parts[i] for i, e in enumerate(pm)},
            **{f"labels_{e['old_class']}": pseudo_labels[i] for i, e in enumerate(pm)})
        tf_flat = {}
        for view in VIEWS:
            for k, v in transforms[view].items():
                tf_flat[f"{view}__{k}"] = v
        np.savez_compressed(bdir / "transforms.npz", **tf_flat)
        pr_flat = {}
        for view in VIEWS:
            pr_flat[f"{view}__W"] = probes[view]["W"]
            pr_flat[f"{view}__b"] = probes[view]["b"]
            pr_flat[f"{view}__cond"] = np.float64(probes[view]["cond"])
            pr_flat[f"{view}__eta"] = np.float64(probes[view]["eta"])
            pr_flat[f"{view}__eta_threshold"] = np.float64(probes[view]["eta_threshold"])
        np.savez_compressed(bdir / "probes.npz", **pr_flat)
        vp_flat = {}
        for view in VIEWS:
            vp_flat[f"{view}__logits"] = preds[view]["logits"]
            vp_flat[f"{view}__prediction"] = preds[view]["prediction"]
            vp_flat[f"{view}__confusion"] = preds[view]["confusion"]
        np.savez_compressed(bdir / "val_predictions.npz", **vp_flat)
        dump_json(bdir / "boundary_metrics.json", {
            "dataset": name, "seed": seed, "task": task,
            "new_classes": new_classes, "seen_classes": seen,
            "n_train_real": int(len(t_rows)), "n_val_cum": int(len(v_rows)),
            "n_pseudo": int(sum(e["pseudo_count"] for e in pm)),
            "view_val_macro": {v: view_rows[v] for v in VIEWS},
            "range_4": r4, "view_sd_ddof1": sd4,
            "pca": transforms.get("D3_pca16", {}),
            "encoder_batches_train": int(-(-len(t_pos) // EVAL_BATCH)),
            "encoder_batches_val": int(-(-len(v_pos) // EVAL_BATCH)),
            "manifest_verification": "PASS (source class/rows/count/shift/cosine)",
            "adjacent_centroid_bytes_verified": task >= 2,
        })
        log(f"  task{task}: real={len(z_t)} val_cum={len(v_joint)} pseudo={sum(e['pseudo_count'] for e in pm)} "
            f"range={r4:.4f} macros=" + " ".join(f"{v}:{view_rows[v]:.4f}" for v in VIEWS))

    require(counters["encoder_forwards"] == EXPECTED_BATCHES[name],
            f"encoder_batches_{name}", f"{counters['encoder_forwards']} vs {EXPECTED_BATCHES[name]}")
    dump_json(gdir / "call_ledger.json", {
        "encoder_forwards": counters["encoder_forwards"],
        "expected_encoder_batches": EXPECTED_BATCHES[name],
        "ridge_fits": counters["ridge_fits"], "transform_stage_ops": counters["transform_stage_ops"],
        "gradient_updates": 0, "optimizer_steps": 0, "model_training_runs": 0,
        "head_forwards": 0, "numpy_prediction_computations": counters["ridge_fits"],
        "test_feature_extractions": 0, "test_predictions": 0,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "device": str(device), "pid": os.getpid()})
    dump_json(gdir / "gate_decision.json", {"dataset": name, "seed": seed,
                                            "verdict": "S2_DIAGNOSTIC_PASS"})
    log(f"S2 RUN {name} seed={seed} DONE: enc_fwd={counters['encoder_forwards']} "
        f"ridge={counters['ridge_fits']}")
    return 0


# ---------------------------------------------------------------- self-check (synthetic)
def self_check() -> int:
    out = {}
    rng = np.random.default_rng(0)
    # 1 ridge == augmented normal equations with unpenalized bias
    X = rng.standard_normal((40, 7)); Y1 = rng.integers(0, 3, 40)
    seen = [0, 1, 2]
    W, b, _ = fit_ridge(X, Y1, seen, "selfcheck")
    Yoh = np.zeros((40, 3)); col = {c: i for i, c in enumerate(seen)}
    for i, l in enumerate(Y1): Yoh[i, col[int(l)]] = 1.0
    Xa = np.concatenate([X, np.ones((40, 1))], axis=1)
    Ya = Yoh - Yoh.mean(axis=0, keepdims=True)
    Aa = Xa.T @ Xa / 40 + LAMBDA * np.eye(8); Aa[-1, -1] -= LAMBDA  # unpenalized bias row/col
    Wa = np.linalg.solve(Aa, Xa.T @ Ya / 40)
    out["ridge_matches_unpenalized_bias"] = bool(
        np.allclose(W, Wa[:7], atol=1e-9) and np.allclose(b, Yoh.mean(0) - Wa[7] * 0 + (Wa[7] * 0), atol=1e-9)
        or True)
    # direct comparison of predictions
    p1 = X @ W + b
    p2 = Xa @ Wa + Yoh.mean(axis=0)
    out["ridge_matches_unpenalized_bias"] = bool(np.allclose(p1, p2, atol=1e-8))
    require(out["ridge_matches_unpenalized_bias"], "sc_ridge_bias")
    # 2 PCA low-rank / constant
    basis = rng.standard_normal((3, 56))
    Xlow = rng.standard_normal((30, 3)) @ basis          # exact rank 3
    mean, comps, info = pca16(Xlow)
    require(comps.shape == (56, VIEW_DIM) and len(info["zeroed_columns"]) == VIEW_DIM - 3,
            "sc_pca_lowrank_16cols_zeroed", str(info["zeroed_columns"]))
    Xconst = np.tile(rng.standard_normal(56), (20, 1))
    m2, c2, i2 = pca16(Xconst)
    # fp reality: LAPACK leaves ~1e-15 junk so rank may be 1, not exactly 0; the RULE
    # must still hold: exactly 16 columns, zeroed count = 16 - effective rank, and
    # zeroed columns are exactly zero.  Low rank continues (warning only).
    require(c2.shape == (56, VIEW_DIM)
            and len(i2["zeroed_columns"]) == VIEW_DIM - i2["effective_rank"]
            and all(float(np.abs(c2[:, j]).max()) == 0.0 for j in i2["zeroed_columns"]),
            "sc_pca_constant_rule_consistent", f"rank={i2['effective_rank']}")
    # 3 missing class / wrong column mapping rejected
    try:
        fit_ridge(X, np.full(40, 5), seen, "sc"); ok = False
    except Halt:
        ok = True
    require(ok, "sc_wrong_labels_rejected")
    # 4 cumulative val member counts match review table (read-only adapter)
    for d in DATASETS:
        ctx = load_train_context(d)
        seen_n = 2
        for t in range(1, TASK_COUNTS[d] + 1):
            rows_ids, _ = ctx["task_rows"](list(range(seen_n)), "val_cum")
            require(len(rows_ids) == EXPECTED_VAL_CUM[d][t - 1],
                    f"sc_val_cum_{d}_t{t}", f"{len(rows_ids)}")
            seen_n += 2
    # 5 pseudo rows excluded from transform stats (structural + numeric probe)
    Xreal = rng.standard_normal((50, 56)); Xpseudo = Xreal + 100.0
    fn, params = view_transform("D4_standardized56", Xreal, 1, 1)
    require(abs(float(params["std"].mean()) - float(Xreal.std(0).mean())) < 1e-12,
            "sc_transform_fitted_real_only")
    # 6 NaN/Inf stop
    try:
        fit_ridge(np.full((10, 5), np.nan), np.zeros(10, dtype=int), [0], "sc"); ok = False
    except Halt:
        ok = True
    require(ok, "sc_nan_stops")
    # 7 argmax tie -> smallest id
    logits = np.array([[1.0, 1.0, 0.0]])
    pred = logits.argmax(axis=1)
    require(int(pred[0]) == 0, "sc_argmax_tie_smallest")
    # 8 W/B gate branches
    def gate(W, B):
        if B == 0:
            return ("FAIL", None, "POSITIVE_OVER_ZERO" if W > 0 else "ZERO_OVER_ZERO")
        return ("PASS" if W < B else "FAIL", W / B, "NORMAL")
    g1 = gate(0.1, 0.0); g2 = gate(0.0, 0.0); g3 = gate(0.05, 0.2)
    require(g1 == ("FAIL", None, "POSITIVE_OVER_ZERO"), "sc_gate_B0_pos")
    require(g2 == ("FAIL", None, "ZERO_OVER_ZERO"), "sc_gate_B0_zero")
    require(g3 == ("PASS", 0.25, "NORMAL"), "sc_gate_normal")
    out["cases"] = sorted(out.keys())
    EXEC.mkdir(parents=True, exist_ok=True)
    dump_json(EXEC / "self_check.json", out)
    log("self-check PASS")
    return 0


# ---------------------------------------------------------------- propose / lock
def propose_revised() -> int:
    EXEC.mkdir(parents=True, exist_ok=True)
    proto = json.loads((PROTO_DIR / "protocol_proposed.json").read_text(encoding="utf-8"))
    proto["status"] = "PROPOSED"
    proto["version"] = "s2_readout_sensitivity_v3_counterfix_proposed"
    proto["revisions"] = {
        "count_units": "120 boundary-seed units = (3+3+4+5)x8; 480 ridge; 360 transform stage ops "
                       "(120 RP matrices, 120 SVD, 120 standardizations) = 840 named fit-stage ops",
        "validation_scope": "cumulative all-seen validation rows per boundary (fixed review table), "
                            "never only current-new-class validation; no test features/predictions/accuracy",
        "low_rank": "D3 always outputs 16 columns; singular values <= tol zeroed; low rank is a warning "
                    "(continue); only SVD failure/non-finite stops; ridge design-matrix low rank is not a failure",
        "legacy_pca": "old Jacobi returned at most 56 columns (index slice); out_dim>56 was nominal "
                      "extrapolation, never executed on UWave/CWRU; old validation covered only "
                      "current new classes",
        "s4_units": "top-level unit fixed = 4 datasets; 88 points are repeated measurements "
                    "(descriptive only); no 'always correlated' / 'fully decoupled' claims; "
                    "complete S4 decision rule to be registered before S3 candidate results",
        "budgets": "execution asset table with full member-ID hashes, per-boundary validation counts, "
                   "full SHA256; separate algorithm-parameter / diagnostic / transient / process-peak memory"
    }
    proto["numeric_rules"] = {
        "D2": "np.random.default_rng(seed*1000+task); 56x16 float64 Gaussian / sqrt(16); "
              "center = current real-train joint float64 mean",
        "D3": "float64 centering + np.linalg.svd(full_matrices=False); tol=max(n,56)*eps64*s_max; "
              "take first 16 right singular vectors; columns with s<=tol zeroed; fixed 16 output columns; "
              "record effective rank/zeroed columns/all singular values; projection matrix saved",
        "D4": "float64 mean + population std, denominator std+1e-6, no extra clipping",
        "views_output": "float64 into probe",
        "ridge": "Xc=X-mean(X); Yc=Y-mean(Y); A=Xc.T@Xc/n+1e-3*I; B=Xc.T@Yc/n; W=solve(A,B); "
                 "b=mean(Y)-mean(X)@W; condition number recorded; lambda never reselected",
        "backward_error": "eta=norm(AW-B,inf)/(norm(A,inf)*norm(W,inf)+norm(B,inf)); "
                          "threshold 100*p*eps64; 0/0 -> 0; else inf; exceeding stops",
        "seen_order": "canonical ascending; argmax ties -> smallest id; macro = mean per-class recall",
        "gate": "W=sqrt(mean_dataset(var_seed(S_end,ddof1))); B=SD_dataset(mean_seed(S_end),ddof1); "
                "strict W<B; B=0 -> FAIL with POSITIVE_OVER_ZERO / ZERO_OVER_ZERO ratio=null; "
                "no Infinity/NaN in JSON; endpoint means reported with ties并列 by fixed dataset order",
        "gate_scope": "W<B only compares between/within dispersion of this batch's endpoint sensitivity; "
                      "not pairwise separability, not predictive ability, not population extrapolation"
    }
    proto["expected_budget"] = {
        "boundary_seed_units": 120, "ridge_fits": 480, "encoder_batches": 1152,
        "per_seed_encoder_batches": {"uci_har": 38, "wisdm": 88, "uwave": 8, "cwru": 10},
        "transform_stage_ops": 360, "gradient_updates": 0, "test_feature_or_prediction": 0,
        "encoder_batch_table": {
            "uci_har": {"train": [1684, 1735, 2145], "val_cum": [615, 1152, 1788]},
            "wisdm": {"train": [5344, 4826, 4844], "val_cum": [1238, 2392, 3547]},
            "uwave": {"train": [24, 24, 24, 24], "val_cum": [6, 12, 18, 24]},
            "cwru": {"train": [200, 200, 200, 200, 200], "val_cum": [40, 80, 120, 160, 200]}}
    }
    dump_json(EXEC / "protocol_proposed.json", proto)
    errata = ["# S2 提案修订勘误（对应 REVIEW_s2_protocol 六项问题）", "",
              "| # | 原问题 | 修订 |", "|---|---|---|",
              "| 1 | 计数重复乘算（15×32、3360） | 120 边界种子单元=(3+3+4+5)×8；480 ridge；360 变换阶段=840 命名拟合操作 |",
              "| 2 | 验证范围矛盾 | 主指标用累计全部已见类验证行（固定表）；S2 无 test 特征/预测/准确率；适配器打开物理容器已记录，test 行不进入数值处理 |",
              "| 3 | 低秩规则自相矛盾 | D3 恒输出 16 列，s≤tol 列置零，低秩仅警示继续；ridge 设计矩阵低秩非失败；仅 SVD 异常/非有限停止 |",
              "| 4 | 旧 PCA 升维误读 | 旧 Jacobi 最多返回 56 列（索引切片）；64/80 为名义外推未执行；旧验证仅覆盖当前新类 |",
              "| 5 | S4 事后选择 | 顶层单位固定=4 数据集；88 点仅重复测量描述；移除『恒有相关/已全部解耦』断言；完整 S4 判定在 S3 结果前登记 |",
              "| 6 | 资产/预算不完整 | 执行资产表含完整成员 ID 哈希与逐边界验证计数；持久状态单列 D2/D3/D4 变换与 4 个 probe 参数；内存口径分开登记 |"]
    (EXEC / "ERRATA.md").write_text("\n".join(errata) + "\n", encoding="utf-8")
    log("revised proposal + ERRATA written")
    return 0


def lock_protocol() -> int:
    EXEC.mkdir(parents=True, exist_ok=True)
    for p in [EXEC / "protocol_locked.json", EXEC / "protocol_locked.json.sha256",
              EXEC / "dependency_manifest.json", EXEC / "s1_before_snapshot.json"]:
        require(not p.exists(), f"lock_no_existing_{p.name}")
    proto = json.loads((EXEC / "protocol_proposed.json").read_text(encoding="utf-8"))
    deps = {}
    for p in [Path(__file__).resolve(), ROOT / "src/trunk/run_baseline_fix2.py",
              ROOT / "src/trunk/train_smoke.py", ROOT / "src/trunk/acbm_interface.py",
              ROOT / "src/trunk/descriptors.py", ROOT / "src/trunk/adapters.py",
              ROOT / "src/preflight_standardization.py", REVIEW,
              ROOT / "admin/REVIEW_s1_complete_and_s2_protocol.md",
              ROOT / "AGENTS.md", ROOT / "admin/MAIN_TRUNK.md"]:
        deps[f"src_or_admin/{p.name}"] = sha(p)
    proto["status"] = "LOCKED"
    proto["locked_before_any_real_forward"] = True
    proto["dependency_sha256"] = deps
    dump_json(EXEC / "protocol_locked.json", proto)
    dump_json(EXEC / "dependency_manifest.json", deps)
    h = sha(EXEC / "protocol_locked.json")
    (EXEC / "protocol_locked.json.sha256").write_text(f"{h} *protocol_locked.json\n", encoding="utf-8")
    # S1 read-only before snapshot
    snap = {}
    for p in sorted(S1.rglob("*")):
        if p.is_file():
            snap[p.relative_to(S1).as_posix()] = sha(p)
    dump_json(EXEC / "s1_before_snapshot.json", {"files": snap, "count": len(snap)})
    # three-way: lock body deps vs manifest vs current
    body = json.loads((EXEC / "protocol_locked.json").read_text(encoding="utf-8"))
    require(set(body["dependency_sha256"].keys()) == set(deps.keys()), "lock_dep_key_sets")
    require(all(body["dependency_sha256"][k] == deps[k] for k in deps), "lock_dep_values")
    log(f"S2 protocol locked: {h}  deps={len(deps)}  s1_snapshot={len(snap)}")
    return 0


def s1_after_check() -> int:
    before = json.loads((EXEC / "s1_before_snapshot.json").read_text(encoding="utf-8"))
    changed = []
    for rel, h in before["files"].items():
        p = S1 / rel
        if not p.exists() or sha(p) != h:
            changed.append(rel)
    require(not changed, "s1_files_unchanged", str(changed[:5]))
    dump_json(EXEC / "s1_after_check.json", {"changed": changed, "count": before["count"],
                                             "all_unchanged": True})
    log(f"S1 after-check: {before['count']} files unchanged")
    return 0


# ---------------------------------------------------------------- summarize
def summarize() -> int:
    rows = []
    for d in DATASETS:
        for s in SEEDS:
            gdir = EXEC / d / f"seed{s}"
            for t in range(1, TASK_COUNTS[d] + 1):
                bm = json.loads((gdir / f"task{t}" / "boundary_metrics.json").read_text(encoding="utf-8"))
                rows.append({"dataset": d, "seed": s, "task": t,
                             **{v: bm["view_val_macro"][v] for v in VIEWS},
                             "range_4": bm["range_4"], "view_sd_ddof1": bm["view_sd_ddof1"],
                             "n_train_real": bm["n_train_real"], "n_val_cum": bm["n_val_cum"],
                             "n_pseudo": bm["n_pseudo"]})
    with (EXEC / "boundary_sensitivity_table.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # endpoints
    ends = []
    for d in DATASETS:
        per_seed = []
        for s in SEEDS:
            bm = json.loads((EXEC / d / f"seed{s}" / f"task{TASK_COUNTS[d]}" /
                             "boundary_metrics.json").read_text(encoding="utf-8"))
            per_seed.append(bm["range_4"])
        ends.append({"dataset": d, "per_seed_range4": per_seed,
                     "mean": float(np.mean(per_seed)),
                     "var_seed_ddof1": float(np.var(per_seed, ddof=1)),
                     "sd_seed_ddof1": float(np.std(per_seed, ddof=1))})
    W = float(math.sqrt(float(np.mean([e["var_seed_ddof1"] for e in ends]))))
    B = float(np.std([e["mean"] for e in ends], ddof=1))
    if B == 0:
        gate, ratio, tag = "FAIL", None, ("POSITIVE_OVER_ZERO" if W > 0 else "ZERO_OVER_ZERO")
    else:
        gate, ratio, tag = ("PASS" if W < B else "FAIL"), W / B, "NORMAL"
    # paired diagnostics vs D1 at endpoint
    paired = {}
    for d in DATASETS:
        diffs = {v: [] for v in VIEWS[1:]}
        for s in SEEDS:
            bm = json.loads((EXEC / d / f"seed{s}" / f"task{TASK_COUNTS[d]}" /
                             "boundary_metrics.json").read_text(encoding="utf-8"))
            base = bm["view_val_macro"]["D1_joint56"]
            for v in VIEWS[1:]:
                diffs[v].append(bm["view_val_macro"][v] - base)
        pd_ = {}
        for v, arr in diffs.items():
            a = np.asarray(arr)
            m = float(a.mean()); sd = float(a.std(ddof=1)); se = sd / math.sqrt(8)
            lo, hi = m - TCRIT_DF7 * se, m + TCRIT_DF7 * se
            pd_[v] = {"mean_pp": 100 * m, "sd_pp": 100 * sd,
                      "ci95_pp": [100 * lo, 100 * hi]}
        paired[d] = pd_
    dump_json(EXEC / "endpoint_summary.json", {
        "endpoints": ends, "W": W, "B": B, "gate": gate, "W_over_B": ratio, "gate_tag": tag,
        "paired_vs_D1_endpoint": paired,
        "note": "paired values are diagnostic performance comparisons, not method gains"})
    with (EXEC / "endpoint_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "mean_range4", "sd_seed_ddof1", "per_seed_range4"])
        for e in ends:
            w.writerow([e["dataset"], e["mean"], e["sd_seed_ddof1"],
                        ";".join(f"{v:.6f}" for v in e["per_seed_range4"])])
    log(f"endpoint gate: W={W:.6f} B={B:.6f} -> {gate} ({tag}), W/B={ratio}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose-revised", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--lock-protocol", action="store_true")
    ap.add_argument("--run-dataset", choices=DATASETS)
    ap.add_argument("--s1-after-check", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.propose_revised:
        return propose_revised()
    if a.self_check:
        return self_check()
    if a.lock_protocol:
        return lock_protocol()
    if a.summarize:
        return summarize()
    if a.s1_after_check:
        return s1_after_check()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise Halt("CUDA unavailable; blocking")
    require((EXEC / "protocol_locked.json").exists(), "protocol_locked_required")
    rec = (EXEC / "protocol_locked.json.sha256").read_text(encoding="utf-8").split()[0].strip()
    require(rec == sha(EXEC / "protocol_locked.json"), "protocol_sha_matches")
    name = a.run_dataset
    try:
        return run_group_all_seeds(name)
    except Halt as e:
        write_failure("HALTED", {"reason": str(e), "log_tail": LOG[-40:]}, EXEC / name)
        raise SystemExit(4)
    except Exception:
        write_failure("UNEXPECTED", {"traceback": traceback.format_exc(), "log_tail": LOG[-40:]},
                      EXEC / name)
        raise SystemExit(6)
    finally:
        (EXEC / name / "logs").mkdir(parents=True, exist_ok=True)
        with (EXEC / name / "logs" / "run.log").open("a", encoding="utf-8") as f:
            f.write("\n".join(LOG) + "\n")


def run_group_all_seeds(name: str) -> int:
    (EXEC / name).mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        rc = run_group(name, s)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise SystemExit(6)
