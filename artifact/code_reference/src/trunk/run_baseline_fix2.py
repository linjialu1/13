"""Formal S1 baseline runner — FIX1 (addresses REVIEW_s1_protocol_final B1–B6).

Keeps the accepted training implementation and fixes the execution chain:

  B1  independent test arrays / labels / row-id space are carried in a split view;
      evaluation reads the TEST view (never the train side).
  B2  the training loop PRODUCES reference outputs before launching the recovery
      consumer (restored producer/consumer chain).
  B3  unified preflight before any research-model forward (LOCKED protocol + SHA +
      input hashes + registration range); --lock-protocol refuses to overwrite an
      existing lock; --seal-training gates evaluation on a complete 32-group seal.
  B4  per-row evaluation evidence, confusion matrix, head epoch logs, per-boundary
      source manifest, output hash manifest, finally-persisted logs, failure records.
  B5  save/load/recover share a full state-package schema (dataset/seed/task identity,
      seen classes, dtype/shape/finite, positive counts, NPZ↔metadata consistency,
      task mapping, centroid value hash, package hash, reference hashes).
  B6  pseudo-feature bytes (separate from sample count), per-stage timing, GPU/RSS
      measured per process with explicit scope, Windows PowerShell entry, test-forward
      scope correction.

Zero research-model training / forward / test evaluation in this round.
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
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/baselines/trunk_baseline_4datasets"          # formal results (uncreated)
PROTO_OUT = ROOT / "experiments/baselines/trunk_baseline_4datasets_protocol_final_fix2"
PROTO_OLD = ROOT / "experiments/baselines/trunk_baseline_4datasets_protocol_final_fix1"
V2_DIR = ROOT / "experiments/baselines/trunk_model_interface_check_v2"
for p in (str(ROOT / "src"), str(ROOT), str(ROOT / ".local_packages")):
    if p not in sys.path:
        sys.path.insert(0, p)

from trunk import train_smoke as ts   # noqa: E402
from trunk import acbm_interface as ai  # noqa: E402
from trunk import adapters as ad        # noqa: E402
from trunk import descriptors as ds     # noqa: E402

F = torch.nn.functional

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DATASETS = ["uci_har", "wisdm", "uwave", "cwru"]
ALL_TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
DATASET_TASK_COUNTS = {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}

HIDDEN = 48
DESCRIPTOR_DIM = 8
JOINT_DIM = 56
ENCODER_LR = 1e-3
BATCH_SIZE = 128
GRAD_CLIP = 5.0
MIN_EPOCHS = 3
MAX_EPOCHS = 12
PATIENCE = 2
VAL_DELTA_ACC = 0.005
VAL_DELTA_LOSS = 0.001
EVAL_BATCH = 256
SPARSE_EPOCHS = 300
SPARSE_LR = 0.05
SPARSITY_LAMBDA = 0.0007
REF_PER_CLASS = 4

LOG: list[str] = []
COUNTERS: dict[str, int] = {}


class Halt(Exception):
    pass


def log(m: str = "") -> None:
    print(m, flush=True)
    LOG.append(m)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sanitize_nan(o):
    if isinstance(o, dict):
        return {k: _sanitize_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize_nan(v) for v in o]
    if isinstance(o, np.ndarray):
        return _sanitize_nan(o.tolist())
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(o, np.integer):
        return int(o)
    return o


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_sanitize_nan(obj), indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


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


# ---------------------------------------------------------------- input inventory
def input_files() -> dict[str, Path]:
    f: dict[str, Path] = {}
    for p in [Path(__file__).resolve(), ROOT / "src/trunk/run_baseline.py",
              ROOT / "src/trunk/train_smoke.py", ROOT / "src/trunk/acbm_interface.py",
              ROOT / "src/trunk/descriptors.py", ROOT / "src/trunk/adapters.py",
              ROOT / "src/preflight_standardization.py"]:
        f[f"src/{p.name}"] = p
    for k, p in {
        "archive/pipeline.py": ROOT / "experiments/archive/2026-10-13/cfc_acbm_t3_component_ablation_3seeds/pipeline.py",
        "archive/stage4_benchmark.py": ROOT / "experiments/archive/stage4/refined/benchmark.py",
        "archive/stage4_run_config.json": ROOT / "experiments/archive/stage4/refined/run_config.json",
        "archive/run_wisdm.py": ROOT / "experiments/archive/2026-10-11/cfc_cbm_wisdm_original_vs_staged/run_wisdm.py",
        "legacy/benchmark.py": ROOT / "src/legacy/benchmark.py",
        "legacy/cicbm_benchmark.py": ROOT / "src/legacy/cicbm/benchmark.py",
    }.items():
        f[k] = p
    f["split/uci_har_fixed_split.npz"] = ad.PREFLIGHT_DIR / "uci_har_fixed_split.npz"
    f["split/wisdm_fixed_split.npz"] = ad.PREFLIGHT_DIR / "wisdm_fixed_split.npz"
    f["split/uwave_fixed_split.npz"] = ad.PREFLIGHT_DIR / "uwave_fixed_split.npz"
    f["split/cwru_window_index.csv"] = ad.PREFLIGHT_DIR / "cwru_window_index.csv"
    f["split/wisdm_row_provenance.csv"] = ad.PREFLIGHT_DIR / "wisdm_row_provenance.csv"
    f["stats/preprocess_statistics.json"] = ad.FIX_DIR / "preprocess_statistics.json"
    f["stats/fit_indices.npz"] = ad.FIX_DIR / "fit_indices.npz"
    for ds_ in DATASETS:
        f[f"v2/interface_check_{ds_}.json"] = V2_DIR / f"interface_check_{ds_}.json"
    f["raw/wisdm.zip"] = ROOT / "data/wisdm.zip"
    f["raw/wisdm_cache.npz"] = ROOT / "data/wisdm_har_windows.npz"
    f["raw/uwave_train.ts"] = ROOT / "data/UWaveGestureLibrary_TRAIN.ts"
    f["raw/uwave_test.ts"] = ROOT / "data/UWaveGestureLibrary_TEST.ts"
    f["raw/cwru_manifest.json"] = ROOT / "data/cwru_lttsnet_seed10.manifest.json"
    f["raw/uci_train_y.txt"] = ROOT / "data/UCI HAR Dataset/train/y_train.txt"
    f["raw/uci_test_y.txt"] = ROOT / "data/UCI HAR Dataset/test/y_test.txt"
    f["raw/uci_train_subject.txt"] = ROOT / "data/UCI HAR Dataset/train/subject_train.txt"
    f["raw/uci_test_subject.txt"] = ROOT / "data/UCI HAR Dataset/test/subject_test.txt"
    for n in ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y",
              "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]:
        for split in ("train", "test"):
            f[f"raw/uci_{split}_{n}.txt"] = ROOT / f"data/UCI HAR Dataset/{split}/Inertial Signals/{n}_{split}.txt"
    for m in sorted((ROOT / "data/cwru_lttsnet_raw").glob("*.mat")):
        f[f"raw/cwru_{m.name}"] = m
    # R1: the PowerShell launcher is a locked dependency (fixed path, not PROTO_OUT which
    # may be monkeypatched during contract regression)
    f["launcher/run_baseline.ps1"] = ROOT / "experiments/baselines/trunk_baseline_4datasets_protocol_final_fix2/run_baseline.ps1"
    return f


# ---------------------------------------------------------------- data context (B1)
def task_list(name: str) -> list[list[int]]:
    return ALL_TASKS[:DATASET_TASK_COUNTS[name]]


def load_context(name: str) -> dict:
    raw = ad.adapt(name)
    x_raw = np.asarray(raw["x"])
    y = np.asarray(raw["y"])
    n_classes = int(len(set(y.tolist())))

    if name == "cwru":
        splits = np.asarray([p["split"] for p in raw["provenance"]])
        train_all = np.flatnonzero(splits == "train")
        val_all = np.flatnonzero(splits == "validation")
        test_all = np.flatnonzero(splits == "test")
        test_view = {"x_raw": x_raw, "y": y, "row_ids": test_all,
                     "row_space": "joint", "n_rows": len(test_all)}
    else:
        sp = ad.accepted_splits()[name]
        train_all = np.asarray(sp["train"])
        val_all = np.asarray(sp["validation"])
        rt = raw["test"]
        test_x = np.asarray(rt["x"])
        if name == "wisdm":
            test_y = np.asarray([p["label"] for p in rt["provenance"]], dtype=np.int64)
        else:
            test_y = np.asarray(rt["y"], dtype=np.int64)
        test_view = {"x_raw": test_x, "y": test_y,
                     "row_ids": np.arange(len(test_y), dtype=np.int64),
                     "row_space": "test", "n_rows": len(test_y)}

    in_stats = ad.accepted_input_stats()[name]
    in_mean = np.asarray(in_stats["mean"], dtype=np.float64)
    in_denom = np.asarray(in_stats["denom"], dtype=np.float64)
    x_std = ai.standardize_input(x_raw, in_mean, in_denom)
    x_std_test = ai.standardize_input(test_view["x_raw"], in_mean, in_denom)
    v2 = json.loads((V2_DIR / f"interface_check_{name}.json").read_text(encoding="utf-8"))
    d_mean = np.asarray(v2["descriptor_stats_mean"], dtype=np.float64)
    d_std = np.asarray(v2["descriptor_stats_std"], dtype=np.float64)

    def task_rows(classes, which="train"):
        if which == "test":
            base = test_view["row_ids"]
            lab = test_view["y"]
            return np.asarray(sorted(int(i) for i in base if int(lab[i]) in classes), dtype=np.int64)
        base = train_all if which == "train" else val_all
        return np.asarray(sorted(int(i) for i in base if int(y[i]) in classes), dtype=np.int64)

    return dict(name=name, x_raw=x_raw, y=y, x_std=x_std, d_mean=d_mean, d_std=d_std,
                n_classes=n_classes, input_channels=int(x_raw.shape[2]),
                in_mean=in_mean, in_denom=in_denom, task_rows=task_rows,
                test_view=test_view, x_std_test=x_std_test,
                train_all=train_all, val_all=val_all,
                test_row_space=test_view["row_space"])


# ---------------------------------------------------------------- preflight (B3)
def preflight(gate: str, dataset: str | None, seed: int | None, task: int | None):
    """Run before ANY research-model forward. Verify protocol lock + 3-way snapshot."""
    if dataset is not None:
        require(dataset in DATASETS, f"{gate}_dataset_registered", dataset)
    if seed is not None:
        require(seed in SEEDS, f"{gate}_seed_registered", str(seed))
    if task is not None:
        require(1 <= task <= DATASET_TASK_COUNTS[dataset], f"{gate}_task_registered", str(task))
    proto = OUT / "protocol_locked.json"
    sha_file = OUT / "protocol_locked.json.sha256"
    require(proto.exists(), f"{gate}_protocol_locked_exists")
    require(sha_file.exists(), f"{gate}_protocol_sha_exists")
    rec = sha_file.read_text(encoding="utf-8").split()[0].strip()
    require(rec == sha(proto), f"{gate}_protocol_sha_matches")
    body = json.loads(proto.read_text(encoding="utf-8"))
    require(body.get("status") == "LOCKED", f"{gate}_protocol_status_locked")
    locked_snapshot = body.get("input_sha256_before") or {}
    before_file = json.loads((OUT / "input_sha256_before.json").read_text(encoding="utf-8"))
    now = {k: sha(v) for k, v in input_files().items()}
    # R3: locked body snapshot, independent before file, and current inventory must
    # agree on BOTH key set and values. Empty / missing / extra / drift all stop.
    keys_ok = (set(locked_snapshot.keys()) == set(before_file.keys()) == set(now.keys()))
    require(keys_ok, f"{gate}_snapshot_key_sets_identical",
            f"locked={len(locked_snapshot)} before={len(before_file)} now={len(now)}")
    require(bool(now), f"{gate}_snapshot_nonempty")
    mismatch = [k for k in now if locked_snapshot.get(k) != before_file.get(k)
                or before_file.get(k) != now[k]]
    require(not mismatch, f"{gate}_snapshot_values_identical", str(mismatch[:5]))


def postflight(gate: str):
    before = json.loads((OUT / "input_sha256_before.json").read_text(encoding="utf-8"))
    now = {k: sha(v) for k, v in input_files().items()}
    changed = [k for k in before if before.get(k) != now.get(k)]
    require(not changed, f"{gate}_input_hashes_unchanged_exit", str(changed))


# ---------------------------------------------------------------- seeds / model / helpers
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(input_channels: int, n_classes: int, seed: int):
    set_seed(seed)
    return ai.pipeline_module().TemporalAttentionSequenceClassifier(
        "CfC", input_channels, HIDDEN, n_classes)


def labels_to_local(yy, active, n_classes, device):
    lookup = torch.full((n_classes,), -1, dtype=torch.long, device=device)
    lookup[torch.as_tensor(active, dtype=torch.long, device=device)] = torch.arange(
        len(active), dtype=torch.long, device=device)
    return lookup[yy]


def full_signature(model) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(b"param:" + name.encode())
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    for name, b in model.named_buffers():
        h.update(b"buf:" + name.encode())
        h.update(np.ascontiguousarray(b.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def assert_grads_finite_present(model, label):
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            raise Halt(f"MISSING gradient {label}: {n}")
        if not bool(torch.isfinite(p.grad).all().item()):
            raise Halt(f"NON-FINITE gradient {label}: {n}")


def assert_params_finite(model, label):
    for n, p in model.named_parameters():
        if not bool(torch.isfinite(p).all().item()):
            raise Halt(f"NON-FINITE parameter {label}: {n}")


def make_loader(x, y, indices, batch_size, generator_seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(generator_seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x[indices]), torch.from_numpy(y[indices]).long())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       drop_last=False, pin_memory=torch.cuda.is_available(),
                                       generator=generator)


def evaluate_validation(model, x, y, indices, new_classes, n_classes, device, batch_size):
    model.eval()
    class_accuracies, total_loss, total_count = [], 0.0, 0
    with torch.no_grad():
        for cls in new_classes:
            class_indices = indices[y[indices] == cls]
            if not len(class_indices):
                continue
            correct = 0
            for start in range(0, len(class_indices), batch_size):
                batch = class_indices[start:start + batch_size]
                xx = torch.from_numpy(x[batch]).to(device)
                yy = torch.from_numpy(y[batch]).long().to(device)
                logits = model(xx)[:, new_classes]
                mapped = labels_to_local(yy, new_classes, n_classes, device)
                total_loss += float(F.cross_entropy(logits, mapped, reduction="sum").item())
                total_count += len(batch)
                correct += int((logits.argmax(1) == mapped).sum().item())
            class_accuracies.append(correct / len(class_indices))
    return float(np.mean(class_accuracies)), total_loss / max(total_count, 1)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def process_memory_bytes() -> dict | None:
    try:
        import ctypes
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        h = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(h, ctypes.byref(c), ctypes.sizeof(c)):
            return {"working_set_bytes": int(c.WorkingSetSize),
                    "peak_working_set_bytes": int(c.PeakWorkingSetSize)}
        return None
    except Exception:
        return None


def gpu_peak_bytes(device) -> dict:
    if device.type != "cuda":
        return {"gpu_available": False}
    return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}


# ---------------------------------------------------------------- sparse head + pseudo (unchanged)
def fit_sparse_head(head, c, y, device, name, counters):
    cc = torch.from_numpy(np.asarray(c, dtype=np.float32)).to(device)
    yy = torch.from_numpy(np.asarray(y, dtype=np.int64)).long().to(device)
    require(head.in_features == cc.shape[1], f"{name}_head_in_features")
    require(int(y.max()) + 1 == head.out_features, f"{name}_head_out_features")
    opt = torch.optim.Adam(head.parameters(), lr=SPARSE_LR)
    w_before = head.weight.detach().clone()
    losses = []
    for _ in range(SPARSE_EPOCHS):
        opt.zero_grad(set_to_none=True)
        logits = head(cc)
        counters["head_forwards"] += 1
        if not bool(torch.isfinite(logits).all().item()):
            raise Halt(f"NON-FINITE {name} logits")
        loss = F.cross_entropy(logits, yy) + SPARSITY_LAMBDA * head.weight.abs().sum()
        if not bool(torch.isfinite(loss).all().item()):
            raise Halt(f"NON-FINITE {name} loss")
        loss.backward()
        counters["head_backward"] += 1
        assert_grads_finite_present(head, f"{name} pre-step")
        opt.step()
        counters["head_optimizer_steps"] += 1
        assert_params_finite(head, f"{name} post-step")
        losses.append(float(loss.detach()))
    return losses, bool(not torch.equal(w_before, head.weight))


def pooled_shift_pseudo(centroids, current_centroids, z_t, y_t, t_rows, old_classes, new_classes):
    pseudo_parts, pseudo_labels, manifest = [], [], []
    for cls in old_classes:
        old_mean = np.asarray(centroids[cls], dtype=np.float64)
        nearest = max(new_classes, key=lambda c: cosine(
            old_mean, np.asarray(current_centroids[c], dtype=np.float64)))
        src_mask = y_t == nearest
        source = z_t[src_mask].astype(np.float64)
        source_rows = t_rows[src_mask]
        shift = old_mean - np.asarray(current_centroids[nearest], dtype=np.float64)
        pseudo = (source + shift).astype(np.float32)
        ds.assert_all_finite(pseudo, f"pseudo old class {cls}")
        manifest.append({
            "old_class": int(cls), "source_class": int(nearest),
            "source_row_ids": [int(r) for r in source_rows],
            "source_count": int(len(source_rows)), "pseudo_count": int(len(pseudo)),
            "shift_vector": [float(v) for v in shift],
            "old_mean": [float(v) for v in old_mean],
            "source_mean": [float(v) for v in np.asarray(current_centroids[nearest], dtype=np.float64)],
            "cosine_similarity": cosine(old_mean, np.asarray(current_centroids[nearest], dtype=np.float64)),
            "formula": "pseudo_z = source - current_mean[nearest] + old_mean (float64 shift -> float32)",
        })
        pseudo_parts.append(pseudo)
        pseudo_labels.append(np.full(len(pseudo), cls, dtype=np.int64))
    return pseudo_parts, pseudo_labels, manifest


# ---------------------------------------------------------------- task1 encoder (unchanged)
def train_task1_encoder(model, optimizer, ctx, t1_train, t1_val, n_classes, device, counters, seed):
    best_key, best_state, best_opt = (-1.0, -float("inf")), None, None
    best_acc, best_loss, stagnant, trained = -1.0, float("inf"), 0, 0
    best_epoch = 0
    best_key_va, best_key_vl = -1.0, float("inf")
    epoch_rows = []
    stop_reason = f"max_epochs_{MAX_EPOCHS}"
    for epoch in range(1, MAX_EPOCHS + 1):
        loader = make_loader(ctx["x_std"], ctx["y"], t1_train, BATCH_SIZE, seed + 100000 + epoch)
        model.train()
        epoch_losses = []
        for xx, yy in loader:
            xx, yy = xx.to(device), yy.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx)[:, [0, 1]]
            mapped = labels_to_local(yy, [0, 1], n_classes, device)
            loss = F.cross_entropy(logits, mapped)
            if not bool(torch.isfinite(loss).all().item()):
                raise Halt(f"NON-FINITE encoder loss epoch {epoch}")
            loss.backward()
            counters["encoder_backward"] += 1
            assert_grads_finite_present(model, f"encoder pre-clip epoch {epoch}")
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            counters["encoder_optimizer_steps"] += 1
            assert_params_finite(model, f"encoder post-step epoch {epoch}")
            epoch_losses.append(float(loss.detach()))
        va, vl = evaluate_validation(model, ctx["x_std"], ctx["y"], t1_val, [0, 1],
                                     n_classes, device, EVAL_BATCH)
        trained = epoch
        key = (va, -vl)
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch
            best_key_va, best_key_vl = va, vl
        if va - best_acc >= VAL_DELTA_ACC or best_loss - vl >= VAL_DELTA_LOSS:
            best_acc, best_loss, stagnant = max(best_acc, va), min(best_loss, vl), 0
        else:
            stagnant += 1
        epoch_rows.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)),
                           "val_macro_accuracy": va, "val_loss": vl})
        if epoch >= MIN_EPOCHS and stagnant >= PATIENCE:
            stop_reason = f"early_stopping_epoch_{epoch}"
            break
    require(best_state is not None, "encoder_best_state")
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_opt)
    return dict(epochs_trained=trained, stop_reason=stop_reason, best_epoch=best_epoch,
                selected_checkpoint_metrics={"val_macro_accuracy": best_key_va, "val_loss": best_key_vl},
                early_stopping_tracker={"best_acc": best_acc, "best_loss": best_loss},
                epoch_rows=epoch_rows, budget_truncated=(trained == MAX_EPOCHS))


# ---------------------------------------------------------------- state package (B5)
def centroid_value_hash(centroids, seen):
    h = hashlib.sha256()
    for c in sorted(seen):
        h.update(np.ascontiguousarray(np.asarray(centroids[c], dtype=np.float32)).tobytes())
    return h.hexdigest()


def save_state_package(ddir, task_idx, centroids, counts, task_of, name, seed, head):
    seen = list(range(2 * task_idx))
    sp_keys = {}
    for c in seen:
        sp_keys[f"centroid_{c}"] = np.asarray(centroids[c], dtype=np.float32)
    sp_keys["seen_classes"] = np.asarray(seen, dtype=np.int64)
    sp_keys["class_counts"] = np.asarray([counts[c] for c in seen], dtype=np.int64)
    sp_keys["task_ids"] = np.asarray([task_of[c] for c in seen], dtype=np.int64)
    pkg = ddir / f"state_package_task{task_idx}.npz"
    np.savez_compressed(pkg, **sp_keys)
    meta = {
        "dataset": name, "seed": seed, "task": task_idx, "seen_classes": seen,
        "class_real_train_counts": {str(c): counts[c] for c in seen},
        "class_task_mapping": {str(c): task_of[c] for c in seen},
        "centroid_dtype": "float32 (uniform)",
        "centroid_value_sha256": centroid_value_hash(centroids, seen),
        "package_file_sha256": sha(pkg),
        "frozen_model": {"file": str(ddir / "task1_encoder_best.pt"),
                         "sha256": sha(ddir / "task1_encoder_best.pt")},
        "current_head": {"file": str(ddir / f"task{task_idx}_head.pt"),
                         "sha256": sha(ddir / f"task{task_idx}_head.pt")},
        "input_stats_sha256": sha(ad.FIX_DIR / "preprocess_statistics.json"),
        "descriptor_stats_sha256": sha(V2_DIR / f"interface_check_{name}.json"),
        "config_version": "s1_baseline_v1",
    }
    dump_json(ddir / f"state_package_task{task_idx}.json", meta)
    return meta


def validate_state_package(ddir, name, seed, task_idx):
    """Full schema check shared by load/recover. Rejects identity/count/mapping/hash errors."""
    meta_path = ddir / f"state_package_task{task_idx}.json"
    pkg = ddir / f"state_package_task{task_idx}.npz"
    require(meta_path.exists(), f"{name}_state_meta_exists")
    require(pkg.exists(), f"{name}_state_pkg_exists")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    require(meta["dataset"] == name, f"{name}_state_dataset_identity", meta["dataset"])
    require(int(meta["seed"]) == seed, f"{name}_state_seed_identity", str(meta["seed"]))
    require(int(meta["task"]) == task_idx, f"{name}_state_task_identity", str(meta["task"]))
    seen = [int(c) for c in meta["seen_classes"]]
    require(seen == list(range(2 * task_idx)), f"{name}_state_seen_classes", str(seen))
    require(meta["package_file_sha256"] == sha(pkg), f"{name}_state_package_hash")
    sp = np.load(pkg)
    centroids, counts, task_of = {}, {}, {}
    for c in seen:
        v = sp[f"centroid_{c}"]
        require(v.dtype == np.float32, f"{name}_state_centroid_{c}_float32")
        require(v.shape == (JOINT_DIM,), f"{name}_state_centroid_{c}_shape")
        require(bool(np.isfinite(v).all()), f"{name}_state_centroid_{c}_finite")
        centroids[c] = v.copy()
        cnt = int(sp["class_counts"][seen.index(c)])
        require(cnt > 0, f"{name}_state_count_{c}_positive", str(cnt))
        require(cnt == int(meta["class_real_train_counts"][str(c)]),
                f"{name}_state_count_{c}_npz_meta_consistent", f"{cnt} vs {meta['class_real_train_counts'][str(c)]}")
        counts[c] = cnt
        tid = int(sp["task_ids"][seen.index(c)])
        expect_tid = (c // 2) + 1
        require(tid == expect_tid, f"{name}_state_task_mapping_{c}", f"{tid} vs {expect_tid}")
        require(tid == int(meta["class_task_mapping"][str(c)]), f"{name}_state_task_mapping_meta_{c}")
        task_of[c] = tid
    require(meta["centroid_value_sha256"] == centroid_value_hash(centroids, seen),
            f"{name}_state_centroid_value_hash")
    # references (from previously saved manifest)
    require(meta["frozen_model"]["sha256"] == sha(Path(meta["frozen_model"]["file"])),
            f"{name}_state_frozen_model_ref")
    require(meta["current_head"]["sha256"] == sha(Path(meta["current_head"]["file"])),
            f"{name}_state_current_head_ref")
    require(meta["input_stats_sha256"] == sha(ad.FIX_DIR / "preprocess_statistics.json"),
            f"{name}_state_input_stats_ref")
    require(meta["descriptor_stats_sha256"] == sha(V2_DIR / f"interface_check_{name}.json"),
            f"{name}_state_descriptor_stats_ref")
    return {"centroids": centroids, "counts": counts, "task_of": task_of, "seen": seen,
            "package_sha256": meta["package_file_sha256"],
            "centroid_value_sha256": meta["centroid_value_sha256"]}


# ---------------------------------------------------------------- metrics (unchanged)
def task_accuracy_matrix(preds_per_task, labels_per_task, num_tasks):
    """R[t][j] (1-indexed) = mean class accuracy over task j's 2 classes, using task t's head.

    A seen class with zero test members is an ERROR (not masked): raises Halt."""
    R = np.full((num_tasks + 1, num_tasks + 1), np.nan)
    for t in range(1, num_tasks + 1):
        pred = np.asarray(preds_per_task[t])
        lab = np.asarray(labels_per_task[t])
        for j in range(1, t + 1):
            accs = []
            for c in ALL_TASKS[j - 1]:
                m = lab == c
                require(m.sum() > 0, f"R_missing_class_{c}_task_{j}")
                accs.append(float((pred[m] == c).mean()))
            R[t, j] = float(np.mean(accs))
    return R


def forgetting_from_R(R, num_tasks):
    Fvec = [np.nan] * (num_tasks + 1)
    for t in range(2, num_tasks + 1):
        vals = []
        for j in range(1, t):
            hist = [R[k, j] for k in range(j, t)]
            if any(np.isnan(v) for v in hist) or np.isnan(R[t, j]):
                vals.append(np.nan)
            else:
                vals.append(max(hist) - R[t, j])
        Fvec[t] = float(np.nanmean(vals)) if vals else np.nan
    return Fvec


def bwt_from_R(R, num_tasks):
    B = [np.nan] * (num_tasks + 1)
    for t in range(2, num_tasks + 1):
        vals = [R[t, j] - R[j, j] for j in range(1, t)
                if not (np.isnan(R[t, j]) or np.isnan(R[j, j]))]
        B[t] = float(np.mean(vals)) if vals else np.nan
    return B


# ---------------------------------------------------------------- training one group
def train_group(name: str, seed: int) -> int:
    require(not (OUT / "training_seal.json").exists(), "train_no_seal_present")
    ddir = OUT / name / f"seed{seed}"
    if ddir.exists():
        raise Halt(f"REFUSE: target dir exists: {ddir}")
    ddir.mkdir(parents=True)
    (ddir / "logs").mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "device_is_cuda", str(device))
    preflight("train", name, seed, None)
    counters = {k: 0 for k in ("encoder_forwards", "encoder_backward", "encoder_optimizer_steps",
                               "head_forwards", "head_backward", "head_optimizer_steps",
                               "test_set_forwards")}
    started = time.perf_counter()
    rss_start = process_memory_bytes()
    log(f"RUN {name} seed={seed} device={device}")

    ctx = load_context(name)
    tasks = task_list(name)
    n_tasks = len(tasks)
    torch.cuda.reset_peak_memory_stats(device)

    model = build_model(ctx["input_channels"], ctx["n_classes"], seed).to(device)
    model.encoder.register_forward_hook(lambda *a: counters.__setitem__(
        "encoder_forwards", counters["encoder_forwards"] + 1))

    centroids, counts, task_of = {}, {}, {}
    handoff_ledger, boundary_memory, head_log_rows, pseudo_manifest_all = [], [], [], []

    for task_idx, new_classes in enumerate(tasks, start=1):
        log(f"--- task{task_idx} new_classes={new_classes} ---")
        seen = list(range(2 * task_idx))
        t_train = ctx["task_rows"](new_classes, "train")
        t_val = ctx["task_rows"](new_classes, "val")
        require(bool(np.isin(ctx["y"][t_train], new_classes).all()), f"{name}s{seed}_t{task_idx}_train_labels")
        require(bool(np.isin(ctx["y"][t_val], new_classes).all()), f"{name}s{seed}_t{task_idx}_val_labels")

        if task_idx >= 2:
            handoff = validate_state_package(ddir, name, seed, task_idx - 1)
            centroids, counts, task_of = handoff["centroids"], handoff["counts"], handoff["task_of"]
            require(handoff["seen"] == list(range(2 * (task_idx - 1))), f"{name}s{seed}_t{task_idx}_handoff_seen")
            handoff_ledger.append({"task": task_idx, "loaded_package": f"state_package_task{task_idx-1}.npz",
                                   "package_sha256": handoff["package_sha256"],
                                   "centroid_value_sha256": handoff["centroid_value_sha256"],
                                   "loaded_seen": handoff["seen"], "used_for_pseudo": True})
            log(f"  loaded prev state: seen={handoff['seen']} pkg={handoff['package_sha256'][:12]}")

        if task_idx == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=ENCODER_LR)
            sig_before = full_signature(model)
            tr = train_task1_encoder(model, optimizer, ctx, t_train, t_val,
                                     ctx["n_classes"], device, counters, seed)
            require(counters["encoder_optimizer_steps"] > 0, f"{name}s{seed}_encoder_updated")
            require(full_signature(model) != sig_before, f"{name}s{seed}_encoder_changed")
            ckpt = {"model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "epochs_trained": tr["epochs_trained"], "stop_reason": tr["stop_reason"],
                    "best_epoch": tr["best_epoch"],
                    "selected_checkpoint_metrics": tr["selected_checkpoint_metrics"],
                    "early_stopping_tracker": tr["early_stopping_tracker"],
                    "budget_truncated": tr["budget_truncated"], "seed": seed}
            torch.save(ckpt, ddir / "task1_encoder_best.pt")
            with (ddir / "encoder_epoch_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_macro_accuracy", "val_loss"])
                w.writeheader(); w.writerows(tr["epoch_rows"])
            fr = ai.freeze_representation(model)
            require(fr["encoder_all_frozen"] and fr["attention_all_frozen"] and fr["head_all_frozen"],
                    f"{name}s{seed}_frozen")
            model.eval()
            model.zero_grad(set_to_none=True)
            require(all(p.grad is None for p in model.parameters()), f"{name}s{seed}_grads_none")
            frozen_sig = full_signature(model)
            dump_json(ddir / "freeze_verification.json", {"frozen_signature_sha256": frozen_sig})

        z_t, _, _, _ = ts.make_joint(model, ctx["x_std"], ctx["x_raw"], t_train,
                                     ctx["d_mean"], ctx["d_std"], name, device)
        require(z_t.shape == (len(t_train), JOINT_DIM), f"{name}s{seed}_t{task_idx}_joint")
        current_centroids = {}
        for c in new_classes:
            m = z_t[ctx["y"][t_train] == c].mean(axis=0).astype(np.float32)
            ds.assert_all_finite(m, f"{name}s{seed} centroid[{c}]")
            centroids[c] = m
            current_centroids[c] = m
            counts[c] = int((ctx["y"][t_train] == c).sum())
            task_of[c] = task_idx

        old_classes = [c for c in seen if c not in new_classes]
        pseudo_parts, pseudo_labels, pm = ([], [], []) if not old_classes else \
            pooled_shift_pseudo(centroids, current_centroids, z_t, ctx["y"][t_train], t_train,
                                old_classes, new_classes)
        pseudo_total = int(sum(m["pseudo_count"] for m in pm))
        pseudo_manifest_all.append({"task": task_idx, "sources": pm})

        torch.manual_seed(seed + 10000 * task_idx)
        head = nn.Linear(JOINT_DIM, 2 * task_idx).to(device)
        X = np.concatenate([z_t.astype(np.float32)] + pseudo_parts, axis=0).astype(np.float32)
        Y = np.concatenate([ctx["y"][t_train].astype(np.int64)] + pseudo_labels, axis=0).astype(np.int64)
        require(sorted(set(Y.tolist())) == seen, f"{name}s{seed}_t{task_idx}_head_labels")
        head_losses, head_changed = fit_sparse_head(head, X, Y, device, f"{name}s{seed}_t{task_idx}_head", counters)
        require(head_changed, f"{name}s{seed}_t{task_idx}_head_updated")
        require(head.in_features == 56 and head.out_features == 2 * task_idx, f"{name}s{seed}_t{task_idx}_head_dim")
        torch.save(head.state_dict(), ddir / f"task{task_idx}_head.pt")
        for i, v in enumerate(head_losses):
            head_log_rows.append({"task": task_idx, "epoch": i + 1, "loss": v})

        require(full_signature(model) == frozen_sig, f"{name}s{seed}_t{task_idx}_frozen_unchanged")
        require(all(not p.requires_grad for p in model.parameters()), f"{name}s{seed}_t{task_idx}_no_trainable")
        require(all(p.grad is None for p in model.parameters()), f"{name}s{seed}_t{task_idx}_grads_none")

        save_state_package(ddir, task_idx, centroids, counts, task_of, name, seed, head)

        # ---- per-boundary budget / memory (B6) ----
        frozen_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        frozen_buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        head_bytes = sum(p.numel() * p.element_size() for p in head.parameters())
        centroid_bytes = sum(centroids[c].nbytes for c in seen)
        pseudo_bytes = sum(p.nbytes for p in pseudo_parts)
        boundary_memory.append({
            "task": task_idx,
            "logical_bytes": {"frozen_params": frozen_param_bytes, "frozen_buffers": frozen_buf_bytes,
                              "current_head": head_bytes, "all_centroids": centroid_bytes,
                              "class_counts": len(seen) * 8, "task_mapping": len(seen) * 8,
                              "input_stats": ctx["in_mean"].nbytes + ctx["in_denom"].nbytes,
                              "descriptor_stats": ctx["d_mean"].nbytes + ctx["d_std"].nbytes},
            "serialized_bytes": {"state_package_npz": (ddir / f"state_package_task{task_idx}.npz").stat().st_size,
                                 "task1_encoder_pt": (ddir / "task1_encoder_best.pt").stat().st_size,
                                 "current_head_pt": (ddir / f"task{task_idx}_head.pt").stat().st_size},
            "real_replay_samples": 0,
            "transient_pseudo_sample_count": pseudo_total,
            "transient_pseudo_array_bytes": pseudo_bytes,
            "gpu_peak_cumulative_group": gpu_peak_bytes(device),
            "rss": process_memory_bytes(),
            "stage_elapsed_cumulative_since_run_start": round(time.perf_counter() - started, 1),
        })

        # ---- reference outputs PRODUCER (B2) then recover consumer ----
        task_dir = ddir / f"task{task_idx}"
        task_dir.mkdir(exist_ok=True)
        ref_rows = []
        for c in new_classes:
            idx = t_train[ctx["y"][t_train] == c][:REF_PER_CLASS]
            ref_rows.extend(int(i) for i in idx)
        ref_rows = np.asarray(ref_rows, dtype=np.int64)
        ref_labels = ctx["y"][ref_rows].astype(np.int64)
        joint_ref, latent_ref, desc_raw_ref, desc_std_ref = ts.make_joint(
            model, ctx["x_std"], ctx["x_raw"], ref_rows, ctx["d_mean"], ctx["d_std"], name, device)
        with torch.inference_mode():
            hlog = head(torch.from_numpy(joint_ref).to(device)).cpu().numpy().astype(np.float32)
            counters["head_forwards"] += 1
        x_std_ref = np.ascontiguousarray(ctx["x_std"][ref_rows]).astype(np.float32)
        raw_sha_ref = [hashlib.sha256(np.ascontiguousarray(ctx["x_raw"][r]).tobytes()).hexdigest()
                       for r in ref_rows]
        ref_npz = task_dir / "reference_outputs.npz"
        np.savez_compressed(ref_npz, row_ids=ref_rows, labels=ref_labels, x_std=x_std_ref,
                            latent=latent_ref, desc_raw=desc_raw_ref, desc_std=desc_std_ref,
                            joint=joint_ref, head_logits=hlog)
        dump_json(task_dir / "reference_outputs_meta.json", {
            "task": task_idx, "new_classes": new_classes, "row_ids": ref_rows.tolist(),
            "labels": ref_labels.tolist(), "raw_window_sha256": raw_sha_ref,
            "reference_npz_sha256": sha(ref_npz),
            "ref_per_class": REF_PER_CLASS,
        })
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--recover",
                            name, str(seed), str(task_idx)],
                           cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            log(r.stdout); log(r.stderr)
            raise Halt(f"recovery failed {name} seed{seed} task{task_idx}: exit {r.returncode}")

        log(f"  task{task_idx} done: real={len(z_t)} pseudo={pseudo_total} seen={seen}")

    dump_json(ddir / "handoff_ledger.json", {"per_boundary": handoff_ledger})
    dump_json(ddir / "boundary_memory.json", boundary_memory)
    dump_json(ddir / "pseudo_feature_manifest.json", {"per_task": pseudo_manifest_all})
    with (ddir / "readout_training_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["task", "epoch", "loss"])
        w.writeheader(); w.writerows(head_log_rows)
    dump_json(ddir / "call_ledger_run.json", {
        "encoder_forwards": counters["encoder_forwards"], "encoder_backward": counters["encoder_backward"],
        "encoder_optimizer_steps": counters["encoder_optimizer_steps"],
        "head_forwards": counters["head_forwards"], "head_backward": counters["head_backward"],
        "head_optimizer_steps": counters["head_optimizer_steps"],
        "test_set_forwards": counters["test_set_forwards"],
        "elapsed_seconds": round(time.perf_counter() - started, 1), "pid": os.getpid()})
    dump_json(ddir / "environment.json", {
        "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "numpy": np.__version__, "os": platform.platform(),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "seed": seed, "rss_at_start": rss_start})
    # R2: write output_hashes (source of seal) then postflight, THEN the PASS marker
    dump_json(ddir / "output_hashes.json", {
        "task1_encoder_best.pt": sha(ddir / "task1_encoder_best.pt"),
        "state_package": {f"task{t}": {"npz": sha(ddir / f"state_package_task{t}.npz"),
                                       "json": sha(ddir / f"state_package_task{t}.json")}
                          for t in range(1, n_tasks + 1)},
        "heads": {f"task{t}": sha(ddir / f"task{t}_head.pt") for t in range(1, n_tasks + 1)},
    })
    postflight("train")
    dump_json(ddir / "gate_decision.json", {"dataset": name, "seed": seed,
                                            "verdict": "TRAINING_PHASE_PASS", "tasks_completed": n_tasks})
    return 0


# ---------------------------------------------------------------- recover (independent)
def recover_group(name: str, seed: int, task_idx: int) -> int:
    ddir = OUT / name / f"seed{seed}"
    task_dir = ddir / f"task{task_idx}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "device_is_cuda", str(device))
    preflight("recover", name, seed, task_idx)
    counters = {"encoder_forwards": 0, "head_forwards": 0}
    started = time.perf_counter()
    log(f"RECOVER {name} seed{seed} task{task_idx}")

    ctx = load_context(name)
    ckpt = torch.load(ddir / "task1_encoder_best.pt", map_location="cpu", weights_only=False)
    head_sd = torch.load(ddir / f"task{task_idx}_head.pt", map_location="cpu", weights_only=False)
    model = build_model(ctx["input_channels"], ctx["n_classes"], 0).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    require(not missing and not unexpected, f"{name}s{seed}_model_keys_exact")
    head = nn.Linear(JOINT_DIM, 2 * task_idx).to(device)
    m, u = head.load_state_dict(head_sd, strict=False)
    require(not m and not u, f"{name}s{seed}_head_keys_exact")
    ai.freeze_representation(model)
    model.eval(); head.eval()
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    require(all(p.grad is None for p in list(model.parameters()) + list(head.parameters())),
            f"{name}s{seed}_grads_none")

    validate_state_package(ddir, name, seed, task_idx)   # full schema (B5)

    new_classes = ALL_TASKS[task_idx - 1]
    t_rows = ctx["task_rows"](new_classes, "train")
    ref_rows = []
    for c in new_classes:
        idx = t_rows[ctx["y"][t_rows] == c][:REF_PER_CLASS]
        ref_rows.extend(int(i) for i in idx)
    ref_rows = np.asarray(ref_rows, dtype=np.int64)
    ref = np.load(task_dir / "reference_outputs.npz")
    require(bool(np.array_equal(ref_rows, ref["row_ids"])), f"{name}s{seed}_rebuilt_row_ids")
    labels = ctx["y"][ref_rows].astype(np.int64)
    require(bool(np.array_equal(labels, ref["labels"])), f"{name}s{seed}_rebuilt_labels")

    model.encoder.register_forward_hook(lambda *a: counters.__setitem__(
        "encoder_forwards", counters["encoder_forwards"] + 1))
    x_std8 = np.ascontiguousarray(ctx["x_std"][ref_rows]).astype(np.float32)
    joint8, latent8, desc_raw8, desc_std8 = ts.make_joint(
        model, ctx["x_std"], ctx["x_raw"], ref_rows, ctx["d_mean"], ctx["d_std"], name, device)
    with torch.inference_mode():
        hlog = head(torch.from_numpy(joint8).to(device)).cpu().numpy().astype(np.float32)
        counters["head_forwards"] += 1

    def bitwise(a, b, label):
        a, b = np.asarray(a), np.asarray(b)
        ok = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        return {"field": label, "bitwise_equal": bool(ok),
                "max_abs_diff": 0.0 if ok else float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())}

    comp = [bitwise(ref_rows, ref["row_ids"], "row_ids"), bitwise(labels, ref["labels"], "labels"),
            bitwise(x_std8, ref["x_std"], "x_std"), bitwise(latent8, ref["latent"], "latent"),
            bitwise(desc_raw8, ref["desc_raw"], "desc_raw"), bitwise(desc_std8, ref["desc_std"], "desc_std"),
            bitwise(joint8, ref["joint"], "joint"), bitwise(hlog, ref["head_logits"], "head_logits")]
    for c in comp:
        require(c["bitwise_equal"], f"{name}s{seed}_task{task_idx}_bitwise_{c['field']}")

    out_npz = task_dir / "recovery_outputs.npz"
    np.savez_compressed(out_npz, row_ids=ref_rows, labels=labels, x_std=x_std8,
                        latent=latent8, desc_raw=desc_raw8, desc_std=desc_std8,
                        joint=joint8, head_logits=hlog)
    dump_json(task_dir / "recovery_comparison.json", {"fields": comp, "all_bitwise_equal": True})
    dump_json(task_dir / "recovery_output_hashes.json", {
        "recovery_outputs.npz": sha(out_npz),
        "state_package": sha(ddir / f"state_package_task{task_idx}.npz")})
    dump_json(task_dir / "call_ledger_recover.json", {
        "encoder_forwards": counters["encoder_forwards"], "head_forwards": counters["head_forwards"],
        "backward": 0, "optimizer_steps": 0, "rss": process_memory_bytes(),
        "gpu_peak": gpu_peak_bytes(device),
        "elapsed_seconds": round(time.perf_counter() - started, 1), "pid": os.getpid()})
    postflight("recover")
    return 0


# ---------------------------------------------------------------- seal training (R2)
def _expected_group_ids() -> set[tuple[str, int]]:
    return {(d, s) for d in DATASETS for s in SEEDS}


def seal_training() -> int:
    preflight("seal", None, None, None)
    require(not (OUT / "training_seal.json").exists(), "seal_no_existing")
    require(not (OUT / "training_seal.json.sha256").exists(), "seal_no_existing_sha")
    groups = []
    for name in DATASETS:
        for seed in SEEDS:
            ddir = OUT / name / f"seed{seed}"
            g = json.loads((ddir / "gate_decision.json").read_text(encoding="utf-8"))
            require(g["verdict"] == "TRAINING_PHASE_PASS", f"seal_{name}_{seed}_pass")
            require(g["tasks_completed"] == DATASET_TASK_COUNTS[name], f"seal_{name}_{seed}_tasks")
            require(not list(ddir.glob("failure_record_*.json")), f"seal_{name}_{seed}_no_failures")
            n = DATASET_TASK_COUNTS[name]
            # per-task recovery success + full state schema (from prior manifest hashes)
            for t in range(1, n + 1):
                cmp = json.loads((ddir / f"task{t}" / "recovery_comparison.json").read_text(encoding="utf-8"))
                require(cmp.get("all_bitwise_equal") is True, f"seal_{name}_{seed}_t{t}_recovery_pass")
                validate_state_package(ddir, name, seed, t)
            # handoff ledger for t>=2
            if n >= 2:
                require((ddir / "handoff_ledger.json").exists(), f"seal_{name}_{seed}_handoff")
            # output_hashes consistency (use PRIOR manifest, not regenerated hash)
            oh = json.loads((ddir / "output_hashes.json").read_text(encoding="utf-8"))
            require(oh["task1_encoder_best.pt"] == sha(ddir / "task1_encoder_best.pt"),
                    f"seal_{name}_{seed}_encoder_hash")
            groups.append({"dataset": name, "seed": seed,
                           "encoder_sha": oh["task1_encoder_best.pt"],
                           "state_sha": {f"task{t}": oh["state_package"][f"task{t}"]["npz"]
                                         for t in range(1, n + 1)},
                           "heads_sha": {f"task{t}": oh["heads"][f"task{t}"] for t in range(1, n + 1)}})
    ids = {(g["dataset"], g["seed"]) for g in groups}
    require(ids == _expected_group_ids(), "seal_exact_32_unique_groups",
            f"got={len(ids)} expected=32")
    require(len(groups) == 32, "seal_32_groups", str(len(groups)))
    seal = {"status": "SEALED", "groups": groups, "protocol_sha256": sha(OUT / "protocol_locked.json"),
            "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    dump_json(OUT / "training_seal.json", seal)
    dump_json(OUT / "training_seal.json.sha256", {"sha256": sha(OUT / "training_seal.json")})
    log("training seal written (32 unique groups)")
    return 0


def check_seal() -> dict:
    require((OUT / "training_seal.json").exists(), "eval_seal_exists")
    require((OUT / "training_seal.json.sha256").exists(), "eval_seal_sha_exists")
    rec = json.loads((OUT / "training_seal.json.sha256").read_text(encoding="utf-8"))
    require(rec["sha256"] == sha(OUT / "training_seal.json"), "eval_seal_hash_matches")
    seal = json.loads((OUT / "training_seal.json").read_text(encoding="utf-8"))
    require(seal.get("status") == "SEALED", "eval_seal_status_sealed")
    require(seal.get("protocol_sha256") == sha(OUT / "protocol_locked.json"),
            "eval_seal_protocol_hash")
    ids = {(g["dataset"], g["seed"]) for g in seal["groups"]}
    require(ids == _expected_group_ids(), "eval_seal_exact_32_unique")
    require(len(seal["groups"]) == 32, "eval_seal_32_groups")
    for g in seal["groups"]:
        ddir = OUT / g["dataset"] / f"seed{g['seed']}"
        require(sha(ddir / "task1_encoder_best.pt") == g["encoder_sha"],
                f"eval_seal_{g['dataset']}_{g['seed']}_encoder_file")
        for t in range(1, DATASET_TASK_COUNTS[g["dataset"]] + 1):
            require(sha(ddir / f"state_package_task{t}.npz") == g["state_sha"][f"task{t}"],
                    f"eval_seal_{g['dataset']}_{g['seed']}_state_{t}_file")
            require(sha(ddir / f"task{t}_head.pt") == g["heads_sha"][f"task{t}"],
                    f"eval_seal_{g['dataset']}_{g['seed']}_head_{t}_file")
    return seal


# ---------------------------------------------------------------- evaluation (B1, B4)
def evaluate_group(name: str, seed: int) -> int:
    ddir = OUT / name / f"seed{seed}"
    require(not (ddir / "evaluation.json").exists(), f"eval_{name}_{seed}_no_overwrite")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "device_is_cuda", str(device))
    preflight("eval", name, seed, None)
    seal = check_seal()   # before any model build / forward
    require(any(g["dataset"] == name and g["seed"] == seed for g in seal["groups"]),
            f"eval_{name}_{seed}_in_seal")
    counters = {"encoder_forwards": 0, "head_forwards": 0}
    started = time.perf_counter()
    log(f"EVALUATE {name} seed{seed} (test phase)")
    ctx = load_context(name)
    tasks = task_list(name)
    n_tasks = len(tasks)
    ckpt = torch.load(ddir / "task1_encoder_best.pt", map_location="cpu", weights_only=False)
    model = build_model(ctx["input_channels"], ctx["n_classes"], 0).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    ai.freeze_representation(model)
    model.eval()
    model.zero_grad(set_to_none=True)
    model.encoder.register_forward_hook(lambda *a: counters.__setitem__(
        "encoder_forwards", counters["encoder_forwards"] + 1))

    preds_per_task, labels_per_task = {}, {}
    per_row, confusion, per_class = {}, {}, {}
    seen_macro, seen_micro = {}, {}
    for t in range(1, n_tasks + 1):
        seen = list(range(2 * t))
        head_sd = torch.load(ddir / f"task{t}_head.pt", map_location="cpu", weights_only=False)
        head = nn.Linear(JOINT_DIM, 2 * t).to(device)
        head.load_state_dict(head_sd, strict=True)
        head.eval()
        tst_idx = ctx["task_rows"](seen, "test")   # test view (B1)
        labels = ctx["test_view"]["y"][tst_idx].astype(np.int64)
        require(len(tst_idx) == len(labels), f"{name}s{seed}_t{t}_test_len")
        require(len(set(zip(tst_idx.tolist(), labels.tolist()))) == len(tst_idx),
                f"{name}s{seed}_t{t}_test_rows_unique")
        if len(tst_idx) == 0:
            raise Halt(f"{name}s{seed}_t{t}_no_test_members")
        for c in seen:
            require((labels == c).sum() > 0, f"{name}s{seed}_t{t}_test_has_class_{c}")
        z, _, _, _ = ts.make_joint(model, ctx["x_std_test"], ctx["test_view"]["x_raw"], tst_idx,
                                   ctx["d_mean"], ctx["d_std"], name, device)
        with torch.inference_mode():
            logits = head(torch.from_numpy(z).to(device)).cpu().numpy().astype(np.float32)
        counters["head_forwards"] += 1
        if not bool(np.isfinite(logits).all()):
            raise Halt(f"{name}s{seed}_t{t}_nonfinite_logits")
        pred = logits.argmax(axis=1).astype(np.int64)
        preds_per_task[t] = pred
        labels_per_task[t] = labels
        per_row[t] = {"row_space": ctx["test_row_space"], "row_ids": tst_idx.tolist(),
                      "labels": labels.tolist(), "prediction": pred.tolist(),
                      "logits": logits.tolist(), "column_mapping": seen}
        # confusion matrix (seen x seen)
        cm = np.zeros((2 * t, 2 * t), dtype=np.int64)
        for i in range(len(labels)):
            cm[labels[i], pred[i]] += 1
        confusion[t] = cm.tolist()
        pc = {}
        for c in seen:
            mm = labels == c
            pc[int(c)] = float((pred[mm] == c).mean()) if mm.sum() else None
        per_class[t] = pc
        seen_macro[t] = float(np.mean([pc[c] for c in seen]))
        seen_micro[t] = float((pred == labels).mean())
    R = task_accuracy_matrix(preds_per_task, labels_per_task, n_tasks)
    Fvec = forgetting_from_R(R, n_tasks)
    B = bwt_from_R(R, n_tasks)
    final_macro = seen_macro[n_tasks]
    avg_incremental = float(np.mean([seen_macro[t] for t in range(1, n_tasks + 1)]))
    dump_json(ddir / "evaluation.json", {
        "dataset": name, "seed": seed,
        "final_task_seen_macro_accuracy": final_macro,
        "per_boundary_macro": {str(t): seen_macro[t] for t in seen_macro},
        "per_boundary_micro": {str(t): seen_micro[t] for t in seen_micro},
        "per_class_accuracy": {str(t): per_class[t] for t in per_class},
        "R_matrix": R.tolist(), "forgetting_F": Fvec, "BWT": B,
        "average_incremental_accuracy": avg_incremental,
        "test_row_space": ctx["test_row_space"],
    })
    dump_json(ddir / "evaluation_per_row.json", {str(t): per_row[t] for t in per_row})
    dump_json(ddir / "evaluation_confusion.json", {str(t): confusion[t] for t in confusion})
    dump_json(ddir / "evaluation_call_ledger.json", {
        "encoder_forwards": counters["encoder_forwards"],
        "head_forwards": counters["head_forwards"],
        "backward": 0, "optimizer_steps": 0,
        "device": str(device), "elapsed_seconds": round(time.perf_counter() - started, 1),
        "rss": process_memory_bytes(), "gpu_peak": gpu_peak_bytes(device), "pid": os.getpid()})
    postflight("eval")
    # R2: evaluation output hash manifest (after postflight)
    dump_json(ddir / "evaluation_output_hashes.json", {
        "evaluation.json": sha(ddir / "evaluation.json"),
        "evaluation_per_row.json": sha(ddir / "evaluation_per_row.json"),
        "evaluation_confusion.json": sha(ddir / "evaluation_confusion.json"),
        "evaluation_call_ledger.json": sha(ddir / "evaluation_call_ledger.json"),
    })
    return 0


# ---------------------------------------------------------------- aggregate (B4, R2)
def aggregate() -> int:
    preflight("aggregate", None, None, None)
    check_seal()
    rows = []
    for name in DATASETS:
        for seed in SEEDS:
            ddir = OUT / name / f"seed{seed}"
            ev = json.loads((ddir / "evaluation.json").read_text(encoding="utf-8"))
            require(ev["dataset"] == name and ev["seed"] == seed, f"agg_{name}_{seed}_identity")
            require((ddir / "evaluation_output_hashes.json").exists(),
                    f"agg_{name}_{seed}_eval_hash_manifest")
            eh = json.loads((ddir / "evaluation_output_hashes.json").read_text(encoding="utf-8"))
            require(eh["evaluation.json"] == sha(ddir / "evaluation.json"),
                    f"agg_{name}_{seed}_eval_file_hash")
            vals = [ev["final_task_seen_macro_accuracy"], ev["average_incremental_accuracy"]]
            require(all(isinstance(v, (int, float)) and np.isfinite(v) for v in vals),
                    f"agg_{name}_{seed}_finite")
            rows.append({"dataset": name, "seed": seed,
                         "final_macro": ev["final_task_seen_macro_accuracy"],
                         "avg_incremental": ev["average_incremental_accuracy"],
                         "F_final": ev["forgetting_F"][-1], "BWT_final": ev["BWT"][-1]})
    with (OUT / "aggregate_per_seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)
    summary = {}
    for name in DATASETS:
        sub = [r for r in rows if r["dataset"] == name]
        require(len(sub) == 8, f"agg_{name}_8_seeds")
        for k in ("final_macro", "avg_incremental", "F_final", "BWT_final"):
            vals = [r[k] for r in sub]
            arr = np.asarray(vals, dtype=np.float64)
            n = len(arr)
            m = float(arr.mean())
            sd = float(arr.std(ddof=1))
            se = sd / np.sqrt(n) if n > 1 else np.nan
            # 95% t interval, df = n-1 = 7
            import math
            tcrit = 2.3646242515927853  # t_{0.975}(7)
            lo, hi = m - tcrit * se, m + tcrit * se
            summary.setdefault(name, {})[k] = {"mean": m, "sd_ddof1": sd,
                                               "ci95_lo": lo, "ci95_hi": hi,
                                               "per_seed": vals}
    dump_json(OUT / "aggregate_summary.json", {"per_dataset": summary,
                                               "note": "internal unrounded ratios; "
                                                       "acc displayed %, diffs pp; no candidate pairing -> NA"})
    log("aggregate done")
    return 0


# ---------------------------------------------------------------- propose / static / regression
def propose_protocol() -> int:
    PROTO_OUT.mkdir(parents=True, exist_ok=True)
    inv = input_files()
    hashes = {k: sha(v) for k, v in inv.items()}
    base = json.loads((PROTO_OLD / "protocol_proposed.json").read_text(encoding="utf-8"))
    base["status"] = "PROPOSED"
    base["version"] = "s1_baseline_4datasets_v1_fix1_proposed"
    base["prep_dir"] = str(PROTO_OUT)
    base["fixes"] = ["B1_test_namespace", "B2_reference_producer", "B3_preflight_seal_lock",
                     "B4_evidence_summary", "B5_state_schema_negatives", "B6_budget_powershell"]
    base["test_forward_scope"] = "0 during prep/train/recover; counted during formal --evaluate"
    base["input_sha256_snapshot"] = hashes
    base["input_count"] = len(hashes)
    dump_json(PROTO_OUT / "protocol_proposed.json", base)
    log(f"protocol_proposed.json written (inputs={len(hashes)})")
    return 0


def static_check() -> int:
    PROTO_OUT.mkdir(parents=True, exist_ok=True)
    import ast
    out = {"in_memory_compile": {}, "import_check": {}, "forbidden_inheritance": {},
           "reference_producer_before_consumer": {}}
    files = [Path(__file__).resolve(), ROOT / "src/trunk/run_baseline.py",
             ROOT / "src/trunk/train_smoke.py", ROOT / "src/trunk/acbm_interface.py",
             ROOT / "src/trunk/descriptors.py", ROOT / "src/trunk/adapters.py",
             ROOT / "src/preflight_standardization.py"]
    for p in files:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
        out["in_memory_compile"][p.name] = "OK"
    import importlib
    for mod in ["trunk.train_smoke", "trunk.acbm_interface", "trunk.adapters", "trunk.descriptors"]:
        importlib.import_module(mod)
        out["import_check"][mod] = "OK"
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    uses_ts_OUT = uses_ts_SEED = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(n.name for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ts":
            if node.attr == "OUT":
                uses_ts_OUT = True
            if node.attr == "SEED":
                uses_ts_SEED = True
    out["forbidden_inheritance"] = {
        "no_full_sequence_import": not any("full_sequence" in m for m in imported),
        "no_ts_OUT": not uses_ts_OUT, "no_ts_SEED": not uses_ts_SEED,
        "imported_modules": sorted(imported)}
    # B2: reference producer (savez_compressed reference_outputs) precedes recover subprocess
    out["reference_producer_before_consumer"] = {
        "producer_present": "reference_outputs.npz" in src,
        "producer_before_recover": src.index("reference_outputs.npz") < src.index("--recover"),
    }
    require(out["reference_producer_before_consumer"]["producer_before_recover"],
            "reference_producer_before_recover")
    dump_json(PROTO_OUT / "static_check.json", out)
    log("static check PASS")
    return 0


def _write_state_pkg(tmp, task_idx, mutate=None):
    """Write a minimal state package for schema negative tests."""
    centroids = {c: np.ones(JOINT_DIM, dtype=np.float32) for c in range(2 * task_idx)}
    counts = {c: 10 for c in range(2 * task_idx)}
    task_of = {c: (c // 2) + 1 for c in range(2 * task_idx)}
    if mutate:
        mutate(centroids, counts, task_of)
    sp = {"seen_classes": np.arange(2 * task_idx, dtype=np.int64),
          "class_counts": np.array([counts[c] for c in range(2 * task_idx)], dtype=np.int64),
          "task_ids": np.array([task_of[c] for c in range(2 * task_idx)], dtype=np.int64)}
    for c in range(2 * task_idx):
        sp[f"centroid_{c}"] = centroids[c]
    np.savez_compressed(tmp / f"state_package_task{task_idx}.npz", **sp)
    meta = {"dataset": "uci_har", "seed": 11, "task": task_idx,
            "seen_classes": list(range(2 * task_idx)),
            "class_real_train_counts": {str(c): counts[c] for c in range(2 * task_idx)},
            "class_task_mapping": {str(c): task_of[c] for c in range(2 * task_idx)},
            "centroid_dtype": "float32", "package_file_sha256": sha(tmp / f"state_package_task{task_idx}.npz"),
            "centroid_value_sha256": centroid_value_hash(centroids, list(range(2 * task_idx))),
            "frozen_model": {"file": str(tmp / "task1_encoder_best.pt"), "sha256": "x"},
            "current_head": {"file": str(tmp / f"task{task_idx}_head.pt"), "sha256": "x"},
            "input_stats_sha256": "x", "descriptor_stats_sha256": "x", "config_version": "x"}
    # write tiny placeholder files for references
    (tmp / "task1_encoder_best.pt").write_bytes(b"x")
    (tmp / f"task{task_idx}_head.pt").write_bytes(b"x")
    meta["frozen_model"]["sha256"] = sha(tmp / "task1_encoder_best.pt")
    meta["current_head"]["sha256"] = sha(tmp / f"task{task_idx}_head.pt")
    meta["input_stats_sha256"] = sha(ad.FIX_DIR / "preprocess_statistics.json")
    meta["descriptor_stats_sha256"] = sha(V2_DIR / "interface_check_uci_har.json")
    dump_json(tmp / f"state_package_task{task_idx}.json", meta)
    return tmp


def zero_data_regression() -> int:
    PROTO_OUT.mkdir(parents=True, exist_ok=True)
    import tempfile
    out = {}
    # 1) Adam param_groups roundtrip
    torch.manual_seed(0)
    lin = nn.Linear(3, 2)
    opt = torch.optim.Adam(lin.parameters(), lr=0.01)
    x = torch.randn(6, 3); y = torch.randint(0, 2, (6,))
    for _ in range(3):
        opt.zero_grad(); F.cross_entropy(lin(x), y).backward(); opt.step()
    snap = copy.deepcopy(opt.state_dict())
    lin2 = nn.Linear(3, 2); opt2 = torch.optim.Adam(lin2.parameters(), lr=0.01)
    opt2.load_state_dict(snap)
    s2 = opt2.state_dict()
    ok = all(all(a[k] == b[k] for k in a) for a, b in zip(snap["param_groups"], s2["param_groups"]))
    for pid in snap["state"]:
        for k in ("step", "exp_avg", "exp_avg_sq"):
            a, b = s2["state"][pid][k], snap["state"][pid][k]
            ok = ok and (torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b)
    out["adam_param_groups_roundtrip"] = bool(ok)
    require(ok, "adam_param_groups_roundtrip")

    # 2) task layout
    out["task_layout"] = {d: task_list(d) for d in DATASETS}

    # 3) metric formula examples
    num_tasks = 3
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    lpt = {t: labels for t in range(1, num_tasks + 1)}
    p2 = {t: np.array(labels) for t in range(1, num_tasks + 1)}
    R = task_accuracy_matrix(p2, lpt, num_tasks)
    for t in range(1, num_tasks + 1):
        for j in range(1, t + 1):
            require(abs(R[t, j] - 1.0) < 1e-9, f"R_{t}_{j}_1")
    p3 = {t: np.array(labels) for t in range(1, num_tasks + 1)}
    p3[3] = np.array([2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5])
    R3 = task_accuracy_matrix(p3, lpt, num_tasks)
    require(abs(forgetting_from_R(R3, num_tasks)[3] - 1.0) < 1e-9, "forgetting_1")
    # negative forgetting example: task1 improves later (R[2,1] > R[1,1])
    pneg = {t: np.array(labels) for t in range(1, num_tasks + 1)}
    pneg[1] = np.array([0, 2, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])  # task1 imperfect at t1
    Rneg = task_accuracy_matrix(pneg, lpt, num_tasks)
    require(forgetting_from_R(Rneg, num_tasks)[2] < 0, "forgetting_negative_preserved",
            str(forgetting_from_R(Rneg, num_tasks)[2]))
    out["metric_examples"] = {"perfect_R": R.tolist(), "forgetting_1": 1.0,
                              "negative_forgetting": forgetting_from_R(Rneg, num_tasks)[2]}

    # 4) state schema negatives (B5) — each injected independently
    neg = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # normal case
        validate_state_package(_write_state_pkg(tmp, 1), "uci_har", 11, 1)
        neg["normal_accepts"] = True
        # wrong dataset
        _write_state_pkg(tmp, 1)
        require(_expect_reject(lambda: validate_state_package(tmp, "wisdm", 11, 1)), "neg_wrong_dataset")
        # wrong seed
        require(_expect_reject(lambda: validate_state_package(tmp, "uci_har", 22, 1)), "neg_wrong_seed")
        # wrong task
        require(_expect_reject(lambda: validate_state_package(tmp, "uci_har", 11, 2)), "neg_wrong_task")
        # negative count
        _write_state_pkg(tmp, 1, mutate=lambda c, cn, to: cn.__setitem__(0, -1))
        require(_expect_reject(lambda: validate_state_package(tmp, "uci_har", 11, 1)), "neg_negative_count")
        # zero count
        _write_state_pkg(tmp, 1, mutate=lambda c, cn, to: cn.__setitem__(0, 0))
        require(_expect_reject(lambda: validate_state_package(tmp, "uci_har", 11, 1)), "neg_zero_count")
        # wrong task mapping
        _write_state_pkg(tmp, 1, mutate=lambda c, cn, to: to.__setitem__(0, 99))
        require(_expect_reject(lambda: validate_state_package(tmp, "uci_har", 11, 1)), "neg_wrong_mapping")
        # tampered centroid (write meta with wrong centroid_value_sha256)
        pkg = _write_state_pkg(tmp, 1)
        m = json.loads((pkg / "state_package_task1.json").read_text(encoding="utf-8"))
        m["centroid_value_sha256"] = "0" * 64
        dump_json(pkg / "state_package_task1.json", m)
        require(_expect_reject(lambda: validate_state_package(tmp, "uci_har", 11, 1)), "neg_tampered_centroid")
        # tampered reference (frozen model hash)
        pkg = _write_state_pkg(tmp, 1)
        m = json.loads((pkg / "state_package_task1.json").read_text(encoding="utf-8"))
        m["frozen_model"]["sha256"] = "0" * 64
        dump_json(pkg / "state_package_task1.json", m)
        require(_expect_reject(lambda: validate_state_package(tmp, "uci_har", 11, 1)), "neg_tampered_reference")
        # missing package
        pkg = _write_state_pkg(tmp, 1)
        (pkg / "state_package_task1.npz").unlink()
        require(_expect_reject(lambda: validate_state_package(tmp, "uci_har", 11, 1)), "neg_missing_package")
        neg["all_negative_cases_rejected"] = True
    out["state_schema_negatives"] = neg
    dump_json(PROTO_OUT / "zero_data_regression.json", out)
    log("zero-data regression PASS")
    return 0


def _expect_reject(fn) -> bool:
    try:
        fn()
        return False
    except (Halt, FileNotFoundError, KeyError, AssertionError):
        return True


# ---------------------------------------------------------------- lock / main
def _require(cond, label):
    if not cond:
        raise Halt(f"contract: {label}")


def launcher_commands() -> list[list[str]]:
    cmds = [["--lock-protocol"]]
    for d in DATASETS:
        for s in SEEDS:
            cmds.append(["--run", d, str(s)])
    cmds.append(["--seal-training"])
    for d in DATASETS:
        for s in SEEDS:
            cmds.append(["--evaluate", d, str(s)])
    cmds.append(["--aggregate"])
    return cmds


def _write_state_pkg_general(tmp, dataset, seed, task_idx):
    seen = list(range(2 * task_idx))
    centroids = {c: np.ones(JOINT_DIM, dtype=np.float32) for c in seen}
    counts = {c: 10 for c in seen}
    task_of = {c: (c // 2) + 1 for c in seen}
    sp = {"seen_classes": np.arange(2 * task_idx, dtype=np.int64),
          "class_counts": np.array([counts[c] for c in seen], dtype=np.int64),
          "task_ids": np.array([task_of[c] for c in seen], dtype=np.int64)}
    for c in seen:
        sp[f"centroid_{c}"] = centroids[c]
    np.savez_compressed(tmp / f"state_package_task{task_idx}.npz", **sp)
    if not (tmp / "task1_encoder_best.pt").exists():
        (tmp / "task1_encoder_best.pt").write_bytes(b"enc")
    (tmp / f"task{task_idx}_head.pt").write_bytes(f"head{task_idx}".encode())
    meta = {"dataset": dataset, "seed": seed, "task": task_idx, "seen_classes": seen,
            "class_real_train_counts": {str(c): counts[c] for c in seen},
            "class_task_mapping": {str(c): task_of[c] for c in seen},
            "centroid_dtype": "float32",
            "package_file_sha256": sha(tmp / f"state_package_task{task_idx}.npz"),
            "centroid_value_sha256": centroid_value_hash(centroids, seen),
            "frozen_model": {"file": str(tmp / "task1_encoder_best.pt"),
                             "sha256": sha(tmp / "task1_encoder_best.pt")},
            "current_head": {"file": str(tmp / f"task{task_idx}_head.pt"),
                             "sha256": sha(tmp / f"task{task_idx}_head.pt")},
            "input_stats_sha256": sha(ad.FIX_DIR / "preprocess_statistics.json"),
            "descriptor_stats_sha256": sha(V2_DIR / f"interface_check_{dataset}.json"),
            "config_version": "x"}
    dump_json(tmp / f"state_package_task{task_idx}.json", meta)
    return meta


def _build_group_fixture(tmp, dataset, seed):
    ddir = tmp / dataset / f"seed{seed}"
    ddir.mkdir(parents=True)
    n = DATASET_TASK_COUNTS[dataset]
    oh = {"state_package": {}, "heads": {}}
    for t in range(1, n + 1):
        _write_state_pkg_general(ddir, dataset, seed, t)
        (ddir / f"task{t}").mkdir(exist_ok=True)
        dump_json(ddir / f"task{t}" / "recovery_comparison.json", {"all_bitwise_equal": True})
        oh["state_package"][f"task{t}"] = {"npz": sha(ddir / f"state_package_task{t}.npz"),
                                           "json": sha(ddir / f"state_package_task{t}.json")}
        oh["heads"][f"task{t}"] = sha(ddir / f"task{t}_head.pt")
    oh["task1_encoder_best.pt"] = sha(ddir / "task1_encoder_best.pt")
    if n >= 2:
        dump_json(ddir / "handoff_ledger.json", {"per_boundary": []})
    dump_json(ddir / "output_hashes.json", oh)
    dump_json(ddir / "gate_decision.json", {"dataset": dataset, "seed": seed,
                                            "verdict": "TRAINING_PHASE_PASS",
                                            "tasks_completed": n})
    return ddir


def _build_seal_fixture(tmp):
    """Build 32 valid groups + a locked protocol, then seal."""
    # protocol + before
    now = {k: sha(v) for k, v in input_files().items()}
    body = {"status": "LOCKED", "input_sha256_before": now}
    dump_json(tmp / "protocol_locked.json", body)
    dump_json(tmp / "input_sha256_before.json", now)
    (tmp / "protocol_locked.json.sha256").write_text(
        f"{sha(tmp / 'protocol_locked.json')} *protocol_locked.json\n", encoding="utf-8")
    for d in DATASETS:
        for s in SEEDS:
            _build_group_fixture(tmp, d, s)


def contract_regression() -> int:
    global OUT, PROTO_OUT
    PROTO_OUT.mkdir(parents=True, exist_ok=True)
    import tempfile
    results = []

    def record(name, expect, fn):
        try:
            fn()
            results.append({"name": name, "expect": expect, "actual": "NO_EXCEPTION",
                            "exit": 0, "outcome": "PASS" if expect == "accept" else "FAIL"})
        except (Halt, FileNotFoundError, KeyError, AssertionError, SystemExit) as e:
            results.append({"name": name, "expect": expect, "actual": type(e).__name__,
                            "reason": str(e)[:160], "exit": 1,
                            "outcome": "PASS" if expect == "reject" else "FAIL"})

    # ---- R1 launcher argv ----
    cmds = launcher_commands()
    record("launcher_67_commands", "accept", lambda: _require(len(cmds) == 67, f"len={len(cmds)}"))
    record("launcher_order", "accept", lambda: _require(
        cmds[0] == ["--lock-protocol"] and cmds[1] == ["--run", "uci_har", "11"]
        and cmds[32] == ["--run", "cwru", "88"] and cmds[33] == ["--seal-training"]
        and cmds[34] == ["--evaluate", "uci_har", "11"] and cmds[66] == ["--aggregate"], "order"))

    saved_out, saved_proto = OUT, PROTO_OUT
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # ---- R3 lock/preflight ----
            OUT = tmp
            PROTO_OUT = tmp / "proto"
            PROTO_OUT.mkdir()
            now = {k: sha(v) for k, v in input_files().items()}
            k0 = next(iter(now))
            dump_json(PROTO_OUT / "protocol_proposed.json",
                      {"status": "PROPOSED", "input_sha256_snapshot": {**now, k0: "0" * 64},
                       "seeds": SEEDS, "datasets": {d: DATASET_TASK_COUNTS[d] for d in DATASETS},
                       "budget": {"encoder": {"lr": ENCODER_LR}, "sparse_head": {"epochs": SPARSE_EPOCHS}}})
            record("lock_wrong_proposed_hash", "reject", lambda: lock_protocol())
            dump_json(PROTO_OUT / "protocol_proposed.json",
                      {"status": "PROPOSED", "input_sha256_snapshot": now, "seeds": SEEDS,
                       "datasets": {d: DATASET_TASK_COUNTS[d] for d in DATASETS},
                       "budget": {"encoder": {"lr": ENCODER_LR}, "sparse_head": {"epochs": SPARSE_EPOCHS}}})
            record("lock_correct_proposed", "accept", lambda: lock_protocol())
            b0 = (OUT / "protocol_locked.json").read_bytes()
            record("lock_double_reject", "reject", lambda: lock_protocol())
            record("lock_double_bytes_unchanged", "accept",
                   lambda: _require((OUT / "protocol_locked.json").read_bytes() == b0, "bytes"))
            dump_json(OUT / "input_sha256_before.json", {})
            record("preflight_empty_before", "reject", lambda: preflight("t", None, None, None))
            dump_json(OUT / "input_sha256_before.json", {**now, "extra": "x"})
            record("preflight_extra_key", "reject", lambda: preflight("t", None, None, None))
            dump_json(OUT / "input_sha256_before.json", {k: v for k, v in now.items() if k != k0})
            record("preflight_missing_key", "reject", lambda: preflight("t", None, None, None))
            dump_json(OUT / "input_sha256_before.json", {**now, k0: "0" * 64})
            record("preflight_wrong_hash", "reject", lambda: preflight("t", None, None, None))
            dump_json(OUT / "input_sha256_before.json", now)
            (OUT / "protocol_locked.json").unlink()
            record("preflight_missing_lock", "reject", lambda: preflight("t", None, None, None))

            # ---- R2 seal/check_seal/aggregate ----
            OUT = tmp / "seal"
            OUT.mkdir()
            _build_seal_fixture(OUT)
            seal_base = OUT
            record("seal_normal_32", "accept", lambda: seal_training())
            record("check_seal_normal", "accept", lambda: check_seal())
            # negatives (each changes ONE condition, on a fresh copy of the clean base)
            import shutil
            def fresh():
                nxt = tmp / f"case{len(results)}"
                shutil.copytree(seal_base, nxt)
                return nxt

            def fresh_noseal():
                nxt = fresh()
                for sf in ("training_seal.json", "training_seal.json.sha256"):
                    p = nxt / sf
                    if p.exists():
                        p.unlink()
                return nxt
            # 31 groups: remove one seed dir, re-seal -> reject
            c = fresh_noseal(); shutil.rmtree(c / "cwru" / "seed88")
            def run_seal_31():
                global OUT
                OUT = c; seal_training()
            record("seal_31_groups", "reject", run_seal_31)
            # existing failure
            c = fresh_noseal(); dump_json(c / "uci_har" / "seed11" / "failure_record_1.json", {"k": "fail"})
            def run_seal_fail():
                global OUT
                OUT = c; seal_training()
            record("seal_existing_failure", "reject", run_seal_fail)
            # recovery not passed
            c = fresh_noseal(); dump_json(c / "uci_har" / "seed11" / "task1" / "recovery_comparison.json",
                                          {"all_bitwise_equal": False})
            def run_seal_rec():
                global OUT
                OUT = c; seal_training()
            record("seal_recovery_not_passed", "reject", run_seal_rec)
            # double seal (on the already-sealed base)
            OUT = seal_base
            record("seal_double_reject", "reject", lambda: seal_training())
            # check_seal: 32 duplicate groups in manifest
            c = fresh()
            se = json.loads((c / "training_seal.json").read_text(encoding="utf-8"))
            se["groups"] = [dict(se["groups"][0]) for _ in range(32)]
            dump_json(c / "training_seal.json", se)
            dump_json(c / "training_seal.json.sha256", {"sha256": sha(c / "training_seal.json")})
            def run_check_dup():
                global OUT
                OUT = c; check_seal()
            record("check_seal_32_dup_groups", "reject", run_check_dup)
            # check_seal: changed model file
            c = fresh(); (c / "uci_har" / "seed11" / "task1_encoder_best.pt").write_bytes(b"TAMPERED")
            def run_check_changed():
                global OUT
                OUT = c; check_seal()
            record("check_seal_changed_encoder", "reject", run_check_changed)
            # check_seal: non-SEALED
            c = fresh()
            se = json.loads((c / "training_seal.json").read_text(encoding="utf-8"))
            se["status"] = "NOT_SEALED"
            dump_json(c / "training_seal.json", se)
            dump_json(c / "training_seal.json.sha256", {"sha256": sha(c / "training_seal.json")})
            def run_check_notsealed():
                global OUT
                OUT = c; check_seal()
            record("check_seal_non_sealed", "reject", run_check_notsealed)
            # check_seal: wrong protocol hash
            c = fresh()
            se = json.loads((c / "training_seal.json").read_text(encoding="utf-8"))
            se["protocol_sha256"] = "0" * 64
            dump_json(c / "training_seal.json", se)
            dump_json(c / "training_seal.json.sha256", {"sha256": sha(c / "training_seal.json")})
            def run_check_proto():
                global OUT
                OUT = c; check_seal()
            record("check_seal_wrong_protocol", "reject", run_check_proto)
            # train after seal
            OUT = seal_base
            record("train_after_seal", "reject", lambda: train_group("uci_har", 11))
            # eval unsealed (no seal file)
            c = fresh_noseal()
            def run_eval_unsealed():
                global OUT
                OUT = c; evaluate_group("uci_har", 11)
            record("evaluate_unsealed", "reject", run_eval_unsealed)

    finally:
        OUT, PROTO_OUT = saved_out, saved_proto

    # metric negatives (pure functions, no model)
    num_tasks = 3
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    lpt = {t: labels for t in range(1, num_tasks + 1)}
    # macro vs micro differ
    pmac = {t: np.array(labels) for t in range(1, num_tasks + 1)}
    pmac[3] = np.array([0, 2, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])
    Rm = task_accuracy_matrix(pmac, lpt, num_tasks)
    record("metric_macro_micro_differ", "accept", lambda: _require(
        abs(Rm[3, 1] - 0.5) < 1e-9, f"R31={Rm[3,1]}"))
    # negative forgetting preserved
    pn = {t: np.array(labels) for t in range(1, num_tasks + 1)}
    pn[1] = np.array([0, 2, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])
    Rn = task_accuracy_matrix(pn, lpt, num_tasks)
    record("metric_negative_forgetting", "accept", lambda: _require(
        forgetting_from_R(Rn, num_tasks)[2] < 0, "neg F"))
    # missing class -> metric RAISES (not masked by nanmean)
    pmiss = {t: np.array(labels) for t in range(1, num_tasks + 1)}
    lbl_miss = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4])  # class 5 missing at t3
    lpt_miss = {t: lbl_miss for t in range(1, num_tasks + 1)}
    record("metric_missing_class_fails", "reject",
           lambda: task_accuracy_matrix(pmiss, lpt_miss, num_tasks))

    dump_json(PROTO_OUT / "contract_regression.json",
              {"cases": results, "note": "direct-call formal functions on temp fixtures; "
                                          "zero research model forward"})
    fails = [r for r in results if r["outcome"] == "FAIL"]
    log(f"contract regression: {len(results)} cases, {len(fails)} failures")
    require(not fails, "contract_regression_all_pass", str(fails[:3]))
    return 0


def lock_protocol() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in [OUT / "protocol_locked.json", OUT / "protocol_locked.json.sha256",
              OUT / "input_sha256_before.json"]:
        require(not p.exists(), f"lock_no_existing_{p.name}", str(p))
    proposed = json.loads((PROTO_OUT / "protocol_proposed.json").read_text(encoding="utf-8"))
    # R3: current dependencies must match the proposed snapshot on key set AND hash
    proposed_snap = proposed.get("input_sha256_snapshot") or {}
    now = {k: sha(v) for k, v in input_files().items()}
    require(set(proposed_snap.keys()) == set(now.keys()), "lock_proposed_key_set_matches",
            f"proposed={len(proposed_snap)} now={len(now)}")
    require(bool(now), "lock_inventory_nonempty")
    mismatch = [k for k in now if proposed_snap.get(k) != now[k]]
    require(not mismatch, "lock_proposed_hash_matches", str(mismatch[:5]))
    # R3: running config constants must equal the proposed protocol
    require(proposed.get("seeds") == SEEDS, "lock_config_seeds")
    require(proposed.get("datasets") == {d: DATASET_TASK_COUNTS[d] for d in DATASETS},
            "lock_config_datasets")
    require(proposed.get("budget", {}).get("encoder", {}).get("lr") == ENCODER_LR, "lock_config_enc_lr")
    require(proposed.get("budget", {}).get("sparse_head", {}).get("epochs") == SPARSE_EPOCHS,
            "lock_config_head_epochs")
    proposed["status"] = "LOCKED"
    proposed["locked_before_any_forward"] = True
    proposed["input_sha256_before"] = now
    dump_json(OUT / "protocol_locked.json", proposed)
    dump_json(OUT / "input_sha256_before.json", now)
    h = sha(OUT / "protocol_locked.json")
    (OUT / "protocol_locked.json.sha256").write_text(f"{h} *protocol_locked.json\n", encoding="utf-8")
    log(f"protocol locked: {h}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose-protocol", action="store_true")
    ap.add_argument("--static-check", action="store_true")
    ap.add_argument("--zero-data-regression", action="store_true")
    ap.add_argument("--contract-regression", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lock-protocol", action="store_true")
    ap.add_argument("--seal-training", action="store_true")
    ap.add_argument("--run", nargs=2, metavar=("DATASET", "SEED"))
    ap.add_argument("--recover", nargs=3, metavar=("DATASET", "SEED", "TASK"))
    ap.add_argument("--evaluate", nargs=2, metavar=("DATASET", "SEED"))
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.propose_protocol:
        return propose_protocol()
    if a.static_check:
        return static_check()
    if a.zero_data_regression:
        return zero_data_regression()
    if a.contract_regression:
        return contract_regression()
    if a.dry_run:
        dump_json(PROTO_OUT / "dry_run_commands.json",
                  {"n_commands": len(launcher_commands()),
                   "commands": [" ".join(c) for c in launcher_commands()]})
        log(f"dry-run: {len(launcher_commands())} commands emitted")
        return 0
    if a.lock_protocol:
        return lock_protocol()
    if a.seal_training:
        return seal_training()
    if a.aggregate:
        return aggregate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise Halt("CUDA unavailable; blocking, not auto-switching to CPU")
    if a.run:
        name, seed = a.run[0], int(a.run[1])
        require(name in DATASETS, "run_dataset_registered")
        require(seed in SEEDS, "run_seed_registered")
        ddir = OUT / name / f"seed{seed}"
        try:
            return train_group(name, seed)
        except Halt as e:
            write_failure("HALTED", {"reason": str(e), "log_tail": LOG[-40:]}, ddir)
            raise SystemExit(4)
        except Exception:
            write_failure("UNEXPECTED", {"traceback": traceback.format_exc(), "log_tail": LOG[-40:]}, ddir)
            raise SystemExit(6)
        finally:
            _persist_log(ddir, "run.log")
    if a.recover:
        name, seed, task = a.recover[0], int(a.recover[1]), int(a.recover[2])
        ddir = OUT / name / f"seed{seed}"
        try:
            return recover_group(name, seed, task)
        except Halt as e:
            write_failure("HALTED_RECOVER", {"reason": str(e), "log_tail": LOG[-40:]}, ddir)
            raise SystemExit(4)
        except Exception:
            write_failure("UNEXPECTED_RECOVER", {"traceback": traceback.format_exc(), "log_tail": LOG[-40:]}, ddir)
            raise SystemExit(6)
        finally:
            _persist_log(ddir, f"recover_task{task}.log")
    if a.evaluate:
        name, seed = a.evaluate[0], int(a.evaluate[1])
        ddir = OUT / name / f"seed{seed}"
        try:
            return evaluate_group(name, seed)
        except Halt as e:
            write_failure("HALTED_EVAL", {"reason": str(e), "log_tail": LOG[-40:]}, ddir)
            raise SystemExit(4)
        except Exception:
            write_failure("UNEXPECTED_EVAL", {"traceback": traceback.format_exc(), "log_tail": LOG[-40:]}, ddir)
            raise SystemExit(6)
        finally:
            _persist_log(ddir, "evaluate.log")
    raise SystemExit("see --help")


def _persist_log(ddir: Path, name: str) -> None:
    try:
        (ddir / "logs").mkdir(parents=True, exist_ok=True)
        with (ddir / "logs" / name).open("a", encoding="utf-8") as f:
            f.write("\n".join(LOG) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise SystemExit(6)
