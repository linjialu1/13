"""Formal S1 baseline runner: 4 datasets x 8 paired seeds (32 groups).

This is the production runner for the full CIL baseline. It differs from the seed11
smoke in the following ways (see admin/REVIEW_full_sequence_smoke_seed11_fix1.md):

  * t>=2 ACTUALLY loads the previous task's on-disk state package (centroids, counts,
    seen/task mapping) and uses those loaded values for pseudo-feature generation --
    the saved state is never a bypass.  A per-boundary handoff ledger records the
    loaded package SHA256 and per-centroid value hashes.
  * per-boundary persistent-budget / GPU-peak / process-RSS are measured and reported;
    main process and recovery sub-process are measured separately.
  * two-phase protocol: training (32 groups, state checks, disk recovery) then a
    SEPARATE, frozen evaluation entry. Test values never feed training/selection.
  * full metric definitions (R matrix, forgetting F, BWT, average incremental accuracy,
    per-boundary macro/micro, per-class, per-old-task) are fixed before any test forward.

Zero research-model training / forward / test evaluation is performed in the
PREPARATION round; the modes used now are --propose-protocol, --static-check and
--zero-data-regression.  All seed / output / budget values are explicit arguments or
module constants of THIS file -- nothing is silently inherited from earlier scripts.

Usage (formal run, after protocol acceptance):
    python src/trunk/run_baseline.py --lock-protocol
    python src/trunk/run_baseline.py --run uci_har 11      # one (dataset,seed) group
    python src/trunk/run_baseline.py --recover uci_har 11 3
    python src/trunk/run_baseline.py --evaluate uci_har 11 # separate frozen phase
    python src/trunk/run_baseline.py --aggregate
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
# planned formal result directory (used only AFTER acceptance)
OUT = ROOT / "experiments/baselines/trunk_baseline_4datasets"
# preparation-round products (this round)
PROTO_OUT = ROOT / "experiments/baselines/trunk_baseline_4datasets_protocol_final"
V2_DIR = ROOT / "experiments/baselines/trunk_model_interface_check_v2"
for p in (str(ROOT / "src"), str(ROOT), str(ROOT / ".local_packages")):
    if p not in sys.path:
        sys.path.insert(0, p)

from trunk import train_smoke as ts   # noqa: E402  (proven numeric path, no OUT/SEED use)
from trunk import acbm_interface as ai  # noqa: E402
from trunk import adapters as ad        # noqa: E402
from trunk import descriptors as ds     # noqa: E402

F = torch.nn.functional

# ---------------------------------------------------------------- explicit constants
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DATASETS = ["uci_har", "wisdm", "uwave", "cwru"]
ALL_TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
DATASET_TASK_COUNTS = {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}

HIDDEN = 48
DESCRIPTOR_DIM = 8
JOINT_DIM = 56

# formal encoder budget (pre-fixed)
ENCODER_LR = 1e-3
BATCH_SIZE = 128
GRAD_CLIP = 5.0
MIN_EPOCHS = 3
MAX_EPOCHS = 12
PATIENCE = 2
VAL_DELTA_ACC = 0.005
VAL_DELTA_LOSS = 0.001
EVAL_BATCH = 256

# formal sparse-head budget (pre-fixed)
SPARSE_EPOCHS = 300
SPARSE_LR = 0.05
SPARSITY_LAMBDA = 0.0007

# seed derivation (explicit, parameterised by seed s)
#   model init          : torch.manual_seed(s)
#   Task1 DataLoader    : generator seed = s + 100000 + epoch
#   per-task head init  : torch.manual_seed(s + 10000 * task_idx)

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


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
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
    for p in [Path(__file__).resolve(), ROOT / "src/trunk/train_smoke.py",
              ROOT / "src/trunk/acbm_interface.py", ROOT / "src/trunk/descriptors.py",
              ROOT / "src/trunk/adapters.py", ROOT / "src/preflight_standardization.py"]:
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
    return f


# ---------------------------------------------------------------- data context
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
    else:
        sp = ad.accepted_splits()[name]
        train_all = np.asarray(sp["train"])
        val_all = np.asarray(sp["validation"])
        test_all = np.asarray(sp["test"])

    def task_rows(classes, which="train"):
        base = train_all if which == "train" else (val_all if which == "val" else test_all)
        return np.asarray(sorted(int(i) for i in base if int(y[i]) in classes), dtype=np.int64)

    in_stats = ad.accepted_input_stats()[name]
    in_mean = np.asarray(in_stats["mean"], dtype=np.float64)
    in_denom = np.asarray(in_stats["denom"], dtype=np.float64)
    x_std = ai.standardize_input(x_raw, in_mean, in_denom)
    v2 = json.loads((V2_DIR / f"interface_check_{name}.json").read_text(encoding="utf-8"))
    d_mean = np.asarray(v2["descriptor_stats_mean"], dtype=np.float64)
    d_std = np.asarray(v2["descriptor_stats_std"], dtype=np.float64)
    return dict(name=name, x_raw=x_raw, y=y, x_std=x_std, d_mean=d_mean, d_std=d_std,
                n_classes=n_classes, input_channels=int(x_raw.shape[2]),
                in_mean=in_mean, in_denom=in_denom, task_rows=task_rows,
                test_row_space=("joint array (0..3199)" if name == "cwru"
                                else "separate official test array"))


# ---------------------------------------------------------------- seeds / model
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


# ---------------------------------------------------------------- process RSS / GPU
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


# ---------------------------------------------------------------- sparse head
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


# ---------------------------------------------------------------- train task1 encoder
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
                epoch_rows=epoch_rows,
                budget_truncated=(trained == MAX_EPOCHS))


# ---------------------------------------------------------------- state package (disk handoff)
def centroid_value_hash(centroids, seen):
    h = hashlib.sha256()
    for c in sorted(seen):
        h.update(np.ascontiguousarray(centroids[c].astype(np.float32)).tobytes())
    return h.hexdigest()


def save_state_package(ddir, task_idx, centroids, counts, task_of, name, device, seed, head):
    seen = list(range(2 * task_idx))
    sp_keys = {}
    for c in seen:
        sp_keys[f"centroid_{c}"] = centroids[c].astype(np.float32)
    sp_keys["seen_classes"] = np.asarray(seen, dtype=np.int64)
    sp_keys["class_counts"] = np.asarray([counts[c] for c in seen], dtype=np.int64)
    sp_keys["task_ids"] = np.asarray([task_of[c] for c in seen], dtype=np.int64)
    np.savez_compressed(ddir / f"state_package_task{task_idx}.npz", **sp_keys)
    dump_json(ddir / f"state_package_task{task_idx}.json", {
        "dataset": name, "task": task_idx, "seed": seed, "seen_classes": seen,
        "class_real_train_counts": {str(c): counts[c] for c in seen},
        "class_task_mapping": {str(c): task_of[c] for c in seen},
        "centroid_dtype": "float32 (uniform)",
        "centroid_value_sha256": centroid_value_hash(centroids, seen),
        "frozen_model": {"file": str(ddir / "task1_encoder_best.pt"),
                         "sha256": sha(ddir / "task1_encoder_best.pt")},
        "current_head": {"file": str(ddir / f"task{task_idx}_head.pt"),
                         "sha256": sha(ddir / f"task{task_idx}_head.pt")},
        "input_stats_sha256": sha(ad.FIX_DIR / "preprocess_statistics.json"),
        "descriptor_stats_sha256": sha(V2_DIR / f"interface_check_{name}.json"),
        "config_version": "s1_baseline_v1",
        "pseudo_numeric_rule": "centroid float32; shift float64; pseudo float32",
    })


def load_state_package(ddir, prev_task_idx, name, device):
    """t>=2: actually load previous task's on-disk state package and rebuild old-class state."""
    pkg = ddir / f"state_package_task{prev_task_idx}.npz"
    meta = json.loads((ddir / f"state_package_task{prev_task_idx}.json").read_text(encoding="utf-8"))
    require(pkg.exists(), f"{name}_prev_state_package_exists")
    sp = np.load(pkg)
    seen = [int(c) for c in sp["seen_classes"]]
    require(seen == list(range(2 * prev_task_idx)), f"{name}_prev_seen_classes", str(seen))
    centroids, counts, task_of = {}, {}, {}
    for c in seen:
        v = sp[f"centroid_{c}"]
        require(v.dtype == np.float32, f"{name}_prev_centroid_{c}_float32")
        require(v.shape == (JOINT_DIM,), f"{name}_prev_centroid_{c}_shape_56")
        require(bool(np.isfinite(v).all()), f"{name}_prev_centroid_{c}_finite")
        centroids[c] = v.copy()
        counts[c] = int(sp["class_counts"][seen.index(c)])
        task_of[c] = int(sp["task_ids"][seen.index(c)])
    # verify references (model/head/stats hashes)
    require(meta["frozen_model"]["sha256"] == sha(Path(meta["frozen_model"]["file"])),
            f"{name}_prev_frozen_model_hash")
    require(meta["current_head"]["sha256"] == sha(Path(meta["current_head"]["file"])),
            f"{name}_prev_current_head_hash")
    require(meta["input_stats_sha256"] == sha(ad.FIX_DIR / "preprocess_statistics.json"),
            f"{name}_prev_input_stats_hash")
    require(meta["descriptor_stats_sha256"] == sha(V2_DIR / f"interface_check_{name}.json"),
            f"{name}_prev_descriptor_stats_hash")
    return {"centroids": centroids, "counts": counts, "task_of": task_of,
            "seen": seen, "package_sha256": sha(pkg),
            "centroid_value_sha256": meta["centroid_value_sha256"]}


# ---------------------------------------------------------------- metrics (fixed before test)
def task_accuracy_matrix(preds_per_task, labels_per_task, num_tasks):
    """R[t][j] (1-indexed) = mean class accuracy over task j's 2 classes, using task t's head.

    preds_per_task[t] and labels_per_task[t] are ALIGNED arrays over the test members of
    all seen classes at task t (so labels_per_task[t] grows with t)."""
    R = np.full((num_tasks + 1, num_tasks + 1), np.nan)
    for t in range(1, num_tasks + 1):
        pred = np.asarray(preds_per_task[t])
        lab = np.asarray(labels_per_task[t])
        for j in range(1, t + 1):
            accs = []
            for c in ALL_TASKS[j - 1]:
                m = lab == c
                accs.append(float((pred[m] == c).mean()) if m.sum() else np.nan)
            R[t, j] = float(np.nanmean(accs))
    return R


def forgetting_from_R(R, num_tasks):
    F = [np.nan] * (num_tasks + 1)
    for t in range(2, num_tasks + 1):
        vals = []
        for j in range(1, t):
            hist = [R[k, j] for k in range(j, t)]
            if any(np.isnan(v) for v in hist) or np.isnan(R[t, j]):
                vals.append(np.nan)
            else:
                vals.append(max(hist) - R[t, j])
        F[t] = float(np.nanmean(vals)) if vals else np.nan
    return F


def bwt_from_R(R, num_tasks):
    B = [np.nan] * (num_tasks + 1)
    for t in range(2, num_tasks + 1):
        vals = [R[t, j] - R[j, j] for j in range(1, t)
                if not (np.isnan(R[t, j]) or np.isnan(R[j, j]))]
        B[t] = float(np.mean(vals)) if vals else np.nan
    return B


# ---------------------------------------------------------------- training one group
def train_group(name: str, seed: int) -> int:
    ddir = OUT / name / f"seed{seed}"
    if ddir.exists():
        raise Halt(f"REFUSE: target dir exists: {ddir}")
    ddir.mkdir(parents=True)
    (ddir / "logs").mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "device_is_cuda", str(device))
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
    handoff_ledger, boundary_memory, pseudo_peaks = [], [], []

    for task_idx, new_classes in enumerate(tasks, start=1):
        log(f"--- task{task_idx} new_classes={new_classes} ---")
        seen = list(range(2 * task_idx))
        t_train = ctx["task_rows"](new_classes, "train")
        t_val = ctx["task_rows"](new_classes, "val")
        require(bool(np.isin(ctx["y"][t_train], new_classes).all()), f"{name}s{seed}_t{task_idx}_train_labels")
        require(bool(np.isin(ctx["y"][t_val], new_classes).all()), f"{name}s{seed}_t{task_idx}_val_labels")

        # ---- t>=2: load previous task's disk state (REQUIRED, not a bypass) ----
        if task_idx >= 2:
            handoff = load_state_package(ddir, task_idx - 1, name, device)
            centroids, counts, task_of = handoff["centroids"], handoff["counts"], handoff["task_of"]
            require(handoff["seen"] == list(range(2 * (task_idx - 1))),
                    f"{name}s{seed}_t{task_idx}_handoff_seen")
            require(handoff["centroid_value_sha256"] == centroid_value_hash(centroids, handoff["seen"]),
                    f"{name}s{seed}_t{task_idx}_handoff_value_hash")
            handoff_ledger.append({"task": task_idx, "loaded_package": f"state_package_task{task_idx-1}.npz",
                                   "package_sha256": handoff["package_sha256"],
                                   "centroid_value_sha256": handoff["centroid_value_sha256"],
                                   "loaded_seen": handoff["seen"],
                                   "used_for_pseudo": True})
            log(f"  loaded prev state: seen={handoff['seen']} pkg={handoff['package_sha256'][:12]}")

        # ---- Task1: train + freeze ----
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
                    "budget_truncated": tr["budget_truncated"],
                    "seed": seed}
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

        # ---- extract current task joint ----
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

        # ---- pseudo features ----
        old_classes = [c for c in seen if c not in new_classes]
        pseudo_parts, pseudo_labels, pm = ([], [], []) if not old_classes else \
            pooled_shift_pseudo(centroids, current_centroids, z_t, ctx["y"][t_train], t_train,
                                old_classes, new_classes)
        pseudo_total = int(sum(m["pseudo_count"] for m in pm))
        pseudo_peaks.append(pseudo_total)

        # ---- train sparse head ----
        torch.manual_seed(seed + 10000 * task_idx)
        head = nn.Linear(JOINT_DIM, 2 * task_idx).to(device)
        X = np.concatenate([z_t.astype(np.float32)] + pseudo_parts, axis=0).astype(np.float32)
        Y = np.concatenate([ctx["y"][t_train].astype(np.int64)] + pseudo_labels, axis=0).astype(np.int64)
        require(sorted(set(Y.tolist())) == seen, f"{name}s{seed}_t{task_idx}_head_labels")
        head_losses, head_changed = fit_sparse_head(head, X, Y, device, f"{name}s{seed}_t{task_idx}_head", counters)
        require(head_changed, f"{name}s{seed}_t{task_idx}_head_updated")
        require(head.in_features == 56 and head.out_features == 2 * task_idx, f"{name}s{seed}_t{task_idx}_head_dim")
        torch.save(head.state_dict(), ddir / f"task{task_idx}_head.pt")

        # ---- boundary checks ----
        require(full_signature(model) == frozen_sig, f"{name}s{seed}_t{task_idx}_frozen_unchanged")
        require(all(not p.requires_grad for p in model.parameters()), f"{name}s{seed}_t{task_idx}_no_trainable")
        require(all(p.grad is None for p in model.parameters()), f"{name}s{seed}_t{task_idx}_grads_none")

        # ---- save state package ----
        save_state_package(ddir, task_idx, centroids, counts, task_of, name, device, seed, head)

        # ---- per-boundary budget / memory ----
        frozen_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        frozen_buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        head_bytes = sum(p.numel() * p.element_size() for p in head.parameters())
        centroid_bytes = sum(centroids[c].nbytes for c in seen)
        count_bytes = len(seen) * 8
        map_bytes = len(seen) * 8
        instat_bytes = ctx["in_mean"].nbytes + ctx["in_denom"].nbytes
        descstat_bytes = ctx["d_mean"].nbytes + ctx["d_std"].nbytes
        meta_bytes = len(json.dumps({"seen": seen, "task_of": [task_of[c] for c in seen]}))
        boundary_memory.append({
            "task": task_idx,
            "logical_bytes": {
                "frozen_params": frozen_param_bytes, "frozen_buffers": frozen_buf_bytes,
                "current_head": head_bytes, "all_centroids": centroid_bytes,
                "class_counts": count_bytes, "task_mapping": map_bytes,
                "input_stats": instat_bytes, "descriptor_stats": descstat_bytes,
                "other_metadata": meta_bytes,
            },
            "serialized_bytes": {
                "state_package_npz": (ddir / f"state_package_task{task_idx}.npz").stat().st_size,
                "task1_encoder_pt": (ddir / "task1_encoder_best.pt").stat().st_size,
                "current_head_pt": (ddir / f"task{task_idx}_head.pt").stat().st_size,
            },
            "real_replay_samples": 0,
            "transient_pseudo_peak": pseudo_total,
            "gpu_peak": gpu_peak_bytes(device),
            "rss": process_memory_bytes(),
            "elapsed_since_run_start": round(time.perf_counter() - started, 1),
        })

        # ---- independent disk recovery ----
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--recover",
                            name, str(seed), str(task_idx)],
                           cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            log(r.stdout); log(r.stderr)
            raise Halt(f"recovery failed {name} seed{seed} task{task_idx}: exit {r.returncode}")

        log(f"  task{task_idx} done: real={len(z_t)} pseudo={pseudo_total} seen={seen}")

    dump_json(ddir / "handoff_ledger.json", {"per_boundary": handoff_ledger,
                                             "note": "t>=2 actually loads prev disk package"})
    dump_json(ddir / "boundary_memory.json", boundary_memory)
    dump_json(ddir / "call_ledger_run.json", {
        "encoder_forwards": counters["encoder_forwards"], "encoder_backward": counters["encoder_backward"],
        "encoder_optimizer_steps": counters["encoder_optimizer_steps"],
        "head_forwards": counters["head_forwards"], "head_backward": counters["head_backward"],
        "head_optimizer_steps": counters["head_optimizer_steps"],
        "test_set_forwards": counters["test_set_forwards"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "pid": os.getpid(),
    })
    dump_json(ddir / "environment.json", {
        "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "numpy": np.__version__, "os": platform.platform(),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "seed": seed, "rss_at_start": rss_start,
    })
    dump_json(ddir / "gate_decision.json", {"dataset": name, "seed": seed,
                                            "verdict": "TRAINING_PHASE_PASS",
                                            "tasks_completed": n_tasks})
    check_inputs_stable("pre-exit")
    return 0


# ---------------------------------------------------------------- recover (independent)
def recover_group(name: str, seed: int, task_idx: int) -> int:
    ddir = OUT / name / f"seed{seed}"
    task_dir = ddir / f"task{task_idx}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "device_is_cuda", str(device))
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

    sp = np.load(ddir / f"state_package_task{task_idx}.npz")
    seen = list(range(2 * task_idx))
    for c in seen:
        require(sp[f"centroid_{c}"].dtype == np.float32, f"{name}s{seed}_recover_centroid_{c}_float32")
        require(sp[f"centroid_{c}"].shape == (JOINT_DIM,), f"{name}s{seed}_recover_centroid_{c}_shape")

    out_npz = task_dir / "recovery_outputs.npz"
    np.savez_compressed(out_npz, row_ids=ref_rows, labels=labels, x_std=x_std8,
                        latent=latent8, desc_raw=desc_raw8, desc_std=desc_std8,
                        joint=joint8, head_logits=hlog)
    dump_json(task_dir / "recovery_comparison.json", {"fields": comp, "all_bitwise_equal": True})
    dump_json(task_dir / "recovery_output_hashes.json", {
        "recovery_outputs.npz": sha(out_npz),
        "state_package": sha(ddir / f"state_package_task{task_idx}.npz"),
    })
    dump_json(task_dir / "call_ledger_recover.json", {
        "encoder_forwards": counters["encoder_forwards"], "head_forwards": counters["head_forwards"],
        "backward": 0, "optimizer_steps": 0, "rss": process_memory_bytes(),
        "elapsed_seconds": round(time.perf_counter() - started, 1), "pid": os.getpid(),
    })
    check_inputs_stable("pre-exit")
    return 0


# ---------------------------------------------------------------- evaluation (separate phase)
def evaluate_group(name: str, seed: int) -> int:
    ddir = OUT / name / f"seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "device_is_cuda", str(device))
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

    preds_per_task = {t: None for t in range(1, n_tasks + 1)}
    per_row = {t: None for t in range(1, n_tasks + 1)}
    seen_macro, seen_micro = {}, {}
    per_class = {}
    for t in range(1, n_tasks + 1):
        seen = list(range(2 * t))
        head_sd = torch.load(ddir / f"task{t}_head.pt", map_location="cpu", weights_only=False)
        head = nn.Linear(JOINT_DIM, 2 * t).to(device)
        head.load_state_dict(head_sd, strict=True)
        head.eval()
        # test members of ALL seen classes (never restrict to task identity)
        tst_idx = ctx["task_rows"](seen, "test")
        labels = ctx["y"][tst_idx].astype(np.int64)
        if len(tst_idx) == 0:
            preds_per_task[t] = np.asarray([], dtype=np.int64)
            per_row[t] = {"row_ids": [], "labels": [], "logits": []}
            continue
        z, _, _, _ = ts.make_joint(model, ctx["x_std"], ctx["x_raw"], tst_idx,
                                   ctx["d_mean"], ctx["d_std"], name, device)
        with torch.inference_mode():
            logits = head(torch.from_numpy(z).to(device)).cpu().numpy().astype(np.float32)
        pred = logits.argmax(axis=1).astype(np.int64)
        preds_per_task[t] = pred
        per_row[t] = {"row_ids": tst_idx.tolist(), "labels": labels.tolist(),
                      "logits": logits.tolist()}
        # per-class accuracy
        pc = {}
        for c in seen:
            m = labels == c
            pc[int(c)] = float((pred[m] == c).mean()) if m.sum() else np.nan
        per_class[t] = pc
        seen_macro[t] = float(np.nanmean([pc[c] for c in seen]))
        seen_micro[t] = float((pred == labels).mean())
    # R matrix + forgetting + BWT
    labels_per_task = {t: np.asarray(per_row[t]["labels"], dtype=np.int64) for t in range(1, n_tasks + 1)}
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
    return 0


# ---------------------------------------------------------------- propose / static / regression
def propose_protocol() -> int:
    PROTO_OUT.mkdir(parents=True, exist_ok=True)
    inv = input_files()
    hashes = {k: sha(v) for k, v in inv.items()}
    protocol = {
        "status": "PROPOSED",
        "note": "待 Codex 验收后由 --lock-protocol 锁定；本轮不锁定、不执行研究训练/前向/测试。",
        "version": "s1_baseline_4datasets_v1_proposed",
        "output_dir": "experiments/baselines/trunk_baseline_4datasets",
        "prep_dir": "experiments/baselines/trunk_baseline_4datasets_protocol_final",
        "seeds": SEEDS,
        "datasets": {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5},
        "tasks": ALL_TASKS,
        "groups": "4 datasets x 8 seeds = 32 groups; canonical class order; 2 new classes/task",
        "no_candidates_this_round": True,
        "model": {"backbone": "CfC hidden=48", "pooling": "temporal attention",
                  "descriptor": "8-dim", "joint": 56, "bottleneck": "off",
                  "freeze": "whole representation path after Task1"},
        "data": {"splits": "accepted fixed train/val/test", "input_stats": "Task1-only frozen",
                 "descriptor_stats": "v2 frozen"},
        "budget": {
            "encoder": {"lr": ENCODER_LR, "batch": BATCH_SIZE, "clip": GRAD_CLIP,
                        "min": MIN_EPOCHS, "max": MAX_EPOCHS, "patience": PATIENCE,
                        "delta_acc": VAL_DELTA_ACC, "delta_loss": VAL_DELTA_LOSS,
                        "checkpoint_rule": "argmax(val_macro,-val_loss); tie->earlier",
                        "budget_truncation_flagged": True},
            "sparse_head": {"lr": SPARSE_LR, "epochs": SPARSE_EPOCHS, "l1": SPARSITY_LAMBDA,
                            "l1_on": "weight only", "full_batch": True, "early_stopping": "none"},
            "note": "两个预算分开记录；达 12 epoch 上限标记 budget_truncated，不宣称收敛",
        },
        "seed_derivation": {
            "model_init": "torch.manual_seed(s)",
            "task1_loader": "generator seed = s + 100000 + epoch",
            "head_init": "torch.manual_seed(s + 10000 * task_idx)",
            "recovery_audit_isolated": True,
        },
        "centroid_rule": "uniform float32", "pseudo_rule":
            "nearest by cosine(eps 1e-8, float64); tie->earlier canonical; "
            "source=ALL current real train rows; shift float64->pseudo float32",
        "disk_handoff": "t>=2 loads prev state_package from disk, verifies classes/dtype/shape/finite/"
                        "reference hashes, and USES loaded centroids (not in-memory leftover)",
        "recovery_criteria": "strict load; dtype+shape+bytes equal; no loosening",
        "test_isolation": {"two_phase": True, "training_phase": "32 groups + recovery",
                           "evaluation_phase": "separate --evaluate, frozen model/hyperparams/selection",
                           "test_feeds_training": False},
        "metrics": {
            "main": "final-task seen-class macro accuracy",
            "secondary": ["per-boundary macro/micro", "per-class", "per-old-task",
                          "R[t,j] lower-triangular", "forgetting F[t]", "BWT[t]",
                          "average incremental accuracy"],
            "forgetting_formula": "F[t]=mean(j<t){max(k=j..t-1) R[k,j] - R[t,j]}; Task1=NA; keep negative",
            "bwt_formula": "BWT[t]=mean(j<t){R[t,j]-R[j,j]}; Task1=NA",
            "R_definition": "head at task t (all seen classes) predicts task j test members; "
                            "R[t,j]=mean class accuracy over task j's 2 classes; no 2-class head slicing",
            "reporting": "per-seed, mean, sample SD(ddof=1), 95% t-interval(df=7); "
                         "internal unrounded ratios; acc in %, diffs in pp; "
                         "no candidate pairing -> NA; per-dataset separate",
            "cwrw_scope": "same-record time-segment extrapolation",
            "uwave_scope": "validation 3 per class",
        },
        "assertions": ["finite input/latent/desc/joint/pseudo/loss",
                       "grad present+finite before step; all params incl bias finite after step",
                       "freeze grad None; params+buffers hash unchanged each boundary",
                       "row_ids/labels rebuilt from split+data",
                       "test forward = 0", "protected input hashes unchanged"],
        "input_sha256_snapshot": hashes,
        "input_count": len(hashes),
    }
    dump_json(PROTO_OUT / "protocol_proposed.json", protocol)
    log(f"protocol_proposed.json written (inputs={len(hashes)})")
    return 0


def static_check() -> int:
    PROTO_OUT.mkdir(parents=True, exist_ok=True)
    out = {"in_memory_compile": {}, "import_check": {}, "forbidden_inheritance": {}}
    files = [Path(__file__).resolve(), ROOT / "src/trunk/train_smoke.py",
             ROOT / "src/trunk/acbm_interface.py", ROOT / "src/trunk/descriptors.py",
             ROOT / "src/trunk/adapters.py", ROOT / "src/preflight_standardization.py"]
    for p in files:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")   # in-memory, never writes pyc
        out["in_memory_compile"][p.name] = "OK"
    # import check
    import importlib
    for mod in ["trunk.train_smoke", "trunk.acbm_interface", "trunk.adapters", "trunk.descriptors"]:
        importlib.import_module(mod)
        out["import_check"][mod] = "OK"
    # forbidden inheritance: this file must not import OUT/SEED globals from smoke scripts.
    # Use AST (not a substring grep) so the check's own literals cannot self-match.
    import ast
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported_modules: set[str] = set()
    uses_ts_OUT = uses_ts_SEED = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(n.name for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ts":
            if node.attr == "OUT":
                uses_ts_OUT = True
            if node.attr == "SEED":
                uses_ts_SEED = True
    out["forbidden_inheritance"] = {
        "no_full_sequence_import": not any("full_sequence" in m for m in imported_modules),
        "no_train_smoke_OUT_use": not uses_ts_OUT,
        "no_train_smoke_SEED_use": not uses_ts_SEED,
        "imported_modules": sorted(imported_modules),
    }
    require(out["forbidden_inheritance"]["no_full_sequence_import"], "no_full_sequence_import")
    require(out["forbidden_inheritance"]["no_train_smoke_OUT_use"], "no_ts_OUT_inherit")
    require(out["forbidden_inheritance"]["no_train_smoke_SEED_use"], "no_ts_SEED_inherit")
    dump_json(PROTO_OUT / "static_check.json", out)
    log("static check PASS")
    return 0


def zero_data_regression() -> int:
    PROTO_OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    # 1) Adam full param_groups roundtrip (CPU, no data/model)
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
    pg_ok = True
    for a, b in zip(snap["param_groups"], s2["param_groups"]):
        pg_ok = pg_ok and all(a[k] == b[k] for k in a)
    for pid in snap["state"]:
        for k in ("step", "exp_avg", "exp_avg_sq"):
            a, b = s2["state"][pid][k], snap["state"][pid][k]
            if isinstance(a, torch.Tensor):
                pg_ok = pg_ok and bool(torch.equal(a, b))
            else:
                pg_ok = pg_ok and (a == b)
    out["adam_param_groups_roundtrip"] = bool(pg_ok)
    require(pg_ok, "adam_param_groups_roundtrip")

    # 2) task layout / state schema
    out["task_layout"] = {ds_: task_list(ds_) for ds_ in DATASETS}
    require(task_list("uci_har") == [[0, 1], [2, 3], [4, 5]], "task_layout_uci_har")
    require(task_list("uwave") == [[0, 1], [2, 3], [4, 5], [6, 7]], "task_layout_uwave")
    require(task_list("cwru") == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], "task_layout_cwru")

    # 3) metric formula (hand-written arrays; engineering example, NOT research results)
    num_tasks = 3
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    labels_per_task = {t: labels for t in range(1, num_tasks + 1)}
    preds2 = {t: np.zeros_like(labels) for t in range(1, num_tasks + 1)}
    for t in range(1, num_tasks + 1):
        seen = list(range(2 * t))
        for i, lab in enumerate(labels):
            preds2[t][i] = lab if lab in seen else seen[i % len(seen)]
    R = task_accuracy_matrix(preds2, labels_per_task, num_tasks)
    for t in range(1, num_tasks + 1):
        for j in range(1, t + 1):
            require(abs(R[t, j] - 1.0) < 1e-9, f"R_example_{t}_{j}_is_1")
    Fvec = forgetting_from_R(R, num_tasks)
    B = bwt_from_R(R, num_tasks)
    require(all(abs(Fvec[t]) < 1e-9 for t in range(2, num_tasks + 1)), "forgetting_example_zero")
    require(all(abs(B[t]) < 1e-9 for t in range(2, num_tasks + 1)), "bwt_example_zero")
    # a non-trivial forgetting example: task1+task2 classes become wrong at task3
    p3 = {t: np.array(labels) for t in range(1, num_tasks + 1)}
    p3[3] = np.array([2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5])
    R3 = task_accuracy_matrix(p3, labels_per_task, num_tasks)
    require(R3[3, 1] == 0.0 and R3[3, 2] == 0.0 and R3[1, 1] == 1.0, "R3_forgetting_setup")
    F3 = forgetting_from_R(R3, num_tasks)
    # F[3] = mean(j in {1,2}) of {max(R[j..2][j]) - R[3,j]} = mean(1-0, 1-0) = 1
    require(abs(F3[3] - 1.0) < 1e-9, "forgetting_example_equals_1", str(F3[3]))
    out["metric_formula_examples"] = {
        "perfect_R_all_ones": R.tolist(), "forgetting_zero_when_no_drop": Fvec,
        "forgetting_one_when_task1_and_2_drop_at_t3": F3[3],
        "note": "engineering examples over hand-written arrays; NOT research results",
    }
    dump_json(PROTO_OUT / "zero_data_regression.json", out)
    log("zero-data regression PASS")
    return 0


# ---------------------------------------------------------------- lock / aggregate
def lock_protocol() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inv = input_files()
    before = {k: sha(v) for k, v in inv.items()}
    dump_json(OUT / "protocol_locked.json",
              json.loads((PROTO_OUT / "protocol_proposed.json").read_text(encoding="utf-8"))
              | {"status": "LOCKED", "locked_before_any_forward": True, "input_sha256_before": before})
    dump_json(OUT / "input_sha256_before.json", before)
    h = sha(OUT / "protocol_locked.json")
    (OUT / "protocol_locked.json.sha256").write_text(f"{h} *protocol_locked.json\n", encoding="utf-8")
    log(f"protocol locked: {h}")
    return 0


def check_inputs_stable(stage: str) -> None:
    before = json.loads((OUT / "input_sha256_before.json").read_text(encoding="utf-8"))
    now = {k: sha(v) for k, v in input_files().items()}
    changed = [k for k in before if before.get(k) != now.get(k)]
    if changed:
        raise Halt(f"input hash drift at {stage}: {changed}")


def aggregate() -> int:
    rows = []
    for name in DATASETS:
        for seed in SEEDS:
            ev = json.loads((OUT / name / f"seed{seed}" / "evaluation.json").read_text(encoding="utf-8"))
            rows.append({"dataset": name, "seed": seed,
                         "final_macro": ev["final_task_seen_macro_accuracy"],
                         "avg_incremental": ev["average_incremental_accuracy"],
                         "BWT_final": ev["BWT"][-1]})
    with (OUT / "aggregate.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)
    log("aggregate.csv written")
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose-protocol", action="store_true")
    ap.add_argument("--static-check", action="store_true")
    ap.add_argument("--zero-data-regression", action="store_true")
    ap.add_argument("--lock-protocol", action="store_true")
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
    if a.lock_protocol:
        return lock_protocol()
    if a.aggregate:
        return aggregate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise Halt("CUDA unavailable; blocking, not auto-switching to CPU")
    if a.run:
        name, seed = a.run[0], int(a.run[1])
        ddir = OUT / name / f"seed{seed}"
        try:
            return train_group(name, seed)
        except Halt as e:
            write_failure("HALTED", {"reason": str(e), "log_tail": LOG[-40:]}, ddir)
            raise SystemExit(4)
        except Exception:
            write_failure("UNEXPECTED", {"traceback": traceback.format_exc(), "log_tail": LOG[-40:]}, ddir)
            raise SystemExit(6)
    if a.recover:
        name, seed, task_idx = a.recover[0], int(a.recover[1]), int(a.recover[2])
        return recover_group(name, seed, task_idx)
    if a.evaluate:
        name, seed = a.evaluate[0], int(a.evaluate[1])
        return evaluate_group(name, seed)
    raise SystemExit("see --help")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise SystemExit(6)
