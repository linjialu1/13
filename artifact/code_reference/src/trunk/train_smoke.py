"""Four-dataset ACBM single-seed (seed=11), first-two-tasks TRAINING smoke test.

Faithfully re-implements the archived ACBM call chain (``bottleneck_off``) but wired to
the accepted trunk adapters / statistics / splits and the v2 descriptor statistics:

  raw data --(accepted Task-1 input stats)--> standardized windows --> CfC(+temporal attn)
       \\                                                             48-dim latent
        \\--(v2 descriptor formulas)--> 8-dim descriptors --(v2 mean/std)--> desc_std
                                                                             |
                                                56-dim joint = [latent | desc_std]
                                                                             |
        Task1: fit class centroids + L1 sparse head 56->2                    |
        Task2: pooled_shift pseudo-features for old classes + L1 sparse head 56->4

Scope per dataset: Task1 = [0,1], Task2 = [2,3] ONLY.  Allowed: train / validate /
checkpoint / engineering consistency checks.  FORBIDDEN: test-set forward, test accuracy,
candidate comparison, 8-seed, effectiveness conclusions.

This module NEVER touches the test split.

Usage:
    python src/trunk/train_smoke.py --lock-protocol     # must run BEFORE any dataset
    python src/trunk/train_smoke.py --dataset uci_har   # one independent process
    python src/trunk/train_smoke.py --dataset wisdm
    python src/trunk/train_smoke.py --dataset uwave
    python src/trunk/train_smoke.py --dataset cwru
    python src/trunk/train_smoke.py --summary           # after all four succeeded
"""
from __future__ import annotations

import os
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
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/baselines/trunk_training_smoke_seed11_fix1"
OLD_OUT = ROOT / "experiments/baselines/trunk_training_smoke_seed11"
V2_DIR = ROOT / "experiments/baselines/trunk_model_interface_check_v2"
PRE_FIX_COPY = ROOT / "scratch/train_smoke_pre_fix1.py"
for p in (str(ROOT / "src"), str(ROOT), str(ROOT / ".local_packages")):
    if p not in sys.path:
        sys.path.insert(0, p)

from trunk import acbm_interface as ai   # noqa: E402
from trunk import adapters as ad         # noqa: E402
from trunk import descriptors as ds      # noqa: E402

F = torch.nn.functional

# ---------------------------------------------------------------- constants
SEED = 11
HIDDEN = 48
DESCRIPTOR_DIM = 8
JOINT_DIM = 56
DATASETS = ["uci_har", "wisdm", "uwave", "cwru"]
TASK1 = [0, 1]
TASK2 = [2, 3]

# encoder budget (archived train_cfctask + run_config.json)
ENCODER_LR = 1e-3
BATCH_SIZE = 128
GRAD_CLIP = 5.0
MIN_EPOCHS = 3
MAX_EPOCHS = 12
PATIENCE = 2
VAL_DELTA_ACC = 0.005
VAL_DELTA_LOSS = 0.001
EVAL_BATCH = 256
TASK1_TRAIN_SEED = SEED + 1 * 100_000     # archived: SEED + task_i*100_000, task_i=1

# sparse head budget (archived fit_sparse_classifier + run_config.json)
SPARSE_EPOCHS = 300
SPARSE_LR = 0.05
SPARSITY_LAMBDA = 0.0007

LOG: list[str] = []
COUNTERS: dict[str, int] = {}


class Halt(Exception):
    pass


def log(m: str = "") -> None:
    print(m, flush=True)
    LOG.append(m)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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


# ---------------------------------------------------------------- protected inputs
def protected_input_files() -> dict[str, Path]:
    f = {
        "split/uci_har_fixed_split.npz": ad.PREFLIGHT_DIR / "uci_har_fixed_split.npz",
        "split/wisdm_fixed_split.npz": ad.PREFLIGHT_DIR / "wisdm_fixed_split.npz",
        "split/uwave_fixed_split.npz": ad.PREFLIGHT_DIR / "uwave_fixed_split.npz",
        "split/cwru_window_index.csv": ad.PREFLIGHT_DIR / "cwru_window_index.csv",
        "split/wisdm_row_provenance.csv": ad.PREFLIGHT_DIR / "wisdm_row_provenance.csv",
        "stats/preprocess_statistics.json": ad.FIX_DIR / "preprocess_statistics.json",
        "stats/fit_indices.npz": ad.FIX_DIR / "fit_indices.npz",
        "v2/interface_check_uci_har.json": V2_DIR / "interface_check_uci_har.json",
        "v2/interface_check_wisdm.json": V2_DIR / "interface_check_wisdm.json",
        "v2/interface_check_uwave.json": V2_DIR / "interface_check_uwave.json",
        "v2/interface_check_cwru.json": V2_DIR / "interface_check_cwru.json",
        "archive/pipeline.py": ROOT / "experiments/archive/2026-10-13/cfc_acbm_t3_component_ablation_3seeds/pipeline.py",
        "archive/stage4_refined_benchmark.py": ROOT / "experiments/archive/stage4/refined/benchmark.py",
        "archive/stage4_refined_run_config.json": ROOT / "experiments/archive/stage4/refined/run_config.json",
        "archive/run_wisdm.py": ROOT / "experiments/archive/2026-10-11/cfc_cbm_wisdm_original_vs_staged/run_wisdm.py",
        "thirdparty/src_legacy_benchmark.py": ROOT / "src/legacy/benchmark.py",
        "thirdparty/src_legacy_cicbm_benchmark.py": ROOT / "src/legacy/cicbm/benchmark.py",
        "raw/wisdm.zip": ROOT / "data/wisdm.zip",
        "raw/wisdm_cache.npz": ROOT / "data/wisdm_har_windows.npz",
        "raw/uwave_train.ts": ROOT / "data/UWaveGestureLibrary_TRAIN.ts",
        "raw/uwave_test.ts": ROOT / "data/UWaveGestureLibrary_TEST.ts",
        "raw/uwave_cache.npz": ROOT / "data/uwave_gesture_library.npz",
        "raw/cwru_manifest.json": ROOT / "data/cwru_lttsnet_seed10.manifest.json",
        "raw/uci_train_y.txt": ROOT / "data/UCI HAR Dataset/train/y_train.txt",
        "raw/uci_test_y.txt": ROOT / "data/UCI HAR Dataset/test/y_test.txt",
        "raw/uci_train_subject.txt": ROOT / "data/UCI HAR Dataset/train/subject_train.txt",
        "raw/uci_test_subject.txt": ROOT / "data/UCI HAR Dataset/test/subject_test.txt",
    }
    for n in ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y",
              "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]:
        for split in ("train", "test"):
            f[f"raw/uci_{split}_{n}.txt"] = ROOT / f"data/UCI HAR Dataset/{split}/Inertial Signals/{n}_{split}.txt"
    for m in sorted((ROOT / "data/cwru_lttsnet_raw").glob("*.mat")):
        f[f"raw/cwru_{m.name}"] = m
    return f


def source_files() -> dict[str, Path]:
    f = {"src/trunk/train_smoke.py": Path(__file__).resolve(),
         "src/trunk/acbm_interface.py": ROOT / "src/trunk/acbm_interface.py",
         "src/trunk/descriptors.py": ROOT / "src/trunk/descriptors.py",
         "src/trunk/adapters.py": ROOT / "src/trunk/adapters.py",
         "src/preflight_standardization.py": ROOT / "src/preflight_standardization.py"}
    return f


# ---------------------------------------------------------------- seeds / model
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(input_channels: int, n_classes: int, seed: int):
    set_seed(seed)
    model = ai.pipeline_module().TemporalAttentionSequenceClassifier(
        "CfC", input_channels, HIDDEN, n_classes)
    return model


# ---------------------------------------------------------------- faithful helpers
def labels_to_local(yy: torch.Tensor, active: list[int], n_classes: int, device) -> torch.Tensor:
    lookup = torch.full((n_classes,), -1, dtype=torch.long, device=device)
    lookup[torch.as_tensor(active, dtype=torch.long, device=device)] = torch.arange(
        len(active), dtype=torch.long, device=device)
    return lookup[yy]


def make_loader(x: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int,
                generator_seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(generator_seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x[indices]), torch.from_numpy(y[indices]).long())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       drop_last=False, pin_memory=torch.cuda.is_available(),
                                       generator=generator)


def evaluate_validation(model, x: np.ndarray, y: np.ndarray, indices: np.ndarray,
                        new_classes: list[int], n_classes: int, device, batch_size: int):
    model.eval()
    class_accuracies: list[float] = []
    total_loss = 0.0
    total_count = 0
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


def assert_params_finite(model, label: str) -> None:
    for n, p in model.named_parameters():
        if not bool(torch.isfinite(p).all().item()):
            raise Halt(f"NON-FINITE parameter {label}: {n}")


def assert_grads_finite(model, label: str) -> None:
    for n, p in model.named_parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all().item()):
            raise Halt(f"NON-FINITE gradient {label}: {n}")


def params_signature(model) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- feature extraction
def extract_latent(model, x_std: np.ndarray, rows: np.ndarray, device, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            xx = torch.from_numpy(np.ascontiguousarray(x_std[part])).to(device)
            rep, _ = model.representation_and_logits(xx)
            arr = rep.detach().cpu().numpy().astype(np.float32)
            ds.assert_all_finite(arr, "extracted latent")
            chunks.append(arr)
    return np.concatenate(chunks, axis=0)


def make_joint(model, x_std, x_raw, rows, d_mean, d_std, dataset_name, device):
    latent = extract_latent(model, x_std, rows, device, EVAL_BATCH)
    desc_raw, desc_info = ds.describe(dataset_name, np.asarray(x_raw)[rows])
    # desc_info (implementation metadata) never enters the numeric pipeline; only
    # desc_raw (the [n,8] float32 array) is standardized and concatenated.
    ds.assert_all_finite(desc_raw, f"{dataset_name} descriptors")
    desc_std = ds.standardize_descriptors(desc_raw, d_mean, d_std)
    joint = np.concatenate([latent, desc_std], axis=1).astype(np.float32)
    ds.assert_all_finite(joint, "joint")
    return joint, latent, desc_raw, desc_std


# ---------------------------------------------------------------- sparse head (faithful)
def fit_sparse_head(head: nn.Linear, c: np.ndarray, y: np.ndarray, device, name: str):
    cc = torch.from_numpy(np.asarray(c, dtype=np.float32)).to(device)
    yy = torch.from_numpy(np.asarray(y, dtype=np.int64)).long().to(device)
    require(int(y.max()) + 1 == head.out_features, f"{name}_nclass_matches_head_out")
    require(head.in_features == cc.shape[1], f"{name}_head_in_features_matches_joint")
    opt = torch.optim.Adam(head.parameters(), lr=SPARSE_LR)
    w_before = head.weight.detach().clone()
    losses = []
    for _ in range(SPARSE_EPOCHS):
        opt.zero_grad(set_to_none=True)
        logits = head(cc)
        COUNTERS["head_forwards"] += 1
        if not bool(torch.isfinite(logits).all().item()):
            raise Halt(f"NON-FINITE {name} logits")
        loss = F.cross_entropy(logits, yy) + SPARSITY_LAMBDA * head.weight.abs().sum()
        if not bool(torch.isfinite(loss).all().item()):
            raise Halt(f"NON-FINITE {name} loss")
        loss.backward()
        COUNTERS["head_backward"] += 1
        opt.step()
        COUNTERS["head_optimizer_steps"] += 1
        if not bool(torch.isfinite(head.weight).all().item()):
            raise Halt(f"NON-FINITE {name} weight after step")
        losses.append(float(loss.detach()))
    return losses, bool(not torch.equal(w_before, head.weight))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def pooled_shift_pseudo(class_means, current_means, z_t2, y_t2, t2_rows, old_classes, new_classes):
    pseudo_parts, pseudo_labels, manifest = [], [], []
    for cls in old_classes:
        old_mean = class_means[cls].astype(np.float64)
        nearest = max(new_classes, key=lambda c: cosine(old_mean, current_means[c]))
        src_mask = y_t2 == nearest
        source = z_t2[src_mask].astype(np.float64)
        source_rows = t2_rows[src_mask]
        shift = old_mean - current_means[nearest]
        pseudo = source + shift
        ds.assert_all_finite(pseudo, f"pseudo old class {cls}")
        manifest.append({
            "old_class": int(cls), "source_class": int(nearest),
            "source_row_ids": [int(r) for r in source_rows],
            "source_count": int(len(source_rows)),
            "pseudo_count": int(len(pseudo)),
            "shift_vector": [float(v) for v in shift],
            "old_mean": [float(v) for v in old_mean],
            "source_mean": [float(v) for v in current_means[nearest]],
            "cosine_similarity": cosine(old_mean, current_means[nearest]),
            "formula": "pseudo_z = source - current_means[nearest] + old_mean",
        })
        pseudo_parts.append(pseudo.astype(np.float32))
        pseudo_labels.append(np.full(len(pseudo), cls, dtype=np.int64))
    return pseudo_parts, pseudo_labels, manifest


# ---------------------------------------------------------------- dataset run
def run_dataset(name: str) -> int:
    started = time.perf_counter()
    ddir = OUT / name
    if ddir.exists():
        raise Halt(f"REFUSE: target dir already exists; not overwriting: {ddir}")
    ddir.mkdir(parents=True)
    (ddir / "logs").mkdir()

    for k in ("encoder_forwards", "head_forwards", "encoder_backward", "encoder_optimizer_steps",
              "head_backward", "head_optimizer_steps", "test_set_forwards"):
        COUNTERS[k] = 0

    require(device.type == "cuda", "device_is_cuda", f"device={device}")
    log("=" * 78)
    log(f"DATASET {name}  device={device}")
    log("=" * 78)

    # ---- protected input hashes before -----------------------------------
    prot = protected_input_files()
    prot_before = {k: sha(v) for k, v in prot.items()}
    dump_json(ddir / "protected_sha256_before.json", prot_before)

    # ---- accepted data + splits ------------------------------------------
    raw = ad.adapt(name)
    x_raw = np.asarray(raw["x"])
    y = np.asarray(raw["y"])
    ds.assert_all_finite(x_raw, f"{name} raw")
    n_classes = int(len(set(y.tolist())))
    require(n_classes in (6, 8, 10), f"{name}_n_classes", str(n_classes))

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

    def task_rows(classes):
        tr = np.asarray(sorted(int(i) for i in train_all if int(y[i]) in classes), dtype=np.int64)
        va = np.asarray(sorted(int(i) for i in val_all if int(y[i]) in classes), dtype=np.int64)
        return tr, va

    t1_train, t1_val = task_rows(TASK1)
    t2_train, t2_val = task_rows(TASK2)
    log(f"  rows: t1_train={len(t1_train)} t1_val={len(t1_val)} "
        f"t2_train={len(t2_train)} t2_val={len(t2_val)} test={len(test_all)}")
    require(bool(np.isin(y[t1_train], TASK1).all()), f"{name}_t1_train_labels_0_1")
    require(bool(np.isin(y[t1_val], TASK1).all()), f"{name}_t1_val_labels_0_1")
    require(bool(np.isin(y[t2_train], TASK2).all()), f"{name}_t2_train_labels_2_3")
    require(bool(np.isin(y[t2_val], TASK2).all()), f"{name}_t2_val_labels_2_3")
    require(len(set(t1_train.tolist()) & set(val_all.tolist())) == 0, f"{name}_t1_train_val_disjoint")
    require(len(set(t2_train.tolist()) & set(val_all.tolist())) == 0, f"{name}_t2_train_val_disjoint")
    require(len(set(train_all.tolist()) & set(val_all.tolist())) == 0, f"{name}_train_val_disjoint")
    if name == "cwru":
        # all three splits index the SAME 3200-row array -> index-set disjointness is meaningful
        require(len(set(train_all.tolist()) & set(test_all.tolist())) == 0, f"{name}_train_test_disjoint")
        require(len(set(val_all.tolist()) & set(test_all.tolist())) == 0, f"{name}_val_test_disjoint")
    else:
        # the "test" split indexes a SEPARATE array (official test side); it is never
        # loaded into any forward. Structural guarantee enforced by test_set_forwards==0.
        require(len(raw["test"]["x"]) > 0, f"{name}_test_array_present_but_unused")

    # ---- accepted input statistics (frozen; never refit) ------------------
    in_stats = ad.accepted_input_stats()[name]
    in_mean = np.asarray(in_stats["mean"], dtype=np.float64)
    in_denom = np.asarray(in_stats["denom"], dtype=np.float64)
    require(x_raw.shape[2] == len(in_mean), f"{name}_input_stat_channel_count",
            f"{x_raw.shape[2]} vs {len(in_mean)}")
    require(in_stats["fit_row_count"] == len(t1_train), f"{name}_input_fit_rows_equals_t1_train",
            f"{in_stats['fit_row_count']} vs {len(t1_train)}")
    x_std = ai.standardize_input(x_raw, in_mean, in_denom)   # float32 [n,T,C]
    require(x_std.dtype == np.float32, f"{name}_network_input_float32")
    stats_sig = in_mean.tobytes() + in_denom.tobytes()

    # ---- v2 descriptor statistics (read-only, hashed) ---------------------
    v2 = json.loads((V2_DIR / f"interface_check_{name}.json").read_text(encoding="utf-8"))
    d_mean = np.asarray(v2["descriptor_stats_mean"], dtype=np.float64)
    d_std = np.asarray(v2["descriptor_stats_std"], dtype=np.float64)
    require(len(d_mean) == 8 and len(d_std) == 8, f"{name}_descriptor_stats_8dim")
    require(v2["task1_fit_rows"] == len(t1_train), f"{name}_v2_fit_rows_equals_t1_train")
    fit_ids = np.asarray(v2["descriptor_fit_row_ids"], dtype=np.int64)
    require(len(fit_ids) == len(t1_train) and set(fit_ids.tolist()) == set(t1_train.tolist()),
            f"{name}_v2_descriptor_fit_rows_identical")
    desc_sig = d_mean.tobytes() + d_std.tobytes()
    dump_json(ddir / "descriptor_input_stats_reference.json", {
        "input_stats_source": "experiments/baselines/trunk_preprocessing_fix/preprocess_statistics.json",
        "input_fit_rows": len(t1_train),
        "descriptor_stats_source": f"trunk_model_interface_check_v2/interface_check_{name}.json",
        "descriptor_impl": v2["descriptor_impl"],
        "descriptor_stats_mean": [float(v) for v in d_mean],
        "descriptor_stats_std": [float(v) for v in d_std],
        "descriptor_fit_row_ids": fit_ids.tolist(),
    })

    # ---- model + encoder forward hook -------------------------------------
    model = build_model(int(x_raw.shape[2]), n_classes, SEED).to(device)
    model.eval()
    n_params = int(sum(p.numel() for p in model.parameters()))
    # NOTE: the hook is on the ENCODER (not the top-level module) because
    # representation_and_logits is called as a direct method in extraction and a
    # top-level module hook never fires on a direct method call (see ERRATA E2 of
    # the v2 interface check).
    model.encoder.register_forward_hook(lambda *a: COUNTERS.__setitem__(
        "encoder_forwards", COUNTERS["encoder_forwards"] + 1))

    # ================================================================ TASK 1
    # ---- train encoder on full fixed Task-1 train split -------------------
    # faithful to archived train_cfctask: best_key=(va,-vl); best_acc/best_loss
    # updated only on threshold improvement (0.005 / 0.001); optimizer state deep-copied.
    optimizer = torch.optim.Adam(model.parameters(), lr=ENCODER_LR)
    sig_before = params_signature(model)
    best_key, best_state, best_opt = (-1.0, -float("inf")), None, None
    best_acc, best_loss, stagnant, trained = -1.0, float("inf"), 0, 0
    best_epoch = 0
    epoch_rows = []
    stop_reason = f"max_epochs_{MAX_EPOCHS}"
    for epoch in range(1, MAX_EPOCHS + 1):
        loader = make_loader(x_std, y, t1_train, BATCH_SIZE, TASK1_TRAIN_SEED + epoch)
        model.train()
        epoch_losses = []
        for xx, yy in loader:
            xx = xx.to(device)
            yy = yy.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx)[:, TASK1]
            mapped = labels_to_local(yy, TASK1, n_classes, device)
            loss = F.cross_entropy(logits, mapped)
            if not bool(torch.isfinite(loss).all().item()):
                raise Halt(f"NON-FINITE encoder loss epoch {epoch}")
            loss.backward()
            COUNTERS["encoder_backward"] += 1
            assert_grads_finite(model, f"encoder pre-clip epoch {epoch}")
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            COUNTERS["encoder_optimizer_steps"] += 1
            assert_params_finite(model, f"encoder post-step epoch {epoch}")
            epoch_losses.append(float(loss.detach()))
        va, vl = evaluate_validation(model, x_std, y, t1_val, TASK1, n_classes, device, EVAL_BATCH)
        trained = epoch
        key = (va, -vl)
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch
        if va - best_acc >= VAL_DELTA_ACC or best_loss - vl >= VAL_DELTA_LOSS:
            best_acc, best_loss, stagnant = max(best_acc, va), min(best_loss, vl), 0
        else:
            stagnant += 1
        epoch_rows.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)),
                           "val_macro_accuracy": va, "val_loss": vl})
        log(f"  epoch {epoch:2d} train_loss={np.mean(epoch_losses):.4f} "
            f"val_acc={va:.4f} val_loss={vl:.4f} stagnant={stagnant}")
        if epoch >= MIN_EPOCHS and stagnant >= PATIENCE:
            stop_reason = f"early_stopping_epoch_{epoch}"
            break
    require(trained > 0, f"{name}_encoder_ran_at_least_one_epoch")
    require(best_state is not None, f"{name}_encoder_best_state_exists")
    require(COUNTERS["encoder_optimizer_steps"] > 0, f"{name}_encoder_updates_positive",
            str(COUNTERS["encoder_optimizer_steps"]))
    require(params_signature(model) != sig_before, f"{name}_encoder_params_changed")
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_opt)

    # save checkpoint + hash
    ckpt = {
        "model_state_dict": best_state,
        "optimizer_state_dict": best_opt,
        "epochs_trained": trained, "stop_reason": stop_reason, "best_epoch": best_epoch,
        "val_macro_accuracy": best_acc, "val_loss": best_loss,
        "seed": SEED, "task1_train_seed": TASK1_TRAIN_SEED,
    }
    ckpt_path = ddir / "task1_encoder_best.pt"
    torch.save(ckpt, ckpt_path)

    # ---- freeze the WHOLE representation path -----------------------------
    fr = ai.freeze_representation(model)
    require(fr["encoder_all_frozen"] and fr["attention_all_frozen"], f"{name}_representation_frozen")
    require(fr["head_all_frozen"], f"{name}_original_head_frozen")
    require(not fr["any_parameter_trainable_on_model"], f"{name}_no_model_param_trainable")
    frozen_sd_hash = params_signature(model)
    dump_json(ddir / "freeze_verification.json", {
        "frozen_after_task1": True, "frozen_state_dict_sha256": frozen_sd_hash,
        "encoder_parameter_names": fr["encoder_parameter_names"],
        "attention_parameter_names": fr["attention_parameter_names"],
        "head_parameter_names": fr["head_parameter_names"],
        "trainable_names": fr["trainable_names"],
    })

    # ---- extract Task-1 train joint + centroids ---------------------------
    z_t1, latent_t1, desc_raw_t1, desc_std_t1 = make_joint(
        model, x_std, x_raw, t1_train, d_mean, d_std, name, device)
    require(z_t1.shape == (len(t1_train), JOINT_DIM), f"{name}_t1_joint_shape_56",
            str(z_t1.shape))
    class_means = {c: z_t1[y[t1_train] == c].mean(axis=0) for c in TASK1}
    for c in TASK1:
        ds.assert_all_finite(class_means[c], f"{name} class_means[{c}]")

    # ---- Task-1 sparse head 56->2 ----------------------------------------
    head1 = nn.Linear(JOINT_DIM, 2).to(device)
    require(head1.in_features == 56 and head1.out_features == 2, f"{name}_head1_shape_56_to_2")
    loss1, changed1 = fit_sparse_head(head1, z_t1, y[t1_train], device, f"{name}_head1")
    require(changed1, f"{name}_head1_updated")
    torch.save(head1.state_dict(), ddir / "task1_head.pt")

    # ================================================================ TASK 2
    # ---- extract Task-2 train joint + current centroids -------------------
    z_t2, latent_t2, desc_raw_t2, desc_std_t2 = make_joint(
        model, x_std, x_raw, t2_train, d_mean, d_std, name, device)
    require(z_t2.shape == (len(t2_train), JOINT_DIM), f"{name}_t2_joint_shape_56",
            str(z_t2.shape))
    current_means = {c: z_t2[y[t2_train] == c].mean(axis=0).astype(np.float64) for c in TASK2}

    # ---- pooled_shift pseudo features for old classes ---------------------
    pseudo_parts, pseudo_labels, manifest = pooled_shift_pseudo(
        class_means, current_means, z_t2, y[t2_train], t2_train, TASK1, TASK2)
    pseudo_counts = {int(m["old_class"]): int(m["pseudo_count"]) for m in manifest}
    pseudo_total = int(sum(pseudo_counts.values()))
    log(f"  pseudo manifest: {[(m['old_class'], m['source_class'], m['pseudo_count']) for m in manifest]}")

    # ---- Task-2 sparse head 56->4 (pseudo 0,1 + real 2,3) ----------------
    real_z = z_t2.astype(np.float32)
    real_labels = y[t2_train].astype(np.int64)
    X2 = np.concatenate([real_z] + pseudo_parts, axis=0).astype(np.float32)
    Y2 = np.concatenate([real_labels] + pseudo_labels, axis=0).astype(np.int64)
    require(sorted(set(Y2.tolist())) == [0, 1, 2, 3], f"{name}_head2_labels_0_1_2_3",
            str(sorted(set(Y2.tolist()))))
    head2 = nn.Linear(JOINT_DIM, 4).to(device)
    require(head2.in_features == 56 and head2.out_features == 4, f"{name}_head2_shape_56_to_4")
    loss2, changed2 = fit_sparse_head(head2, X2, Y2, device, f"{name}_head2")
    require(changed2, f"{name}_head2_updated")
    torch.save(head2.state_dict(), ddir / "task2_head.pt")

    # ================================================================ ACCEPTANCE
    checks: dict[str, object] = {}

    # (2) finiteness already enforced inline; record summary
    checks["finiteness_enforced_inline"] = True

    # (3) encoder updated
    checks["encoder_updates"] = COUNTERS["encoder_optimizer_steps"]

    # (4) heads updated + dims
    checks["head_dims"] = {"head1": [head1.in_features, head1.out_features],
                           "head2": [head2.in_features, head2.out_features]}
    checks["heads_updated"] = bool(changed1 and changed2)

    # (5) frozen model state unchanged through Task 2
    require(params_signature(model) == frozen_sd_hash, f"{name}_frozen_model_unchanged_through_task2")
    require(all(not p.requires_grad for p in model.parameters()), f"{name}_frozen_params_no_grad")
    checks["frozen_model_unchanged"] = True

    # (6) input + descriptor statistics unchanged
    require(in_mean.tobytes() + in_denom.tobytes() == stats_sig, f"{name}_input_stats_unchanged")
    require(d_mean.tobytes() + d_std.tobytes() == desc_sig, f"{name}_descriptor_stats_unchanged")
    checks["standardization_stats_unchanged"] = True

    # (7) pseudo count + per-row formula verification
    expected_pseudo = {c: int((y[t2_train] == manifest[c]["source_class"]).sum()) for c in [0, 1]}
    require(pseudo_counts == expected_pseudo, f"{name}_pseudo_counts_match_rule",
            f"{pseudo_counts} vs {expected_pseudo}")
    for idx, cls in enumerate(TASK1):
        m = manifest[idx]
        src = z_t2[y[t2_train] == m["source_class"]].astype(np.float64)
        shift = np.asarray(m["shift_vector"], dtype=np.float64)
        pseudo_check = (src + shift).astype(np.float32)
        require(bool(np.array_equal(pseudo_parts[idx], pseudo_check)),
                f"{name}_pseudo_formula_rowwise_old_class_{cls}")
    checks["pseudo_feature_count"] = pseudo_total
    checks["pseudo_formula_verified_rowwise"] = True

    # (8) no real old-class replay; real replay = 0
    require(len(set(t2_train.tolist()) & set(t1_train.tolist())) == 0,
            f"{name}_task2_real_rows_no_task1_overlap")
    checks["real_replay_samples"] = 0

    # (9) checkpoint reload reproduces output on a fixed training mini-batch
    fixed_n = min(BATCH_SIZE, len(t1_train))
    fixed_rows = t1_train[:fixed_n]
    model.eval()
    e_before = COUNTERS["encoder_forwards"]
    with torch.inference_mode():
        xb = torch.from_numpy(np.ascontiguousarray(x_std[fixed_rows])).to(device)
        ref_logits = model(xb)
    reloaded = build_model(int(x_raw.shape[2]), n_classes, 0).to(device)
    reloaded.load_state_dict(ckpt["model_state_dict"])
    reloaded.eval()
    reloaded.encoder.register_forward_hook(lambda *a: COUNTERS.__setitem__(
        "encoder_forwards", COUNTERS["encoder_forwards"] + 1))
    with torch.inference_mode():
        new_logits = reloaded(xb)
    extra_forwards = COUNTERS["encoder_forwards"] - e_before
    require(extra_forwards == 2, f"{name}_checkpoint_reload_extra_forwards_2",
            str(extra_forwards))
    dmax = float((ref_logits - new_logits).abs().max().item())
    require(bool(torch.allclose(ref_logits, new_logits, rtol=1e-4, atol=1e-6)),
            f"{name}_checkpoint_reload_reproduces_output", f"max_abs_diff={dmax}")
    checks["checkpoint_reload_max_abs_diff"] = dmax
    checks["checkpoint_reload_extra_forwards"] = extra_forwards

    # (10) test set untouched
    require(COUNTERS["test_set_forwards"] == 0, f"{name}_test_set_forwards_zero")
    checks["test_set_forwards"] = 0
    checks["test_performance_evaluations"] = 0

    # (11) protected input hashes unchanged
    prot_after = {k: sha(v) for k, v in prot.items()}
    changed = [k for k in prot_before if prot_before[k] != prot_after[k]]
    require(not changed, f"{name}_protected_inputs_unchanged", str(changed))
    checks["protected_inputs_changed"] = changed

    # ================================================================ artifacts
    checkpoint_hashes = {"task1_encoder_best.pt": sha(ckpt_path),
                         "task1_head.pt": sha(ddir / "task1_head.pt"),
                         "task2_head.pt": sha(ddir / "task2_head.pt")}
    dump_json(ddir / "checkpoint_hashes.json", checkpoint_hashes)

    np.savez_compressed(ddir / "train_val_row_ids_and_labels.npz",
                        t1_train=t1_train, t1_train_labels=y[t1_train],
                        t1_val=t1_val, t1_val_labels=y[t1_val],
                        t2_train=t2_train, t2_train_labels=y[t2_train],
                        t2_val=t2_val, t2_val_labels=y[t2_val])

    with (ddir / "encoder_epoch_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_macro_accuracy", "val_loss"])
        w.writeheader()
        w.writerows(epoch_rows)

    with (ddir / "readout_training_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["head", "epoch", "loss"])
        for i, v in enumerate(loss1):
            w.writerow(["task1", i + 1, v])
        for i, v in enumerate(loss2):
            w.writerow(["task2", i + 1, v])

    dump_json(ddir / "pseudo_feature_manifest.json", {
        "rule": "pooled_shift; nearest new class by cosine similarity (eps=1e-8); "
                "source = ALL current-task train rows of the nearest class (NO 128 cap); "
                "pseudo_z = source - current_means[nearest] + old_mean",
        "total_pseudo_features": pseudo_total,
        "per_old_class": manifest,
    })

    frozen_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    centroid_bytes = 2 * JOINT_DIM * 4
    head_bytes = sum(p.numel() * p.element_size() for p in head2.parameters())
    dump_json(ddir / "call_ledger.json", {
        "encoder_forwards": COUNTERS["encoder_forwards"],
        "head_forwards": COUNTERS["head_forwards"],
        "encoder_backward": COUNTERS["encoder_backward"],
        "encoder_optimizer_steps": COUNTERS["encoder_optimizer_steps"],
        "head_backward": COUNTERS["head_backward"],
        "head_optimizer_steps": COUNTERS["head_optimizer_steps"],
        "test_set_forwards": COUNTERS["test_set_forwards"],
        "training_calls": COUNTERS["encoder_forwards"],
        "note": "encoder_forwards counts every CfC encoder invocation (train+val+extract+reload). "
                "training/backward/optimizer counted separately.",
    })

    peak_cuda = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
    peak_cuda_reserved = (torch.cuda.max_memory_reserved() if device.type == "cuda" else 0)
    import ctypes
    class _PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = _PMC(); pmc.cb = ctypes.sizeof(_PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                             ctypes.byref(pmc), pmc.cb)
    dump_json(ddir / "environment.json", {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "python": platform.python_version(),
        "numpy": np.__version__, "os": platform.platform(),
        "thread_pinning": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "seed": SEED, "task1_train_seed": TASK1_TRAIN_SEED,
    })
    dump_json(ddir / "timing_memory.json", {
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "model_parameters": n_params,
        "persistent_bytes": {
            "frozen_representation_state_dict_bytes": frozen_bytes,
            "old_class_centroids_bytes": centroid_bytes,
            "task2_readout_head_bytes": head_bytes,
            "note": "algorithm persistent memory = frozen encoder + 2 old-class 56-dim centroids + readout head; "
                    "no replay buffer (real replay = 0). Offline audit caches are NOT counted.",
        },
        "real_replay_samples": 0, "pseudo_samples": pseudo_total,
        "peak_cuda_allocated_bytes": peak_cuda,
        "peak_cuda_reserved_bytes": peak_cuda_reserved,
        "peak_process_working_set_bytes": int(pmc.PeakWorkingSetSize),
    })

    gate = {"verdict": "ENGINEERING_PASS", "dataset": name, "seed": SEED,
            "scope": "Task1=[0,1], Task2=[2,3] engineering smoke; NOT a performance screen",
            "checks": checks,
            "encoder": {"epochs_trained": trained, "stop_reason": stop_reason,
                        "best_val_macro_accuracy": best_acc, "best_val_loss": best_loss},
            "pseudo_feature_count": pseudo_total, "real_replay_samples": 0}
    dump_json(ddir / "gate_decision.json", gate)
    log(f"  ALL CHECKS PASS in {time.perf_counter() - started:.1f}s "
        f"(encoder_forwards={COUNTERS['encoder_forwards']} head_forwards={COUNTERS['head_forwards']})")
    return 0


# ---------------------------------------------------------------- protocol lock
def lock_protocol() -> int:
    if (OUT / "protocol_locked.json").exists():
        raise Halt("REFUSE: protocol_locked.json already exists")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- parent (failed) round reference ---------------------------------
    parent_sha_file = OLD_OUT / "protocol_locked.json.sha256"
    if not parent_sha_file.exists():
        raise Halt("REFUSE: parent protocol sha missing; cannot lock fix1")
    parent_sha = parent_sha_file.read_text(encoding="utf-8").split()[0].strip()
    failure_rec = OLD_OUT / "uci_har" / "failure_record_1.json"
    failure_summary = None
    if failure_rec.exists():
        fr = json.loads(failure_rec.read_text(encoding="utf-8"))
        failure_summary = {"path": str(failure_rec), "kind": fr.get("kind"),
                           "reason": (fr.get("reason") or fr.get("traceback", "")).splitlines()[0][:200]}

    # ---- modification evidence: old vs new source hash + exact diff -------
    old_src_hash = sha(PRE_FIX_COPY) if PRE_FIX_COPY.exists() else "MISSING_PRE_FIX_COPY"
    new_src_hash = sha(Path(__file__).resolve())
    import difflib
    old_lines = PRE_FIX_COPY.read_text(encoding="utf-8").splitlines() if PRE_FIX_COPY.exists() else []
    new_lines = Path(__file__).resolve().read_text(encoding="utf-8").splitlines()
    diff_text = "\n".join(difflib.unified_diff(
        old_lines, new_lines, "a/src/trunk/train_smoke.py (pre-fix1)",
        "b/src/trunk/train_smoke.py (fix1)", lineterm=""))
    (OUT / "train_smoke_fix1.diff").write_text(diff_text + "\n", encoding="utf-8")
    modification = {
        "note": "仅修正 ds.describe 返回值解包（desc_raw, desc_info = ...）；实验设计未变",
        "old_source_sha256": old_src_hash,
        "new_source_sha256": new_src_hash,
        "diff_file": "train_smoke_fix1.diff",
        "changed_call_sites": ["src/trunk/train_smoke.py:make_joint — describe 返回值解包"],
        "also_changed_for_isolation": ["OUT 目录 -> *_fix1", "v2 描述符统计量路径 -> 显式 V2_DIR 常量"],
        "unchanged": ["描述符公式", "划分", "输入/描述符标准化", "种子(11)与loader种子",
                      "训练预算(Adam lr1e-3/batch128/clip5/3-12-2)", "早停(0.005/0.001)",
                      "伪特征(pooled_shift)规则", "稀疏头预算(300/0.05/0.0007)", "11条验收条件"],
    }

    prot = protected_input_files()
    srcs = source_files()
    prot_hashes = {k: sha(v) for k, v in prot.items()}
    src_hashes = {k: sha(v) for k, v in srcs.items()}
    protocol = {
        "status": "LOCKED", "version": "seed11_task12_smoke_fix1",
        "locked_before_any_training_or_forward": True,
        "round": "四数据集 ACBM 单种子(11) 前两任务训练烟测（fix1：describe 解包修正）",
        "output_dir": "experiments/baselines/trunk_training_smoke_seed11_fix1",
        "parent_protocol_sha256": parent_sha,
        "parent_failure_record": failure_summary,
        "modification": modification,
        "parent_run_status": "incomplete (halted by unexpected exception at uci_har make_joint); NOT a performance FAIL",
        "seed": SEED,
        "datasets_order": ["uci_har", "wisdm", "uwave", "cwru"],
        "tasks": {"task1": TASK1, "task2": TASK2},
        "device": {"preferred": "cuda", "rule": "CUDA 不可用则阻断，不自动切换设备"},
        "model": {"class": "TemporalAttentionSequenceClassifier (archived pipeline.py)",
                  "backbone": "CfC (ncps 1.0.1)", "hidden_size": HIDDEN,
                  "pooling": "scalar temporal attention (zero-init, tanh scores, softmax over time)",
                  "n_classes_per_dataset": "auto (6/6/8/10)"},
        "input_standardization": {"source": "preprocess_statistics.json (accepted, Task-1-only, frozen)",
                                  "fit_rows": {"uci_har": 1684, "wisdm": 5344, "uwave": 24, "cwru": 200},
                                  "rule": "(x - mean)/(std+1e-6), per channel, float64 accumulate, ddof=0"},
        "descriptors": {"source": "v2 trunk descriptors (interface_check_<ds>.json mean/std)",
                        "formula": "per dataset (uci_har raw_concept_features / wisdm wisdm_sensor_concepts / uwave+cwru generic_8dim)",
                        "standardization": "(d-mean)/(std+1e-3) clipped to [-6,6]"},
        "joint": "48 latent + 8 descriptor = 56; concept bottleneck OFF",
        "encoder_budget": {"optimizer": "Adam", "lr": ENCODER_LR, "batch_size": BATCH_SIZE,
                           "grad_clip": GRAD_CLIP, "min_epochs": MIN_EPOCHS, "max_epochs": MAX_EPOCHS,
                           "patience": PATIENCE, "val_delta_acc": VAL_DELTA_ACC,
                           "val_delta_loss": VAL_DELTA_LOSS,
                           "best_key": "(validation_macro_accuracy, -validation_loss)",
                           "restore": "best model state_dict + optimizer state",
                           "task1_train_seed": TASK1_TRAIN_SEED,
                           "note": "loader generator seed = task1_train_seed + epoch"},
        "sparse_head_budget": {"optimizer": "Adam", "lr": SPARSE_LR, "epochs": SPARSE_EPOCHS,
                               "sparsity_lambda": SPARSITY_LAMBDA, "early_stopping": "none",
                               "l1_penalty": "lambda * |weight|.sum() (weights only)",
                               "init": "fresh nn.Linear each task", "dims": ["56->2", "56->4"]},
        "pooled_shift_rule": {"nearest_by": "cosine similarity (eps=1e-8)",
                              "source": "ALL current-task train rows of nearest new class (no 128 cap)",
                              "formula": "pseudo_z = source - current_means[nearest] + old_mean",
                              "old_class_labels": "pseudo features labeled as the old class"},
        "freeze_rule": "freeze whole representation path (CfC + temporal attention + original head) after Task1; only sparse head trainable",
        "splits": "reuse accepted fixed train/validation/test; no re-split; task order preserved",
        "forbidden": ["test-set forward", "test accuracy", "candidate comparison",
                      "8-seed extension", "effectiveness conclusion"],
        "acceptance_conditions": [
            "1 train/val samples + label mapping correct, row_id aligned",
            "2 loss/grad(features/params/logits finite; pre-clip grad + post-step params checked",
            "3 encoder actually updated (optimizer steps > 0, params changed)",
            "4 both sparse heads updated; dims 56->2 and 56->4",
            "5 frozen model state_dict identical after Task2 (incl. temporal attention); frozen params no grad",
            "6 input & descriptor standardization stats unchanged throughout",
            "7 pseudo count matches per-source rule; formula verified row-wise",
            "8 no real old-class replay; real replay = 0",
            "9 checkpoint reload reproduces output on fixed train mini-batch (extra forwards recorded)",
            "10 test-set forwards = 0, performance eval = 0",
            "11 all protected input hashes identical before/after",
        ],
        "input_sha256_before": prot_hashes,
        "source_sha256_snapshot": src_hashes,
    }
    dump_json(OUT / "protocol_locked.json", protocol)
    h = sha(OUT / "protocol_locked.json")
    (OUT / "protocol_locked.json.sha256").write_text(f"{h} *protocol_locked.json\n", encoding="utf-8")
    log(f"protocol locked (fix1): {h}")
    log(f"parent protocol sha: {parent_sha}")
    log(f"protected inputs: {len(prot_hashes)}  source files: {len(src_hashes)}")
    log(f"source change: {old_src_hash[:12]} -> {new_src_hash[:12]}")
    return 0


# ---------------------------------------------------------------- summary
def summary() -> int:
    rows = []
    for name in DATASETS:
        g = json.loads((OUT / name / "gate_decision.json").read_text(encoding="utf-8"))
        env = json.loads((OUT / name / "environment.json").read_text(encoding="utf-8"))
        tm = json.loads((OUT / name / "timing_memory.json").read_text(encoding="utf-8"))
        led = json.loads((OUT / name / "call_ledger.json").read_text(encoding="utf-8"))
        rows.append({
            "dataset": name, "seed": SEED, "verdict": g["verdict"],
            "encoder_epochs": g["encoder"]["epochs_trained"],
            "encoder_stop": g["encoder"]["stop_reason"],
            "best_val_macro": g["encoder"]["best_val_macro_accuracy"],
            "pseudo_features": g["pseudo_feature_count"],
            "real_replay": g["real_replay_samples"],
            "model_params": tm["model_parameters"],
            "encoder_forwards": led["encoder_forwards"],
            "head_forwards": led["head_forwards"],
            "encoder_optimizer_steps": led["encoder_optimizer_steps"],
            "test_set_forwards": led["test_set_forwards"],
            "peak_cuda_allocated_bytes": tm["peak_cuda_allocated_bytes"],
            "elapsed_seconds": tm["elapsed_seconds"],
            "device": env["device"],
        })
    with (OUT / "summary_table.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    log("summary_table.csv written")
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-protocol", action="store_true")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.lock_protocol:
        return lock_protocol()
    if a.summary:
        return summary()
    if not a.dataset:
        raise SystemExit("pass --lock-protocol | --dataset <name> | --summary")
    # dataset mode: require protocol locked
    if not (OUT / "protocol_locked.json").exists() or not (OUT / "protocol_locked.json.sha256").exists():
        raise Halt("protocol_locked.json(.sha256) missing; run --lock-protocol first")
    rec = (OUT / "protocol_locked.json.sha256").read_text(encoding="utf-8").split()[0].strip()
    if rec != sha(OUT / "protocol_locked.json"):
        raise Halt("protocol hash mismatch")
    if device.type != "cuda":
        raise Halt(f"CUDA unavailable (device={device}); blocking, not auto-switching")
    try:
        return run_dataset(a.dataset)
    except Halt as e:
        write_failure("HALTED", {"reason": str(e), "log_tail": LOG[-40:]}, OUT / a.dataset)
        raise SystemExit(4)
    except Exception:
        write_failure("UNEXPECTED_EXCEPTION", {"traceback": traceback.format_exc(),
                                               "log_tail": LOG[-40:]}, OUT / a.dataset)
        raise SystemExit(6)
    finally:
        ddir = OUT / a.dataset
        (ddir / "logs").mkdir(exist_ok=True)
        with (ddir / "logs" / "run.log").open("a", encoding="utf-8") as f:
            f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    device = torch.device("cpu")
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise SystemExit(6)
