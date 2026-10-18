"""Disk checkpoint recovery + persistent-state completion for the seed11 smoke test.

Zero training, zero backward, zero optimizer.step, zero performance evaluation.
This round ONLY verifies that the existing on-disk checkpoints can recover the full
inference flow, and completes the Task-2 boundary persistent state (4 class centroids).

Key facts recorded in the protocol:
  1. this round verifies "existing disk checkpoints recover the full inference flow";
  2. the original smoke test did NOT save full small-batch outputs, so we CANNOT claim
     to reproduce a non-existent old run output;
  3. two independent processes (A then B) recover from the SAME on-disk files and check
     that NEWLY generated reference outputs agree;
  4. method / split / standardization / seed / budget / verdict rules are unchanged.

Usage:
    python src/trunk/recover_checkpoint.py --lock-protocol
    python src/trunk/recover_checkpoint.py --addendum
    python src/trunk/recover_checkpoint.py --process-a uci_har
    python src/trunk/recover_checkpoint.py --process-b uci_har
    python src/trunk/recover_checkpoint.py --finalize
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.dont_write_bytecode = True

import argparse
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
OUT = ROOT / "experiments/baselines/trunk_checkpoint_recovery_seed11"
FIX1_DIR = ROOT / "experiments/baselines/trunk_training_smoke_seed11_fix1"
V2_DIR = ROOT / "experiments/baselines/trunk_model_interface_check_v2"
for p in (str(ROOT / "src"), str(ROOT), str(ROOT / ".local_packages")):
    if p not in sys.path:
        sys.path.insert(0, p)

from trunk import train_smoke as ts            # noqa: E402  (reuse proven numeric path)
from trunk import acbm_interface as ai         # noqa: E402
from trunk import adapters as ad               # noqa: E402
from trunk import descriptors as ds            # noqa: E402

DATASETS = ["uci_har", "wisdm", "uwave", "cwru"]
SEEN = [0, 1, 2, 3]
TASK1 = [0, 1]
TASK2 = [2, 3]
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
    f["stats/preprocess_statistics.json"] = ad.FIX_DIR / "preprocess_statistics.json"
    f["stats/fit_indices.npz"] = ad.FIX_DIR / "fit_indices.npz"
    for ds_ in DATASETS:
        f[f"v2/interface_check_{ds_}.json"] = V2_DIR / f"interface_check_{ds_}.json"
    # raw data
    f["raw/wisdm.zip"] = ROOT / "data/wisdm.zip"
    f["raw/wisdm_cache.npz"] = ROOT / "data/wisdm_har_windows.npz"
    f["raw/uwave_train.ts"] = ROOT / "data/UWaveGestureLibrary_TRAIN.ts"
    f["raw/uwave_test.ts"] = ROOT / "data/UWaveGestureLibrary_TEST.ts"
    f["raw/uwave_cache.npz"] = ROOT / "data/uwave_gesture_library.npz"
    f["raw/cwru_manifest.json"] = ROOT / "data/cwru_lttsnet_seed10.manifest.json"
    f["raw/uci_train_y.txt"] = ROOT / "data/UCI HAR Dataset/train/y_train.txt"
    f["raw/uci_test_y.txt"] = ROOT / "data/UCI HAR Dataset/test/y_test.txt"
    for n in ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y",
              "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]:
        for split in ("train", "test"):
            f[f"raw/uci_{split}_{n}.txt"] = ROOT / f"data/UCI HAR Dataset/{split}/Inertial Signals/{n}_{split}.txt"
    for m in sorted((ROOT / "data/cwru_lttsnet_raw").glob("*.mat")):
        f[f"raw/cwru_{m.name}"] = m
    # fix1 artifacts (checkpoints, manifest, logs)
    for ds_ in DATASETS:
        d = FIX1_DIR / ds_
        for fn in ["task1_encoder_best.pt", "task1_head.pt", "task2_head.pt",
                   "pseudo_feature_manifest.json", "encoder_epoch_log.csv",
                   "readout_training_log.csv", "train_val_row_ids_and_labels.npz",
                   "descriptor_input_stats_reference.json", "freeze_verification.json",
                   "checkpoint_hashes.json", "gate_decision.json", "environment.json",
                   "timing_memory.json", "call_ledger.json"]:
            f[f"fix1/{ds_}/{fn}"] = d / fn
    f["fix1/protocol_locked.json"] = FIX1_DIR / "protocol_locked.json"
    return f


# ---------------------------------------------------------------- data context (faithful)
def load_context(name: str) -> dict:
    raw = ad.adapt(name)
    x_raw = np.asarray(raw["x"])
    y = np.asarray(raw["y"])
    n_classes = int(len(set(y.tolist())))
    if name == "cwru":
        splits = np.asarray([p["split"] for p in raw["provenance"]])
        train_all = np.flatnonzero(splits == "train")
        val_all = np.flatnonzero(splits == "validation")
    else:
        sp = ad.accepted_splits()[name]
        train_all = np.asarray(sp["train"])
        val_all = np.asarray(sp["validation"])

    def task_rows(classes):
        tr = np.asarray(sorted(int(i) for i in train_all if int(y[i]) in classes), dtype=np.int64)
        va = np.asarray(sorted(int(i) for i in val_all if int(y[i]) in classes), dtype=np.int64)
        return tr, va

    t1_train, t1_val = task_rows(TASK1)
    t2_train, t2_val = task_rows(TASK2)
    in_stats = ad.accepted_input_stats()[name]
    in_mean = np.asarray(in_stats["mean"], dtype=np.float64)
    in_denom = np.asarray(in_stats["denom"], dtype=np.float64)
    x_std = ai.standardize_input(x_raw, in_mean, in_denom)
    v2 = json.loads((V2_DIR / f"interface_check_{name}.json").read_text(encoding="utf-8"))
    d_mean = np.asarray(v2["descriptor_stats_mean"], dtype=np.float64)
    d_std = np.asarray(v2["descriptor_stats_std"], dtype=np.float64)
    return dict(name=name, x_raw=x_raw, y=y, x_std=x_std, t1_train=t1_train, t1_val=t1_val,
                t2_train=t2_train, t2_val=t2_val, d_mean=d_mean, d_std=d_std,
                n_classes=n_classes, input_channels=int(x_raw.shape[2]),
                in_mean=in_mean, in_denom=in_denom)


def reference_rows(ctx: dict) -> np.ndarray:
    rows: list[int] = []
    for c in TASK1:
        idx = ctx["t1_train"][ctx["y"][ctx["t1_train"]] == c][:2]
        rows.extend(int(i) for i in idx)
    for c in TASK2:
        idx = ctx["t2_train"][ctx["y"][ctx["t2_train"]] == c][:2]
        rows.extend(int(i) for i in idx)
    return np.asarray(rows, dtype=np.int64)


# ---------------------------------------------------------------- metadata addendum
def addendum() -> int:
    for name in DATASETS:
        d = FIX1_DIR / name
        ckpt = torch.load(d / "task1_encoder_best.pt", map_location="cpu", weights_only=False)
        rows = list(csv.DictReader((d / "encoder_epoch_log.csv").open(encoding="utf-8-sig")))
        key = lambda r: (float(r["val_macro_accuracy"]), -float(r["val_loss"]))
        selected = max(rows, key=key)
        selected_epoch = int(selected["epoch"])
        require(selected_epoch == int(ckpt["best_epoch"]), f"{name}_selected_epoch_matches_ckpt_best_epoch",
                f"selected={selected_epoch} ckpt_best_epoch={ckpt['best_epoch']}")
        out = {
            "dataset": name,
            "selection_rule": "selected_epoch = argmax(validation_macro_accuracy, -validation_loss)",
            "selected_epoch": selected_epoch,
            "selected_checkpoint_metrics": {
                "validation_macro_accuracy": float(selected["val_macro_accuracy"]),
                "validation_loss": float(selected["val_loss"]),
            },
            "early_stopping_tracker": {
                "best_acc_threshold_tracker": float(ckpt["val_macro_accuracy"]),
                "best_loss_threshold_tracker": float(ckpt["val_loss"]),
                "note": "ckpt 的 val_macro_accuracy / val_loss 字段来自早停阈值跟踪器 "
                        "(max(best_acc,va) / min(best_loss,vl) 按 0.005/0.001 阈值门控)，"
                        "不是 argmax(va,-vl) 选中 epoch 的指标；两者不应混称“最佳检查点指标”。",
            },
            "conflation_flag": bool(abs(float(selected["val_loss"]) - float(ckpt["val_loss"])) > 1e-12
                                    or abs(float(selected["val_macro_accuracy"]) - float(ckpt["val_macro_accuracy"])) > 1e-12),
            "checkpoint_file_unchanged": True,
        }
        p = OUT / name / "checkpoint_metadata_addendum.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise Halt(f"REFUSE: {p} already exists")
        dump_json(p, out)
        log(f"  {name}: selected_epoch={selected_epoch} "
            f"selected_va={selected['val_macro_accuracy']} selected_vl={selected['val_loss']} "
            f"| tracker_vl={ckpt['val_loss']} conflation={out['conflation_flag']}")
    return 0


# ---------------------------------------------------------------- build + load helpers
def build_and_load(ctx: dict, device):
    ckpt = torch.load(FIX1_DIR / ctx["name"] / "task1_encoder_best.pt", map_location="cpu",
                      weights_only=False)
    head1_sd = torch.load(FIX1_DIR / ctx["name"] / "task1_head.pt", map_location="cpu",
                          weights_only=False)
    head2_sd = torch.load(FIX1_DIR / ctx["name"] / "task2_head.pt", map_location="cpu",
                          weights_only=False)
    model = ts.build_model(ctx["input_channels"], ctx["n_classes"], ts.SEED).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    require(not missing and not unexpected, f"{ctx['name']}_model_keys_exact",
            f"missing={missing} unexpected={unexpected}")
    head1 = nn.Linear(ts.JOINT_DIM, 2).to(device)
    head2 = nn.Linear(ts.JOINT_DIM, 4).to(device)
    m1, u1 = head1.load_state_dict(head1_sd, strict=False)
    m2, u2 = head2.load_state_dict(head2_sd, strict=False)
    require(not m1 and not u1, f"{ctx['name']}_head1_keys_exact", f"{m1} {u1}")
    require(not m2 and not u2, f"{ctx['name']}_head2_keys_exact", f"{m2} {u2}")
    require(head1.in_features == 56 and head1.out_features == 2, f"{ctx['name']}_head1_56_to_2")
    require(head2.in_features == 56 and head2.out_features == 4, f"{ctx['name']}_head2_56_to_4")
    return model, head1, head2, ckpt


def freeze_and_check(model, head1, head2, name):
    fr = ai.freeze_representation(model)
    require(fr["encoder_all_frozen"] and fr["attention_all_frozen"] and fr["head_all_frozen"],
            f"{name}_representation_frozen")
    require(not fr["any_parameter_trainable_on_model"], f"{name}_no_model_param_trainable")
    model.eval()
    head1.eval()
    head2.eval()
    model.zero_grad(set_to_none=True)
    head1.zero_grad(set_to_none=True)
    head2.zero_grad(set_to_none=True)
    all_params = list(model.parameters()) + list(head1.parameters()) + list(head2.parameters())
    require(all(p.grad is None for p in all_params), f"{name}_all_grads_none_after_zero_grad")
    return fr


# ---------------------------------------------------------------- process A
def process_a(name: str) -> int:
    ddir = OUT / name
    ddir.mkdir(parents=True, exist_ok=True)
    (ddir / "logs").mkdir(exist_ok=True)
    for k in ("encoder_forwards", "head_forwards"):
        COUNTERS[k] = 0
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "device_is_cuda", str(device))
    log("=" * 78)
    log(f"PROCESS A  {name}  device={device}")
    log("=" * 78)

    ctx = load_context(name)
    model, head1, head2, ckpt = build_and_load(ctx, device)
    model.encoder.register_forward_hook(lambda *a: COUNTERS.__setitem__(
        "encoder_forwards", COUNTERS["encoder_forwards"] + 1))
    fr = freeze_and_check(model, head1, head2, name)

    # ---- full joint extraction (same row order + EVAL_BATCH as smoke) ----
    z_t1, _, _, _ = ts.make_joint(model, ctx["x_std"], ctx["x_raw"], ctx["t1_train"],
                                  ctx["d_mean"], ctx["d_std"], name, device)
    z_t2, _, _, _ = ts.make_joint(model, ctx["x_std"], ctx["x_raw"], ctx["t2_train"],
                                  ctx["d_mean"], ctx["d_std"], name, device)
    require(z_t1.shape == (len(ctx["t1_train"]), ts.JOINT_DIM), f"{name}_z_t1_shape")
    require(z_t2.shape == (len(ctx["t2_train"]), ts.JOINT_DIM), f"{name}_z_t2_shape")

    # ---- 4-class centroids (smoke's actual arithmetic order + dtype) ----
    centroid: dict[int, np.ndarray] = {}
    for c in TASK1:
        centroid[c] = z_t1[ctx["y"][ctx["t1_train"]] == c].mean(axis=0)          # float32
    for c in TASK2:
        centroid[c] = z_t2[ctx["y"][ctx["t2_train"]] == c].mean(axis=0).astype(np.float64)  # float64
    counts = {c: int((ctx["y"][ctx["t1_train"]] == c).sum()) for c in TASK1}
    counts.update({c: int((ctx["y"][ctx["t2_train"]] == c).sum()) for c in TASK2})
    for c in SEEN:
        ds.assert_all_finite(centroid[c], f"{name} centroid[{c}]")

    # ---- verify recovered means vs manifest ----
    manifest = json.loads((FIX1_DIR / name / "pseudo_feature_manifest.json").read_text(encoding="utf-8"))
    ver = []
    for entry in manifest["per_old_class"]:
        cls, src_cls = int(entry["old_class"]), int(entry["source_class"])
        old_mean_rec = centroid[cls].astype(np.float64)
        src_mean_rec = centroid[src_cls].astype(np.float64)
        src_rows = ctx["t2_train"][ctx["y"][ctx["t2_train"]] == src_cls]
        require(bool(np.array_equal(old_mean_rec, np.asarray(entry["old_mean"], dtype=np.float64))),
                f"{name}_old_mean_matches_manifest_class_{cls}")
        require(bool(np.array_equal(src_mean_rec, np.asarray(entry["source_mean"], dtype=np.float64))),
                f"{name}_source_mean_matches_manifest_class_{cls}")
        require(bool(np.array_equal(src_rows, np.asarray(entry["source_row_ids"], dtype=np.int64))),
                f"{name}_source_row_ids_match_class_{cls}")
        require(int(entry["source_count"]) == counts[src_cls], f"{name}_source_count_matches_class_{cls}")
        shift_rec = old_mean_rec - src_mean_rec
        require(bool(np.array_equal(shift_rec, np.asarray(entry["shift_vector"], dtype=np.float64))),
                f"{name}_shift_vector_matches_class_{cls}")
        cos_rec = ts.cosine(old_mean_rec, src_mean_rec)
        require(abs(cos_rec - float(entry["cosine_similarity"])) <= 1e-12,
                f"{name}_cosine_matches_class_{cls}", f"{cos_rec} vs {entry['cosine_similarity']}")
        ver.append({"old_class": cls, "source_class": src_cls, "verified": True})
    log(f"  manifest verification: {[(v['old_class'], v['source_class']) for v in ver]}")

    # ---- 8-row reference outputs ----
    rows8 = reference_rows(ctx)
    require(len(rows8) == 8 and len(set(rows8.tolist())) == 8, f"{name}_reference_8_rows")
    labels8 = ctx["y"][rows8].astype(np.int64)
    require(bool(np.array_equal(labels8, np.array([0, 0, 1, 1, 2, 2, 3, 3]))),
            f"{name}_reference_labels_0_0_1_1_2_2_3_3", str(labels8.tolist()))
    joint8, latent8, desc_raw8, desc_std8 = ts.make_joint(
        model, ctx["x_std"], ctx["x_raw"], rows8, ctx["d_mean"], ctx["d_std"], name, device)
    with torch.inference_mode():
        t2l = head2(torch.from_numpy(joint8).to(device)).cpu().numpy().astype(np.float32)
        COUNTERS["head_forwards"] += 1
        t1l = head1(torch.from_numpy(joint8[:4]).to(device)).cpu().numpy().astype(np.float32)
        COUNTERS["head_forwards"] += 1
    x_std8 = np.ascontiguousarray(ctx["x_std"][rows8]).astype(np.float32)
    raw_sha8 = [hashlib.sha256(np.ascontiguousarray(ctx["x_raw"][r]).tobytes()).hexdigest()
                for r in rows8]

    np.savez_compressed(ddir / "reference_outputs.npz",
                        row_ids=rows8, labels=labels8, x_std=x_std8,
                        latent=latent8, desc_raw=desc_raw8, desc_std=desc_std8,
                        joint=joint8, task2_logits=t2l,
                        task1_row_ids=rows8[:4], task1_logits=t1l)
    dump_json(ddir / "reference_outputs_meta.json", {
        "row_ids": rows8.tolist(), "labels": labels8.tolist(),
        "raw_window_sha256": raw_sha8,
        "note": "原始输入以 row_id + raw_window_sha256 引用；标准化输入 x_std 由 process A 从 raw+输入统计量生成",
    })

    # ---- persistent state package ----
    np.savez_compressed(ddir / "recovered_state_package.npz",
                        centroid_0=centroid[0], centroid_1=centroid[1],
                        centroid_2=centroid[2], centroid_3=centroid[3],
                        class_counts=np.asarray([counts[c] for c in SEEN], dtype=np.int64),
                        seen_classes=np.asarray(SEEN, dtype=np.int64),
                        task1=np.asarray(TASK1, dtype=np.int64),
                        task2=np.asarray(TASK2, dtype=np.int64))
    dump_json(ddir / "state_package_manifest.json", {
        "dataset": name,
        "seen_classes": SEEN, "task1": TASK1, "task2": TASK2,
        "centroid_dtype": {"0": str(centroid[0].dtype), "1": str(centroid[1].dtype),
                           "2": str(centroid[2].dtype), "3": str(centroid[3].dtype)},
        "class_real_train_counts": counts,
        "frozen_model": {"file": f"{FIX1_DIR / name / 'task1_encoder_best.pt'}",
                         "sha256": sha(FIX1_DIR / name / "task1_encoder_best.pt")},
        "task2_head": {"file": f"{FIX1_DIR / name / 'task2_head.pt'}",
                       "sha256": sha(FIX1_DIR / name / "task2_head.pt")},
        "task1_head": {"file": f"{FIX1_DIR / name / 'task1_head.pt'}",
                       "sha256": sha(FIX1_DIR / name / "task1_head.pt")},
        "input_stats_source": "experiments/baselines/trunk_preprocessing_fix/preprocess_statistics.json",
        "input_stats_sha256": sha(ad.FIX_DIR / "preprocess_statistics.json"),
        "descriptor_stats_source": f"trunk_model_interface_check_v2/interface_check_{name}.json",
        "descriptor_stats_sha256": sha(V2_DIR / f"interface_check_{name}.json"),
        "row_id_source": f"{FIX1_DIR / name / 'train_val_row_ids_and_labels.npz'}",
        "row_id_source_sha256": sha(FIX1_DIR / name / "train_val_row_ids_and_labels.npz"),
        "config_version": "seed11_task12_smoke_fix1",
        "pooled_shift_numeric_rule": "smoke 的 pooled_shift 用 float64 中间计算 "
                                     "(source/old_mean/current_mean 均 astype float64) 再 cast float32；"
                                     "与归档 float32 逐位运算不同，不能称逐位相同。",
    })

    # ---- persistent state bytes ----
    frozen_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    frozen_buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    head2_bytes = sum(p.numel() * p.element_size() for p in head2.parameters())
    centroid_bytes = sum(centroid[c].nbytes for c in SEEN)
    count_bytes = 4 * 8
    input_stat_bytes = ctx["in_mean"].nbytes + ctx["in_denom"].nbytes
    desc_stat_bytes = ctx["d_mean"].nbytes + ctx["d_std"].nbytes
    dump_json(ddir / "persistent_state_bytes.json", {
        "algorithm_required_bytes": {
            "frozen_representation_params": frozen_param_bytes,
            "frozen_representation_buffers": frozen_buf_bytes,
            "task2_readout_head": head2_bytes,
            "four_class_centroids": centroid_bytes,
            "class_counts": count_bytes,
            "input_standardization_stats": input_stat_bytes,
            "descriptor_standardization_stats": desc_stat_bytes,
            "total_algorithm_required": (frozen_param_bytes + frozen_buf_bytes + head2_bytes
                                         + centroid_bytes + count_bytes + input_stat_bytes
                                         + desc_stat_bytes),
        },
        "offline_audit_only_not_required": {
            "note": "reference outputs / CSVs / logs are offline audit artifacts, NOT "
                    "algorithm persistent memory",
        },
        "serialized_file_sizes": {
            "task1_encoder_best.pt": (FIX1_DIR / name / "task1_encoder_best.pt").stat().st_size,
            "task1_head.pt": (FIX1_DIR / name / "task1_head.pt").stat().st_size,
            "task2_head.pt": (FIX1_DIR / name / "task2_head.pt").stat().st_size,
            "recovered_state_package.npz": (ddir / "recovered_state_package.npz").stat().st_size,
            "note": "序列化文件大小 ≠ 张量逻辑字节（压缩/容器开销）",
        },
    })

    dump_json(ddir / "call_ledger_process_a.json", {
        "encoder_forwards": COUNTERS["encoder_forwards"],
        "head_forwards": COUNTERS["head_forwards"],
        "backward": 0, "optimizer_steps": 0, "performance_evaluations": 0,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    })
    dump_json(ddir / "gate_decision_process_a.json", {"dataset": name, "stage": "process_a",
                                                      "verdict": "RECOVERY_PASS"})
    log(f"  PROCESS A {name} DONE: encoder_forwards={COUNTERS['encoder_forwards']} "
        f"head_forwards={COUNTERS['head_forwards']}")
    return 0


# ---------------------------------------------------------------- process B
def process_b(name: str) -> int:
    ddir = OUT / name
    require((ddir / "reference_outputs.npz").exists(), f"{name}_reference_outputs_present")
    for k in ("encoder_forwards", "head_forwards"):
        COUNTERS[k] = 0
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "device_is_cuda", str(device))
    log("=" * 78)
    log(f"PROCESS B  {name}  device={device}")
    log("=" * 78)

    ctx = load_context(name)
    model, head1, head2, ckpt = build_and_load(ctx, device)
    model.encoder.register_forward_hook(lambda *a: COUNTERS.__setitem__(
        "encoder_forwards", COUNTERS["encoder_forwards"] + 1))
    freeze_and_check(model, head1, head2, name)

    ref = np.load(ddir / "reference_outputs.npz")
    rows8 = ref["row_ids"]
    labels8 = ref["labels"]
    # verify raw input reference (same data as process A)
    for r in rows8:
        require(np.isfinite(ctx["x_raw"][r]).all(), f"{name}_raw_row_finite_{int(r)}")
    # regenerate standardized input (deterministic)
    x_std8 = np.ascontiguousarray(ctx["x_std"][rows8]).astype(np.float32)
    require(bool(np.array_equal(x_std8, ref["x_std"])), f"{name}_x_std_bitwise_equal")

    joint8, latent8, desc_raw8, desc_std8 = ts.make_joint(
        model, ctx["x_std"], ctx["x_raw"], rows8, ctx["d_mean"], ctx["d_std"], name, device)
    with torch.inference_mode():
        t2l = head2(torch.from_numpy(joint8).to(device)).cpu().numpy().astype(np.float32)
        COUNTERS["head_forwards"] += 1
        t1l = head1(torch.from_numpy(joint8[:4]).to(device)).cpu().numpy().astype(np.float32)
        COUNTERS["head_forwards"] += 1

    def bitwise(a, b, label):
        ok = bool(np.array_equal(np.asarray(a), np.asarray(b)))
        return {"field": label, "bitwise_equal": ok,
                "max_abs_diff": float(np.abs(np.asarray(a, dtype=np.float64)
                                             - np.asarray(b, dtype=np.float64)).max()) if not ok else 0.0}

    comp = [
        bitwise(rows8, ref["row_ids"], "row_ids"),
        bitwise(labels8, ref["labels"], "labels"),
        bitwise(x_std8, ref["x_std"], "x_std"),
        bitwise(latent8, ref["latent"], "latent"),
        bitwise(desc_raw8, ref["desc_raw"], "desc_raw"),
        bitwise(desc_std8, ref["desc_std"], "desc_std"),
        bitwise(joint8, ref["joint"], "joint"),
        bitwise(t1l, ref["task1_logits"], "task1_logits"),
        bitwise(t2l, ref["task2_logits"], "task2_logits"),
    ]
    for c in comp:
        require(c["bitwise_equal"], f"{name}_bitwise_{c['field']}", f"max_abs_diff={c['max_abs_diff']}")
    dump_json(ddir / "comparison_report.json", {"fields": comp, "all_bitwise_equal": True})

    # ---- verify state package recoverability ----
    sp = np.load(ddir / "recovered_state_package.npz")
    for c in SEEN:
        require(f"centroid_{c}" in sp.files, f"{name}_state_pkg_has_centroid_{c}")
        ds.assert_all_finite(sp[f"centroid_{c}"], f"{name} state centroid[{c}]")
        require(sp[f"centroid_{c}"].shape == (ts.JOINT_DIM,), f"{name}_state_centroid_{c}_shape_56")
    require(bool(np.array_equal(sp["seen_classes"], np.array(SEEN))), f"{name}_state_seen_classes")
    require(bool(np.array_equal(sp["class_counts"], np.asarray(
        [int((ctx["y"][ctx["t1_train"]] == c).sum()) for c in TASK1]
        + [int((ctx["y"][ctx["t2_train"]] == c).sum()) for c in TASK2]))), f"{name}_state_class_counts")
    spm = json.loads((ddir / "state_package_manifest.json").read_text(encoding="utf-8"))
    require(spm["frozen_model"]["sha256"] == sha(FIX1_DIR / name / "task1_encoder_best.pt"),
            f"{name}_state_frozen_model_hash")
    require(spm["task2_head"]["sha256"] == sha(FIX1_DIR / name / "task2_head.pt"),
            f"{name}_state_task2_head_hash")
    # no trainable representation params
    require(all(not p.requires_grad for p in model.parameters()), f"{name}_no_trainable_rep_params")

    dump_json(ddir / "call_ledger_process_b.json", {
        "encoder_forwards": COUNTERS["encoder_forwards"],
        "head_forwards": COUNTERS["head_forwards"],
        "backward": 0, "optimizer_steps": 0, "performance_evaluations": 0,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "note": "特征重建(8行)+恢复核验的前向次数；质心从状态包读回，不重算",
    })
    dump_json(ddir / "gate_decision_process_b.json", {"dataset": name, "stage": "process_b",
                                                      "verdict": "RECOVERY_PASS"})
    log(f"  PROCESS B {name} DONE: encoder_forwards={COUNTERS['encoder_forwards']} "
        f"head_forwards={COUNTERS['head_forwards']}")
    return 0


# ---------------------------------------------------------------- lock protocol
def lock_protocol() -> int:
    if (OUT / "protocol_locked.json").exists():
        raise Halt("REFUSE: protocol_locked.json already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    inv = input_files()
    before = {k: sha(v) for k, v in inv.items()}
    protocol = {
        "status": "LOCKED", "version": "checkpoint_recovery_seed11_v1",
        "locked_before_any_model_forward": True,
        "round": "已有烟测结果的磁盘恢复验证与持久状态补齐",
        "output_dir": "experiments/baselines/trunk_checkpoint_recovery_seed11",
        "parent_smoke": "experiments/baselines/trunk_training_smoke_seed11_fix1",
        "seed": ts.SEED,
        "device": {"rule": "CUDA 不可用则阻断，不自动切 CPU", "threads": {k: os.environ.get(k) for k in
                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "claims": [
            "验证现有磁盘检查点能恢复完整推断流程",
            "原烟测未保存完整小批输出，故不声称复现一份不存在的旧运行输出",
            "两个独立进程(A/B)从同一批已保存文件恢复，检查新生成的参考输出一致",
            "不改变方法/划分/标准化/种子/训练预算/判定规则",
        ],
        "scope": {"training": False, "backward": False, "optimizer_step": False,
                  "performance_evaluation": False,
                  "frozen_forward_on_fixed_train_members": "allowed"},
        "forbidden": ["test-set forward", "accuracy/forgetting/pair-diff computation",
                      "Task3", "8-seed"],
        "datasets_order": DATASETS,
        "tasks": {"task1": TASK1, "task2": TASK2, "seen": SEEN},
        "checkpoints_per_dataset": ["task1_encoder_best.pt", "task1_head.pt", "task2_head.pt"],
        "acceptance_conditions": [
            "checkpoint keys exact (no missing/unexpected)",
            "heads 56->2 / 56->4, not random weights",
            "eval + freeze + zero_grad(set_to_none=True); all .grad None",
            "recovered old/source means match pseudo_feature_manifest (source row_id/count/cosine/shift)",
            "4-class centroids + counts saved as numeric artifact",
            "process A/B reference outputs bitwise equal (8 fixed rows)",
            "state package recovers centroids/counts/stats; no trainable rep params",
            "protected input hashes unchanged before/after",
        ],
        "input_sha256_before": before,
        "input_count": len(before),
    }
    dump_json(OUT / "protocol_locked.json", protocol)
    dump_json(OUT / "input_sha256_before.json", before)
    h = sha(OUT / "protocol_locked.json")
    (OUT / "protocol_locked.json.sha256").write_text(f"{h} *protocol_locked.json\n", encoding="utf-8")
    log(f"protocol locked: {h}  inputs={len(before)}")
    return 0


# ---------------------------------------------------------------- finalize
def finalize() -> int:
    inv = input_files()
    after = {k: sha(v) for k, v in inv.items()}
    before = json.loads((OUT / "input_sha256_before.json").read_text(encoding="utf-8"))
    changed = [k for k in before if before.get(k) != after.get(k)]
    require(not changed, "protected_inputs_unchanged", str(changed))
    dump_json(OUT / "input_sha256_after.json", after)
    rows = []
    all_pass = True
    for name in DATASETS:
        ga = json.loads((OUT / name / "gate_decision_process_a.json").read_text(encoding="utf-8"))
        gb = json.loads((OUT / name / "gate_decision_process_b.json").read_text(encoding="utf-8"))
        cmp_ = json.loads((OUT / name / "comparison_report.json").read_text(encoding="utf-8"))
        ok = ga["verdict"] == "RECOVERY_PASS" and gb["verdict"] == "RECOVERY_PASS" \
            and cmp_["all_bitwise_equal"]
        all_pass = all_pass and ok
        la = json.loads((OUT / name / "call_ledger_process_a.json").read_text(encoding="utf-8"))
        lb = json.loads((OUT / name / "call_ledger_process_b.json").read_text(encoding="utf-8"))
        rows.append({"dataset": name, "process_a": ga["verdict"], "process_b": gb["verdict"],
                     "bitwise_equal": cmp_["all_bitwise_equal"],
                     "encoder_fwd_A": la["encoder_forwards"], "encoder_fwd_B": lb["encoder_forwards"],
                     "head_fwd_A": la["head_forwards"], "head_fwd_B": lb["head_forwards"],
                     "result": "RECOVERY_PASS" if ok else "FAIL_ENGINEERING"})
    with (OUT / "recovery_summary_table.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    verdict = "RECOVERY_PASS" if all_pass else "FAIL_ENGINEERING"
    dump_json(OUT / "gate_decision.json", {"verdict": verdict, "four_datasets": all_pass,
                                           "scope": "恢复检查；不是性能 PASS/FAIL"})
    log(f"finalize: verdict={verdict}  changed_inputs={changed}")
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-protocol", action="store_true")
    ap.add_argument("--addendum", action="store_true")
    ap.add_argument("--process-a", choices=DATASETS)
    ap.add_argument("--process-b", choices=DATASETS)
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.lock_protocol:
        return lock_protocol()
    if a.addendum:
        return addendum()
    if a.finalize:
        return finalize()
    stage = "process_a" if a.process_a else ("process_b" if a.process_b else None)
    if stage is None:
        raise SystemExit("pass --lock-protocol | --addendum | --process-a <ds> | --process-b <ds> | --finalize")
    if device.type != "cuda":
        raise Halt("CUDA unavailable; blocking, not auto-switching to CPU")
    if not (OUT / "protocol_locked.json").exists():
        raise Halt("protocol_locked.json missing; run --lock-protocol first")
    rec = (OUT / "protocol_locked.json.sha256").read_text(encoding="utf-8").split()[0].strip()
    if rec != sha(OUT / "protocol_locked.json"):
        raise Halt("protocol hash mismatch")
    name = a.process_a or a.process_b
    ddir = OUT / name
    try:
        if a.process_a:
            return process_a(name)
        return process_b(name)
    except Halt as e:
        write_failure("HALTED", {"stage": stage, "reason": str(e), "log_tail": LOG[-40:]}, ddir)
        raise SystemExit(4)
    except Exception:
        write_failure("UNEXPECTED_EXCEPTION", {"stage": stage, "traceback": traceback.format_exc(),
                                               "log_tail": LOG[-40:]}, ddir)
        raise SystemExit(6)
    finally:
        (ddir / "logs").mkdir(exist_ok=True)
        with (ddir / "logs" / f"{stage}.log").open("a", encoding="utf-8") as f:
            f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise SystemExit(6)
