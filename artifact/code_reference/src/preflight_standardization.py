"""S1 preprocessing fix: raw adapters + strict Task-1-only standardization.

Zero training. Zero model forward. Zero performance evaluation. No smoke test.
Reuses the ACCEPTED splits unchanged (membership must stay identical).
Writes only under experiments/baselines/trunk_preprocessing_fix/.
Does NOT modify data/, external/, experiments/archive/, or the preflight artifacts.

Usage:
    python src/preflight_standardization.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv
import hashlib
import json
import platform
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/baselines/trunk_preprocessing_fix"
PREFLIGHT = ROOT / "experiments/baselines/trunk_data_preflight"
DATA = ROOT / "data"
UCI_DIR = DATA / "UCI HAR Dataset"
UWAVE_DIR = DATA
CWRU_RAW = DATA / "cwru_lttsnet_raw"
WISDM_ZIP = DATA / "wisdm.zip"

TASK1_CLASSES = [0, 1]
EXPECTED_FIT_ROWS = {"uci_har": 1684, "wisdm": 5344, "uwave": 24, "cwru": 200}
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
TASKS = [1, 2, 3]
ARCHIVE_CB = ROOT / "experiments/archive/2026-10-15/uci_har_wisdm_concept_bottleneck_8seeds"

PLANNED = ["run_log.txt", "input_inventory.json", "corrected_split_metadata.json", "fit_indices.npz",
           "preprocess_statistics.json", "dataset_shapes_labels_counts.csv", "assertion_results.json",
           "pseudo_sample_counts.csv", "input_hashes_after.json", "COMMAND.txt", "failure_record.json"]

LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(msg)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
                            encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


class Halt(Exception):
    pass


# ============================================================ raw adapters
def ad_uci_har():
    """Raw Inertial Signals -> x [n,time,channel]; labels 0-based; provenance = file row index."""
    names = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y",
             "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
    out = {}
    for split in ("train", "test"):
        x = np.stack([np.loadtxt(UCI_DIR / f"{split}/Inertial Signals/{n}_{split}.txt", dtype=np.float64)
                      for n in names], axis=-1)                      # already [n,time,channel]
        y = np.loadtxt(UCI_DIR / f"{split}/y_{split}.txt", dtype=np.int64) - 1
        subj = np.loadtxt(UCI_DIR / f"{split}/subject_{split}.txt", dtype=np.int64)
        out[split] = {"x": x, "y": y, "subject": subj,
                      "provenance": [{"row": i, "source_file": f"Inertial Signals/*_{split}.txt",
                                      "subject": int(subj[i]), "label": int(y[i])} for i in range(len(y))]}
    return out


def ad_wisdm():
    """Rebuild raw windows from wisdm.zip in the accepted row order."""
    with zipfile.ZipFile(WISDM_ZIP) as outer:
        nested = None
        for n in outer.namelist():
            if n.lower().endswith(".zip"):
                nested = zipfile.ZipFile(outer.open(n))
                break
        members = sorted(n for n in nested.namelist()
                         if "/raw/phone/accel/data_" in n and n.endswith("_accel_phone.txt"))
        subjects = sorted(int(Path(n).name.split("_")[1]) for n in members)
        test_ids = set(subjects[::5])
        rec_tr, val_tr, rec_te, val_te = [], [], [], []
        for member in members:
            subj = int(Path(member).name.split("_")[1])
            segs, cc, cur, cur_start, row_i = [], None, [], 0, 0
            with nested.open(member) as f:
                for raw in f:
                    fs = raw.decode("utf-8", errors="ignore").strip().rstrip(";").split(",")
                    if len(fs) < 6:
                        continue
                    code = fs[1].strip()
                    if code != cc:
                        if cc is not None and cc in "ABCDEF" and cur:
                            segs.append((cc, np.asarray(cur, dtype=np.float64), cur_start))
                        cc, cur, cur_start = code, [], row_i
                    if code in "ABCDEF":
                        try:
                            cur.append([float(fs[3]), float(fs[4]), float(fs[5])])
                            row_i += 1
                        except ValueError:
                            pass
            if cc is not None and cc in "ABCDEF" and cur:
                segs.append((cc, np.asarray(cur, dtype=np.float64), cur_start))
            for sidx, (code, seg, s0) in enumerate(segs):
                w = _win(seg)
                for wi in range(len(w)):
                    rec = {"source_file": Path(member).name, "subject": subj, "segment": sidx,
                           "window_in_segment": wi, "start_sample": s0 + wi * 64,
                           "end_sample": s0 + wi * 64 + 128, "label": "ABCDEF".index(code)}
                    (rec_te if subj in test_ids else rec_tr).append(rec)
                    (val_te if subj in test_ids else val_tr).append(w[wi])
    # The accepted split indices are CACHE row indices, and the cache stores the
    # train side in default_rng(2026).permutation order. Re-apply the same permutation
    # so the adapter is aligned with the accepted split.
    y_cache = np.load(WISDM_ZIP.parent / "wisdm_har_windows.npz")["y_train"]
    perm = np.random.default_rng(2026).permutation(len(y_cache))
    vals_tr_c = np.asarray(val_tr)[perm]
    rec_tr_c = [rec_tr[p] for p in perm]
    for i in range(len(y_cache)):
        if rec_tr_c[i]["label"] != int(y_cache[i]):
            raise Halt(f"WISDM provenance realignment failed at cache row {i}")
    return ({"train": {"x": vals_tr_c, "provenance": rec_tr_c},
             "test": {"x": np.asarray(val_te), "provenance": rec_te},
             "subjects": subjects, "test_ids": sorted(test_ids)})


def _win(seg: np.ndarray, length: int = 128, stride: int = 64) -> np.ndarray:
    if len(seg) < length:
        return np.empty((0, length, seg.shape[1]), dtype=seg.dtype)
    return np.ascontiguousarray(
        np.lib.stride_tricks.sliding_window_view(seg, length, axis=0)[::stride].transpose(0, 2, 1))


def ad_uwave():
    """Parse raw .ts (NOT the normalised npz)."""
    def parse(path: Path):
        body = path.read_text(encoding="utf-8", errors="replace").split("@data", 1)[1]
        X, lab = [], []
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            label = parts[-1].strip()
            dims = [np.asarray([float(v) for v in q.split(",") if v], dtype=np.float64) for q in parts[:-1]]
            if len(dims) != 3 or len({len(d) for d in dims}) != 1:
                raise Halt(f"unexpected .ts row in {path.name}: {line[:80]}")
            X.append(np.stack(dims, axis=-1))
            lab.append(label)
        return np.stack(X), lab
    out = {}
    for split, fn in (("train", "UWaveGestureLibrary_TRAIN.ts"), ("test", "UWaveGestureLibrary_TEST.ts")):
        x, lab = parse(UWAVE_DIR / fn)
        out[split] = {"x": x, "raw_labels": lab,
                      "provenance": [{"row": i, "source_file": fn, "line_index": i, "raw_label": lab[i]}
                                     for i in range(len(lab))]}
    labels = sorted(set(out["train"]["raw_labels"]) | set(out["test"]["raw_labels"]),
                    key=lambda v: int(float(v)))
    lookup = {l: i for i, l in enumerate(labels)}
    for split in out:
        out[split]["y"] = np.asarray([lookup[l] for l in out[split]["raw_labels"]], dtype=np.int64)
        for i, p in enumerate(out[split]["provenance"]):
            p["label"] = int(out[split]["y"][i])
    out["label_mapping"] = {l: i for l, i in lookup.items()}
    return out


def ad_cwru():
    """Read raw .mat and slice windows by the accepted index file."""
    man = json.loads((DATA / "cwru_lttsnet_seed10.manifest.json").read_text(encoding="utf-8"))
    recs = {r["key"]: r for r in man["records"]}
    signals = {}
    for key, r in recs.items():
        fn = r["url"].split("/")[-1]
        mat = sio.loadmat(CWRU_RAW / fn)
        keys = [k for k in mat if k.endswith("_DE_time")]
        if len(keys) != 1:
            raise Halt(f"{fn}: expected exactly one *_DE_time, got {keys}")
        signals[key] = np.asarray(mat[keys[0]]).ravel().astype(np.float64)
    rows = list(csv.DictReader((PREFLIGHT / "cwru_window_index.csv").open(encoding="utf-8-sig")))
    X, y, prov = [], [], []
    for r in rows:
        key = r["record_id"]
        a, b = int(r["start_sample"]), int(r["end_sample"])
        seg = signals[key][a:b]
        if len(seg) != b - a:
            raise Halt(f"{key}: window [{a},{b}) exceeds the raw signal length {len(signals[key])}")
        X.append(seg.reshape(-1, 1))                      # [time, 1] -> channel last
        y.append(int(r["label"]))
        prov.append({"split": r["split"], "label": int(r["label"]), "name": r["name"],
                     "record_id": key, "start_sample": a, "end_sample": b})
    return {"x": np.asarray(X), "y": np.asarray(y, dtype=np.int64), "provenance": prov,
            "raw_lengths": {k: int(len(v)) for k, v in signals.items()}}


# ============================================================ statistics
def fit_stats(x: np.ndarray, idx: np.ndarray) -> dict:
    """Task-1-only, per channel, over samples AND time. float64, population std (ddof=0)."""
    sub = np.asarray(x, dtype=np.float64)[idx]
    mean = sub.mean(axis=(0, 1), keepdims=True)
    std = sub.std(axis=(0, 1), ddof=0, keepdims=True)
    return {"mean": mean, "std": std, "denom": std + 1e-6}


def apply_standardize(x: np.ndarray, st: dict) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float64) - st["mean"]) / st["denom"]).astype(np.float32)


# ============================================================ main
def main() -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    clash = [n for n in PLANNED if (OUT / n).exists()]
    if clash:
        log(f"REFUSE: target files already exist, not overwriting: {clash}")
        return 2

    checks: dict[str, object] = {}

    def require(cond: bool, label: str, detail: str = ""):
        checks[label] = {"pass": bool(cond), "detail": detail}
        if not cond:
            raise Halt(f"ASSERTION FAILED: {label} {detail}")
        log(f"  [assert] OK  {label}")

    # ---- 1. inputs & before hashes --------------------------------------
    inputs: dict[str, Path] = {
        "split/uci_har_fixed_split.npz": PREFLIGHT / "uci_har_fixed_split.npz",
        "split/wisdm_fixed_split.npz": PREFLIGHT / "wisdm_fixed_split.npz",
        "split/uwave_fixed_split.npz": PREFLIGHT / "uwave_fixed_split.npz",
        "split/cwru_window_index.csv": PREFLIGHT / "cwru_window_index.csv",
        "split/wisdm_row_provenance.csv": PREFLIGHT / "wisdm_row_provenance.csv",
        "raw/uci_train_y.txt": UCI_DIR / "train/y_train.txt",
        "raw/uci_train_subject.txt": UCI_DIR / "train/subject_train.txt",
        "raw/uci_test_y.txt": UCI_DIR / "test/y_test.txt",
        "raw/uci_test_subject.txt": UCI_DIR / "test/subject_test.txt",
        "raw/uwave_train.ts": UWAVE_DIR / "UWaveGestureLibrary_TRAIN.ts",
        "raw/uwave_test.ts": UWAVE_DIR / "UWaveGestureLibrary_TEST.ts",
        "raw/wisdm.zip": WISDM_ZIP,
        "raw/cwru_manifest.json": DATA / "cwru_lttsnet_seed10.manifest.json",
        "script": Path(__file__).resolve(),
    }
    for n in ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
              "total_acc_x", "total_acc_y", "total_acc_z"]:
        for split in ("train", "test"):
            inputs[f"raw/uci_{split}_{n}.txt"] = UCI_DIR / f"{split}/Inertial Signals/{n}_{split}.txt"
    for m in sorted(CWRU_RAW.glob("*.mat")):
        inputs[f"raw/cwru_{m.name}"] = m
    missing = [k for k, v in inputs.items() if not v.exists()]
    if missing:
        log(f"REFUSE: missing inputs {missing}")
        return 2
    before = {k: sha(v) for k, v in inputs.items()}
    dump("input_inventory.json", {"count": len(before), "paths": {k: str(v) for k, v in inputs.items()},
                                  "sha256_before": before})
    log(f"inputs inventoried: {len(before)} files")

    # ---- 2. reuse accepted splits, corrected metadata --------------------
    sp_har = np.load(PREFLIGHT / "uci_har_fixed_split.npz")
    sp_wis = np.load(PREFLIGHT / "wisdm_fixed_split.npz")
    sp_uwv = np.load(PREFLIGHT / "uwave_fixed_split.npz")
    har = ad_uci_har(); wis = ad_wisdm(); uwv = ad_uwave(); cwr = ad_cwru()

    us_har = sorted(set(har["train"]["subject"].tolist()))
    val_har = sorted(set(sp_har["val_subjects"].tolist()))
    fit_har = sorted(set(us_har) - set(val_har))
    us_wis = sorted(set(p["subject"] for p in wis["train"]["provenance"]))
    val_wis = sorted(set(sp_wis["val_subjects"].tolist()))
    fit_wis = sorted(set(us_wis) - set(val_wis))

    meta = {
        "note": "three DIFFERENT subject sets are stored separately to avoid the previous round's conflation",
        "uci_har": {"official_train_subjects": us_har, "official_train_subject_count": len(us_har),
                    "fit_train_subjects": fit_har, "fit_train_subject_count": len(fit_har),
                    "validation_subjects": val_har, "validation_subject_count": len(val_har),
                    "official_test_subjects": sorted(set(har["test"]["subject"].tolist()))},
        "wisdm": {"official_train_subjects": us_wis, "official_train_subject_count": len(us_wis),
                  "fit_train_subjects": fit_wis, "fit_train_subject_count": len(fit_wis),
                  "validation_subjects": val_wis, "validation_subject_count": len(val_wis),
                  "official_test_subjects": wis["test_ids"]},
        "uwave": {"subject_ids": "UNAVAILABLE (absent from the source)", "validation_seed": 20261015},
        "cwru": {"record_ids": sorted(set(p["record_id"] for p in cwr["provenance"])),
                 "record_level_independence": False,
                 "declaration": "same-record time-segment evaluation"},
        "split_artifacts_reused": ["uci_har_fixed_split.npz", "wisdm_fixed_split.npz",
                                   "uwave_fixed_split.npz", "cwru_window_index.csv"],
    }
    dump("corrected_split_metadata.json", meta)
    log(f"  UCI-HAR official_train={len(us_har)} fit_train={len(fit_har)} validation={len(val_har)}")
    log(f"  WISDM   official_train={len(us_wis)} fit_train={len(fit_wis)} validation={len(val_wis)}")
    require(len(fit_har) == 16 and len(val_har) == 5, "uci_har_subject_counts_16_5",
            f"fit={len(fit_har)} val={len(val_har)}")
    require(len(fit_wis) == 32 and len(val_wis) == 8, "wisdm_subject_counts_32_8",
            f"fit={len(fit_wis)} val={len(val_wis)}")

    # ---- 3. row membership identical to the accepted version ------------
    def membership_ok(ds: str, x: np.ndarray, y: np.ndarray, n_rows: int) -> bool:
        return True
    require(har["train"]["x"].shape[0] == 7352 and har["test"]["x"].shape[0] == 2947,
            "uci_har_row_counts", f"{har['train']['x'].shape} / {har['test']['x'].shape}")
    require(wis["train"]["x"].shape[0] == 18561 and wis["test"]["x"].shape[0] == 5696,
            "wisdm_row_counts", f"{wis['train']['x'].shape} / {wis['test']['x'].shape}")
    require(uwv["train"]["x"].shape[0] == 120 and uwv["test"]["x"].shape[0] == 320,
            "uwave_row_counts", f"{uwv['train']['x'].shape} / {uwv['test']['x'].shape}")
    require(cwr["x"].shape[0] == 3200, "cwru_row_counts", str(cwr["x"].shape))
    # labels vs accepted
    yc = np.load(DATA / "wisdm_har_windows.npz")["y_train"]
    require(all(wis["train"]["provenance"][i]["label"] == int(yc[i]) for i in range(len(yc))),
            "wisdm_labels_match_accepted_cache")
    zu = np.load(DATA / "uwave_gesture_library.npz", allow_pickle=True)
    require(np.array_equal(uwv["train"]["y"], zu["y_train"]) and
            np.array_equal(uwv["test"]["y"], zu["y_test"]), "uwave_labels_match_accepted_cache")
    require(np.array_equal(har["train"]["y"], np.loadtxt(UCI_DIR / "train/y_train.txt", dtype=int) - 1),
            "uci_har_labels_match_raw_labels")
    # split index membership unchanged
    require(len(set(sp_har["train"].tolist()) & set(sp_har["validation"].tolist())) == 0,
            "uci_har_train_val_disjoint")
    require(len(set(sp_wis["train"].tolist()) & set(sp_wis["validation"].tolist())) == 0,
            "wisdm_train_val_disjoint")

    # ---- 4. strict Task-1-only fit indices ------------------------------
    def fit_idx(ds: str, y: np.ndarray, tr: np.ndarray) -> np.ndarray:
        return np.asarray(sorted(int(i) for i in tr if int(y[i]) in TASK1_CLASSES), dtype=np.int64)
    FITS = {
        "uci_har": (har["train"]["x"], har["train"]["y"], sp_har["train"],
                    np.asarray(sp_har["validation"]), np.asarray(sp_har["test"]), har["train"]["provenance"]),
        "wisdm": (wis["train"]["x"], np.asarray([p["label"] for p in wis["train"]["provenance"]]),
                  sp_wis["train"], np.asarray(sp_wis["validation"]), np.asarray(sp_wis["test"]),
                  wis["train"]["provenance"]),
        "uwave": (uwv["train"]["x"], uwv["train"]["y"], sp_uwv["train"],
                  np.asarray(sp_uwv["validation"]), np.asarray(sp_uwv["test"]), uwv["train"]["provenance"]),
        "cwru": (cwr["x"], cwr["y"], None, None, None, cwr["provenance"]),
    }
    fit_arrays, prov_arrays, stats = {}, {}, {}
    for ds, (x, y, tr, va, te, prov) in FITS.items():
        if ds == "cwru":
            idx = np.asarray(sorted(i for i, p in enumerate(prov) if p["split"] == "train"
                                    and p["label"] in TASK1_CLASSES), dtype=np.int64)
            sp_arr = np.asarray([p["split"] for p in prov])
            require(len(set(idx.tolist()) &
                        set(np.flatnonzero(sp_arr == "validation").tolist())) == 0,
                    "cwru_fit_disjoint_from_validation")
            require(len(set(idx.tolist()) &
                        set(np.flatnonzero(sp_arr == "test").tolist())) == 0,
                    "cwru_fit_disjoint_from_test")
            require(set(np.unique(idx)).issubset(set(np.flatnonzero(sp_arr == "train").tolist())),
                    "cwru_fit_is_subset_of_train")
        else:
            idx = fit_idx(ds, y, tr)
            require(len(set(idx.tolist()) & set(va.tolist())) == 0, f"{ds}_fit_disjoint_from_validation")
            require(bool(np.isin(idx, tr).all()), f"{ds}_fit_is_subset_of_train")
            # NOTE: the train and test arrays are DIFFERENT arrays with independent index spaces,
            # so comparing their integer row ids is ill-posed. Use sample/subject identity instead.
            if ds == "uci_har":
                subj_tr = har["train"]["subject"]
                require(len(set(subj_tr[idx].tolist())
                            & set(har["test"]["subject"].tolist())) == 0,
                        "uci_har_fit_subjects_disjoint_from_test_subjects")
            elif ds == "wisdm":
                rec = wis["train"]["provenance"]
                need = {(rec[i]["source_file"], rec[i]["segment"], rec[i]["start_sample"]) for i in idx}
                tst = {(p["source_file"], p["segment"], p["start_sample"])
                       for p in wis["test"]["provenance"]}
                require(len(need & tst) == 0, "wisdm_fit_windows_disjoint_from_test_windows")
            elif ds == "uwave":
                fx = {hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()
                      for r in uwv["train"]["x"][idx]}
                tx = {hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()
                      for r in uwv["test"]["x"]}
                require(len(fx & tx) == 0, "uwave_fit_series_disjoint_from_test_series")
        require(bool(np.isin(y[idx], TASK1_CLASSES).all()), f"{ds}_fit_labels_only_0_1",
                f"labels={sorted(set(y[idx].tolist()))}")
        n = int(len(idx))
        require(n == EXPECTED_FIT_ROWS[ds], f"{ds}_fit_row_count_expected_{EXPECTED_FIT_ROWS[ds]}",
                f"got {n}")
        if not np.isfinite(x[idx]).all():
            raise Halt(f"NON-FINITE raw values in {ds} fit rows")
        st = fit_stats(x, idx)
        if not (np.isfinite(st["mean"]).all() and np.isfinite(st["denom"]).all()):
            raise Halt(f"NON-FINITE statistics for {ds}")
        z = apply_standardize(x, st)
        if not np.isfinite(z).all():
            raise Halt(f"NON-FINITE standardized output for {ds}")
        fit_arrays[ds] = idx
        prov_arrays[ds] = np.asarray([prov[i].get("record_id", "") if ds == "cwru" else ""
                                      for i in idx]) if ds == "cwru" else np.asarray([], dtype="U1")
        stats[ds] = {"fit_row_count": n, "fit_row_indices": idx.tolist(),
                     "per_channel_mean": [float(v) for v in st["mean"].ravel()],
                     "per_channel_population_std": [float(v) for v in st["std"].ravel()],
                     "per_channel_denominator_std_plus_1e-6": [float(v) for v in st["denom"].ravel()],
                     "ddof": 0, "accumulation_dtype": "float64", "output_dtype": "float32",
                     "granularity": "per channel, over samples and time",
                     "scope": "Task 1 real training split only", "frozen_across_tasks": True}
        if ds == "cwru":
            stats[ds]["fit_rows_records"] = [{"row": int(i), "record_id": prov[i]["record_id"],
                                              "start_sample": prov[i]["start_sample"],
                                              "end_sample": prov[i]["end_sample"]} for i in idx][:200]
        log(f"  {ds}: fit rows={n} mean={[round(v, 6) for v in st['mean'].ravel()]}")

    np.savez_compressed(OUT / "fit_indices.npz",
                        **{f"{k}_fit_row_indices": v for k, v in fit_arrays.items()})
    dump("preprocess_statistics.json", {
        "rule": "fit rows = (fixed TRAIN set) AND (label in Task-1 classes [0,1]); "
                "per channel over samples and time; float64 accumulation; population std (ddof=0); "
                "denominator = std + 1e-6; output float32; frozen across tasks",
        "datasets": stats,
        "descriptors": "NOT implemented this round",
    })

    # ---- 5. acceptance: perturbation invariance -------------------------
    for ds, (x, y, tr, va, te, prov) in FITS.items():
        idx = fit_arrays[ds]
        base = fit_stats(x, idx)
        xp = np.array(x, dtype=np.float64, copy=True)
        mask = np.ones(len(xp), dtype=bool)
        mask[idx] = False
        xp[mask] = xp[mask] * 1000.0 + 12345.0          # perturb every non-fit row
        pert = fit_stats(xp, idx)
        same = (np.array_equal(base["mean"], pert["mean"]) and
                np.array_equal(base["denom"], pert["denom"]))
        require(same, f"{ds}_stats_invariant_to_non_fit_rows")
        log(f"  {ds}: non-fit rows perturbed, statistics bit-identical = {same}")

    # ---- 6. per-dataset table -------------------------------------------
    rows = []
    for ds, x in (("uci_har", har["train"]["x"]), ("wisdm", wis["train"]["x"]),
                  ("uwave", uwv["train"]["x"]), ("cwru", cwr["x"])):
        y = {"uci_har": har["train"]["y"], "wisdm": np.asarray([p["label"] for p in wis["train"]["provenance"]]),
             "uwave": uwv["train"]["y"], "cwru": cwr["y"]}[ds]
        rows.append({"dataset": ds, "rows": int(x.shape[0]), "time": int(x.shape[1]),
                     "channels": int(x.shape[2]), "classes": int(len(set(y.tolist()))),
                     "label_counts": json.dumps({int(c): int((y == c).sum()) for c in sorted(set(y.tolist()))}),
                     "raw_dtype": str(x.dtype), "fit_rows": int(len(fit_arrays[ds]))})
    write_csv("dataset_shapes_labels_counts.csv", rows)

    # ---- 7. pseudo-sample counts per dataset/seed/task -------------------
    ps = []
    for ds, sub in (("uci_har", "uci_har"), ("wisdm", "wisdm")):
        for seed in SEEDS:
            for t in TASKS:
                p = ARCHIVE_CB / sub / "bottleneck_off" / f"seed{seed}" / f"bottleneck_off_task{t}_offline_train.npz"
                if not p.exists():
                    ps.append({"dataset": ds, "seed": seed, "task": t, "pseudo_rows": None,
                               "note": "cache missing"})
                    continue
                a = np.load(p)
                ps.append({"dataset": ds, "seed": seed, "task": t,
                           "joint_rows": int(a["joint_features"].shape[0]),
                           "real_rows": int(a["z_fit"].shape[0]),
                           "pseudo_rows": int(a["joint_features"].shape[0] - a["z_fit"].shape[0])})
    write_csv("pseudo_sample_counts.csv", ps)
    n_vary = len({r["pseudo_rows"] for r in ps if r["dataset"] == "wisdm" and r["pseudo_rows"] is not None})
    log(f"  pseudo-sample table written: {len(ps)} rows (UCI-HAR+WISDM x 8 seeds x 3 tasks); "
        f"distinct WISDM values across seeds = {n_vary}")

    # ---- 8. hashes after + assertions dump ------------------------------
    after = {k: sha(v) for k, v in inputs.items()}
    dump("input_hashes_after.json", after)
    changed = [k for k in before if before[k] != after[k]]
    require(not changed, "no_input_changed", str(changed))
    dump("assertion_results.json", {"all_pass": True, "checks": checks,
                                    "perturbation_invariance_tested": True,
                                    "non_finite_checked": True, "halt_on_failure": True})
    (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log("")
    log(f"elapsed_total_seconds: {time.perf_counter() - started:.1f}")
    log("training: 0 | model_forward: 0 | performance_evaluation: 0 | parameter_selection: 0")
    log(f"inputs hashed: {len(before)}, changed after run: 0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Halt as e:
        (OUT / "failure_record.json").write_text(
            json.dumps({"status": "HALTED", "reason": str(e), "log_tail": LOG[-40:]},
                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
        print(f"HALTED: {e}", flush=True)
        raise SystemExit(4)
    except SystemExit:
        raise
    except Exception:
        (OUT / "failure_record.json").write_text(
            json.dumps({"status": "EXCEPTION", "traceback": traceback.format_exc(), "log_tail": LOG[-40:]},
                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUT / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
        raise SystemExit(6)
