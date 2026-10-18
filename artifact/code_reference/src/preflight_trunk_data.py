"""S1 trunk data preflight: row-level provenance, fixed splits, CWRU CIL build, preprocessing scope.

Zero training. Zero model forward. Zero performance evaluation.
Reads only. Writes only under experiments/baselines/trunk_data_preflight/.
Does NOT modify data/, external/, or any archived experiment.

Usage:
    python src/preflight_trunk_data.py --phase hash
    python src/preflight_trunk_data.py --phase wisdm
    python src/preflight_trunk_data.py --phase uci_har
    python src/preflight_trunk_data.py --phase uwave
    python src/preflight_trunk_data.py --phase cwru
    python src/preflight_trunk_data.py --phase preprocess
    python src/preflight_trunk_data.py --phase all
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/baselines/trunk_data_preflight"
DATA = ROOT / "data"
OLD_SPLIT = ROOT / "experiments/archive/2026-10-15/rls_analytic_classifier_offline/wisdm/split_indices.npz"
WISDM_ZIP = DATA / "wisdm.zip"
WISDM_CACHE = DATA / "wisdm_har_windows.npz"
CWRU_NPZ = DATA / "cwru_lttsnet_seed10.npz"
CWRU_MANIFEST = DATA / "cwru_lttsnet_seed10.manifest.json"
UCI_DIR = DATA / "UCI HAR Dataset"
UWAVE_CACHE = DATA / "uwave_gesture_library.npz"

TASKS = [[0, 1], [2, 3], [4, 5]]
CWRU_TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
CWRU_W, CWRU_S, CWRU_GUARD = 1024, 512, 1024
CWRU_TARGET_TRAIN, CWRU_TARGET_VAL = 100, 20
SEED_CWRU = 20261015
SEED_UWAVE = 20261015


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


def make_windows(seg: np.ndarray, length: int = 128, stride: int = 64) -> np.ndarray:
    """Match the archived loader's window layout: (n_windows, length, channels)."""
    if len(seg) < length:
        return np.empty((0, length, seg.shape[1]), dtype=seg.dtype)
    w = np.lib.stride_tricks.sliding_window_view(seg, length, axis=0)[::stride]
    return np.ascontiguousarray(w.transpose(0, 2, 1))


# --------------------------------------------------------------------------- WISDM
def _wisdm_replay():
    """Replay the archived loader order, returning per-record provenance AND window values."""
    with zipfile.ZipFile(WISDM_ZIP) as outer:
        nested = None
        for n in outer.namelist():
            if n.lower().endswith(".zip"):
                nested = zipfile.ZipFile(outer.open(n))
                break
        if nested is None:
            raise SystemExit("wisdm.zip: nested archive not found")
        members = sorted(n for n in nested.namelist()
                         if "/raw/phone/accel/data_" in n and n.endswith("_accel_phone.txt"))
        subjects = sorted(int(Path(n).name.split("_")[1]) for n in members)
        test_ids = set(subjects[::5])
        recs_tr, vals_tr, recs_te, vals_te = [], [], [], []
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
                            segs.append((cc, np.asarray(cur, dtype=np.float32), cur_start))
                        cc, cur, cur_start = code, [], row_i
                    if code in "ABCDEF":
                        try:
                            cur.append([float(fs[3]), float(fs[4]), float(fs[5])])
                            row_i += 1
                        except ValueError:
                            pass
            if cc is not None and cc in "ABCDEF" and cur:
                segs.append((cc, np.asarray(cur, dtype=np.float32), cur_start))
            for sidx, (code, seg, s0) in enumerate(segs):
                wins = make_windows(seg)
                for wi in range(len(wins)):
                    rec = {"subject": subj, "source_file": Path(member).name, "segment": sidx,
                           "segment_code": code, "window_in_segment": wi,
                           "start_sample": s0 + wi * 64, "end_sample": s0 + wi * 64 + 128,
                           "activity": "ABCDEF".index(code)}
                    (recs_te if subj in test_ids else recs_tr).append(rec)
                    (vals_te if subj in test_ids else vals_tr).append(wins[wi])
    return (recs_tr, np.asarray(vals_tr, dtype=np.float32),
            recs_te, np.asarray(vals_te, dtype=np.float32),
            subjects, sorted(test_ids))


def phase_wisdm() -> None:
    print("=" * 78)
    print("PHASE wisdm: per-row provenance + interval intersection with the OLD split")
    print("=" * 78)
    cache = np.load(WISDM_CACHE, allow_pickle=True)
    ytr, yte, xtr, xte = cache["y_train"], cache["y_test"], cache["x_train"], cache["x_test"]
    meta = json.loads(str(cache["meta"]))
    print(f"  cache: train={xtr.shape} test={xte.shape} | meta sample_rate={meta.get('sample_rate_hz')} Hz")

    recs_tr, vals_tr, recs_te, vals_te, subjects, test_ids = _wisdm_replay()
    print(f"  replayed: train={len(recs_tr)} test={len(recs_te)} subjects={len(subjects)} test_subjects={len(test_ids)}")

    perm = np.random.default_rng(2026).permutation(len(ytr))
    lab_ok = all(recs_tr[perm[r]]["activity"] == int(ytr[r]) for r in range(len(ytr)))
    print(f"  label/order verification: {lab_ok}")
    if not lab_ok:
        raise SystemExit("VERIFY FAILED: replayed labels/order do not match the cache")

    raw_train = vals_tr[perm]                      # align to cache row order
    mean = raw_train.mean(axis=(0, 1), keepdims=True)
    std = raw_train.std(axis=(0, 1), keepdims=True) + 1e-6
    err_tr = float(np.max(np.abs(((raw_train - mean) / std).astype(np.float32) - xtr)))
    err_te = float(np.max(np.abs(((vals_te - mean) / std).astype(np.float32) - xte)))
    print(f"  per-channel mean={mean.ravel().round(6).tolist()} std={std.ravel().round(6).tolist()}")
    print(f"  max|recompute-cache| train={err_tr:.3e} test={err_te:.3e} (archived loader bound 2e-5)")

    rows = [{"cache_row": r, "split": "train", **recs_tr[perm[r]]} for r in range(len(ytr))]
    rows += [{"cache_row": r, "split": "test", **recs_te[r]} for r in range(len(yte))]
    write_csv("wisdm_row_provenance.csv", rows)
    print(f"  wrote wisdm_row_provenance.csv ({len(rows)} rows)")

    # OLD split intersection, half-open intervals keyed by (source_file, segment)
    spr = np.load(OLD_SPLIT)
    subj_of = np.array([recs_tr[perm[r]]["subject"] for r in range(len(ytr))])
    segk = np.array([recs_tr[perm[r]]["segment"] for r in range(len(ytr))])
    win = np.array([recs_tr[perm[r]]["window_in_segment"] for r in range(len(ytr))])
    per_task, examples = [], []
    for t in (1, 2, 3):
        tr = [int(v) for v in spr[f"train_task{t}"]]
        va = [int(v) for v in spr[f"val_task{t}"]]
        trk = {}
        for r in tr:
            rec = recs_tr[perm[r]]
            trk.setdefault((rec["source_file"], rec["segment"]), []).append(
                (rec["start_sample"], rec["end_sample"], r))
        ov = 0
        for r in va:
            rec = recs_tr[perm[r]]
            a0, a1 = rec["start_sample"], rec["end_sample"]
            for b0, b1, rr in trk.get((rec["source_file"], rec["segment"]), []):
                if a0 < b1 and b0 < a1:
                    ov += 1
                    if len(examples) < 20:
                        examples.append({"task": t, "val_row": r, "val_interval": [a0, a1],
                                         "train_row": rr, "train_interval": [b0, b1],
                                         "source_file": rec["source_file"], "segment": rec["segment"]})
                    break
        trs = set(tr)
        prev = sum(1 for i in va if ((i - 1 in trs and segk[i - 1] == segk[i] and abs(win[i - 1] - win[i]) == 1)
                                     or (i + 1 in trs and segk[i + 1] == segk[i] and abs(win[i + 1] - win[i]) == 1)))
        so = len(set(subj_of[tr].tolist()) & set(subj_of[va].tolist()))
        per_task.append({"task": t, "class_pair": TASKS[t - 1], "train_n": len(tr), "val_n": len(va),
                         "val_overlapping_count": ov,
                         "val_overlapping_ratio": round(ov / len(va), 6),
                         "prev_style_adjacent_count": prev,
                         "prev_style_adjacent_ratio": round(prev / len(va), 6),
                         "subjects_in_both": so,
                         "subjects_train": len(set(subj_of[tr].tolist())),
                         "subjects_val": len(set(subj_of[va].tolist()))})
        print(f"  task{t} {TASKS[t-1]}: train={len(tr)} val={len(va)} | "
              f"(source_file,segment) 半开区间重叠={ov} ({ov/len(va):.4%}) | "
              f"上轮口径(仅段号+|Δwin|=1)={prev} ({prev/len(va):.4%}) | 受试者跨集={so}")
    dump("wisdm_interval_intersection.json", {
        "key_definition": "two windows overlap iff same source_file AND same segment AND half-open [start,end) intersect",
        "forbidden_practice": "cache row-index adjacency was NOT used as a proxy for temporal adjacency",
        "row_provenance_verification": {"label_order_match": lab_ok,
                                        "max_abs_recompute_train": err_tr,
                                        "max_abs_recompute_test": err_te,
                                        "archived_loader_bound": 2e-5},
        "per_task": per_task, "examples": examples,
        "diagnosis_of_previous_round": "the previous round keyed on (segment_index, |Δ window_in_segment| == 1) and did NOT require the same subject/source_file; the key was therefore incomplete. The (source_file, segment) key above is the source of truth; both counts are reported for comparison."})
    print("  wrote wisdm_interval_intersection.json")


# --------------------------------------------------------------------------- UCI-HAR
def phase_uci_har() -> None:
    print("=" * 78)
    print("PHASE uci_har: subject intersection (exact) + row-order overlap (ESTIMATE)")
    print("=" * 78)
    subj = np.loadtxt(UCI_DIR / "train/subject_train.txt", dtype=int)
    y = np.loadtxt(UCI_DIR / "train/y_train.txt", dtype=int) - 1
    subj_te = np.loadtxt(UCI_DIR / "test/subject_test.txt", dtype=int)
    y_te = np.loadtxt(UCI_DIR / "test/y_test.txt", dtype=int) - 1
    us = sorted(set(subj.tolist()))
    print(f"  train rows={len(subj)} subjects={len(us)} | test rows={len(subj_te)} subjects={len(set(subj_te.tolist()))}")
    print(f"  official train/test subject overlap (exact) = {len(set(us) & set(subj_te.tolist()))}")

    sys.path.insert(0, str(ROOT / "src"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("lb_preflight", ROOT / "src/legacy/benchmark.py")
    lb = importlib.util.module_from_spec(spec)
    sys.modules["lb_preflight"] = lb
    spec.loader.exec_module(lb)
    lb.CLASSES = [f"c{i}" for i in range(6)]
    pool, _ = lb.split_probe_and_task_pool(y, {"diagnostic_split_seed": 20261005,
                                               "diagnostic_fraction_per_class": 0.1})
    runs, s = [], 0
    for i in range(1, len(subj) + 1):
        if i == len(subj) or subj[i] != subj[s] or y[i] != y[s]:
            runs.append((s, i - 1))
            s = i
    run_of = np.zeros(len(subj), dtype=int)
    for ri, (a, b) in enumerate(runs):
        run_of[a:b + 1] = ri
    print(f"  contiguous (subject,activity) runs = {len(runs)}")

    old_rows = []
    for t, cc in enumerate(TASKS, 1):
        tr, va = lb.split_task_train_validation(y, pool, cc, 0.2, 20261004 + t * 100)
        trs = set(tr.tolist())
        adj = sum(1 for i in va if (i - 1 in trs and run_of[i - 1] == run_of[i])
                  or (i + 1 in trs and run_of[i + 1] == run_of[i]))
        so = len(set(subj[tr].tolist()) & set(subj[va].tolist()))
        old_rows.append({"task": t, "class_pair": cc, "train_n": len(tr), "val_n": len(va),
                         "subject_intersection_EXACT": so,
                         "subjects_train": len(set(subj[tr].tolist())),
                         "subjects_val": len(set(subj[va].tolist())),
                         "val_adjacent_rows_ESTIMATE": adj,
                         "val_adjacent_ratio_ESTIMATE": round(adj / len(va), 6)})
        print(f"  OLD task{t} {cc}: train={len(tr)} val={len(va)} | subject intersection (exact)={so} | "
              f"adjacent rows (ESTIMATE)={adj} ({adj/len(va):.2%})")

    val_s = set(us[::5])
    m_tr = ~np.isin(subj, list(val_s))
    tr_idx, va_idx = np.flatnonzero(m_tr), np.flatnonzero(~m_tr)
    inter = len(set(subj[tr_idx].tolist()) & set(subj[va_idx].tolist()))
    per_class = [{"class": c, "train": int(((y == c) & m_tr).sum()),
                  "val": int(((y == c) & ~m_tr).sum()),
                  "test": int((y_te == c).sum())} for c in range(6)]
    print(f"  NEW split: val_subjects={sorted(val_s)} subject_intersection={inter} "
          f"train={len(tr_idx)} val={len(va_idx)} hash={hashlib.sha256(np.asarray(sorted(tr_idx.tolist())).tobytes()).hexdigest()[:16]}")
    np.savez_compressed(OUT / "uci_har_fixed_split.npz", train=tr_idx, validation=va_idx,
                        test=np.arange(len(y_te)), val_subjects=np.array(sorted(val_s)),
                        train_subjects=np.array(us))
    write_csv("uci_har_per_class_counts.csv", per_class)
    dump("uci_har_split_report.json", {
        "rule": "validation subjects = sorted(train_subjects)[::5]; NOT reselected by model performance",
        "val_subjects": sorted(val_s), "subject_intersection_train_val_EXACT": inter,
        "official_train_test_subject_overlap_EXACT": len(set(us) & set(subj_te.tolist())),
        "train_rows": int(len(tr_idx)), "val_rows": int(len(va_idx)), "test_rows": int(len(y_te)),
        "per_class": per_class, "old_split_rows": old_rows,
        "row_order_overlap_is_ESTIMATE": "on-disk window positions are not provided by the official dataset; "
                                         "the estimate assumes consecutive rows within a (subject,activity) run "
                                         "are consecutive 50%-overlapping windows",
        "subject_intersection_is_EXACT": True,
    })
    print("  wrote uci_har_fixed_split.npz / uci_har_per_class_counts.csv / uci_har_split_report.json")


# --------------------------------------------------------------------------- UWave
def phase_uwave() -> None:
    print("=" * 78)
    print("PHASE uwave: original train/test boundary kept; fixed 20% per class validation")
    print("=" * 78)
    z = np.load(UWAVE_CACHE, allow_pickle=True)
    xtr, ytr, xte, yte = z["x_train"], z["y_train"], z["x_test"], z["y_test"]
    rng = np.random.default_rng(SEED_UWAVE)
    tr_idx, va_idx, per_class = [], [], []
    for c in range(8):
        idx = np.flatnonzero(ytr == c)
        idx = idx[rng.permutation(len(idx))]
        n_va = max(1, int(round(len(idx) * 0.2)))
        va_idx.extend(idx[:n_va].tolist())
        tr_idx.extend(idx[n_va:].tolist())
        per_class.append({"class": c, "train": len(idx) - n_va, "val": n_va,
                          "test": int((yte == c).sum())})
    tr_idx, va_idx = np.array(sorted(tr_idx)), np.array(sorted(va_idx))
    print(f"  seed={SEED_UWAVE} train={len(tr_idx)} val={len(va_idx)} test={len(yte)}")

    def fps(a):
        return {hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest() for r in a}
    ftr, fte = fps(xtr), fps(xte)
    dup = {"within_train": len(xtr) - len(ftr), "within_test": len(xte) - len(fte),
           "train_test_cross": len(ftr & fte)}
    print(f"  exact duplicate series: {dup}")
    np.savez_compressed(OUT / "uwave_fixed_split.npz", train=tr_idx, validation=va_idx,
                        test=np.arange(len(yte)))
    write_csv("uwave_per_class_counts.csv", per_class)
    dump("uwave_split_report.json", {
        "original_train_test_boundary_kept": True,
        "validation_rule": "per class, 20% of the ORIGINAL TRAIN set, fixed seed",
        "seed": SEED_UWAVE, "per_class": per_class, "duplicate_check": dup,
        "caveat": "{'no exact duplicates found' does NOT imply subject independence; subject ids are absent from this cache}",
        "no_window_overlap": "each instance is a whole 315-length gesture trial; no sub-windowing -> no window leakage",
    })
    print("  wrote uwave_fixed_split.npz / uwave_per_class_counts.csv / uwave_split_report.json")


# --------------------------------------------------------------------------- CWRU
def phase_cwru() -> None:
    print("=" * 78)
    print("PHASE cwru: capacity FIRST, then same-record time-segment CIL build")
    print("=" * 78)
    man = json.loads(CWRU_MANIFEST.read_text(encoding="utf-8"))
    cap = []
    for r in man["records"]:
        end = int(r["train_end"])
        t_min = CWRU_W + (CWRU_TARGET_TRAIN - 1) * CWRU_S
        t_max = end - CWRU_GUARD - (CWRU_W + (CWRU_TARGET_VAL - 1) * CWRU_S)
        cap.append({"label": int(r["label"]), "name": r["name"], "train_end": end,
                    "t_min_for_target_train": t_min, "t_max_for_target_val": t_max,
                    "feasible": bool(t_max >= t_min)})
    write_csv("cwru_capacity.csv", cap)
    dump("cwru_capacity.json", {"window": CWRU_W, "stride": CWRU_S, "guard": CWRU_GUARD,
                                "target_train": CWRU_TARGET_TRAIN, "target_val": CWRU_TARGET_VAL,
                                "per_record": cap,
                                "infeasible": [c for c in cap if not c["feasible"]]})
    print(f"  {'lab':<4}{'name':<20}{'train_end':>10}{'t_min':>9}{'t_max':>9}{'feasible':>10}")
    for c in cap:
        print(f"  {c['label']:<4}{c['name']:<20}{c['train_end']:>10}{c['t_min_for_target_train']:>9}"
              f"{c['t_max_for_target_val']:>9}{str(c['feasible']):>10}")
    bad = [c for c in cap if not c["feasible"]]
    if bad:
        print(f"STOP: {len(bad)} class(es) cannot meet 100/20 with a {CWRU_GUARD}-sample guard.")
        dump("cwru_build_status.json", {"status": "STOP_INSUFFICIENT_CAPACITY", "infeasible": bad,
                                        "no_mitigation": ["guard NOT shortened", "classes NOT reduced",
                                                          "target NOT lowered"]})
        return

    rng = np.random.default_rng(SEED_CWRU)
    rows, counts = [], []
    for r, c in zip(man["records"], cap):
        end, t = int(r["train_end"]), c["t_max_for_target_val"]
        tr_all = np.arange(0, (t - CWRU_W) // CWRU_S + 1)
        va_all = np.arange(0, (end - (t + CWRU_GUARD) - CWRU_W) // CWRU_S + 1)
        tr_sel = np.sort(rng.permutation(tr_all)[:CWRU_TARGET_TRAIN])
        va_sel = np.sort(rng.permutation(va_all)[:CWRU_TARGET_VAL])
        lab = int(r["label"])
        for w in tr_sel:
            rows.append({"split": "train", "label": lab, "name": r["name"], "record_id": r["key"],
                         "start_sample": int(w * CWRU_S), "end_sample": int(w * CWRU_S + CWRU_W)})
        for w in va_sel:
            rows.append({"split": "validation", "label": lab, "name": r["name"], "record_id": r["key"],
                         "start_sample": int(t + CWRU_GUARD + w * CWRU_S),
                         "end_sample": int(t + CWRU_GUARD + w * CWRU_S + CWRU_W)})
        for w in r["test_window_indices"]:
            rows.append({"split": "test", "label": lab, "name": r["name"], "record_id": r["key"],
                         "start_sample": int(int(r["test_start"]) + w * CWRU_S),
                         "end_sample": int(int(r["test_start"]) + w * CWRU_S + CWRU_W)})
        counts.append({"label": lab, "name": r["name"], "train": len(tr_sel), "validation": len(va_sel),
                       "test": len(r["test_window_indices"]), "t_split": int(t),
                       "train_segment": [0, int(t)], "guard": [int(t), int(t + CWRU_GUARD)],
                       "val_segment": [int(t + CWRU_GUARD), int(r["train_end"])],
                       "test_segment_starts": int(r["test_start"])})
    write_csv("cwru_window_index.csv", rows)
    write_csv("cwru_per_class_counts.csv", counts)

    by = {}
    for x in rows:
        by.setdefault((x["record_id"], x["split"]), []).append((x["start_sample"], x["end_sample"]))
    inter = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        n = 0
        for (rid, sp), iv in by.items():
            if sp != a:
                continue
            for y0, y1 in by.get((rid, b), []):
                if any(x0 < y1 and y0 < x1 for x0, x1 in iv):
                    n += 1
                    break
        inter[f"{a}_vs_{b}"] = n
    print(f"  built rows={len(rows)} | half-open interval intersections (record-keyed): {inter}")
    print(f"  per class: {[(c['label'], c['train'], c['validation'], c['test']) for c in counts]}")
    dump("cwru_interval_intersection.json", {
        "key": "record_id (one official .mat per class)",
        "interval_rule": "half-open [start,end); overlap iff x0 < y1 and y0 < x1",
        "intersections": inter, "must_be_zero": True, "sampling_seed": SEED_CWRU,
        "declaration": "same-record time-segment evaluation; NOT independent-record generalization",
        "not_inherited": ["LTTSNet 100/10-shot budget", "unlabelled arrays", "subset / _10shot variants"],
        "tasks": CWRU_TASKS,
    })
    dump("cwru_build_status.json", {"status": "BUILT", "per_class": counts, "intersections": inter,
                                    "sampling_seed": SEED_CWRU, "tasks": CWRU_TASKS})
    print("  wrote cwru_window_index.csv / cwru_per_class_counts.csv / cwru_interval_intersection.json")


# --------------------------------------------------------------------------- WISDM fixed split
def phase_wisdm_split() -> None:
    print("=" * 78)
    print("PHASE wisdm_split: subject-grouped fixed train/validation split")
    print("=" * 78)
    z = np.load(WISDM_CACHE, allow_pickle=True)
    y, x = z["y_train"], z["x_train"]
    meta = json.loads(str(z["meta"]))
    recs_tr, _v, _rt, _vt, _s, test_ids = _wisdm_replay()
    perm = np.random.default_rng(2026).permutation(len(y))
    subj_of = np.array([recs_tr[perm[r]]["subject"] for r in range(len(y))])
    us = sorted(set(subj_of.tolist()))
    val_s = set(us[::5])
    m_tr = ~np.isin(subj_of, list(val_s))
    tr_idx, va_idx = np.flatnonzero(m_tr), np.flatnonzero(~m_tr)
    inter = len(set(subj_of[tr_idx].tolist()) & set(subj_of[va_idx].tolist()))
    per_class = [{"class": c, "train": int(((y == c) & m_tr).sum()),
                  "val": int(((y == c) & ~m_tr).sum()),
                  "test": int((z["y_test"] == c).sum())} for c in range(6)]
    print(f"  train_subjects={len(us)} val_subjects={sorted(val_s)} subject_intersection={inter}")
    print(f"  train={len(tr_idx)} val={len(va_idx)} test={len(z['y_test'])}")
    print(f"  per class: {[(d['class'], d['train'], d['val'], d['test']) for d in per_class]}")
    assert inter == 0, "subject intersection must be zero"
    np.savez_compressed(OUT / "wisdm_fixed_split.npz", train=tr_idx, validation=va_idx,
                        test=np.arange(len(z["y_test"])), val_subjects=np.array(sorted(val_s)),
                        train_subjects=np.array(us))
    write_csv("wisdm_per_class_counts.csv", per_class)
    dump("wisdm_split_report.json", {
        "rule": "validation subjects = sorted(train_subjects)[::5]; NOT reselected by model performance",
        "val_subjects": sorted(val_s), "train_subjects": us,
        "subject_intersection_train_val_EXACT": inter,
        "train_rows": int(len(tr_idx)), "val_rows": int(len(va_idx)),
        "test_rows": int(len(z["y_test"])), "per_class": per_class,
        "test_subjects_from_cache_meta": meta["test_subjects"],
        "cost_note": "training rows drop from 18561 to %d (-%.1f%%)" % (
            len(tr_idx), 100 * (1 - len(tr_idx) / 18561)),
    })
    print("  wrote wisdm_fixed_split.npz / wisdm_per_class_counts.csv / wisdm_split_report.json")


# --------------------------------------------------------------------------- preprocessing
def phase_preprocess() -> None:
    print("=" * 78)
    print("PHASE preprocess: split-then-fit; Task-1-only, per channel, frozen")
    print("=" * 78)
    names = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y",
             "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
    out = {}

    x_har = np.stack([np.loadtxt(UCI_DIR / f"train/Inertial Signals/{n}_train.txt", dtype=np.float32)
                      for n in names], axis=-1)
    sp = np.load(OUT / "uci_har_fixed_split.npz")
    tr = sp["train"]
    out["uci_har"] = _fit_stats(x_har, tr, "raw Inertial Signals (train folder), rows of the fixed train split")
    print(f"  uci_har: fit_rows={len(tr)} per-channel mean={[round(v, 6) for v in out['uci_har']['per_channel_mean']]}")

    z = np.load(WISDM_CACHE, allow_pickle=True)
    sp_w = np.load(OUT / "wisdm_fixed_split.npz")
    yw = z["y_train"]
    t1_w = np.array([r for r in sp_w["train"].tolist() if int(yw[r]) in TASKS[0]], dtype=np.int64)
    recs_tr, vals_tr, _a, _b, _c, _d = _wisdm_replay()
    perm_w = np.random.default_rng(2026).permutation(len(yw))
    raw_w = np.asarray([vals_tr[perm_w[r]] for r in t1_w], dtype=np.float32)
    out["wisdm"] = _fit_stats(raw_w, np.arange(len(raw_w)),
                              "RAW windows replayed from wisdm.zip for the Task-1 train rows only")
    out["wisdm"]["row_ids_are_cache_indices"] = True
    out["wisdm"]["cache_row_indices_sha256"] = _hash_rows(t1_w)
    print(f"  wisdm: fit_rows={len(t1_w)} (RAW, Task-1 train split) per-channel mean="
          f"{[round(v, 6) for v in out['wisdm']['per_channel_mean']]}")

    zu = np.load(UWAVE_CACHE, allow_pickle=True)
    sp_u = np.load(OUT / "uwave_fixed_split.npz")
    yu = zu["y_train"]
    t1_u = np.array([r for r in sp_u["train"].tolist() if int(yu[r]) in TASKS[0]], dtype=np.int64)
    out["uwave"] = _fit_stats(zu["x_train"], t1_u,
                              "cache x_train for Task-1 train rows -- the cached array is ALREADY z-scored, "
                              "so these numbers describe the cache, not raw data")
    out["uwave"]["must_refit_from"] = "raw .ts parsed values for the same rows before any training"
    print(f"  uwave: fit_rows={len(t1_u)} (cache already normalised; refit from raw .ts required)")

    out["cwru"] = {"fit_rows_count": None, "status": "NOT_COMPUTED_THIS_ROUND",
                   "must_refit_from": "raw .mat signals for the built Task-1 train windows",
                   "reason": "the archived npz is normalised by a single scalar fitted on the LTTSNet base split; "
                             "the new pipeline must refit per channel from raw .mat",
                   "fit_rows_sha256": None,
                   "rows_source": "cwru_window_index.csv (split == train, label in {0,1})"}
    print("  cwru: declared only (needs raw .mat; per-channel refit required)")

    dump("preprocess_statistics.json", {
        "rule": "split first, then fit. Statistics come from the Task 1 real training split only, per channel, "
                "then frozen across tasks. No validation, no probe, no future-task samples.",
        "not_carried_over": "the historical per-task refit normalization scheme is NOT used",
        "datasets": out,
        "uwave_cwru_descriptors": "NOT decided this round; no model instantiated; interfaces only",
        "required_interfaces": [
            "descriptor(x_raw) -> (n, k) using that dataset's real channels only",
            "descriptor standardization fitted on the Task 1 training split, per descriptor, then frozen",
            "sampling rate must be registered before any spectral descriptor is used",
            "UCI-HAR descriptors may use acc/gyro/total_acc at 50 Hz; WISDM has its own 8-dim set at 20 Hz; "
            "UWave and CWRU need new definitions or an explicit 'no descriptor' decision",
        ],
    })
    print("  wrote preprocess_statistics.json")


def _hash_rows(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(sorted(np.asarray(idx, dtype=np.int64).tolist())).tobytes()).hexdigest()


def _fit_stats(x: np.ndarray, idx: np.ndarray, note: str) -> dict:
    sub = x[idx]
    mean = sub.mean(axis=(0, 1), keepdims=True)
    std = sub.std(axis=(0, 1), keepdims=True) + 1e-6
    return {"fit_rows_count": int(len(idx)), "fit_rows_sha256": _hash_rows(idx),
            "per_channel_mean": [round(float(v), 8) for v in mean.ravel()],
            "per_channel_std": [round(float(v), 8) for v in std.ravel()],
            "granularity": "per channel", "scope": "Task 1 real training split only",
            "frozen_across_tasks": True, "note": note}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["hash", "wisdm", "wisdm_split", "uci_har", "uwave", "cwru", "preprocess", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase in ("hash", "all"):
        files = {"data/wisdm.zip": WISDM_ZIP, "data/wisdm_har_windows.npz": WISDM_CACHE,
                 "data/cwru_lttsnet_seed10.npz": CWRU_NPZ,
                 "data/cwru_lttsnet_seed10.manifest.json": CWRU_MANIFEST,
                 "data/uwave_gesture_library.npz": UWAVE_CACHE,
                 "data/UCI_HAR/train/subject_train.txt": UCI_DIR / "train/subject_train.txt",
                 "data/UCI_HAR/test/subject_test.txt": UCI_DIR / "test/subject_test.txt",
                 "old_split/wisdm/split_indices.npz": OLD_SPLIT,
                 "script": Path(__file__).resolve()}
        dump("input_hashes.json", {k: sha(v) for k, v in files.items()})
        print("wrote input_hashes.json")
        if a.phase == "hash":
            return 0
    for ph in (["wisdm", "wisdm_split", "uci_har", "uwave", "cwru", "preprocess"]
               if a.phase == "all" else [a.phase]):
        {"wisdm": phase_wisdm, "wisdm_split": phase_wisdm_split, "uci_har": phase_uci_har,
         "uwave": phase_uwave, "cwru": phase_cwru, "preprocess": phase_preprocess}[ph]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
