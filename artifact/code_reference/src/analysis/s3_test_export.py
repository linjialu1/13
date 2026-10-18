"""S3 test + S4 export — registered main table, descriptive attachment, S4 units.

Reads the sealed `aggregate.json` (never rewrites it) and re-derives every published
number from `test_results.json` + accepted S1 references + accepted S2 endpoints, with an
independent scipy t-test / Holm / Spearman recomputation as a cross-check.

Rules enforced here (REVIEW_s3_sealed_20261017.md §4):

  * 9 cells with 8 seeds get mean, SD(ddof=1), strictly-positive count, nominal df7 95 %
    paired CI, two-sided raw p, and Holm-adjusted p over the FIXED family of 24;
  * the 15 SCREEN_FAIL cells keep their 3-seed values as a descriptive attachment only —
    ci95 / raw p / adjusted p are written as explicit nulls;
  * `aggregate.json` stores the CELL state in the row's `state` field, which overwrites the
    statistic's degenerate marker; the export re-derives the degeneracy flag from SD/mean
    instead of trusting that field;
  * R2 - R1 is a same-dataset same-seed descriptive contrast and is NOT part of family 24;
  * S4 uses 4 top-level dataset units, average ranks for ties, rho null when constant.

Outputs (all new, under <exec>/test_s4/):
    test_main_table.csv / .json
    test_screen_fail_3seed.csv
    test_per_seed_raw.csv
    test_row_asset_manifest.csv
    r2_minus_r1.csv
    s4_units.csv / s4_summary.json
    statistics_recompute_check.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EX = ROOT / "experiments/matrix/trunk_s3_run_ucihar_seed11"
OUT = EX / "test_s4"
S2 = ROOT / "experiments/sensitivity/trunk_readout_sensitivity_4datasets"
S2_ACC = ROOT / "experiments/sensitivity/trunk_readout_sensitivity_acceptance_recovery"
DATASETS = ["uci_har", "wisdm", "uwave", "cwru"]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
TASK_COUNTS = {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}
READOUT = ("R1_analytic_head", "R2_pca16_readout")
TCRIT = 2.3646242515927853
FAMILY = 24


def sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    """Empty cell would read as a missing value; a null statistic is written as NULL."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("NULL" if r.get(k) is None else r.get(k)) for k in fields})


def fmt(x, nd=6):
    return None if x is None else float(f"{x:.{nd}f}")


def main() -> int:
    seal = read_json(EX / "candidate_seal.json")
    agg = read_json(EX / "aggregate.json")
    results = read_json(EX / "test_results.json")
    s1 = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        for s in SEEDS:
            s1[ds][s] = float(read_json(
                ROOT / f"experiments/baselines/trunk_baseline_4datasets/{ds}/seed{s}"
                       "/evaluation.json")["final_task_seen_macro_accuracy"])
    s2_mean = {}
    for ds in DATASETS:
        vals = []
        for s in SEEDS:
            bm = read_json(S2 / ds / f"seed{s}" / f"task{TASK_COUNTS[ds]}"
                           / "boundary_metrics.json")
            vals.append(float(bm["range_4"]))
        s2_mean[ds] = float(np.mean(vals))

    from scipy import stats as st

    OUT.mkdir(parents=True, exist_ok=True)
    checks = {"cells": {}, "problems": []}

    def bad(msg):
        checks["problems"].append(msg)

    # ---------------------------------------------------------------- main table
    main_rows, per_seed_rows, manifest_rows = [], [], []
    pvals = {}
    for cell in sorted(agg["cells"]):
        a = agg["cells"][cell]
        rec = seal["cells"][cell]
        cand, ds = a["candidate"], a["dataset"]
        seeds = sorted(int(x) for x in results[cell])
        candv = [float(results[cell][str(s)]) for s in seeds]
        basev = [s1[ds][s] for s in seeds]
        diff = [c - b for c, b in zip(candv, basev)]
        if [round(x, 15) for x in diff] != [round(x, 15) for x in a["paired_gain"]]:
            bad(f"{cell}: paired gain vector differs from aggregate")
        mean = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1))
        se = sd / math.sqrt(len(diff)) if len(diff) > 1 else None
        n_pos = int(sum(1 for x in diff if x > 0))
        state = rec["state"]
        row = {"cell": cell, "candidate": cand, "dataset": ds, "final_task":
               TASK_COUNTS[ds], "state": state, "n_seeds": len(seeds),
               "seeds": " ".join(str(s) for s in seeds),
               "raw_mean": mean, "raw_sd": sd,
               "mean_gain": fmt(mean), "mean_gain_pp": fmt(mean * 100),
               "sd_gain": fmt(sd), "sd_gain_pp": fmt(sd * 100),
               "strictly_positive": n_pos}
        # degenerate flag re-derived (aggregate's row["state"] is the CELL state)
        if sd == 0.0:
            deg = "DEGENERATE_ZERO_DIFF" if mean == 0.0 else "DEGENERATE_CONSTANT_NONZERO"
            row["degenerate_flag"] = deg
        else:
            row["degenerate_flag"] = ""

        if len(seeds) == 8:
            hi = mean + TCRIT * sd / math.sqrt(8)
            lo = mean - TCRIT * sd / math.sqrt(8)
            if sd == 0.0:
                p = 1.0 if mean == 0.0 else 0.0
                tval = None
            else:
                tval = mean / se
                p = float(2 * st.t.sf(abs(tval), df=7))
            row.update({"raw_ci_low": lo, "raw_ci_high": hi,
                        "ci95_low": fmt(lo), "ci95_high": fmt(hi),
                        "ci95_low_pp": fmt(lo * 100), "ci95_high_pp": fmt(hi * 100),
                        "t_stat": fmt(tval), "raw_p": fmt(p, 12),
                        "stat_state": "OK" if sd else row["degenerate_flag"]})
            pvals[cell] = p
            row["cell_type"] = "main_8seed"
        else:
            row.update({"raw_ci_low": None, "raw_ci_high": None,
                        "ci95_low": None, "ci95_high": None,
                        "ci95_low_pp": None, "ci95_high_pp": None,
                        "t_stat": None, "raw_p": None,
                        "stat_state": "NULL_THREE_SEED_DESCRIPTIVE",
                        "cell_type": "screen_fail_3seed"})
        main_rows.append(row)

        for s, c, b, d in zip(seeds, candv, basev, diff):
            rec_row = {"cell": cell, "candidate": cand, "dataset": ds, "seed": s,
                       "state": state, "candidate_test_macro": c,
                       "s1_test_macro": b, "paired_gain": d,
                       "paired_gain_pp": d * 100,
                       "kind": "readout" if cand in READOUT else "trained"}
            per_seed_rows.append(rec_row)
            adir = Path(rec["assets"][str(s)]["dir"])
            npz = adir / f"test_task{TASK_COUNTS[ds]}.npz"
            manifest_rows.append({
                "cell": cell, "candidate": cand, "dataset": ds, "seed": s,
                "test_npz": str(npz), "npz_sha256": sha(npz),
                "npz_bytes": npz.stat().st_size,
                "result_json": str(adir / "result.json"),
                "result_sha256": sha(adir / "result.json"),
                "final_task": TASK_COUNTS[ds]})

    # ---------------------------------------------------------------- Holm (fixed 24)
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    holm = {}
    running = 0.0
    for i, (cell, p) in enumerate(items, start=1):
        running = max(running, (FAMILY - i + 1) * p)
        holm[cell] = {"raw_p": p, "adjusted_p": float(min(1.0, running)),
                      "rank": i, "weight": FAMILY - i + 1}
    for row in main_rows:
        if row["cell"] in holm:
            row["p_adjusted_holm24"] = fmt(holm[row["cell"]]["adjusted_p"], 12)
            row["holm_rank"] = holm[row["cell"]]["rank"]
        else:
            row["p_adjusted_holm24"] = None
            row["holm_rank"] = None

    # cross-check against the locked aggregate
    if len(pvals) != 9:
        bad(f"expected 9 cells with 8 seeds, got {len(pvals)}")
    for cell, h in holm.items():
        got = agg["holm_family_24"]["results"].get(cell)
        if got is None:
            bad(f"aggregate misses holm entry for {cell}")
        else:
            if abs(got["raw_p"] - h["raw_p"]) > 1e-15:
                bad(f"raw p differs for {cell}")
            if abs(got["adjusted_p"] - h["adjusted_p"]) > 1e-15:
                bad(f"holm p differs for {cell}")
    if agg["holm_family_24"]["family_size"] != FAMILY:
        bad("aggregate holm family size != 24")
    for cell in agg["cells"]:
        if agg["cells"][cell]["n_seeds"] != 8 and \
                agg["cells"][cell].get("p") is not None:
            bad(f"{cell}: 3-seed cell carries a p value")
    # CI cross-check (raw values, not the rounded display strings)
    for row in main_rows:
        if row["cell_type"] != "main_8seed":
            continue
        a = agg["cells"][row["cell"]]
        if abs(a["ci95"][0] - row["raw_ci_low"]) > 1e-12 or \
                abs(a["ci95"][1] - row["raw_ci_high"]) > 1e-12:
            bad(f"{row['cell']}: CI differs from aggregate")
        if abs(a["sd"] - row["raw_sd"]) > 1e-12:
            bad(f"{row['cell']}: SD differs from aggregate")
        if abs(a["mean"] - row["raw_mean"]) > 1e-12:
            bad(f"{row['cell']}: mean differs from aggregate")
        if a["strictly_positive"] != row["strictly_positive"]:
            bad(f"{row['cell']}: strictly_positive differs")
        if a.get("state") != row["state"]:
            bad(f"{row['cell']}: cell state differs from seal")

    checks["holm"] = {"family_size": FAMILY, "n_cells_corrected": len(holm),
                      "cells": {k: {"raw_p": v["raw_p"],
                                    "adjusted_p": v["adjusted_p"],
                                    "rank": v["rank"]} for k, v in holm.items()},
                      "note": "fixed family 24; only the 9 cells that reached 8 seeds "
                              "receive p-values; weights run 24 down to 16"}
    checks["status"] = "PASS" if not checks["problems"] else "FAIL"

    # ---------------------------------------------------------------- R2 - R1
    r2r1_rows = []
    r2r1_stats = {}
    for ds in DATASETS:
        c1, c2 = f"R1_analytic_head__{ds}", f"R2_pca16_readout__{ds}"
        d = [float(results[c2][str(s)]) - float(results[c1][str(s)]) for s in SEEDS]
        m = float(np.mean(d)); sd = float(np.std(d, ddof=1))
        t = m / (sd / math.sqrt(8)) if sd else None
        p = float(2 * st.t.sf(abs(t), df=7)) if sd else (1.0 if m == 0 else 0.0)
        r2r1_stats[ds] = {"mean": m, "sd": sd, "t": t, "p": p,
                          "ci95": [m - TCRIT * sd / math.sqrt(8),
                                   m + TCRIT * sd / math.sqrt(8)]}
        for s, v in zip(SEEDS, d):
            r2r1_rows.append({"dataset": ds, "seed": s,
                              "r2_test_macro": float(results[c2][str(s)]),
                              "r1_test_macro": float(results[c1][str(s)]),
                              "r2_minus_r1": v, "r2_minus_r1_pp": v * 100})
    checks["r2_minus_r1_note"] = ("descriptive paired contrast, NOT part of family 24; "
                                  "no adjusted p, no rejection claim")

    # ---------------------------------------------------------------- S4
    s4_rows, s4 = [], {}
    for cand in READOUT:
        gains = []
        for ds in DATASETS:
            cell = f"{cand}__{ds}"
            g = float(agg["cells"][cell]["mean"])
            gains.append(g)
            s4_rows.append({"candidate": cand, "dataset": ds,
                            "s2_endpoint_range_4_mean": s2_mean[ds],
                            "s2_endpoint_range_4_mean_pp": s2_mean[ds] * 100,
                            "test_gain_mean": g, "test_gain_mean_pp": g * 100,
                            "n_seeds": 8})
        from trunk import s3_candidates_numpy_fix1 as cn
        rho = cn.spearman([s2_mean[d] for d in DATASETS], gains)
        s4[cand] = {"rho": rho, "gains": gains}
        agg_rho = agg["s4"]["spearman_" + cand.split("_")[0]]["rho"]
        if agg_rho is None or rho["rho"] is None:
            if agg_rho is not None or rho["rho"] is not None:
                bad(f"S4 constant/null status differs from aggregate for {cand}")
        elif abs(agg_rho - rho["rho"]) > 1e-12:
            bad(f"S4 rho differs from aggregate for {cand}")
    checks["s4"] = {"units": 4, "n_s2_endpoint_files": 32,
                    "per_dataset_range_4_mean": s2_mean,
                    "rho_R1": s4["R1_analytic_head"]["rho"],
                    "rho_R2": s4["R2_pca16_readout"]["rho"],
                    "note": "top-level units are the 4 datasets; n=4 gives no "
                            "population inference and no n=88 independence"}

    # ---------------------------------------------------------------- write
    fields_main = ["cell", "candidate", "dataset", "final_task", "state", "cell_type",
                   "n_seeds", "seeds", "mean_gain", "mean_gain_pp", "sd_gain",
                   "sd_gain_pp", "strictly_positive", "ci95_low", "ci95_high",
                   "ci95_low_pp", "ci95_high_pp", "t_stat", "raw_p",
                   "p_adjusted_holm24", "holm_rank", "stat_state", "degenerate_flag"]
    write_csv(OUT / "test_main_table.csv", main_rows, fields_main)
    (OUT / "test_main_table.json").write_text(json.dumps(
        {"note": "registered main table; 9 cells = 8 seeds with full paired statistics, "
                 "15 cells = SCREEN_FAIL with 3-seed descriptive values and null p/CI",
         "t_crit_df7": TCRIT, "holm_family": FAMILY, "rows": main_rows},
        indent=1, ensure_ascii=False), encoding="utf-8")

    write_csv(OUT / "test_screen_fail_3seed.csv",
              [r for r in main_rows if r["cell_type"] == "screen_fail_3seed"],
              ["cell", "candidate", "dataset", "state", "n_seeds", "seeds",
               "mean_gain", "mean_gain_pp", "sd_gain", "strictly_positive",
               "stat_state"])
    write_csv(OUT / "test_per_seed_raw.csv", per_seed_rows,
              ["cell", "candidate", "dataset", "seed", "state", "kind",
               "candidate_test_macro", "s1_test_macro", "paired_gain", "paired_gain_pp"])
    write_csv(OUT / "test_row_asset_manifest.csv", manifest_rows,
              ["cell", "candidate", "dataset", "seed", "final_task", "test_npz",
               "npz_sha256", "npz_bytes", "result_json", "result_sha256"])
    write_csv(OUT / "r2_minus_r1.csv", r2r1_rows,
              ["dataset", "seed", "r2_test_macro", "r1_test_macro",
               "r2_minus_r1", "r2_minus_r1_pp"])
    (OUT / "r2_minus_r1_stats.json").write_text(json.dumps({
        "note": "descriptive paired contrast per dataset; NOT part of family 24; "
                "nominal p only, no adjusted p, no rejection claim",
        "t_crit_df7": TCRIT, "n_seeds": 8,
        "per_dataset": {d: {"mean": v["mean"], "mean_pp": v["mean"] * 100,
                            "sd": v["sd"], "sd_pp": v["sd"] * 100,
                            "ci95": v["ci95"], "ci95_pp": [x * 100 for x in v["ci95"]],
                            "t": v["t"], "nominal_p": v["p"]}
                        for d, v in r2r1_stats.items()},
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    write_csv(OUT / "s4_units.csv", s4_rows,
              ["candidate", "dataset", "s2_endpoint_range_4_mean",
               "s2_endpoint_range_4_mean_pp", "test_gain_mean", "test_gain_mean_pp",
               "n_seeds"])
    (OUT / "s4_summary.json").write_text(json.dumps({
        "note": "descriptive only; 4 top-level units; no n=88 inference, no causal claim",
        "units": 4,
        "r1": s4["R1_analytic_head"]["rho"],
        "r2": s4["R2_pca16_readout"]["rho"],
        "per_dataset": s4_rows,
        "aggregate_json_rho": {"r1": agg["s4"]["spearman_R1"],
                               "r2": agg["s4"]["spearman_R2"]},
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    (OUT / "statistics_recompute_check.json").write_text(
        json.dumps(checks, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": checks["status"], "n_main": len(main_rows),
                      "n_8seed": len(pvals), "n_3seed": len(main_rows) - len(pvals),
                      "holm_min_weight": FAMILY - len(pvals) + 1,
                      "rho_R1": s4["R1_analytic_head"]["rho"]["rho"],
                      "rho_R2": s4["R2_pca16_readout"]["rho"]["rho"],
                      "r2_minus_r1_means_pp": {d: round(v["mean"] * 100, 4)
                                               for d, v in r2r1_stats.items()},
                      "problems": checks["problems"][:8]}, indent=1, ensure_ascii=False))
    return 0 if checks["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
