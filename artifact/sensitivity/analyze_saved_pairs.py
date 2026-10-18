#!/usr/bin/env python3
"""Recompute the explicitly post-hoc saved-pair sensitivity outputs offline.

Python >= 3.8; standard library only. This script never imports training code.
Required inputs are the two original S4 tables, the fixed plan, and its lock.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPECTED_RAW = "30723dda78f6e83ff3904be45a6213779d4b77b56c12a662d64f9dc9d9beef92"
EXPECTED_MAIN = "844248f68c5aef5ff07d8cced4ae86300b98be9ff18eddb47677c099caaaa39b"
EXPECTED_PLAN = "16f937c8f4013832d2b9bc2094f227c0c04730ee72bc55412bf63ab5dce07728"
FAMILY = 24
ALPHA = 0.05
SEEDS_8 = [11, 22, 33, 44, 55, 66, 77, 88]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def exact_sign_p(positive, negative):
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def dump_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")


def dump_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--main", dest="primary", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Validate exact plan and experimental bytes before reading numeric data.
    source_before = {"test_per_seed_raw.csv": sha256(args.raw),
                     "test_main_table.json": sha256(args.primary)}
    require(source_before["test_per_seed_raw.csv"] == EXPECTED_RAW, "Raw source hash mismatch")
    require(source_before["test_main_table.json"] == EXPECTED_MAIN, "Primary source hash mismatch")
    plan_hash = sha256(args.plan)
    lock_hash = args.lock.read_text(encoding="ascii").strip().split()[0]
    require(plan_hash == lock_hash == EXPECTED_PLAN, "Plan hash/lock mismatch")
    require(args.out.resolve() != args.raw.parent.resolve(), "Output must be separate from experiment source")
    require(args.raw.resolve() != args.primary.resolve(), "Source paths must differ")
    args.out.mkdir(parents=True, exist_ok=True)

    primary = json.loads(args.primary.read_text(encoding="utf-8-sig"))
    primary_rows = primary["rows"]
    require(len(primary_rows) == 24, "Expected 24 primary cells")
    primary_by_cell = {row["cell"]: row for row in primary_rows}
    require(len(primary_by_cell) == 24, "Duplicate primary cell")
    require(primary["holm_family"] == FAMILY, "Unexpected primary family size")

    with args.raw.open(encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    groups = {}
    for row in raw_rows:
        groups.setdefault(row["cell"], []).append(row)
    require(set(groups) == set(primary_by_cell), "Raw/primary cell identities differ")

    result_rows = []
    loo_rows = []
    for cell in sorted(groups):
        entries = sorted(groups[cell], key=lambda row: int(row["seed"]))
        saved = primary_by_cell[cell]
        seeds = [int(row["seed"]) for row in entries]
        n = len(entries)
        require(n == int(saved["n_seeds"]) and n in (3, 8), "Seed count mismatch: " + cell)
        require(seeds == (SEEDS_8 if n == 8 else SEEDS_8[:3]), "Unexpected seed set: " + cell)
        require(len(set(seeds)) == n, "Duplicate seed: " + cell)
        gains = [float(row["paired_gain_pp"]) for row in entries]
        require(all(math.isfinite(x) for x in gains), "Nonfinite gain: " + cell)
        for entry, gain in zip(entries, gains):
            require(entry["candidate"] == saved["candidate"] and
                    entry["dataset"] == saved["dataset"] and
                    entry["state"] == saved["state"], "Cell metadata mismatch: " + cell)
            cand, base = float(entry["candidate_test_macro"]), float(entry["s1_test_macro"])
            paired = float(entry["paired_gain"])
            require(all(math.isfinite(x) for x in (cand, base, paired)), "Nonfinite accuracy")
            require(0 <= cand <= 1 and 0 <= base <= 1, "Accuracy outside [0,1]")
            require(math.isclose(gain, 100 * paired, rel_tol=0, abs_tol=1e-10), "Saved gain mismatch")
            require(math.isclose(gain, 100 * (cand - base), rel_tol=0, abs_tol=1e-10), "Pair difference mismatch")
        mean = statistics.fmean(gains)
        require(math.isclose(mean, 100 * float(saved["raw_mean"]), rel_tol=0, abs_tol=1e-9),
                "Primary mean mismatch: " + cell)
        positive, negative, zero = sum(x > 0 for x in gains), sum(x < 0 for x in gains), sum(x == 0 for x in gains)
        require(positive == int(saved["strictly_positive"]), "Positive count mismatch: " + cell)
        eligible = n == 8 and saved["candidate"].startswith(("R1_", "R2_"))
        if n == 8 and not eligible:
            require(saved["candidate"].startswith("C1_") and saved["dataset"] == "wisdm",
                    "Unexpected selected eight-seed cell")
        if n == 3:
            for field in ("raw_p", "raw_ci_low", "raw_ci_high", "p_adjusted_holm24"):
                require(saved[field] is None, "Three-seed inference unexpectedly present: " + cell)

        loo = []
        if n == 8:
            for omitted_index, omitted_seed in enumerate(seeds):
                loo_mean = statistics.fmean(gains[:omitted_index] + gains[omitted_index + 1:])
                loo.append(loo_mean)
                loo_rows.append({"cell": cell, "candidate": saved["candidate"], "dataset": saved["dataset"],
                                 "omitted_seed": omitted_seed, "retained_n": 7,
                                 "mean_gain_pp": loo_mean, "full_mean_pp": mean,
                                 "sign_changed": sign(loo_mean) != sign(mean)})

        row = {"cell": cell, "candidate": saved["candidate"], "dataset": saved["dataset"],
               "n_seeds": n, "seeds": seeds, "gains_pp": gains,
               "mean_gain_pp": mean, "median_gain_pp": statistics.median(gains),
               "min_gain_pp": min(gains), "max_gain_pp": max(gains),
               "positive": positive, "negative": negative, "zero": zero,
               "near_zero_nonzero_count": sum(0 < abs(x) < 1e-10 for x in gains),
               "loo_mean_min_pp": min(loo) if loo else None,
               "loo_mean_max_pp": max(loo) if loo else None,
               "loo_sign_changed": any(sign(x) != sign(mean) for x in loo) if loo else None,
               "loo_sign_change_count": sum(sign(x) != sign(mean) for x in loo) if loo else None,
               "secondary_sign_test_eligible": eligible,
               "sign_nonzero_n": positive + negative if eligible else None,
               "sign_p_two_sided": exact_sign_p(positive, negative) if eligible else None,
               "sign_p_holm24": None, "sign_holm_rank": None,
               "secondary_reject_005": False,
               "analysis_status": "SECONDARY_READOUT_REFERENCE" if eligible else
                                  "SELECTED_POST_SCREENING_DESCRIPTIVE_ONLY" if n == 8 else
                                  "THREE_SEED_DESCRIPTIVE_ONLY"}
        result_rows.append(row)

    eight = [row for row in result_rows if row["n_seeds"] == 8]
    tested = sorted([row for row in result_rows if row["secondary_sign_test_eligible"]],
                    key=lambda row: (row["sign_p_two_sided"], row["cell"]))
    require(len(raw_rows) == 117 and len(eight) == 9 and len(tested) == 8 and len(loo_rows) == 72,
            "Unexpected analysis dimensions")
    require(sum(row["n_seeds"] == 3 for row in result_rows) == 15, "Expected 15 three-seed cells")
    running = 0.0
    for rank, row in enumerate(tested, 1):
        running = max(running, (FAMILY - rank + 1) * row["sign_p_two_sided"])
        row["sign_p_holm24"] = min(1.0, running)
        row["sign_holm_rank"] = rank
        row["secondary_reject_005"] = row["sign_p_holm24"] < ALPHA

    # Independent implementation check: literal p=1 computational placeholders.
    padded = sorted([(row["sign_p_two_sided"], row["cell"]) for row in tested] +
                    [(1.0, "RESERVED_SLOT_" + str(i)) for i in range(16)])
    padded_adj = {}
    running = 0.0
    for rank, (p_value, cell) in enumerate(padded, 1):
        running = max(running, (FAMILY - rank + 1) * p_value)
        padded_adj[cell] = min(1.0, running)
    require(all(row["sign_p_holm24"] == padded_adj[row["cell"]] for row in tested),
            "Reserved-slot Holm cross-check failed")
    require(all(not row["secondary_reject_005"] for row in tested), "Unexpected rejection at n<=8")
    require(all(not row["secondary_sign_test_eligible"] or row["sign_p_holm24"] >= 0.1875
                for row in result_rows), "Exact test resolution check failed")

    source_after = {"test_per_seed_raw.csv": sha256(args.raw),
                    "test_main_table.json": sha256(args.primary)}
    require(source_after == source_before, "Experimental source changed during analysis")
    metadata = {"analysis_type": "POST_HOC_SECONDARY_SAVED_PAIR_SENSITIVITY",
                "preregistered": False, "primary_results_seen_before_plan": True,
                "plan_sha256": plan_hash, "source_sha256_before": source_before,
                "source_sha256_after": source_after, "sources_unchanged": True,
                "script_sha256": sha256(__file__), "paired_rows": len(raw_rows),
                "cells_total": 24, "eight_seed_descriptive_cells": 9,
                "secondary_sign_test_cells": 8, "three_seed_descriptive_cells": 15,
                "leave_one_out_rows": 72, "secondary_holm_family": 24,
                "secondary_reserved_nonrejection_slots": 16,
                "minimum_two_sided_sign_p_n8": 2 / (2 ** 8),
                "minimum_possible_holm24_sign_p_n8": 24 * 2 / (2 ** 8),
                "alpha": ALPHA, "primary_t_holm_results_changed": False,
                "training_or_forward_passes": 0, "independent_model_reproduction": False}
    dump_json(args.out / "sensitivity_results.json", {"metadata": metadata, "rows": result_rows})
    dump_json(args.out / "source_hashes.json", metadata)
    fields = [key for key in result_rows[0] if key not in ("seeds", "gains_pp")]
    dump_csv(args.out / "sensitivity_all_24_cells.csv", result_rows, fields)
    dump_csv(args.out / "sensitivity_8seed_summary.csv", eight, fields)
    dump_csv(args.out / "leave_one_out_means.csv", loo_rows, list(loo_rows[0]))

    table = ["| Cell | Mean | Median | Min / max | LOO mean range | + / - / 0 | Sign p | Holm24 p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in eight:
        raw_p = "descriptive only" if row["sign_p_two_sided"] is None else f"{row['sign_p_two_sided']:.7g}"
        adj_p = "—" if row["sign_p_holm24"] is None else f"{row['sign_p_holm24']:.7g}"
        table.append(f"| {row['candidate'].split('_')[0]}/{row['dataset']} | {row['mean_gain_pp']:.4f} | "
                     f"{row['median_gain_pp']:.4f} | {row['min_gain_pp']:.4f} / {row['max_gain_pp']:.4f} | "
                     f"[{row['loo_mean_min_pp']:.4f}, {row['loo_mean_max_pp']:.4f}] | "
                     f"{row['positive']} / {row['negative']} / {row['zero']} | {raw_p} | {adj_p} |")
    stable = sum(not row["loo_sign_changed"] for row in eight)
    changed = [row["cell"] for row in eight if row["loo_sign_changed"]]
    report = """# Saved-pair post-hoc sensitivity analysis

This analysis was planned and SHA256-locked after the primary results had been seen. It is not a preregistration, a new experiment, or an independent model reproduction. The original t intervals, t p values, and primary Holm table were read only and remain unchanged.

## Coverage and results

All 117 saved paired rows were retained: 9 eight-seed cells and 15 three-seed descriptive-only cells. All 9 eight-seed cells received median/range, exact sign counts, and all 8 leave-one-out means (72 rows). Only the 8 complete readout cells received secondary exact sign-test references; selected C1/WISDM and all 15 three-seed cells have null sign p values.

Values below are percentage-point candidate-minus-B0 differences. LOO deletes one of the 8 seeds, leaving 7 pairs.

""" + "\n".join(table) + f"\n\nThe mean sign was retained after every single-seed deletion in {stable}/9 cells. Cells with any sign change: {', '.join(changed) if changed else 'none'}.\n\n" + """## Interpretation and limits

- Leave-one-out means overlap heavily and are not additional independent replicates. Retaining a mean sign under deletion is a descriptive property of these 8 observed pairs, not a confidence guarantee, distributional diagnostic, or proof of robustness to a new split, order, dataset, or seed population.
- The exact sign-test reference assumes independent nonzero paired-gain signs with probability 1/2 under the null. Its null differs from the primary t test's zero mean paired gain. No normality check or alternative test was used to select a more favorable primary result.
- For at most 8 nonzero pairs, the minimum two-sided sign p is 2/256 = 0.0078125. With the fixed 24-slot secondary family, the smallest attainable adjusted p is 0.1875. Thus the absence of adjusted rejections is forced by test resolution in this design and is not evidence of a null effect. This limited inferential resolution is more informative here than an unqualified assertion of robustness.
- The secondary family retains 24 slots: 8 computed readout p values and 16 reserved non-rejection slots. The latter retain null data fields, even though the eligible adjustments are computationally equivalent to appending 16 p=1 placeholders. This secondary computation is separate from the original primary family, which had 9 t p values and 15 reserved slots.
- C1/WISDM was promoted by screening and retained its screening seeds. It is descriptive only here; neither t intervals in the original paper nor this sensitivity analysis resolve selection bias.
- Exact zero means exactly zero in the saved CSV; values were not thresholded. Nonzero values smaller than 1e-10 pp are explicitly counted in the complete output.

## Reproduction and provenance

Run with Python 3.8 or newer and no third-party packages:

```text
python analyze_saved_pairs.py --raw <test_per_seed_raw.csv> --main <test_main_table.json> --plan POSTHOC_ANALYSIS_PLAN.md --lock POSTHOC_ANALYSIS_PLAN.sha256 --out <separate_output_directory>
```

The plan SHA256 is `""" + plan_hash + """`. Source and script hashes, dimension checks, and before/after identities are in `source_hashes.json`. The script also checks the Holm result through a separate padded-vector implementation. Local existence and successful offline recomputation do not establish a public download URL or independent model reproduction.
"""
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")

    def get(candidate_prefix, dataset):
        return next(row for row in eight if row["candidate"].startswith(candidate_prefix) and row["dataset"] == dataset)
    r1u, r2u, c1w = get("R1_", "uwave"), get("R2_", "uwave"), get("C1_", "wisdm")
    range_text = lambda row: f"[{row['loo_mean_min_pp']:.4f}, {row['loo_mean_max_pp']:.4f}]"
    paragraph = ("A post-hoc sensitivity analysis used all saved eight-seed paired differences without refitting models "
                 "or changing the primary t-based analysis. "
                 f"The mean sign was unchanged after every single-seed deletion in {stable} of the 9 cells. "
                 f"For R1 and R2 on UWave, median gains were {r1u['median_gain_pp']:.4f} and {r2u['median_gain_pp']:.4f} pp, "
                 f"and the eight seven-seed means ranged over {range_text(r1u)} and {range_text(r2u)} pp, respectively. "
                 f"For selected C1/WISDM, the median was {c1w['median_gain_pp']:.4f} pp and the leave-one-out mean range was "
                 f"{range_text(c1w)} pp; this cell remains descriptive only. "
                 "Secondary two-sided exact sign-test references were computed for the 8 fully covered readout cells, "
                 "with 16 reserved non-rejection slots in a 24-slot Holm family. Their null concerns positive-sign "
                 "probability among nonzero gains, rather than the t test's mean gain. At n<=8, the smallest possible "
                 "two-sided p value is 0.0078125 and its smallest possible 24-slot Holm adjustment is 0.1875; "
                 "therefore no secondary adjusted rejection at 0.05 is attainable. This finite resolution does not "
                 "establish absence of an effect. Leave-one-out summaries are overlapping descriptions of the same "
                 "pairs, not additional independent observations, and neither analysis removes screening-selection bias.")
    (args.out / "MANUSCRIPT_SUMMARY_EN.md").write_text(paragraph + "\n\n" + "\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sources_unchanged": True, "cells": 24,
                      "eight_seed_cells": 9, "sign_tests": 8, "loo_rows": 72,
                      "loo_sign_stable_cells": stable, "loo_sign_changed_cells": changed,
                      "plan_sha256": plan_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
