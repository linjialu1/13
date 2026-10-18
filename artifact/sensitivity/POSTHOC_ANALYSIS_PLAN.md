# Saved-pair sensitivity analysis plan

Date: 2026-10-19 (Asia/Shanghai). Status: post hoc, fixed before this sensitivity computation. This is not a preregistration: the primary results, review comments, and a few source CSV rows were already visible when this plan was written.

## Inputs and scope

Read only these existing experimental outputs, verified by SHA256:

- `test_per_seed_raw.csv`: `30723dda78f6e83ff3904be45a6213779d4b77b56c12a662d64f9dc9d9beef92`.
- `test_main_table.json`: `844248f68c5aef5ff07d8cced4ae86300b98be9ff18eddb47677c099caaaa39b`.

Both are from `experiments/matrix/trunk_s3_run_ucihar_seed11/test_s4/`. The analysis uses the saved final-task candidate-minus-B0 paired gain in percentage points. No model fitting, network forward pass, prediction selection, change to the primary t test/Holm results, or rewrite of the experimental sources is authorized by this plan.

Validate source identities, 24 distinct cells, 9 cells with 8 distinct paired seeds, 15 cells with 3 distinct seeds, finite numeric inputs, and the consistency of saved paired gains and summary means. Halt on a validation failure instead of changing the criteria.

## Fixed descriptive outputs

For each of the 9 eight-seed cells report all saved gains, mean, median, minimum, maximum, and strictly positive/negative/exact-zero counts. A zero is the saved floating-point value exactly equal to zero; no tolerance-based recoding will be introduced. Flag any nonzero absolute value below `1e-10` pp without changing its sign.

For each eight-seed cell, delete each seed in turn and calculate the eight seven-seed means. Report every omitted seed and resulting mean, their minimum and maximum, and whether any deletion changes the full-mean sign. These overlapping leave-one-out summaries describe sensitivity to individual observations; they are not additional independent observations or replicated experiments. Retain the 15 three-seed cells as descriptive-only records with all new inferential fields null.

## Fixed secondary sign analysis

Apply a two-sided exact sign test only to the eight readout cells (R1 and R2 on all four datasets), all of which had complete eight-seed coverage. Drop exact-zero gains from the sign statistic and report the resulting nonzero sample size. Under the reference null, independent nonzero paired gains have positive-sign probability 1/2. For `b = min(n_positive, n_negative)` and `n = n_nonzero`, use `min(1, 2 * sum(comb(n, k), k=0..b) / 2**n)`; use p=1 if n=0.

Retain a 24-slot secondary comparison family. Sort the eight eligible p values by `(p, cell)` and apply monotone Holm multipliers 24, 23, ..., 17. The other 16 slots (the 15 three-seed cells and selected C1/WISDM) are reserved non-rejections and retain null p values. This is computationally equivalent for the eligible adjustments to appending 16 p=1 values, without storing fabricated p values for untested cells. C1/WISDM is a post-screening observation and receives descriptive outputs only, including null raw and adjusted sign p values.

The sign null concerns the probability of a positive nonzero difference, unlike the primary t test's zero mean difference. The two procedures need not agree. With at most 8 nonzero pairs, the smallest two-sided exact p value is `2/2**8 = 0.0078125`. With 24-slot Holm correction, the smallest possible adjusted p is `24 * 0.0078125 = 0.1875`, so rejection at 0.05 is unattainable in this secondary family. This finite resolution and limited sample size preclude treating a non-rejection as evidence of no effect. The secondary analysis does not replace or upgrade the primary analysis and does not correct candidate-selection bias.

## Outputs and integrity checks

Create a portable standard-library-only `analyze_saved_pairs.py`, full JSON and CSV results, all leave-one-out rows, English manuscript-ready descriptive summary, `REPORT.md`, and source hashes. Preserve all results regardless of signs or p values. Save the bytes of this plan and its SHA256 lock before running the computation. After completion, verify both experimental source hashes remain identical, include the script/plan hashes and command, and distinguish deterministic numerical reproducibility from independent model reproduction or public release.
