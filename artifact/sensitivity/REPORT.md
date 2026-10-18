# Saved-pair post-hoc sensitivity analysis

This analysis was planned and SHA256-locked after the primary results had been seen. It is not a preregistration, a new experiment, or an independent model reproduction. The original t intervals, t p values, and primary Holm table were read only and remain unchanged.

## Coverage and results

All 117 saved paired rows were retained: 9 eight-seed cells and 15 three-seed descriptive-only cells. All 9 eight-seed cells received median/range, exact sign counts, and all 8 leave-one-out means (72 rows). Only the 8 complete readout cells received secondary exact sign-test references; selected C1/WISDM and all 15 three-seed cells have null sign p values.

Values below are percentage-point candidate-minus-B0 differences. LOO deletes one of the 8 seeds, leaving 7 pairs.

| Cell | Mean | Median | Min / max | LOO mean range | + / - / 0 | Sign p | Holm24 p |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1/wisdm | -0.4183 | -0.3421 | -6.0353 / 3.1704 | [-0.9309, 0.3842] | 2 / 6 / 0 | descriptive only | — |
| R1/cwru | -9.9250 | -8.8750 | -15.9000 / -5.7000 | [-10.5286, -9.0714] | 0 / 8 / 0 | 0.0078125 | 0.1875 |
| R1/uci_har | 1.3491 | 1.6646 | -0.8446 / 2.4762 | [1.1880, 1.6625] | 7 / 1 / 0 | 0.0703125 | 1 |
| R1/uwave | 3.9063 | 3.9062 | 0.9375 / 7.8125 | [3.3482, 4.3304] | 8 / 0 / 0 | 0.0078125 | 0.1875 |
| R1/wisdm | -8.5753 | -8.6667 | -13.5795 / -4.9812 | [-9.0887, -7.8604] | 0 / 8 / 0 | 0.0078125 | 0.1875 |
| R2/cwru | -10.0875 | -9.0500 | -15.9000 / -5.9000 | [-10.6857, -9.2571] | 0 / 8 / 0 | 0.0078125 | 0.1875 |
| R2/uci_har | -1.0344 | -0.7083 | -2.9393 / 0.7397 | [-1.2879, -0.7623] | 2 / 6 / 0 | 0.2890625 | 1 |
| R2/uwave | 3.1250 | 2.8125 | -0.3125 / 6.8750 | [2.5893, 3.6161] | 7 / 1 / 0 | 0.0703125 | 1 |
| R2/wisdm | -9.2848 | -9.0767 | -13.7620 / -4.2024 | [-10.0109, -8.6453] | 0 / 8 / 0 | 0.0078125 | 0.1875 |

The mean sign was retained after every single-seed deletion in 8/9 cells. Cells with any sign change: C1_backbone_freeze__wisdm.

## Interpretation and limits

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

The plan SHA256 is `16f937c8f4013832d2b9bc2094f227c0c04730ee72bc55412bf63ab5dce07728`. Source and script hashes, dimension checks, and before/after identities are in `source_hashes.json`. The script also checks the Holm result through a separate padded-vector implementation. Local existence and successful offline recomputation do not establish a public download URL or independent model reproduction.
