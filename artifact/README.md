# Fixed-protocol time-series class-incremental evaluation: reproducibility supplement v1

This directory is a standalone public distribution copy prepared on 2026-10-19.
It accompanies an evaluation of readout changes on four selected datasets under
one fixed split/order protocol. It is not a package claiming a new general method.
Original experiment records and previous manuscript versions were not rewritten.

## What can be reproduced offline

1. All 117 saved cell/seed scalar observations, covering 24 candidate/dataset cells.
2. The nine eight-seed rows: paired mean, sample SD, positive counts, nominal df=7
   95% t interval, two-sided t p value, and Holm adjustment in the fixed family 24.
3. Explicit null inferential fields for all fifteen three-seed descriptive rows.
4. The saved S4 rank associations using four dataset units, never 32 seed units.
5. Numerical inputs and the quantitative layouts of Figures 2–4.
6. The separate, post-hoc saved-pair sensitivity analysis (median/range,
   single-seed deletions and secondary sign-test references), with its own plan lock.

These are recomputations of stored scalar evidence. They do not re-evaluate a
model on test data and do not establish independent replication from raw data.

## Quick start from a fresh copy

Use Python 3.10 or newer. The reported execution environment is retained in
`paper_tables/reproducibility.csv`; the lightweight audit does not require CUDA,
PyTorch, ncps, datasets, model checkpoints, or network access after installation.

```sh
python -m venv .venv
# Activate .venv using the command appropriate to your operating system.
python -m pip install -r requirements.txt
python scripts/verify_manifest.py
python scripts/recompute_tables.py --output ../table_recomputation.json
python scripts/plot_quantitative_figures.py --output ../rendered_quantitative_figures
python sensitivity/analyze_saved_pairs.py --raw records/s3/test_s4/test_per_seed_raw.csv --main records/s3/test_s4/test_main_table.json --plan sensitivity/POSTHOC_ANALYSIS_PLAN.md --lock sensitivity/POSTHOC_ANALYSIS_PLAN.sha256 --out ../saved_pair_sensitivity_recomputed
```

Run commands from this directory. The first two audit scripts resolve inputs from
their own location and also support `--root PATH`, so they can run from any cwd.
Use a new output directory for the plotting and sensitivity commands. Outputs are
kept outside this immutable distribution so `SHA256SUMS.txt` remains meaningful.
The dependency installation command accesses package indexes; none of the scripts
above uploads manuscript text, records, credentials, or raw data.

`scripts/recompute_tables.py` must exit 0 and report `PASS`. Its detailed JSON
lists every comparison and input hash. Source floats are compared at full
precision, while the saved table's declared six-/twelve-decimal display fields
are compared at their corresponding rounding precision. The statistical inputs
are read from the raw CSV rather than copied from the saved main table.

## Directory map

| Directory/file | Contents |
|---|---|
| `records/s3/test_s4/test_per_seed_raw.csv` | Original scalar CSV, byte-identical |
| `records/s3/test_s4/test_main_table.json` | Original 24-row reported table, byte-identical |
| `records/s3/aggregate.json` | Original aggregate, byte-identical |
| `records/s3/` | Selection, evaluation and locked scientific protocol records |
| `records/s2/` | Readout-view sensitivity endpoints and boundary summaries |
| `paper_tables/` | All eleven manuscript CSV tables, byte-identical |
| `figure_sources/` | Numeric source ledger for quantitative Figures 2–4 |
| `figures/` | Original exported quantitative Figures 2–4 |
| `sensitivity/` | Separate post-hoc plan, its SHA256 lock, script and saved outputs |
| `code_reference/` | Authored training/preprocessing source and inspected archive dependencies |
| `IMPORT_DEPENDENCIES.json` | Static source dependency inventory and stated limitations |
| `SOURCE_MANIFEST.json` | Original-source and distribution hashes plus any path anonymization |
| `SHA256SUMS.txt` | Checksums of all included files except itself |
| `LICENSE_SCOPE.md` | Existing license/attribution scope; no invented blanket license |
| `DATA_SOURCES.md` | Dataset references, expected layout, and reconstruction limitations |

## Statistical interpretation

Seed pairs share the same fixed split, task order and protocol. Seed-level
intervals describe training randomness conditional on those choices; they do not
sample a population of datasets, orders, or subjects. Intervals are nominal,
not simultaneous. R1/R2 are composite implementation contrasts against B0:
loss, regularization, solver and (for R2) dimension change together.

Only nine rows reached eight seeds. The main adjustment orders those nine p
values and uses weights 24 through 16. Keeping fifteen additional p=1 entries
would give the same adjustments for the nine observed p values, but the saved
three-seed rows retain null p and interval fields. No three-seed rejection is
reported. The selected C1/WISDM row reuses the screening seeds and is not
selection-adjusted; neither Holm weighting nor additional descriptive checks
convert it into an independent confirmation.

The post-hoc sign-test reference has a different null from the primary t test.
Its smallest attainable two-sided raw p at eight nonzero pairs is 0.0078125;
the family-24 first-step reference is 0.1875. Its lack of adjusted significance
therefore does not establish zero effect or negate the saved primary analysis.
Its plan was locked after the primary results were known, not preregistered.

## Training code: available for inspection, not validated end-to-end here

`code_reference/` preserves the project-relative layout used by the actual
accepted S1/S2/S3 implementation. The static inventory follows local imports at
all nesting levels, manually accounts for file-based `importlib` loaders, and
includes the archived pipeline, both benchmark roles, and their configuration
files. The stage4 benchmark intentionally remains at its archive-relative path
because it reads its neighboring configuration at module scope. This is not a
blind copy of the full project or of third-party repositories.

The original author-side entry points, supplied for inspection, are:

```sh
python code_reference/src/preflight_trunk_data.py --help
python code_reference/src/trunk/run_baseline_fix2.py --help
python code_reference/src/trunk/readout_sensitivity.py --help
python code_reference/src/trunk/s3_cli_fix6.py --help
```

They require the optional model dependencies and further inputs. **They were not
executed or advertised as turnkey retraining commands in this release.** Raw
archives, normalized caches, all trained checkpoints, S2 feature/probe arrays,
some historical protocol/audit inputs and third-party implementations are absent.
The original protective hash gates can intentionally reject an incomplete or
relocated tree. Sanitized historical locks are provenance copies, not newly
valid execution seals, and should never be edited merely to make a run pass.
The public supplement supplies fixed split indices and preprocessing statistics
but does not promise that those files alone recreate every cache and prior seal.

To attempt a new independent training replication, first retrieve the official
datasets, reconstruct the exact subset and layout in `DATA_SOURCES.md`, install
and record compatible model dependencies, resolve the explicitly missing prior
inputs, then create a **new** run directory and lock a new replication protocol.
Do not overwrite the saved results or describe such a new run as the historical
run. No end-to-end retraining, runtime compatibility or predictive equivalence
for that new environment has been verified here.

## Anonymization and provenance

No raw signal samples, checkpoint/weight files, `.git` directory, external
third-party repository, account credentials, personal email, username, or
machine-specific project/home prefix is included. Dataset subject identifiers
in derived split/provenance metadata are pseudonymous dataset identifiers,
not author identities. Source-level copyright/license notices are retained.
Machine path strings in selected distribution copies use `${PROJECT_ROOT}` or
`${USER_HOME}` and both original and changed hashes are recorded. Numerical
values are not anonymized or recomputed in those copies.

This content-level anonymization is not a claim of anonymous hosting: the public
repository owner's account identifies the host. Public posting does not itself
establish a dataset redistribution license, a blanket software license, a DOI,
archival permanence, or journal acceptance.
