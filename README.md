# Fixed-protocol time-series class-incremental evaluation

Public code and saved-evidence distribution, v1 (2026-10-19).

- [Reproducibility supplement and exact commands](artifact/README.md)
- [Download the anonymized supplement ZIP](downloads/reproducibility_supplement_v1_20261019.zip)
- [Original saved scalar observations](artifact/records/s3/test_s4/test_per_seed_raw.csv)
- [Portable primary-table recomputation](artifact/scripts/recompute_tables.py)
- [Post-hoc sensitivity plan and results](artifact/sensitivity/REPORT.md)
- [Training/preprocessing source reference](artifact/code_reference)
- [Data sources and reconstruction limits](artifact/DATA_SOURCES.md)
- [License and attribution scope](artifact/LICENSE_SCOPE.md)

## Verify the saved results

```sh
cd artifact
python -m pip install -r requirements.txt
python scripts/verify_manifest.py
python scripts/recompute_tables.py --output ../table_recomputation.json
```

The distribution preserves 117 scalar observations across 24 cells. Nine eight-seed cells have the original paired t summaries; fifteen three-seed cells retain null inferential fields. Secondary saved-pair sensitivity analyses are identified as post hoc and do not replace the primary analysis.

Offline scalar and figure recomputation has been checked in a separate directory using the installed scientific Python environment. This is not an independent model retraining or a newly provisioned environment test. Raw datasets, checkpoints and third-party repositories are not included. The public repository identifies its owner; the ZIP content omits owner links and personal machine paths, and may be used as a separate anonymized review supplement. No OpenReview submission or journal acceptance is implied. Public visibility is not an invented open-source license grant; see the attribution scope above.
