# Data sources and preprocessing scope

Original dataset archives and amplitudes are not redistributed. These references
identify the datasets used in the manuscript; access terms belong to their
respective maintainers. This file is not a license determination.

| Dataset | Official reference | Protocol instance in this evaluation |
|---|---|---|
| UCI-HAR | https://doi.org/10.24432/C54S4K | 6 classes, 9 channels, 128 samples; official test boundary; 16 training / 5 validation / 9 test subjects; 5564 / 1788 / 2947 rows |
| WISDM | https://doi.org/10.24432/C5HK59 | 2019 smartphone/smartwatch archive, phone acceleration only, activity codes A–F; 6 classes, 3 channels, 128-sample windows, stride 64; 32 / 8 / 11 subjects and 15014 / 3547 / 5696 rows |
| UWaveGestureLibrary | https://doi.org/10.5281/zenodo.11206282 | Specific multivariate archive with 120 official training and 320 test cases, 8 classes, 3 channels, length 315; fixed 96/24 training/validation rows (12/3 per class), 40 test cases per class |
| CWRU | https://engineering.case.edu/bearingdatacenter/download-data-file | 10 selected records/classes, drive-end signals, windows 1024; 1000/200/2000 training/validation/test windows; nonoverlapping raw intervals but same-record segment evaluation |

The WISDM F code is typing in this selected six-code protocol. Do not replace the
dataset with another WISDM release or infer a different six-activity label list
from the common dataset name. UWave has no subject identifier in the supplied
split; row separation does not establish subject independence. CWRU interval
separation does not establish independent operating-condition or record holdout.

For the source tree, expected filenames under `code_reference/data/` are:

```text
UCI HAR Dataset/{train,test}/... (official text files and subject files)
wisdm.zip
wisdm_har_windows.npz                 # derived normalized historical cache, absent
UWaveGestureLibrary_TRAIN.ts
UWaveGestureLibrary_TEST.ts
uwave_gesture_library.npz             # derived historical cache, absent
cwru_lttsnet_raw/*.mat
cwru_lttsnet_seed10.npz               # derived historical cache, absent
cwru_lttsnet_seed10.manifest.json     # selected record URLs/hashes; included metadata
```

The included CWRU manifest lists the exact record download URLs, signal keys,
record checksums and historical boundaries. The accepted final CIL window index
in `code_reference/experiments/baselines/trunk_data_preflight/cwru_window_index.csv`
is authoritative for the CIL windows. Do not substitute legacy random-window
splits from a previous phase for the accepted index.

Fixed split index NPZs, WISDM row provenance, CWRU window indices, input
standardization statistics and descriptor-interface metadata are included in the
project-relative source tree. The NPZs contain indices/fit row identifiers rather
than sample amplitudes. Input and descriptor fitting is restricted to the
accepted Task-1 training rows, then frozen for subsequent tasks. The implemented
PCA16 transforms use current-task real training rows only; shifted pseudo-features
are generated in joint56 before applying the readout transform.

The exact preprocessing implementation is in `src/preflight_standardization.py`
and `src/preflight_trunk_data.py` under `code_reference/`; model adapters use the
accepted split metadata rather than refitting a validation split. Some code also
checks historical archives and hash/seal inputs that are not included. Thus the
metadata documents membership and computations without claiming complete
automatic reconstruction or verified end-to-end retraining from these downloads.
