"""Raw-data adaptation for the four datasets.

Reuses the ACCEPTED adapters from ``src/preflight_standardization.py`` verbatim
(no re-implementation, no re-splitting). Importing that module is safe: its
``main()`` is guarded and it writes nothing at import time.

Axis order is fixed to [n, time, channel] everywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import preflight_standardization as _ps  # noqa: E402  (accepted adapters; import-safe)

ACCEPTED_ADAPTER_SOURCE = Path(_ps.__file__).resolve()
PREFLIGHT_DIR = ROOT / "experiments/baselines/trunk_data_preflight"
FIX_DIR = ROOT / "experiments/baselines/trunk_preprocessing_fix"

# re-exported verbatim
ad_uci_har = _ps.ad_uci_har
ad_wisdm = _ps.ad_wisdm
ad_uwave = _ps.ad_uwave
ad_cwru = _ps.ad_cwru
fit_stats = _ps.fit_stats
apply_standardize = _ps.apply_standardize


def accepted_splits() -> dict:
    """Load the accepted split artifacts (read-only; membership must stay unchanged)."""
    import numpy as np
    return {
        "uci_har": np.load(PREFLIGHT_DIR / "uci_har_fixed_split.npz"),
        "wisdm": np.load(PREFLIGHT_DIR / "wisdm_fixed_split.npz"),
        "uwave": np.load(PREFLIGHT_DIR / "uwave_fixed_split.npz"),
    }


def accepted_input_stats() -> dict:
    """Accepted Task-1-only INPUT standardization statistics (frozen; never refit here)."""
    import json
    d = json.loads((FIX_DIR / "preprocess_statistics.json").read_text(encoding="utf-8"))
    out = {}
    for name, v in d["datasets"].items():
        out[name] = {"mean": v["per_channel_mean"],
                     "denom": v["per_channel_denominator_std_plus_1e-6"],
                     "fit_row_count": v["fit_row_count"]}
    return out


def adapt(dataset: str) -> dict:
    """Return the raw material for one dataset, with labels and per-row provenance."""
    if dataset == "uci_har":
        a = ad_uci_har()
        y = a["train"]["y"]
        return {"x": a["train"]["x"], "y": y, "provenance": a["train"]["provenance"],
                "test": a["test"], "subjects": a["train"]["subject"]}
    if dataset == "wisdm":
        a = ad_wisdm()
        y = [p["label"] for p in a["train"]["provenance"]]
        return {"x": a["train"]["x"], "y": y, "provenance": a["train"]["provenance"],
                "test": a["test"], "subjects": [p["subject"] for p in a["train"]["provenance"]]}
    if dataset == "uwave":
        a = ad_uwave()
        return {"x": a["train"]["x"], "y": a["train"]["y"], "provenance": a["train"]["provenance"],
                "test": a["test"], "label_mapping": a["label_mapping"], "subjects": None}
    if dataset == "cwru":
        a = ad_cwru()
        train_rows = [i for i, p in enumerate(a["provenance"]) if p["split"] == "train"]
        return {"x": a["x"], "y": a["y"], "provenance": a["provenance"],
                "train_rows": train_rows, "subjects": None}
    raise ValueError(dataset)
