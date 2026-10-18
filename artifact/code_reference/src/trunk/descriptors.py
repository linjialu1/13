"""Descriptor implementations (v2).

Fixes over v1:
  * NO nan_to_num-style cleaning anywhere. Inputs, intermediates and outputs are
    explicitly checked; any NaN/Inf raises NonFiniteError and stops the run.
  * descriptor 7 is named for what it is: the RMS of diff(x), not "mean_abs_diff".
  * the ACTUAL computation dtype is recorded (the two reused descriptors compute in
    float32 because their call site has always passed float32; the generic one uses float64).

* UCI-HAR : reuses ``raw_concept_features`` verbatim.
* WISDM   : reuses ``wisdm_sensor_concepts`` verbatim (never the UCI-HAR version).
* UWave / CWRU : generic 8-dim, per real channel then arithmetic mean over channels.

Input is ALWAYS the raw window, never the standardized network input.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

CICBM_SRC = ROOT / "experiments/archive/stage4/refined/benchmark.py"
WISDM_LEGACY_DIR = ROOT / "experiments/archive/2026-10-11/cfc_cbm_wisdm_original_vs_staged"
WISDM_MODULE = WISDM_LEGACY_DIR / "run_wisdm.py"
EPS = 1e-08

GENERIC_NAMES = ["mean", "std", "rms", "absmax", "crest", "kurtosis_like",
                 "diff_rms", "spectral_centroid_cycles_per_sample"]
GENERIC_UNITS = ["raw", "raw", "raw", "raw", "ratio", "ratio", "raw", "cycles/sample"]
GENERIC_DIM7_DEFINITION = "sqrt(mean(diff(x)^2))  -- RMS of the first difference"


class NonFiniteError(Exception):
    """Raised when a NaN/Inf is found in an input, an intermediate or an output."""


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assert_all_finite(a: np.ndarray, where: str) -> np.ndarray:
    """Explicit stop on NaN/Inf. No cleaning, no substitution."""
    arr = np.asarray(a)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise NonFiniteError(f"non-finite values in {where}: {n_bad} of {arr.size}")
    return arr


def top_level_side_effect_audit(path: Path) -> dict:
    """AUXILIARY static check only -- it is not a proof that a module has no side effects."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    suspicious = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = ast.unparse(node.value.func)
            if any(k in f for k in ("open", "write", "mkdir", "save", "remove", "unlink", "dump")):
                suspicious.append(ast.unparse(node))
    return {"module": str(path), "sha256": sha(path),
            "module_level_write_like_calls": suspicious, "clean": not suspicious,
            "status": "AUXILIARY_STATIC_CHECK_ONLY — not a proof of absence of side effects"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_cicbm = None
_wisdm_mod = None


def cicbm_module():
    global _cicbm
    if _cicbm is None:
        _cicbm = _load("trunk_v2_cicbm_impl", CICBM_SRC)
    return _cicbm


def wisdm_module():
    global _wisdm_mod
    if _wisdm_mod is None:
        saved = list(sys.path)
        try:
            sys.path.insert(0, str(WISDM_LEGACY_DIR))
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            _wisdm_mod = _load("trunk_v2_wisdm_impl", WISDM_MODULE)
        finally:
            sys.path[:] = saved
    return _wisdm_mod


# --------------------------------------------------------------------- generic (float64)
def _generic_1ch(s: np.ndarray) -> np.ndarray:
    """s: [n, T] float64 -> [n, 8] float64. Constant/all-zero give finite values;
    NaN/Inf raise. NO cleaning is applied."""
    x = assert_all_finite(np.asarray(s, dtype=np.float64), "generic_1ch input")
    n, T = x.shape
    m = x.mean(axis=1)
    assert_all_finite(m, "generic_1ch mean")
    c = x - m[:, None]
    var = (c ** 2).mean(axis=1)
    rms = np.sqrt((x ** 2).mean(axis=1))
    abmax = np.abs(x).max(axis=1)
    m4 = (c ** 4).mean(axis=1)
    mean_abs_diff = np.sqrt((np.diff(x, axis=1) ** 2).mean(axis=1)) if T > 1 else np.zeros(n)
    P = np.abs(np.fft.rfft(c, axis=1)) ** 2
    f = np.fft.rfftfreq(T, d=1.0)                       # cycles/sample, [0, 0.5]
    centroid = (f[None, :] * P).sum(axis=1) / (P.sum(axis=1) + EPS)
    out = np.stack([m, np.sqrt(var), rms, abmax, abmax / (rms + EPS),
                    m4 / (var ** 2 + EPS), mean_abs_diff, centroid], axis=1)
    return assert_all_finite(out, "generic_1ch output")


def generic_8dim(x: np.ndarray) -> np.ndarray:
    """x: [n, T, C] raw windows -> [n, 8] float32 (per channel, then arithmetic mean)."""
    x = assert_all_finite(np.asarray(x, dtype=np.float64), "generic_8dim input")
    if x.ndim != 3:
        raise ValueError(f"expected [n,T,C], got {x.shape}")
    per_ch = np.stack([_generic_1ch(x[:, :, c]) for c in range(x.shape[2])], axis=1)
    assert_all_finite(per_ch, "generic_8dim per-channel")
    out = per_ch.mean(axis=1)
    assert_all_finite(out, "generic_8dim channel mean")
    return out.astype(np.float32)


# --------------------------------------------------------------------- dispatch
def describe(dataset: str, raw_windows: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (descriptors float32 [n,8], implementation info incl. ACTUAL dtypes)."""
    raw = np.asarray(raw_windows)
    assert_all_finite(raw, f"{dataset} raw windows passed to descriptors")
    if dataset == "uci_har":
        cm = cicbm_module()
        fn = getattr(cm, "raw_concept_features")
        inp = np.asarray(raw, dtype=np.float32)          # historical call site used float32
        assert_all_finite(inp, "uci_har descriptor input (float32 cast)")
        out = fn(inp)
        assert_all_finite(out, "uci_har descriptor output")
        d = np.asarray(out).astype(np.float32)
        info = {"implementation": "raw_concept_features (reused verbatim)", "modified_formula": False,
                "module_path": str(CICBM_SRC), "module_sha256": sha(CICBM_SRC),
                "function_qualname": f"{fn.__module__}.{fn.__name__}",
                "function_source_sha256": hashlib.sha256(
                    inspect.getsource(fn).encode("utf-8")).hexdigest(),
                "input_dtype_passed": str(inp.dtype),
                "internal_accumulation_dtype": "float32 (inherited from the float32 input; the "
                                               "function only casts at the end)",
                "output_dtype": str(d.dtype),
                "dtype_probe_float32_equals_float64": None}
        probe = fn(np.asarray(raw, dtype=np.float64))
        info["dtype_probe_float32_equals_float64"] = bool(
            np.array_equal(np.asarray(probe, dtype=np.float32), d))
        return d, info
    if dataset == "wisdm":
        wm = wisdm_module()
        fn = getattr(wm, "wisdm_sensor_concepts")
        inp = np.asarray(raw, dtype=np.float32)
        assert_all_finite(inp, "wisdm descriptor input (float32 cast)")
        out = fn(inp)
        assert_all_finite(out, "wisdm descriptor output")
        d = np.asarray(out).astype(np.float32)
        info = {"implementation": "wisdm_sensor_concepts (reused verbatim; NOT the UCI-HAR version)",
                "modified_formula": False,
                "module_path": str(WISDM_MODULE), "module_sha256": sha(WISDM_MODULE),
                "function_qualname": f"{fn.__module__}.{fn.__name__}",
                "function_source_sha256": hashlib.sha256(
                    inspect.getsource(fn).encode("utf-8")).hexdigest(),
                "input_dtype_passed": str(inp.dtype),
                "internal_accumulation_dtype": "float32 (inherited from the float32 input)",
                "output_dtype": str(d.dtype),
                "dtype_probe_float32_equals_float64": None}
        probe = fn(np.asarray(raw, dtype=np.float64))
        info["dtype_probe_float32_equals_float64"] = bool(
            np.array_equal(np.asarray(probe, dtype=np.float32), d))
        return d, info
    if dataset in ("uwave", "cwru"):
        d = generic_8dim(raw)
        info = {"implementation": "generic_8dim (new this round)", "modified_formula": "N/A (new definition)",
                "module_path": str(Path(__file__).resolve()),
                "names": GENERIC_NAMES, "units": GENERIC_UNITS, "eps": EPS,
                "dim7_name": GENERIC_NAMES[6], "dim7_definition": GENERIC_DIM7_DEFINITION,
                "spectral_centroid_unit": "cycles/sample, range [0, 0.5], NOT Hz",
                "sampling_rate_registered": False,
                "previous_clause_revision": "前轮『频谱类描述符必须先登记 Hz 采样率』在本方案下被显式修订："
                                            "使用归一化频率，不比较物理频率",
                "input_dtype_passed": "float64 (converted internally to float64)",
                "internal_accumulation_dtype": "float64",
                "output_dtype": str(d.dtype)}
        return d, info
    raise ValueError(dataset)


def standardize_descriptors(d: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """d_std = clip((d - mean) / (std + 1e-3), -6, 6), float32. Explicit finiteness, no cleaning."""
    dd = assert_all_finite(np.asarray(d, dtype=np.float64), "standardize_descriptors input")
    num = dd - np.asarray(mean, dtype=np.float64)
    den = np.asarray(std, dtype=np.float64) + 1e-3
    assert_all_finite(num, "standardize_descriptors numerator")
    assert_all_finite(den, "standardize_descriptors denominator")
    if not (den > 0).all():
        raise NonFiniteError("non-positive descriptor standardization denominator")
    out = np.clip(num / den, -6.0, 6.0)
    assert_all_finite(out, "standardize_descriptors output")
    return out.astype(np.float32)


def fit_descriptor_stats(d: np.ndarray) -> dict:
    dd = assert_all_finite(np.asarray(d, dtype=np.float64), "fit_descriptor_stats input")
    return {"mean": dd.mean(axis=0), "std": dd.std(axis=0, ddof=0),
            "n_rows": int(dd.shape[0]), "accumulation_dtype": "float64", "ddof": 0}
