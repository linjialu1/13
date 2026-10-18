"""S3 lock writer — FIX6.  Consumes a real self-check report; no default all-True.

No default all-True.  The writer CONSUMES a real self-check report:

  * report must exist and contain every required case as literal `True`;
  * the report's recorded source SHA256 must equal the CURRENT source files
    (so a report produced against older code is rejected);
  * the report file's bytes are hashed and bound into the lock body;
  * the science protocol JSON and this locker are also hashed and bound;
  * fixture mode is explicitly labelled in the body and is refused for production.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trunk import execution_gate_v7 as g3  # noqa: E402

LOCKER = Path(__file__).resolve()


class LockWriteError(RuntimeError):
    pass


def _load_report(path: Path) -> dict:
    if not Path(path).exists():
        raise LockWriteError(f"self-check report missing: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_report(report_path: Path, required=tuple(g3.REQUIRED_SELF_CHECK_CASES)) -> dict:
    """Validate a real self-check report; return its case map.  No defaults."""
    rep = _load_report(report_path)
    cases = rep.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise LockWriteError("report has no cases map")
    missing = [c for c in required if c not in cases]
    if missing:
        raise LockWriteError(f"report missing required cases: {missing[:4]}")
    for c in required:
        v = cases[c]
        if not isinstance(v, bool) or v is not True:
            raise LockWriteError(f"report case {c} is not literal True ({type(v).__name__})")
    # source SHA must match the CURRENT files
    src = rep.get("source_sha256")
    if not isinstance(src, dict) or not src:
        raise LockWriteError("report has no source_sha256 map")
    # full required key set — a report cannot cite an arbitrary non-empty subset
    need_src = list(getattr(g3, "REQUIRED_REPORT_SOURCES", []))
    lacking = [n for n in need_src if n not in src]
    if lacking:
        raise LockWriteError(f"report missing required source keys: {lacking[:4]}")
    for name, recorded in src.items():
        p = ROOT / "src/trunk" / name
        if not p.exists():
            raise LockWriteError(f"report cites unknown source {name}")
        if g3.sha(p) != recorded:
            raise LockWriteError(f"report was produced against different code: {name}")
    # production-interface results are part of the hard verdict
    prod_cases = rep.get("production_interface_cases")
    need_prod = list(getattr(g3, "REQUIRED_PRODUCTION_CASES", []))
    if not isinstance(prod_cases, dict) or not prod_cases:
        raise LockWriteError("report has no production_interface_cases map")
    missing_p = [c for c in need_prod if c not in prod_cases]
    if missing_p:
        raise LockWriteError(f"report missing production cases: {missing_p[:4]}")
    for c in need_prod:
        v = prod_cases[c]
        if not isinstance(v, bool):
            raise LockWriteError(f"production case {c} is {type(v).__name__}, require bool")
        if v is not True:
            raise LockWriteError(f"production case {c} is False")

    return {"cases": {c: True for c in required},
            "production_interface_cases": {c: True for c in need_prod},
            "source_sha256": src,
            "report_sha256": g3.sha(Path(report_path))}


def write_lock(out_dir: Path, *, s1_dir: Path, report_path: Path, science_protocol: Path,
               spec=None, required_s1_paths=None, fixture_mode: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ("protocol_locked.json", "protocol_locked.json.sha256", "dependency_manifest.json",
             "s1_before_snapshot.json", "self_check.json", "science_protocol.json")
    existing = [n for n in names if (out / n).exists()]
    if existing:
        raise LockWriteError(f"refusing to overwrite: {existing}")

    verified = verify_report(Path(report_path))

    if spec is None:
        if fixture_mode:
            raise LockWriteError("fixture mode requires an explicit spec")
        spec = g3.production_dependency_spec(ROOT)
    missing = [k for k, p in spec if not Path(p).exists()]
    if missing:
        raise LockWriteError(f"dependency missing: {missing[:4]}")
    manifest = {k: g3.sha(p) for k, p in spec}
    (out / "dependency_manifest.json").write_text(json.dumps(manifest, indent=1),
                                                  encoding="utf-8")

    s1 = Path(s1_dir)
    snap = {}
    for p in sorted(s1.rglob("*")):
        if p.is_file():
            snap[p.relative_to(s1).as_posix()] = g3.sha(p)
    if not snap:
        raise LockWriteError("empty S1 tree")
    (out / "s1_before_snapshot.json").write_text(
        json.dumps({"files": snap, "count": len(snap)}, indent=1), encoding="utf-8")

    (out / "self_check.json").write_text(json.dumps(
        {"cases": verified["cases"],
         "production_interface_cases": verified["production_interface_cases"],
         "source_sha256": verified["source_sha256"],
         "report_sha256": verified["report_sha256"],
         "report_path": str(report_path), "fixture_mode": bool(fixture_mode)},
        indent=1), encoding="utf-8")

    sp = Path(science_protocol)
    if not sp.exists():
        raise LockWriteError(f"science protocol missing: {sp}")
    (out / "science_protocol.json").write_text(sp.read_text(encoding="utf-8"),
                                               encoding="utf-8")

    body = {
        "status": "LOCKED",
        "version": "s3_candidate_matrix_v8_fix6",
        "fixture_mode": bool(fixture_mode),
        "dependency_sha256": manifest,
        "dependency_manifest_sha256": g3.sha(out / "dependency_manifest.json"),
        "before_snapshot_sha256": g3.sha(out / "s1_before_snapshot.json"),
        "self_check_sha256": g3.sha(out / "self_check.json"),
        "self_check_report_sha256": verified["report_sha256"],
        "science_protocol_sha256": g3.sha(out / "science_protocol.json"),
        "locker_sha256": g3.sha(LOCKER),
        "required_s1_artifacts": (required_s1_paths if required_s1_paths is not None
                                  else ([] if fixture_mode
                                        else g3.production_s1_artifacts(ROOT))),
    }
    (out / "protocol_locked.json").write_text(json.dumps(body, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    (out / "protocol_locked.json.sha256").write_text(
        f"{g3.sha(out / 'protocol_locked.json')} *protocol_locked.json\n", encoding="utf-8")
    return {"locked_sha256": g3.sha(out / "protocol_locked.json"),
            "n_dependencies": len(manifest), "n_s1_files": len(snap),
            "report_sha256": verified["report_sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--s1-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--science-protocol", required=True)
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--confirm-real-lock", action="store_true")
    a = ap.parse_args(argv)
    if not a.fixture and not a.confirm_real_lock:
        print("ERROR: production lock requires --confirm-real-lock", file=sys.stderr)
        return 3
    try:
        info = write_lock(Path(a.out), s1_dir=Path(a.s1_dir), report_path=Path(a.report),
                          science_protocol=Path(a.science_protocol), fixture_mode=a.fixture)
    except (LockWriteError, g3.GateError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
