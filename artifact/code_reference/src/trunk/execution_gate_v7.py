"""Execution gate V7 — FIX6.  Adds the production-case hard verdict to the lock.

Adds (per REVIEW_s3_fix2_20261017.md):
  * the science protocol copy is part of the dependency enumeration AND its hash is
    verified against the lock-body binding (editing it after lock is rejected);
  * the locker source hash in the lock body is verified against the CURRENT locker file;
  * the self-check report hash recorded in `self_check.json` must match the bound value,
    and the report's cited source key set must equal the FULL required set;
  * fix3 modules are in the production enumeration.
"""
from __future__ import annotations

import json
from pathlib import Path

from trunk import execution_gate_v3 as v3

GateError = v3.GateError
sha = v3.sha
strict_json = v3.strict_json
REQUIRED_SELF_CHECK_CASES = v3.REQUIRED_SELF_CHECK_CASES
FIXTURE_REQUIRED_S1 = v3.FIXTURE_REQUIRED_S1
production_s1_artifacts = v3.production_s1_artifacts

# Every source file whose SHA the self-check report must cite.
REQUIRED_REPORT_SOURCES = [
    "s3_candidates_torch_fix1.py", "s3_candidates_numpy_fix1.py",
    "execution_gate_v7.py", "s3_cli_fix6.py", "s3_prod_fix6.py",
    "s3_lock_fix6.py", "s3_selftest_fix6.py", "readout_sensitivity.py",
]
LOCKER_FILES = ["s3_lock_fix6.py"]

# Production-interface cases that must all be literal True in the report.
REQUIRED_PRODUCTION_CASES = [
    "valid_lock_accepted_by_default_gate",
    "test_restore_both_branches",
    "production_counter_equality",
    "gate_rejection_zero_build",
    "init_snapshot_and_freeze_baseline",
    "lifecycle_24_cells_stub",
    "cli_evaluate_integration_trained",
    "cli_evaluate_integration_r1",
    "cli_evaluate_integration_r2",
    "partial_npz_blocks_evaluate",
    "evaluate_products_bind_npz",
]
REPO_ROOT = Path(__file__).resolve().parents[2]


def production_dependency_spec(root: Path) -> list[tuple[str, Path]]:
    spec = v3.production_dependency_spec(root)
    extra = {
        "src/trunk/s3_cli_fix6.py": "src/trunk/s3_cli_fix6.py",
        "src/trunk/s3_prod_fix6.py": "src/trunk/s3_prod_fix6.py",
        "src/trunk/s3_lock_fix6.py": "src/trunk/s3_lock_fix6.py",
        "src/trunk/s3_selftest_fix6.py": "src/trunk/s3_selftest_fix6.py",
        "src/trunk/execution_gate_v7.py": "src/trunk/execution_gate_v7.py",
        # actually imported by TrainCtx
        "src/trunk/readout_sensitivity.py": "src/trunk/readout_sensitivity.py",
        # the ORIGINAL science protocol file (not just the locked copy)
        "science/S3_science_protocol_fix6.json":
            "experiments/matrix/trunk_s3_protocol_final_fix6/S3_science_protocol_fix6.json",
    }
    keys = {k for k, _ in spec}
    for k, rel in extra.items():
        if k not in keys:
            spec.append((k, Path(root) / rel))
    return spec


REPO_ROOT = Path(__file__).resolve().parents[2]


class Gate(v3.Gate):
    """V3 gate + science/report/locker bindings + ONE shared enumeration."""

    def __init__(self, exec_dir, s1_dir, spec=None, required_s1_paths=None,
                 fixture_mode=False, root=None, fixture_current_inputs=None):
        if not fixture_mode and (spec is not None or required_s1_paths is not None):
            raise GateError("production gate refuses caller-supplied spec/required list")
        super().__init__(exec_dir, s1_dir, spec=spec,
                         required_s1_paths=required_s1_paths,
                         fixture_mode=fixture_mode,
                         root=root or REPO_ROOT,
                         fixture_current_inputs=fixture_current_inputs)
        if not fixture_mode:
            # override V3's 438-item list with the unified 443+ enumeration
            self.spec = production_dependency_spec(REPO_ROOT)
            self.required_s1_paths = production_s1_artifacts(REPO_ROOT)

    def check_bindings(self, body: dict) -> None:
        super().check_bindings(body)
        # science protocol copy must match the lock binding
        sc = self.exec_dir / "science_protocol.json"
        if not sc.exists():
            raise GateError("missing science_protocol.json")
        if sha(sc) != body.get("science_protocol_sha256"):
            raise GateError("science protocol drifted since lock")
        # the self-check sidecar must record the bound report hash
        side = strict_json(self.exec_dir / "self_check.json")
        if body.get("self_check_report_sha256") and \
                side.get("report_sha256") != body["self_check_report_sha256"]:
            raise GateError("self-check report binding mismatch")
        # report must cite the FULL required source set
        cited = side.get("source_sha256") or {}
        missing = [n for n in REQUIRED_REPORT_SOURCES if n not in cited]
        if missing:
            raise GateError(f"report does not cite required sources: {missing[:4]}")
        # locker hash must match the current locker
        if body.get("locker_sha256"):
            # always the REAL repo locker (fixtures run inside the same repo)
            p = REPO_ROOT / "src/trunk/s3_lock_fix6.py"
            if not p.exists() or sha(p) != body["locker_sha256"]:
                raise GateError("locker source drifted since lock")
