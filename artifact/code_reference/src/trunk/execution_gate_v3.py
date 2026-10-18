"""Execution gate V3 — FIX1 for admin/REVIEW_s3_protocol_final_20261017.md §P1.

Changes vs V2:
  * production dependency enumeration is built HERE from real paths and covers every
    algorithm module (including the candidate modules and CLI), all raw inputs the
    S1/S3 chain reads, fixed splits, statistics, S1 models/state, and the S2 D1/D3
    assets that S3 reuses.  A caller cannot supply a shorter list outside fixture mode.
  * S1 inputs are verified THREE-WAY against CURRENTLY RECOMPUTED hashes
    (locked body / before file / current inventory), not just the two stored files.
  * JSON is parsed strictly: duplicate keys raise.
  * required S1 artifacts are enumerated by this module (32 encoders + 120 heads +
    120 state packages + protocol/seal/input + the S2 assets S3 consumes).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REQUIRED_SELF_CHECK_CASES = [
    "cand_c1_mask_protocol_direction", "cand_c1_backward_step", "cand_attention_baseline_formula",
    "cand_ln_wiring", "cand_pool_wiring", "cand_gru_wiring", "cand_unknown_candidate_rejected",
    "cand_pca_after_pseudo", "cand_ridge_reuse_no_refit", "cand_centroid_isolation",
    "cand_screen_promotion", "cand_s4_spearman_ties", "cand_s4_holm_family",
    "gate_entry_zero_builds_on_reject", "cli_state_machine",
]

DATASETS = ("uci_har", "wisdm", "uwave", "cwru")
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
TASK_COUNTS = {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}


class GateError(PermissionError):
    pass


# Fixture trees are synthetic, so the "model weights" leg is a single stand-in file.
# Production uses `production_s1_artifacts()` (32 encoders + 120 heads + 120 state pkgs).
FIXTURE_REQUIRED_S1 = ["protocol_locked.json", "input_sha256_before.json",
                       "training_seal.json", "training_seal.json.sha256", "w.bin"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def strict_json(path: Path):
    """json.loads with duplicate-key rejection (plain json silently keeps the last)."""
    def hook(pairs):
        seen = set()
        dup = []
        for k, _ in pairs:
            if k in seen:
                dup.append(k)
            seen.add(k)
        if dup:
            raise GateError(f"duplicate JSON keys in {Path(path).name}: {sorted(set(dup))[:5]}")
        return dict(pairs)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=hook)


# ---------------------------------------------------------------- production enumeration
def production_dependency_spec(root: Path) -> list[tuple[str, Path]]:
    root = Path(root)
    spec: list[tuple[str, Path]] = []
    src = {
        "src/trunk/s3_candidates_torch_fix1.py": "src/trunk/s3_candidates_torch_fix1.py",
        "src/trunk/s3_candidates_numpy_fix1.py": "src/trunk/s3_candidates_numpy_fix1.py",
        "src/trunk/s3_cli.py": "src/trunk/s3_cli.py",
        "src/trunk/s3_lock.py": "src/trunk/s3_lock.py",
        "src/trunk/s3_cli_fix2.py": "src/trunk/s3_cli_fix2.py",
        "src/trunk/s3_prod_fix2.py": "src/trunk/s3_prod_fix2.py",
        "src/trunk/s3_lock_fix2.py": "src/trunk/s3_lock_fix2.py",
        "src/trunk/s3_selftest_fix2.py": "src/trunk/s3_selftest_fix2.py",
        "src/trunk/s3_selftest_fix1.py": "src/trunk/s3_selftest_fix1.py",
        "src/trunk/execution_gate_v3.py": "src/trunk/execution_gate_v3.py",
        "src/trunk/run_baseline_fix2.py": "src/trunk/run_baseline_fix2.py",
        "src/trunk/train_smoke.py": "src/trunk/train_smoke.py",
        "src/trunk/acbm_interface.py": "src/trunk/acbm_interface.py",
        "src/trunk/descriptors.py": "src/trunk/descriptors.py",
        "src/trunk/adapters.py": "src/trunk/adapters.py",
        "src/preflight_standardization.py": "src/preflight_standardization.py",
        "archive/pipeline.py": ("experiments/archive/2026-10-13/"
                                "cfc_acbm_t3_component_ablation_3seeds/pipeline.py"),
        "archive/benchmark_stage3.py": "src/legacy/benchmark.py",
        "archive/cicbm_benchmark.py": "src/legacy/cicbm/benchmark.py",
        "archive/stage4_benchmark.py": "experiments/archive/stage4/refined/benchmark.py",
        "archive/stage4_run_config.json": "experiments/archive/stage4/refined/run_config.json",
        "archive/run_wisdm.py": ("experiments/archive/2026-10-11/"
                                 "cfc_cbm_wisdm_original_vs_staged/run_wisdm.py"),
        "stats/preprocess_statistics.json":
            "experiments/baselines/trunk_preprocessing_fix/preprocess_statistics.json",
        "stats/fit_indices.npz": "experiments/baselines/trunk_preprocessing_fix/fit_indices.npz",
        "split/uci_har_fixed_split.npz":
            "experiments/baselines/trunk_data_preflight/uci_har_fixed_split.npz",
        "split/wisdm_fixed_split.npz":
            "experiments/baselines/trunk_data_preflight/wisdm_fixed_split.npz",
        "split/uwave_fixed_split.npz":
            "experiments/baselines/trunk_data_preflight/uwave_fixed_split.npz",
        "split/cwru_window_index.csv":
            "experiments/baselines/trunk_data_preflight/cwru_window_index.csv",
        "split/wisdm_row_provenance.csv":
            "experiments/baselines/trunk_data_preflight/wisdm_row_provenance.csv",
        "raw/wisdm.zip": "data/wisdm.zip",
        "raw/wisdm_cache.npz": "data/wisdm_har_windows.npz",
        "raw/uwave_train.ts": "data/UWaveGestureLibrary_TRAIN.ts",
        "raw/uwave_test.ts": "data/UWaveGestureLibrary_TEST.ts",
        "raw/cwru_manifest.json": "data/cwru_lttsnet_seed10.manifest.json",
        "raw/uci_train_y.txt": "data/UCI HAR Dataset/train/y_train.txt",
        "raw/uci_test_y.txt": "data/UCI HAR Dataset/test/y_test.txt",
        "raw/uci_train_subject.txt": "data/UCI HAR Dataset/train/subject_train.txt",
        "raw/uci_test_subject.txt": "data/UCI HAR Dataset/test/subject_test.txt",
        "s1/protocol_locked.json":
            "experiments/baselines/trunk_baseline_4datasets/protocol_locked.json",
        "s1/input_sha256_before.json":
            "experiments/baselines/trunk_baseline_4datasets/input_sha256_before.json",
        "s1/training_seal.json":
            "experiments/baselines/trunk_baseline_4datasets/training_seal.json",
        "s1/training_seal.json.sha256":
            "experiments/baselines/trunk_baseline_4datasets/training_seal.json.sha256",
        "s2/evaluation_protocol_locked.json": ("experiments/baselines/trunk_baseline_4datasets/"
                                               "evaluation_phase/evaluation_protocol_locked.json"),
        "review/s3_protocol_final_20261017.md": "admin/REVIEW_s3_protocol_final_20261017.md",
        "review/s2_closed_s3_design.md": "admin/REVIEW_s2_closed_s3_design.md",
        "review/s2_results_and_recovery.md": "admin/REVIEW_s2_results_and_recovery.md",
    }
    for key, rel in src.items():
        spec.append((key, root / rel))
    for ds in DATASETS:
        spec.append((f"v2/interface_check_{ds}.json",
                     root / f"experiments/baselines/trunk_model_interface_check_v2/"
                            f"interface_check_{ds}.json"))
    for n in ("body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y",
              "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"):
        for split in ("train", "test"):
            spec.append((f"raw/uci_{split}_{n}.txt",
                         root / f"data/UCI HAR Dataset/{split}/Inertial Signals/"
                                f"{n}_{split}.txt"))
    for m in sorted((root / "data/cwru_lttsnet_raw").glob("*.mat")):
        spec.append((f"raw/cwru_{m.name}", m))
    # S2 per-boundary assets reused by R1/R2 (120 boundaries x probes + transforms)
    s2 = root / "experiments/sensitivity/trunk_readout_sensitivity_4datasets"
    for d in DATASETS:
        for s in SEEDS:
            for t in range(1, TASK_COUNTS[d] + 1):
                bdir = s2 / d / f"seed{s}" / f"task{t}"
                spec.append((f"s2/{d}/seed{s}/task{t}/probes.npz", bdir / "probes.npz"))
                spec.append((f"s2/{d}/seed{s}/task{t}/transforms.npz", bdir / "transforms.npz"))
                spec.append((f"s2/{d}/seed{s}/task{t}/features.npz", bdir / "features.npz"))
    return spec


def production_s1_artifacts(root: Path) -> list[str]:
    out = ["protocol_locked.json", "input_sha256_before.json",
           "training_seal.json", "training_seal.json.sha256"]
    for d in DATASETS:
        for s in SEEDS:
            out.append(f"{d}/seed{s}/task1_encoder_best.pt")
            for t in range(1, TASK_COUNTS[d] + 1):
                out.append(f"{d}/seed{s}/task{t}_head.pt")
                out.append(f"{d}/seed{s}/state_package_task{t}.npz")
    return out


class Gate:
    """Production gate. `spec`/`required_s1_paths` may only be overridden in fixture mode."""

    def __init__(self, exec_dir: Path, s1_dir: Path, spec=None,
                 required_s1_paths=None, fixture_mode: bool = False,
                 root: Path | None = None, fixture_current_inputs=None):
        self.exec_dir = Path(exec_dir)
        self.s1_dir = Path(s1_dir)
        self.fixture_mode = bool(fixture_mode)
        self.fixture_current_inputs = fixture_current_inputs
        if self.fixture_mode and fixture_current_inputs is None:
            raise GateError("fixture mode requires fixture_current_inputs for the 3rd leg")
        self.root = Path(root) if root else self.exec_dir.parents[3]
        if spec is None:
            if self.fixture_mode:
                raise GateError("fixture mode requires an explicit spec")
            spec = production_dependency_spec(self.root)
        elif not self.fixture_mode:
            raise GateError("callers may not supply a dependency spec outside fixture mode")
        self.spec = list(spec)
        if required_s1_paths is None:
            self.required_s1_paths = (production_s1_artifacts(self.root)
                                      if not self.fixture_mode
                                      else list(FIXTURE_REQUIRED_S1))
        else:
            if not self.fixture_mode:
                raise GateError("callers may not shrink the required S1 artifact list")
            self.required_s1_paths = list(required_s1_paths)

    # ---------- 1 lock ----------
    def check_lock(self) -> dict:
        lp = self.exec_dir / "protocol_locked.json"
        sp = self.exec_dir / "protocol_locked.json.sha256"
        if not lp.exists() or not sp.exists():
            raise GateError("missing lock files")
        if sp.read_text(encoding="utf-8").split()[0].strip() != sha(lp):
            raise GateError("lock sha text mismatch")
        body = strict_json(lp)
        if body.get("status") != "LOCKED":
            raise GateError(f"lock status {body.get('status')!r} != 'LOCKED'")
        for f in ("dependency_sha256", "dependency_manifest_sha256",
                  "before_snapshot_sha256", "self_check_sha256"):
            if not body.get(f):
                raise GateError(f"lock body missing binding {f}")
        return body

    # ---------- 2 sidecar bindings ----------
    def check_bindings(self, body: dict) -> None:
        for fld, name in (("dependency_manifest_sha256", "dependency_manifest.json"),
                          ("before_snapshot_sha256", "s1_before_snapshot.json"),
                          ("self_check_sha256", "self_check.json")):
            p = self.exec_dir / name
            if not p.exists():
                raise GateError(f"missing sidecar {name}")
            if sha(p) != body[fld]:
                raise GateError(f"{name} hash != lock binding")

    # ---------- 3 dependencies ----------
    def check_dependencies(self, body: dict) -> None:
        manifest = strict_json(self.exec_dir / "dependency_manifest.json")
        keys = [k for k, _ in self.spec]
        if len(keys) != len(set(keys)):
            raise GateError("duplicate keys in dependency spec")
        if set(manifest) != set(keys):
            miss = sorted(set(keys) - set(manifest))[:4]
            extra = sorted(set(manifest) - set(keys))[:4]
            raise GateError(f"manifest key set != enumeration (missing={miss}, extra={extra})")
        if set(body["dependency_sha256"]) != set(keys):
            raise GateError("lock body dependency keys != enumeration")
        for key, p in self.spec:
            if not p.exists():
                raise GateError(f"dependency missing: {key}")
            actual = sha(p)
            if manifest[key] != actual:
                raise GateError(f"dependency drift: {key}")
            if body["dependency_sha256"][key] != actual:
                raise GateError(f"lock body hash != current: {key}")

    # ---------- 4 self-check ----------
    def check_self_check(self) -> None:
        sc = strict_json(self.exec_dir / "self_check.json")
        cases = sc.get("cases")
        if not isinstance(cases, dict) or not cases:
            raise GateError("self-check cases missing/empty")
        if len(cases) != len(REQUIRED_SELF_CHECK_CASES):
            raise GateError(f"case count {len(cases)} != {len(REQUIRED_SELF_CHECK_CASES)}")
        for c in REQUIRED_SELF_CHECK_CASES:
            if c not in cases:
                raise GateError(f"missing case {c}")
            v = cases[c]
            if not isinstance(v, bool):
                raise GateError(f"{c} is {type(v).__name__}, require bool")
            if v is not True:
                raise GateError(f"{c} is False")

    # ---------- 5 before snapshot ----------
    def check_before_snapshot(self) -> None:
        snap = strict_json(self.exec_dir / "s1_before_snapshot.json")
        files = snap.get("files")
        if not isinstance(files, dict) or not files:
            raise GateError("before snapshot empty")
        for req in self.required_s1_paths:
            if req not in files:
                raise GateError(f"snapshot missing required S1 artifact {req}")
        for rel, h in files.items():
            p = self.s1_dir / rel
            if not p.exists():
                raise GateError(f"snapshot file missing on disk: {rel}")
            if sha(p) != h:
                raise GateError(f"snapshot drift: {rel}")

    # ---------- 6 S1 three-way with CURRENT recomputation ----------
    def check_s1_inputs(self, body: dict) -> None:
        inner = strict_json(self.s1_dir / "protocol_locked.json").get("input_sha256_before")
        before = strict_json(self.s1_dir / "input_sha256_before.json")
        if not isinstance(inner, dict) or not inner:
            raise GateError("S1 locked body has no input snapshot")
        if set(inner) != set(before):
            raise GateError("S1 locked vs before key mismatch")
        if any(inner[k] != before[k] for k in before):
            raise GateError("S1 locked vs before value mismatch")
        # third leg: recompute the CURRENT hashes of the same inventory
        if self.fixture_mode:
            current = dict(self.fixture_current_inputs)
        else:
            from trunk import run_baseline_fix2 as rb1
            current = {k: sha(v) for k, v in rb1.input_files().items()}
        if set(current) != set(before):
            raise GateError("S1 current inventory key set != locked/before")
        drift = [k for k in before if before[k] != current[k]]
        if drift:
            raise GateError(f"S1 input drift (current recomputed): {drift[:3]}")
        seal = self.s1_dir / "training_seal.json"
        rec = self.s1_dir / "training_seal.json.sha256"
        if not seal.exists() or not rec.exists():
            raise GateError("missing S1 seal")
        if strict_json(rec).get("sha256") != sha(seal):
            raise GateError("S1 seal hash mismatch")
        sbody = strict_json(seal)
        if sbody.get("status") != "SEALED":
            raise GateError("S1 seal not SEALED")
        if int(sbody.get("n_groups", len(sbody.get("groups", [])))) != 32:
            raise GateError("S1 seal group count != 32")

    # ---------- 7 no overwrite ----------
    def check_no_existing_output(self, *paths: Path) -> None:
        for p in paths:
            if Path(p).exists():
                raise GateError(f"output exists (no-overwrite): {p}")

    def full(self, *forbidden_outputs: Path) -> dict:
        body = self.check_lock()
        self.check_bindings(body)
        self.check_dependencies(body)
        self.check_self_check()
        self.check_before_snapshot()
        self.check_s1_inputs(body)
        if forbidden_outputs:
            self.check_no_existing_output(*forbidden_outputs)
        return body


def guarded_entry(gate: Gate, build_fn, *args, **kwargs):
    gate.full()
    return build_fn(*args, **kwargs)
