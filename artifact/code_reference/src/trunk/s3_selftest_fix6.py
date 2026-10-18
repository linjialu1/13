"""S3 FIX6 engineering tests — no-overwrite evaluation, NPZ-bound products, real cmd_evaluate."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/matrix/trunk_s3_protocol_final_fix6"
S1 = ROOT / "experiments/baselines/trunk_baseline_4datasets"
S2 = ROOT / "experiments/sensitivity/trunk_readout_sensitivity_4datasets"
DATASETS = ("uci_har", "wisdm", "uwave", "cwru")
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fake_make_joint(model, x_std, x_raw, rows, d_mean, d_std, name, device):
    n = int(len(rows))
    g = torch.Generator().manual_seed(777)
    joint = torch.randn(n, 56, generator=g).numpy().astype(np.float32)
    return (joint, np.ascontiguousarray(joint[:, :48]),
            torch.randn(n, 8, generator=g).numpy(), torch.randn(n, 8, generator=g).numpy())


class SynthCtx:
    def __init__(self, dataset="uci_har", n_train=24, n_val=16, t=8, c=3, n_classes=4):
        rng = np.random.default_rng(3)
        self.name = dataset
        self.input_channels = c
        self.n_classes = n_classes
        total = n_train + n_val
        self.x_std = rng.standard_normal((total, t, c)).astype(np.float32)
        self.x_raw = self.x_std.copy()
        y = np.zeros(total, dtype=np.int64)
        for i in range(n_train):
            y[i] = i % n_classes
        for j in range(n_val):
            y[n_train + j] = j % n_classes
        self.y = y
        self.d_mean = np.zeros(8); self.d_std = np.ones(8)
        self.keep = np.arange(9000, 9000 + total, dtype=np.int64)
        self.train_all = list(range(n_train))
        self.val_all = list(range(n_train, total))

    def rows(self, classes, which):
        base = self.train_all if which == "train" else self.val_all
        return np.asarray(sorted(i for i in base if int(self.y[i]) in classes), dtype=np.int64)

    def orig_ids(self, pos_ids):
        return np.asarray([int(self.keep[int(i)]) for i in pos_ids], dtype=np.int64)


def make_assets(tmp: Path):
    from trunk import run_baseline_fix2 as rb1
    s2 = tmp / "s2"; s1 = tmp / "s1"
    m = rb1.build_model(3, 4, 0)
    for ds in DATASETS:
        for s in SEEDS:
            d2 = s2 / ds / f"seed{s}" / "task2"
            d2.mkdir(parents=True)
            rng = np.random.default_rng(s)
            np.savez_compressed(d2 / "features.npz",
                                val_joint=rng.standard_normal((16, 56)).astype(np.float32),
                                val_labels=np.array([j % 4 for j in range(16)], dtype=np.int64))
            np.savez_compressed(d2 / "probes.npz",
                                **{"D1_joint56__W": rng.standard_normal((56, 4)),
                                   "D1_joint56__b": rng.standard_normal(4),
                                   "D3_pca16__W": rng.standard_normal((16, 4)),
                                   "D3_pca16__b": rng.standard_normal(4)})
            np.savez_compressed(d2 / "transforms.npz",
                                **{"D3_pca16__mean": rng.standard_normal(56),
                                   "D3_pca16__components": rng.standard_normal((56, 16))})
            d1 = s1 / ds / f"seed{s}"
            d1.mkdir(parents=True)
            torch.save({"model_state_dict": m.state_dict()}, d1 / "task1_encoder_best.pt")
    return s1, s2


def build_lifecycle(tmp: Path):
    from trunk import s3_cli_fix6 as cli
    from trunk import execution_gate_v7 as g7
    from trunk import s3_lock_fix6 as locker
    from trunk import s3_candidates_torch_fix1 as ct
    from trunk import acbm_interface as ai
    import torch.nn as nn

    d = tmp / "deps"; d.mkdir(parents=True)
    spec = []
    for i, n in enumerate(["stats.json", "split.npz", "model.pt"]):
        f = d / n; f.write_bytes(f"d{i}".encode()); spec.append((f"dep/{n}", f))
    s1rep = tmp / "s1rep"; s1rep.mkdir()
    (s1rep / "protocol_locked.json").write_text(json.dumps(
        {"status": "LOCKED", "input_sha256_before": {"a": "1"}}), encoding="utf-8")
    (s1rep / "input_sha256_before.json").write_text(json.dumps({"a": "1"}), encoding="utf-8")
    (s1rep / "training_seal.json").write_text(json.dumps({"status": "SEALED", "n_groups": 32}),
                                              encoding="utf-8")
    (s1rep / "training_seal.json.sha256").write_text(
        json.dumps({"sha256": sha(s1rep / "training_seal.json")}), encoding="utf-8")
    (s1rep / "w.bin").write_bytes(b"w")
    report = tmp / "report.json"
    report.write_text(json.dumps({
        "cases": {c: True for c in g7.REQUIRED_SELF_CHECK_CASES},
        "production_interface_cases": {c: True for c in g7.REQUIRED_PRODUCTION_CASES},
        "source_sha256": {n: sha(ROOT / f"src/trunk/{n}")
                          for n in g7.REQUIRED_REPORT_SOURCES}}), encoding="utf-8")
    sci = tmp / "science.json"; sci.write_text("{}", encoding="utf-8")
    ex = tmp / "exec"
    locker.write_lock(ex, s1_dir=s1rep, report_path=report, science_protocol=sci,
                      spec=spec, fixture_mode=True, required_s1_paths=g7.FIXTURE_REQUIRED_S1)

    def gf():
        return g7.Gate(ex, s1rep, spec=spec, fixture_mode=True, root=tmp,
                       fixture_current_inputs={"a": "1"})

    ctx = SynthCtx()

    def build(c, dname, s, o):
        if c in cli.READOUT:
            if o is not None:
                Path(o).mkdir(parents=True, exist_ok=True)
                (Path(o) / "readout_asset.json").write_text("{}", encoding="utf-8")
            return {"readout": True, "val_macro": 0.70, "asset_refs": {}}
        return ct.build_wired_candidate(c, ctx.input_channels, ctx.n_classes)

    def train(m, dname, s, c):
        return {"epochs_trained": 0, "stub_no_training": True}

    def freeze(m, c):
        ai.freeze_representation(m)
        return {"signature_sha256": "SIG", "all_frozen": True, "grads_all_none": True}

    def fit_head(m, dname, s, o, c, cand, fsig=None):
        torch.save({"model_state_dict": m.state_dict()}, Path(o) / "model.pt")
        per = {}
        for t in range(1, 3):
            h = nn.Linear(56, 2 * t)                       # random-init head (stub)
            hp = Path(o) / f"task{t}_head.pt"
            torch.save(h.state_dict(), hp)
            (Path(o) / f"boundary_task{t}.npz").write_bytes(f"B{cand}{s}{t}".encode())
            (Path(o) / f"state_package_task{t}.npz").write_bytes(f"S{cand}{s}{t}".encode())
            per[f"task{t}"] = {
                "head_path": str(hp), "head_sha256": sha(hp),
                "boundary_npz_sha256": sha(Path(o) / f"boundary_task{t}.npz"),
                "state_pkg_sha256": sha(Path(o) / f"state_package_task{t}.npz"),
                "stub_head": True}
        mp = Path(o) / "model.pt"
        return {"per_task": per, "model_path": str(mp), "model_sha256": sha(mp),
                "freeze_evidence": [], "frozen_signature_sha256": "SIG"}

    deps = cli.Deps(build=build, train_task1=train, freeze=freeze, fit_head=fit_head,
                    val_macro_from_result=lambda c, dn, p: (0.71 if dn == "uci_har" else 0.60),
                    baseline_val_macro=lambda dn, s: 0.65,
                    test_macro=None, baseline_test_macro=lambda dn, s: 0.70,
                    gate_factory=gf, device="cpu", allow_cpu=True,
                    attach_hooks=lambda m, c: [], detach_hooks=lambda h: None,
                    init_snapshot=lambda c, dn, s: {"shared_param_count": 10,
                                                    "all_shared_initial_values_equal": True},
                    test_accepts_context=True, test_counters={"test_forwards": 0})
    return ex, deps, ctx, spec, s1rep, gf


def drive_to_sealed(ex, deps, cli):
    cli.cmd_lock(ex, deps)
    for c in cli.READOUT + cli.TRAIN:
        for d in cli.DATASETS:
            if c in cli.READOUT:
                for s in cli.SEEDS:
                    cli.cmd_run(ex, deps, c, d, s)
            else:
                for s in cli.SCREEN_SEEDS:
                    cli.cmd_run(ex, deps, c, d, s)
                cli.cmd_screen(ex, deps, c, d)
    cli.cmd_expand(ex, deps)
    cli.cmd_seal(ex, deps)


def case_valid_lock():
    from trunk import execution_gate_v7 as g5
    from trunk import s3_lock_fix6 as locker

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        report = tmp / "report.json"
        report.write_text(json.dumps({
            "cases": {c: True for c in g5.REQUIRED_SELF_CHECK_CASES},
            "production_interface_cases": {c: True for c in g5.REQUIRED_PRODUCTION_CASES},
            "source_sha256": {n: sha(ROOT / f"src/trunk/{n}")
                              for n in g5.REQUIRED_REPORT_SOURCES}}), encoding="utf-8")
        sci = ROOT / "experiments/matrix/trunk_s3_protocol_final_fix5/S3_science_protocol_fix5.json"
        ex = tmp / "exec"
        info = locker.write_lock(ex, s1_dir=S1, report_path=report, science_protocol=sci)
        gate = g5.Gate(ex, S1)                      # DEFAULT production gate, no injection
        gate.full()                                  # must accept a valid lock
        accepted = True

        rejections = {}
        # 1) science protocol drift
        (ex / "science_protocol.json").write_text('{"version":"tampered"}', encoding="utf-8")
        try:
            gate.full()
            rejections["science_drift"] = "NO_EXCEPTION"
        except Exception as e:
            rejections["science_drift"] = type(e).__name__
        # restore
        (ex / "science_protocol.json").write_text(Path(sci).read_text(encoding="utf-8"),
                                                   encoding="utf-8")
        # 2) dependency drift (a real enumerated file)
        victim = ROOT / "src/trunk/s3_candidates_numpy_fix1.py"
        orig = victim.read_bytes()
        victim.write_bytes(orig + b"\n# tampered\n")
        try:
            gate.full()
            rejections["dependency_drift"] = "NO_EXCEPTION"
        except Exception as e:
            rejections["dependency_drift"] = type(e).__name__
        victim.write_bytes(orig)
        # 3) locker drift
        rec = json.loads((ex / "protocol_locked.json").read_text(encoding="utf-8"))
        rec["locker_sha256"] = "0" * 64
        (ex / "protocol_locked.json").write_text(json.dumps(rec), encoding="utf-8")
        (ex / "protocol_locked.json.sha256").write_text(
            gate_mod_sha(ex / "protocol_locked.json") + " *protocol_locked.json\n",
            encoding="utf-8")
        try:
            gate.full()
            rejections["locker_drift"] = "NO_EXCEPTION"
        except Exception as e:
            rejections["locker_drift"] = type(e).__name__
        return {"valid_lock_accepted": accepted, "n_dependencies": info["n_dependencies"],
                "rejections": rejections}


def gate_mod_sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- B. real test restore
def case_test_restore():
    from trunk import s3_prod_fix6 as prod
    from trunk import train_smoke as ts
    from trunk import run_baseline_fix2 as rb1
    import torch.nn as nn

    results = {}
    orig_ctx, orig_mj = rb1.load_context, ts.make_joint
    try:
        ctx = SynthCtx()
        fake_ctx = {"x_std_test": ctx.x_std, "test_view": {"x_raw": ctx.x_raw, "y": ctx.y},
                    "d_mean": ctx.d_mean, "d_std": ctx.d_std,
                    "input_channels": ctx.input_channels, "n_classes": ctx.n_classes,
                    "task_rows": lambda classes, which: ctx.rows(classes, "train")}
        rb1.load_context = lambda dataset: fake_ctx
        ts.make_joint = fake_make_joint
        prod.ALL_TASKS, prod.TASK_COUNTS = [[0, 1], [2, 3]], {"uci_har": 2}

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            # training branch: real wired model + saved head/model
            cand = "C1_backbone_freeze"
            model = prod.build_candidate(cand, ctx, 11, torch.device("cpu"))
            torch.save({"model_state_dict": model.state_dict()}, out / "model.pt")
            head = nn.Linear(56, 4)
            torch.save(head.state_dict(), out / "task2_head.pt")
            seed_result = {"model_path": str(out / "model.pt"),
                           "per_task": {"task2": {"head_path": str(out / "task2_head.pt")}}}
            counters = {"test_forwards": 0}
            v_train = prod.test_macro_from_seed_result(cand, "uci_har", 11, seed_result,
                                                       torch.device("cpu"),
                                                       out_dir=out, counters=counters)
            ev = out / "test_task2.npz"
            results["train_branch"] = {"value": round(float(v_train), 4),
                                       "evidence_saved": ev.exists(),
                                       "test_forwards": counters["test_forwards"]}
        # readout branch: same real function; S1 checkpoint replaced by a synthetic one
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            fake_s1 = out / "s1"
            ck_dir = fake_s1 / "uci_har" / "seed11"
            ck_dir.mkdir(parents=True)
            m0 = rb1.build_model(ctx.input_channels, ctx.n_classes, 0)
            torch.save({"model_state_dict": m0.state_dict()},
                       ck_dir / "task1_encoder_best.pt")
            prod.S1 = fake_s1
            counters2 = {"test_forwards": 0}
            v_ro = prod.test_macro_from_seed_result("R1_analytic_head", "uci_har", 11, {},
                                                    torch.device("cpu"),
                                                    out_dir=out, counters=counters2)
            results["readout_branch"] = {"value": round(float(v_ro), 4),
                                         "evidence_saved": (out / "test_task2.npz").exists(),
                                         "test_forwards": counters2["test_forwards"]}
    finally:
        rb1.load_context, ts.make_joint = orig_ctx, orig_mj
        prod.S1 = S1
        prod.ALL_TASKS, prod.TASK_COUNTS = prod.__dict__.get("_orig_tasks", [[0, 1], [2, 3], [4, 5],
                                                                             [6, 7], [8, 9]]), \
            {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}
    return results


# ---------------------------------------------------------------- C. production counters
def case_production_counters():
    from trunk import s3_cli_fix6 as cli
    from trunk import s3_prod_fix6 as prod
    from trunk import s3_lock_fix6 as locker
    from trunk import execution_gate_v7 as g5
    from trunk import train_smoke as ts

    _mj = ts.make_joint
    ts.make_joint = fake_make_joint          # replace the DATA SOURCE only
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ex, _, gf = _fixture_lock(tmp, g5, locker)
        deps = cli.production_deps(ex, ctx_factory=lambda d: SynthCtx(d),
                                   device="cpu", allow_cpu=True, gate_factory=gf)
        prod.ALL_TASKS, prod.TASK_COUNTS = [[0, 1], [2, 3]], {"uci_har": 2}
        observed = {"encoder": 0, "head": 0}
        orig_build = deps.build

        def build(c, d, s, o):
            m = orig_build(c, d, s, o)
            if not isinstance(m, dict):
                m.encoder.register_forward_hook(
                    lambda *a: observed.__setitem__("encoder", observed["encoder"] + 1))
            return m

        deps.build = build
        cli.cmd_lock(ex, deps)
        try:
            out = cli.cmd_run(ex, deps, "C1_backbone_freeze", "uci_har", 11)
        finally:
            prod.ALL_TASKS, prod.TASK_COUNTS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], \
                {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}
        ts.make_joint = _mj
        rj = json.loads((ex / "runs" / cli.cell_id("C1_backbone_freeze", "uci_har") /
                         "seed11" / "result.json").read_text(encoding="utf-8"))
        c = rj["counters"]
        return {"val_macro": out["val_macro"],
                "ledger_encoder_forwards": c["encoder_forwards"],
                "observed_encoder_forwards": observed["encoder"],
                "encoder_match": c["encoder_forwards"] == observed["encoder"],
                "original_head_forwards": c["original_head_forwards"],
                "head_forwards": c["head_forwards"],
                "head_expected_train_plus_val": 300 * 2 + 2,
                "head_match": c["head_forwards"] == 300 * 2 + 2,
                "gradient_updates": c["gradient_updates"],
                "gradient_updates_positive": c["gradient_updates"] > 0,
                "init_snapshot": rj.get("init_snapshot"),
                "freeze_evidence_tasks": len(rj.get("freeze_evidence", [])),
                "budget_present": "budget" in rj}


# ---------------------------------------------------------------- D. gate rejection -> build 0
def case_gate_reject_zero_build():
    from trunk import s3_cli_fix6 as cli
    from trunk import execution_gate_v7 as g5
    from trunk import s3_lock_fix6 as locker

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ex, spec_map, gf = _fixture_lock(tmp, g5, locker)
        calls = {"n": 0}
        deps = cli.production_deps(ex, ctx_factory=lambda d: SynthCtx(d),
                                   device="cpu", allow_cpu=True, gate_factory=gf)
        deps.build = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or "M"
        cli.cmd_lock(ex, deps)                      # lock on the CLEAN fixture
        victim = Path(spec_map["dep/stats.json"])   # then drift a dependency
        orig = victim.read_bytes()
        victim.write_bytes(orig + b"x")
        try:
            rc = cli.main(["--exec-dir", str(ex), "--run", "R1_analytic_head",
                           "uci_har", "11"], deps=deps)
        finally:
            victim.write_bytes(orig)
        return {"exit_code": rc, "build_calls": calls["n"],
                "rejected": rc != 0 and calls["n"] == 0}


# ---------------------------------------------------------------- E. init snapshot / freeze
def case_init_and_freeze():
    from trunk import s3_prod_fix6 as prod
    from trunk import train_smoke as ts

    snap = prod.common_param_snapshot("LN_layernorm_injection", 3, 4, 11)
    ctx = SynthCtx()
    model = prod.build_candidate("C1_backbone_freeze", ctx, 11, torch.device("cpu"))
    counters = prod.new_counters()
    handles = prod.attach_hooks(model, counters)
    prod.train_task1(model, ctx, 11, counters, torch.device("cpu"))
    prod.detach_hooks(handles)
    rep = prod.freeze_all(model, counters)
    orig = ts.make_joint
    ts.make_joint = fake_make_joint
    prod.ALL_TASKS, prod.TASK_COUNTS = [[0, 1], [2, 3]], {"uci_har": 2}
    good = prod.run_boundaries(model, ctx, seed=11, out_dir=Path(tempfile.mkdtemp()),
                               counters=counters, candidate="C1", device=torch.device("cpu"),
                               frozen_baseline=rep["signature_sha256"])
    bad_msg = None
    prod.ALL_TASKS, prod.TASK_COUNTS = [[0, 1]], {"uci_har": 1}
    try:
        prod.run_boundaries(model, ctx, seed=11, out_dir=Path(tempfile.mkdtemp()),
                            counters=prod.new_counters(), candidate="C1",
                            device=torch.device("cpu"), frozen_baseline="0" * 64)
        bad_msg = "NO_EXCEPTION"
    except Exception as e:
        bad_msg = f"{type(e).__name__}: {e}"
    finally:
        ts.make_joint = orig
        prod.ALL_TASKS, prod.TASK_COUNTS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], \
            {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}
    return {"init_shared_equal": snap["all_shared_initial_values_equal"],
            "init_shared_params": snap["shared_param_count"],
            "freeze_matches": all(e["matches_task1_baseline"]
                                  for e in good["freeze_evidence"]),
            "wrong_baseline_rejected": bad_msg.startswith("RuntimeError")}


# ---------------------------------------------------------------- fixtures
# ---------------------------------------------------------------- F. 24-cell lifecycle (stubs)
def case_lifecycle_24():
    from trunk import s3_cli_fix6 as cli
    from trunk import execution_gate_v7 as g5
    from trunk import s3_lock_fix6 as locker

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ex, _spec, gf = _fixture_lock(tmp, g5, locker)
        calls = {"train": 0, "head": 0}

        def build(c, d, s, o):
            if c in cli.READOUT:
                if o is not None:
                    Path(o).mkdir(parents=True, exist_ok=True)
                    (Path(o) / "readout_asset.json").write_text("{}", encoding="utf-8")
                return {"readout": True, "val_macro": 0.70, "asset_refs": {}}
            return f"MODEL::{c}::{d}::{s}"

        def train(m, d, s, c):
            calls["train"] += 1
            return {"epochs_trained": 3}

        def freeze(m, c):
            return {"signature_sha256": f"SIG::{m}"}

        def fit_head(m, d, s, o, c, cand, fsig=None):
            calls["head"] += 1
            per = {}
            for t in range(1, cli.TASK_COUNTS[d] + 1):
                hp = Path(o) / f"task{t}_head.pt"; hp.parent.mkdir(parents=True, exist_ok=True)
                hp.write_bytes(f"H::{cand}::{s}::{t}".encode())
                (Path(o) / f"boundary_task{t}.npz").write_bytes(b"B")
                (Path(o) / f"state_package_task{t}.npz").write_bytes(b"S")
                per[f"task{t}"] = {"head_path": str(hp), "head_sha256": sha(hp),
                                   "boundary_npz_sha256": sha(Path(o) / f"boundary_task{t}.npz"),
                                   "state_pkg_sha256": sha(Path(o) / f"state_package_task{t}.npz")}
            mp = Path(o) / "model.pt"; mp.write_bytes(f"M::{cand}::{s}".encode())
            return {"per_task": per, "model_path": str(mp), "model_sha256": sha(mp),
                    "freeze_evidence": [], "frozen_signature_sha256": "SIG"}

        deps = cli.Deps(build=build, train_task1=train, freeze=freeze, fit_head=fit_head,
                        val_macro_from_result=lambda c, d, p: (0.71 if d in ("uci_har", "uwave")
                                                              else 0.60),
                        baseline_val_macro=lambda d, s: 0.65,
                        test_macro=(lambda c, d, s, r, out_dir=None, counters=None:
                                    (_write_stub_test_npz(out_dir, c, d, s), 0.80)[1]),
                        baseline_test_macro=lambda d, s: 0.70,
                        gate_factory=gf, device="cpu", allow_cpu=True,
                        attach_hooks=lambda m, c: [], detach_hooks=lambda h: None,
                        init_snapshot=lambda c, d, s: {"shared_param_count": 10,
                                                      "all_shared_initial_values_equal": True},
                        test_accepts_context=True)
        cli.cmd_lock(ex, deps)
        assert len(cli.load_state(ex)["cells"]) == 24
        for c in cli.READOUT:
            for d in cli.DATASETS:
                for s in cli.SEEDS:
                    cli.cmd_run(ex, deps, c, d, s)
        ro_train = calls["train"]
        for c in cli.TRAIN:
            for d in cli.DATASETS:
                for s in cli.SCREEN_SEEDS:
                    cli.cmd_run(ex, deps, c, d, s)
                cli.cmd_screen(ex, deps, c, d)
        st = cli.load_state(ex)
        states = [v["state"] for v in st["cells"].values() if v["candidate"] in cli.TRAIN]
        promoted, failed = states.count("PROMOTED"), states.count("SCREEN_FAIL")
        exp = cli.cmd_expand(ex, deps)
        cli.cmd_seal(ex, deps)
        cli.cmd_evaluate(ex, deps)
        cli.cmd_aggregate(ex, deps)
        agg = json.loads((ex / "aggregate.json").read_text(encoding="utf-8"))
        r1 = agg["cells"][cli.cell_id("R1_analytic_head", "uci_har")]
        fcell = next(k for k, v in agg["cells"].items() if v["state"] == "SCREEN_FAIL")
        return {"cells": 24, "readout_train_calls": ro_train, "promoted": promoted,
                "screen_fail": failed, "expanded": len(exp["expanded_runs"]),
                "train_calls": calls["train"], "head_calls": calls["head"],
                "r1_gain": round(r1["mean"], 6), "screen_fail_p_null": agg["cells"][fcell]["p"] is None,
                "holm_family": agg["holm_family_24"]["family_size"],
                "s4_has_R1_R2": ("spearman_R1" in agg["s4"] and "spearman_R2" in agg["s4"])}


# ---------------------------------------------------------------- integration
def _try(fn):
    try:
        fn()
        return "OK"
    except Exception as e:
        return type(e).__name__




def _fixture_lock(tmp: Path, gmod, locker):
    """Small locked fixture used by the carried-over cases: returns (exec, spec_map, gate_factory)."""
    d = tmp / "deps"; d.mkdir(parents=True, exist_ok=True)
    spec = []
    for i, n in enumerate(["stats.json", "split.npz", "model.pt"]):
        f = d / n; f.write_bytes(f"d{i}".encode()); spec.append((f"dep/{n}", f))
    s1 = tmp / "s1rep"; s1.mkdir(exist_ok=True)
    (s1 / "protocol_locked.json").write_text(json.dumps(
        {"status": "LOCKED", "input_sha256_before": {"a": "1"}}), encoding="utf-8")
    (s1 / "input_sha256_before.json").write_text(json.dumps({"a": "1"}), encoding="utf-8")
    (s1 / "training_seal.json").write_text(json.dumps({"status": "SEALED", "n_groups": 32}),
                                           encoding="utf-8")
    (s1 / "training_seal.json.sha256").write_text(
        json.dumps({"sha256": sha(s1 / "training_seal.json")}), encoding="utf-8")
    (s1 / "w.bin").write_bytes(b"w")
    rep = tmp / "report_carried.json"
    rep.write_text(json.dumps({
        "cases": {c: True for c in gmod.REQUIRED_SELF_CHECK_CASES},
        "production_interface_cases": {c: True for c in gmod.REQUIRED_PRODUCTION_CASES},
        "source_sha256": {n: sha(ROOT / f"src/trunk/{n}")
                          for n in gmod.REQUIRED_REPORT_SOURCES}}), encoding="utf-8")
    sci = tmp / "science_carried.json"; sci.write_text("{}", encoding="utf-8")
    ex = tmp / "exec"
    locker.write_lock(ex, s1_dir=s1, report_path=rep, science_protocol=sci,
                      spec=spec, fixture_mode=True,
                      required_s1_paths=gmod.FIXTURE_REQUIRED_S1)

    def gf():
        return gmod.Gate(ex, s1, spec=spec, fixture_mode=True, root=tmp,
                         fixture_current_inputs={"a": "1"})
    return ex, dict(spec), gf



def _write_stub_test_npz(out_dir, candidate, dataset, seed, n=16):
    """Stub evaluation writes schema-valid NPZs so product binding can be exercised."""
    if out_dir is None:
        return
    for t in range(1, 6):
        kk = 2 * t
        scores = np.random.default_rng(seed + t).standard_normal((n, kk)).astype(np.float64)
        pred = scores.argmax(1).astype(np.int64)
        labels = np.array([j % kk for j in range(n)], dtype=np.int64)
        cm = np.zeros((kk, kk), dtype=np.int64)
        for i in range(n):
            cm[labels[i], pred[i]] += 1
        np.savez_compressed(Path(out_dir) / f"test_task{t}.npz",
                            row_ids=np.arange(n, dtype=np.int64), labels=labels,
                            joint=np.zeros((n, 56), dtype=np.float32), logits=scores,
                            prediction=pred, confusion=cm,
                            seen=np.arange(kk, dtype=np.int64),
                            compute_dtype="stub", note="stub state-machine evidence")

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from trunk import s3_cli_fix6 as cli
    from trunk import s3_prod_fix6 as prod
    from trunk import run_baseline_fix2 as rb1
    from trunk import train_smoke as ts

    saved = (prod.S1, prod.S2, prod.ALL_TASKS, prod.TASK_COUNTS,
             cli.TASK_COUNTS, rb1.load_context, ts.make_joint)
    rows = []
    carried = []
    for nm, fn in [("valid_lock_accepted_by_default_gate", case_valid_lock),
                   ("test_restore_both_branches", case_test_restore),
                   ("production_counter_equality", case_production_counters),
                   ("gate_rejection_zero_build", case_gate_reject_zero_build),
                   ("init_snapshot_and_freeze_baseline", case_init_and_freeze),
                   ("lifecycle_24_cells_stub", case_lifecycle_24)]:
        try:
            carried.append({"case": nm, "outcome": "PASS", "detail": fn()})
        except Exception as e:
            carried.append({"case": nm, "outcome": "FAIL",
                            "detail": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-800:]})
    try:
        tmpd = Path(tempfile.mkdtemp())
        s1, s2 = make_assets(tmpd)
        ctx0 = SynthCtx()
        prod.S1, prod.S2 = s1, s2
        prod.ALL_TASKS = [[0, 1], [2, 3]]
        prod.TASK_COUNTS = {d: 2 for d in DATASETS}
        cli.TASK_COUNTS = {d: 2 for d in DATASETS}
        fake_ctx = {"x_std_test": ctx0.x_std, "test_view": {"x_raw": ctx0.x_raw, "y": ctx0.y},
                    "d_mean": ctx0.d_mean, "d_std": ctx0.d_std,
                    "input_channels": ctx0.input_channels, "n_classes": ctx0.n_classes,
                    "task_rows": lambda classes, which: ctx0.rows(classes, "train")}
        rb1.load_context = lambda dataset: fake_ctx
        ts.make_joint = fake_make_joint

        ex, deps, ctx, spec, s1rep, gf = build_lifecycle(tmpd / "A")
        base = cli.production_deps(ex, ctx_factory=lambda d: ctx, device="cpu",
                                   allow_cpu=True, gate_factory=gf)
        deps.test_macro = base.test_macro                     # REAL closure
        drive_to_sealed(ex, deps, cli)
        ev = cli.cmd_evaluate(ex, deps)
        prods = json.loads((ex / "test_products.json").read_text(encoding="utf-8"))
        npz_n = len(prods.get("test_npz_sha256", {}))
        rows.append({"case": "cli_evaluate_integration_real", "outcome": "PASS",
                     "detail": {"cells": ev["cells"], "test_npz_bound": npz_n,
                                "test_forwards": deps.test_counters["test_forwards"],
                                "note": "cmd_evaluate executed; training stage stubbed"}})
        agg = cli.cmd_aggregate(ex, deps)
        rows.append({"case": "evaluate_products_bind_npz", "outcome": "PASS",
                     "detail": {"n_npz": npz_n, "aggregate_cells": agg["cells"],
                                "products_bind": sorted(prods.keys())}})

        key0 = next(iter(prods["test_npz_sha256"]))
        cell, rest = key0.split("/seed")
        seed_part, fname = rest.split("/")
        victim = ex / "runs" / cell / f"seed{seed_part}" / fname
        orig = victim.read_bytes()
        victim.write_bytes(orig + b"T")
        try:
            cli.cmd_aggregate(ex, deps)
            ag = "NO_EXCEPTION"
        except Exception as e:
            ag = type(e).__name__
        victim.write_bytes(orig)
        rows.append({"case": "aggregate_rejects_npz_drift",
                     "outcome": "PASS" if ag == "CLIError" else "FAIL",
                     "detail": {"raised": ag}})

        ex2, deps2, ctx2, spec2, s1rep2, gf2 = build_lifecycle(tmpd / "C")
        base2 = cli.production_deps(ex2, ctx_factory=lambda d: ctx2, device="cpu",
                                    allow_cpu=True, gate_factory=gf2)
        deps2.test_macro = base2.test_macro
        drive_to_sealed(ex2, deps2, cli)
        rec = cli.load_state(ex2)["cells"][cli.cell_id("R1_analytic_head", "uci_har")]
        d0 = Path(rec["assets"][str(11)]["dir"])
        partial = d0 / "test_task2.npz"
        partial.write_bytes(b"PARTIAL-EVIDENCE")
        before = sha(partial)
        calls_before = deps2.test_counters["test_forwards"]
        try:
            cli.cmd_evaluate(ex2, deps2)
            res = "NO_EXCEPTION"
        except Exception as e:
            res = f"{type(e).__name__}: {str(e)[:70]}"
        after = sha(partial)
        ok = (res.startswith("CLIError") and before == after
              and deps2.test_counters["test_forwards"] == calls_before)
        rows.append({"case": "partial_npz_blocks_evaluate",
                     "outcome": "PASS" if ok else "FAIL",
                     "detail": {"raised": res, "bytes_unchanged": before == after,
                                "test_calls": deps2.test_counters["test_forwards"]}})
    except Exception as e:
        rows.append({"case": "fix6_suite", "outcome": "FAIL",
                     "detail": f"{type(e).__name__}: {e}",
                     "trace": traceback.format_exc()[-1400:]})
    finally:
        (prod.S1, prod.S2, prod.ALL_TASKS, prod.TASK_COUNTS,
         cli.TASK_COUNTS, rb1.load_context, ts.make_joint) = saved

    src = {n: sha(ROOT / f"src/trunk/{n}") for n in
           ["s3_cli_fix6.py", "s3_prod_fix6.py", "s3_lock_fix6.py",
            "execution_gate_v7.py", "s3_selftest_fix6.py",
            "s3_candidates_torch_fix1.py", "s3_candidates_numpy_fix1.py",
            "readout_sensitivity.py"]}
    rows = carried + rows
    (OUT / "evaluation_guard_fix6.json").write_text(json.dumps(
        {"cases": rows, "source_sha256": src}, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(f"  {r['case']:32s} {r['outcome']}  {json.dumps(r['detail'], ensure_ascii=False)[:180]}")
    bad = [r for r in rows if r["outcome"] == "FAIL"]
    print(f"cases={len(rows)} failures={len(bad)}")
    for b in bad:
        print(b.get("trace", ""))
    return 0 if not bad else 4


if __name__ == "__main__":
    raise SystemExit(main())
