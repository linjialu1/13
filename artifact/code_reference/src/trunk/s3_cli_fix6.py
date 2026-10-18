"""S3 lifecycle CLI — FIX6.

Fixes from REVIEW_s3_fix2_20261017.md:
  * `main()` builds a production Gate bound to `--exec-dir` + repo ROOT when none is injected;
  * `cmd_run` computes validation from the IN-MEMORY per_task values and only then writes
    `result.json` atomically — no read of a not-yet-written file;
  * counters carry every key the accepted trainer/head fitter increments;
  * `cmd_evaluate` hands the production test callback the VERIFIED per-seed record
    (model_path + per_task), not the whole cell record;
  * device is resolved once and threaded through every adapter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trunk import execution_gate_v7 as gate_mod  # noqa: E402

DATASETS = ["uci_har", "wisdm", "uwave", "cwru"]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
SCREEN_SEEDS = [11, 22, 33]
READOUT = ["R1_analytic_head", "R2_pca16_readout"]
TRAIN = ["LN_layernorm_injection", "POOL_mean_pooling", "C1_backbone_freeze",
         "C2_gru_backbone"]
CANDIDATES = READOUT + TRAIN
TASK_COUNTS = {"uci_har": 3, "wisdm": 3, "uwave": 4, "cwru": 5}
TCRIT_DF7 = 2.3646242515927853
S1 = ROOT / "experiments/baselines/trunk_baseline_4datasets"


class CLIError(RuntimeError):
    pass


class StageUnavailable(CLIError):
    pass


@dataclass
class Deps:
    """Compute adapters.  `device`/`ctx_factory` make the engineering path explicit."""
    build: object = None            # (candidate, dataset, seed, out_dir_or_None) -> model/dict
    train_task1: object = None      # (model, dataset, seed, counters) -> info
    freeze: object = None           # (model, counters) -> report
    fit_head: object = None         # (model, dataset, seed, out_dir, counters, candidate) -> dict
    val_macro_from_result: object = None  # (candidate, dataset, per_task) -> float
    baseline_val_macro: object = None     # (dataset, seed) -> float
    test_macro: object = None       # (candidate, dataset, seed, seed_result) -> float
    baseline_test_macro: object = None    # (dataset, seed) -> float
    budget: object = None                 # (out_dir, counters) -> dict
    test_accepts_context: bool = False    # production test writes per-row evidence
    test_counters: dict = field(default_factory=dict)
    gate_factory: object = None
    attach_hooks: object = None     # (model, counters) -> handles
    detach_hooks: object = None     # (handles) -> None
    init_snapshot: object = None    # (candidate, dataset, seed) -> dict with all_shared_..._equal
    device: str = "cpu"
    allow_cpu: bool = True


def cell_id(c: str, d: str) -> str:
    return f"{c}__{d}"


def all_cells():
    return [(c, d) for c in CANDIDATES for d in DATASETS]


# ---------------------------------------------------------------- state
def state_file(ex: Path) -> Path:
    return Path(ex) / "state.json"


def load_state(ex: Path) -> dict:
    p = state_file(ex)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else \
        {"stage": "none", "cells": {}}


def save_state(ex: Path, st: dict) -> None:
    state_file(ex).write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")


def _need(st, allowed, action):
    if st["stage"] not in allowed:
        raise CLIError(f"{action} requires stage in {allowed}, current={st['stage']}")
    if len(st["cells"]) != 24:
        raise CLIError(f"state has {len(st['cells'])} cells, require exactly 24")


def _terminal(rec) -> bool:
    return rec["state"] in ("COMPLETE", "SCREEN_FAIL")


# ---------------------------------------------------------------- commands
def cmd_lock(ex: Path, deps: Deps) -> dict:
    st = load_state(ex)
    if st["stage"] != "none":
        raise CLIError(f"already locked (stage={st['stage']})")
    need = ("protocol_locked.json", "protocol_locked.json.sha256",
            "dependency_manifest.json", "s1_before_snapshot.json", "self_check.json",
            "science_protocol.json")
    missing = [n for n in need if not (Path(ex) / n).exists()]
    if missing:
        raise CLIError(f"lock artifacts missing: {missing}")
    body = deps.gate_factory().full()
    st.update({"stage": "locked",
               "locked_sha256": gate_mod.sha(Path(ex) / "protocol_locked.json"),
               "protocol": {"science": gate_mod.sha(Path(ex) / "science_protocol.json")},
               "cells": {cell_id(c, d): {"candidate": c, "dataset": d, "seeds_run": [],
                                         "state": "PENDING", "assets": {}}
                         for c, d in all_cells()}})
    save_state(ex, st)
    return {"stage": "locked", "cells": 24, "locked_sha256": st["locked_sha256"]}


def cmd_run(ex: Path, deps: Deps, candidate: str, dataset: str, seed: int) -> dict:
    from trunk import s3_prod_fix6 as prod
    st = load_state(ex)
    _need(st, ["locked", "running"], "run")
    if candidate not in CANDIDATES:
        raise CLIError(f"unknown candidate {candidate}")
    if dataset not in DATASETS:
        raise CLIError(f"unknown dataset {dataset}")
    if seed not in SEEDS:
        raise CLIError(f"unknown seed {seed}")
    cell = cell_id(candidate, dataset)
    rec = st["cells"][cell]
    if rec["state"] == "SCREEN_FAIL":
        raise CLIError(f"{cell} SCREEN_FAIL; no further seeds")
    if rec["state"] == "COMPLETE":
        raise CLIError(f"{cell} already COMPLETE")
    if seed in rec["seeds_run"]:
        raise CLIError(f"{cell} seed {seed} already run")
    is_train = candidate in TRAIN
    if is_train and not rec.get("promoted") and seed not in SCREEN_SEEDS:
        raise CLIError(f"{cell}: seed {seed} not allowed before promotion")
    out_dir = Path(ex) / "runs" / cell / f"seed{seed}"
    if out_dir.exists():
        raise CLIError(f"run output exists: {out_dir}")

    import time as _t
    _t0 = _t.perf_counter()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.reset_peak_memory_stats()      # per-seed peak, not cumulative
    except Exception:
        pass
    deps.gate_factory().full(out_dir)
    counters = prod.new_counters() if hasattr(prod, "new_counters") else {}
    counters.update({"encoder_forwards": 0, "encoder_backward": 0,
                     "encoder_optimizer_steps": 0, "head_forwards": 0,
                     "head_backward": 0, "head_optimizer_steps": 0,
                     "gradient_updates": 0})
    record = {"candidate": candidate, "dataset": dataset, "seed": seed,
              "device": deps.device}

    if is_train:
        if None in (deps.build, deps.train_task1, deps.freeze, deps.fit_head):
            raise StageUnavailable("training adapters not wired")
        # shared-parameter initialization must be verified BEFORE any training
        if deps.init_snapshot is not None:
            snap = deps.init_snapshot(candidate, dataset, seed)
            record["init_snapshot"] = {
                "shared_param_count": snap.get("shared_param_count"),
                "all_shared_initial_values_equal": snap.get("all_shared_initial_values_equal"),
                "per_candidate_shared_sha256": snap.get("per_candidate_shared_sha256")}
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "init_snapshot.json").write_text(
                json.dumps(snap, indent=1, sort_keys=True), encoding="utf-8")
            record["init_snapshot_file"] = str(out_dir / "init_snapshot.json")
            record["init_snapshot_sha256"] = gate_mod.sha(out_dir / "init_snapshot.json")
            if not snap.get("all_shared_initial_values_equal", False):
                raise CLIError(f"shared parameter init mismatch: {snap.get('mismatched')}")
        model = deps.build(candidate, dataset, seed, None)
        # hooks installed ONCE here (production assembly), before Task1
        handles = deps.attach_hooks(model, counters) if deps.attach_hooks else []
        try:
            record["task1"] = deps.train_task1(model, dataset, seed, counters)
            record["freeze"] = deps.freeze(model, counters)
            frozen_sig = record["freeze"].get("signature_sha256")
            heads = deps.fit_head(model, dataset, seed, out_dir, counters, candidate,
                                  frozen_sig)
            record.update(heads)
        finally:
            if deps.detach_hooks:
                deps.detach_hooks(handles)
        if deps.val_macro_from_result is None:
            raise StageUnavailable("validation adapter not wired")
        val = float(deps.val_macro_from_result(candidate, dataset, heads["per_task"]))
    else:
        if deps.build is None:
            raise StageUnavailable("readout adapter not wired")
        info = deps.build(candidate, dataset, seed, out_dir)
        record.update(info)
        val = float(info["val_macro"])

    if not math.isfinite(val) or not (0.0 <= val <= 1.0):
        raise CLIError(f"validation macro out of range: {val}")
    import time as _t
    if deps.budget is not None:
        record["budget"] = deps.budget(out_dir, counters)
    record.update({"val_macro": val, "counters": counters,
                   "elapsed_seconds": round(_t.perf_counter() - _t0, 1)})
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ".result.json.tmp"
    tmp.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, out_dir / "result.json")          # atomic publish after success

    rec["seeds_run"].append(seed)
    rec["assets"][str(seed)] = {"dir": str(out_dir),
                                "result_sha256": gate_mod.sha(out_dir / "result.json")}
    if rec["state"] != "PROMOTED":
        rec["state"] = "RUNNING"
    if len(rec["seeds_run"]) == 8:
        rec["state"] = "COMPLETE"
    st["stage"] = "running"
    save_state(ex, st)
    return {"cell": cell, "seed": seed, "val_macro": val, "state": rec["state"],
            "seeds_run": rec["seeds_run"]}


def cmd_screen(ex: Path, deps: Deps, candidate: str, dataset: str) -> dict:
    from trunk import s3_candidates_numpy_fix1 as cn
    st = load_state(ex)
    _need(st, ["running"], "screen")
    if candidate not in TRAIN:
        raise CLIError("screening is for training candidates only")
    cell = cell_id(candidate, dataset)
    rec = st["cells"][cell]
    if sorted(rec["seeds_run"]) != SCREEN_SEEDS:
        raise CLIError(f"screen needs {SCREEN_SEEDS}, have {rec['seeds_run']}")
    if deps.baseline_val_macro is None:
        raise StageUnavailable("baseline validation adapter not wired")
    diffs = []
    for s in SCREEN_SEEDS:
        rj = json.loads((Path(rec["assets"][str(s)]["dir"]) / "result.json")
                        .read_text(encoding="utf-8"))
        diffs.append(float(rj["val_macro"]) - float(deps.baseline_val_macro(dataset, s)))
    verdict = cn.screen_promote(diffs)
    rec.update({"screen_diffs": diffs, "state": verdict["state"],
                "promoted": bool(verdict["state"] == "PROMOTED")})
    save_state(ex, st)
    out = Path(ex) / "runs" / cell / "screen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"diffs": diffs, **verdict}, indent=1), encoding="utf-8")
    return {"cell": cell, "diffs": diffs, "state": rec["state"]}


def cmd_expand(ex: Path, deps: Deps) -> dict:
    st = load_state(ex)
    _need(st, ["running"], "expand")
    expanded, done = [], []
    for cell, rec in sorted(st["cells"].items()):
        if not rec.get("promoted") or rec["state"] in ("COMPLETE", "SCREEN_FAIL"):
            continue
        for s in SEEDS:
            if s in rec["seeds_run"]:
                continue
            cmd_run(ex, deps, rec["candidate"], rec["dataset"], s)
            expanded.append(f"{cell}/seed{s}")
        done.append(cell)
    return {"expanded_runs": expanded, "completed_cells": done}


def cmd_seal(ex: Path, deps: Deps) -> dict:
    st = load_state(ex)
    _need(st, ["running"], "seal")
    problems = []
    for cell, rec in sorted(st["cells"].items()):
        if not _terminal(rec):
            problems.append(f"{cell}:{rec['state']}")
            continue
        want = SCREEN_SEEDS if rec["state"] == "SCREEN_FAIL" else SEEDS
        if sorted(rec["seeds_run"]) != sorted(want):
            problems.append(f"{cell}:seeds={sorted(rec['seeds_run'])}")
            continue
        for s in rec["seeds_run"]:
            a = rec["assets"][str(s)]
            rj = Path(a["dir"]) / "result.json"
            if not rj.exists() or gate_mod.sha(rj) != a["result_sha256"]:
                problems.append(f"{cell}/seed{s}: result hash")
                continue
            data = json.loads(rj.read_text(encoding="utf-8"))
            if data.get("init_snapshot_sha256"):
                ip = Path(data["init_snapshot_file"])
                if not ip.exists() or gate_mod.sha(ip) != data["init_snapshot_sha256"]:
                    problems.append(f"{cell}/seed{s}: init snapshot")
            if rec["candidate"] in TRAIN:
                mp = Path(data.get("model_path", ""))
                if not mp.exists() or gate_mod.sha(mp) != data.get("model_sha256"):
                    problems.append(f"{cell}/seed{s}: model hash")
                for t, info in data["per_task"].items():
                    hp = Path(info["head_path"])
                    if not hp.exists() or gate_mod.sha(hp) != info["head_sha256"]:
                        problems.append(f"{cell}/seed{s}/{t}: head hash")
                    sp_ = Path(info["head_path"]).parent / f"state_package_task{t.split('task')[-1]}.npz"
                    if not sp_.exists() or gate_mod.sha(sp_) != info.get("state_pkg_sha256"):
                        problems.append(f"{cell}/seed{s}/{t}: state package")
                    bp_ = Path(info["head_path"]).parent / f"boundary_{t}.npz"
                    if not bp_.exists() or gate_mod.sha(bp_) != info.get("boundary_npz_sha256"):
                        problems.append(f"{cell}/seed{s}/{t}: boundary evidence")
                    bp = Path(info["head_path"]).parent / f"boundary_{t}.npz"
                    if not bp.exists() or gate_mod.sha(bp) != info["boundary_npz_sha256"]:
                        problems.append(f"{cell}/seed{s}/{t}: boundary evidence")
    if problems:
        raise CLIError(f"seal refused: {len(problems)} problem(s): {problems[:4]}")
    sp = Path(ex) / "candidate_seal.json"
    if sp.exists():
        raise CLIError("candidate_seal.json already exists")
    seal = {"status": "SEALED", "locked_sha256": st["locked_sha256"],
            "science_protocol_sha256": st["protocol"]["science"],
            "n_cells": 24, "cells": st["cells"]}
    sp.write_text(json.dumps(seal, indent=1, default=str), encoding="utf-8")
    (Path(ex) / "candidate_seal.json.sha256").write_text(
        json.dumps({"sha256": gate_mod.sha(sp)}), encoding="utf-8")
    st["stage"] = "sealed"
    save_state(ex, st)
    return {"stage": "sealed", "n_cells": 24, "seal_sha256": gate_mod.sha(sp)}


def _verify_seal(ex: Path) -> dict:
    sp = Path(ex) / "candidate_seal.json"
    rec = Path(ex) / "candidate_seal.json.sha256"
    if not sp.exists() or not rec.exists():
        raise CLIError("candidate seal missing; test evaluation refused")
    if json.loads(rec.read_text(encoding="utf-8"))["sha256"] != gate_mod.sha(sp):
        raise CLIError("candidate seal hash mismatch")
    seal = json.loads(sp.read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED" or len(seal.get("cells", {})) != 24:
        raise CLIError("candidate seal incomplete")
    if seal.get("locked_sha256") != gate_mod.sha(Path(ex) / "protocol_locked.json"):
        raise CLIError("seal produced under a different locked protocol")
    for cell, r in seal["cells"].items():
        for s, a in r["assets"].items():
            rj = Path(a["dir"]) / "result.json"
            if not rj.exists() or gate_mod.sha(rj) != a["result_sha256"]:
                raise CLIError(f"sealed asset changed: {cell}/seed{s}")
            data = json.loads(rj.read_text(encoding="utf-8"))
            if r["candidate"] in TRAIN:
                mp = Path(data["model_path"])
                if not mp.exists() or gate_mod.sha(mp) != data["model_sha256"]:
                    raise CLIError(f"sealed model changed: {cell}/seed{s}")
                for t, info in data["per_task"].items():
                    hp = Path(info["head_path"])
                    if not hp.exists() or gate_mod.sha(hp) != info["head_sha256"]:
                        raise CLIError(f"sealed head changed: {cell}/seed{s}/{t}")
                    tno = t.split("task")[-1]
                    sp_ = hp.parent / f"state_package_task{tno}.npz"
                    if not sp_.exists() or gate_mod.sha(sp_) != info.get("state_pkg_sha256"):
                        raise CLIError(f"sealed state package changed: {cell}/seed{s}/{t}")
                    bp_ = hp.parent / f"boundary_{t}.npz"
                    if not bp_.exists() or gate_mod.sha(bp_) != info.get("boundary_npz_sha256"):
                        raise CLIError(f"sealed boundary evidence changed: {cell}/seed{s}/{t}")
    return seal


def cmd_evaluate(ex: Path, deps: Deps) -> dict:
    st = load_state(ex)
    _need(st, ["sealed"], "evaluate")
    out = Path(ex) / "test_results.json"
    if out.exists():
        raise CLIError("test results already exist (no overwrite)")
    # ---- pre-flight: enumerate EVERY target of this evaluation and refuse if any exists
    # (a previous mid-way failure may have left some NPZs without the summary file)
    targets = [out, Path(ex) / "test_ledger.json", Path(ex) / "test_products.json"]
    for cell, rec in st["cells"].items():
        if rec["state"] not in ("COMPLETE", "SCREEN_FAIL"):
            continue
        for s0 in rec["seeds_run"]:
            d = Path(rec["assets"][str(s0)]["dir"])
            tcount = TASK_COUNTS[rec["dataset"]]
            targets.append(d / f"test_task{tcount}.npz")
    existing = [str(p) for p in targets if Path(p).exists()]
    if existing:
        raise CLIError(f"evaluation targets already exist (partial run): {existing[:3]}")
    deps.gate_factory().full()            # full gate before any test read
    seal = _verify_seal(ex)
    # init snapshots must still match before any test read
    for cell, rec in seal["cells"].items():
        for s0 in rec["seeds_run"]:
            rj = Path(rec["assets"][str(s0)]["dir"]) / "result.json"
            data = json.loads(rj.read_text(encoding="utf-8"))
            if data.get("init_snapshot_sha256"):
                ip = Path(data["init_snapshot_file"])
                if not ip.exists() or gate_mod.sha(ip) != data["init_snapshot_sha256"]:
                    raise CLIError(f"init snapshot changed: {cell}/seed{s0}")
    if deps.test_macro is None:
        raise StageUnavailable("test adapter not wired")
    res = {}
    for cell, rec in sorted(seal["cells"].items()):
        for s in rec["seeds_run"]:
            sr = json.loads((Path(rec["assets"][str(s)]["dir"]) / "result.json")
                            .read_text(encoding="utf-8"))
            kw = {}
            if deps.test_accepts_context:
                kw = {"out_dir": Path(rec["assets"][str(s)]["dir"]),
                      "counters": deps.test_counters}
            v = float(deps.test_macro(rec["candidate"], rec["dataset"], int(s), sr, **kw))
            if not math.isfinite(v) or not (0.0 <= v <= 1.0):
                raise CLIError(f"test macro out of range {cell}/seed{s}: {v}")
            res.setdefault(cell, {})[str(s)] = v
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    ledger = {"test_forwards": dict(deps.test_counters),
              "n_cells": len(res),
              "note": "test_forwards counts test INFERENCE CALLS per candidate/seed, "
                      "not encoder batch invocations; sealed training results untouched"}
    (Path(ex) / "test_ledger.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    npz_hashes = {}
    for cell, rec in seal["cells"].items():
        for s0 in rec["seeds_run"]:
            d = Path(rec["assets"][str(s0)]["dir"])
            f = d / f"test_task{TASK_COUNTS[rec['dataset']]}.npz"
            if not f.exists():
                raise CLIError(f"missing test evidence after evaluation: {cell}/seed{s0}")
            npz_hashes[f"{cell}/seed{s0}/test_task{TASK_COUNTS[rec['dataset']]}.npz"] = \
                gate_mod.sha(f)
    (Path(ex) / "test_products.json").write_text(json.dumps(
        {"test_results_sha256": gate_mod.sha(out),
         "test_ledger_sha256": gate_mod.sha(Path(ex) / "test_ledger.json"),
         "seal_sha256": gate_mod.sha(Path(ex) / "candidate_seal.json"),
         "test_npz_sha256": npz_hashes,
         "n_test_npz": len(npz_hashes)}, indent=1), encoding="utf-8")
    st["stage"] = "evaluated"
    save_state(ex, st)
    return {"stage": "evaluated", "cells": len(res)}


def cmd_aggregate(ex: Path, deps: Deps) -> dict:
    from trunk import s3_candidates_numpy_fix1 as cn
    st = load_state(ex)
    _need(st, ["evaluated"], "aggregate")
    res = json.loads((Path(ex) / "test_results.json").read_text(encoding="utf-8"))
    seal = _verify_seal(ex)
    prod_p = Path(ex) / "test_products.json"
    if not prod_p.exists():
        raise CLIError("test products manifest missing")
    prodm = json.loads(prod_p.read_text(encoding="utf-8"))
    if prodm["test_results_sha256"] != gate_mod.sha(Path(ex) / "test_results.json"):
        raise CLIError("test results drifted since evaluation")
    if prodm["test_ledger_sha256"] != gate_mod.sha(Path(ex) / "test_ledger.json"):
        raise CLIError("test ledger drifted since evaluation")
    drift = [k for k, v in prodm.get("test_npz_sha256", {}).items()
             if not (Path(ex) / "runs" / k.split("/seed")[0] / ("seed" + k.split("/seed")[1].split("/")[0])
                     / k.split("/")[-1]).exists()]
    for key, h in prodm.get("test_npz_sha256", {}).items():
        cell, rest = key.split("/seed")
        seed_part, fname = rest.split("/")
        d = Path(ex) / "runs" / cell / f"seed{seed_part}"
        f = d / fname
        if not f.exists() or gate_mod.sha(f) != h:
            raise CLIError(f"test evidence drifted: {key}")
    rows, holm_input = {}, {}
    for cell, per_seed in sorted(res.items()):
        rec = seal["cells"][cell]
        cand, ds = rec["candidate"], rec["dataset"]
        diffs, cands, bases = [], [], []
        for s_str, v in sorted(per_seed.items(), key=lambda kv: int(kv[0])):
            s = int(s_str)
            if deps.baseline_test_macro is not None:
                base = float(deps.baseline_test_macro(ds, s))
            else:
                base = float(json.loads((S1 / ds / f"seed{s}" / "evaluation.json")
                                        .read_text(encoding="utf-8"))
                             ["final_task_seen_macro_accuracy"])
            diffs.append(float(v) - base); cands.append(float(v)); bases.append(base)
        stat = cn.paired_t_summary(diffs)
        row = {"candidate": cand, "dataset": ds, "n_seeds": len(diffs),
               "candidate_test": cands, "s1_baseline_test": bases,
               "paired_gain": diffs, **stat,
               "strictly_positive": int(sum(1 for d in diffs if d > 0)),
               "state": rec["state"]}
        if len(diffs) == 8:
            row["ci95"] = [stat["mean"] - TCRIT_DF7 * stat["sd"] / math.sqrt(8),
                           stat["mean"] + TCRIT_DF7 * stat["sd"] / math.sqrt(8)]
            if stat.get("p") is not None:
                holm_input[cell] = stat["p"]
        else:
            row["ci95"] = None; row["p"] = None
            row["note"] = "SCREEN_FAIL: main p/CI null"
        rows[cell] = row
    holm = cn.holm(holm_input, family_size=24)
    for cell, info in holm["results"].items():
        rows[cell]["p_adjusted_holm24"] = info["adjusted_p"]
    r2r1 = {}
    for ds in DATASETS:
        a, b = rows.get(cell_id("R1_analytic_head", ds)), rows.get(cell_id("R2_pca16_readout", ds))
        if a and b:
            r2r1[ds] = cn.paired_t_summary([y - x for x, y in
                                           zip(a["candidate_test"], b["candidate_test"])])
    sens = {}
    for ds in DATASETS:
        vals = []
        for s in SEEDS:
            bm = json.loads((ROOT / "experiments/sensitivity/"
                             "trunk_readout_sensitivity_4datasets" / ds / f"seed{s}" /
                             f"task{TASK_COUNTS[ds]}" / "boundary_metrics.json").read_text())
            vals.append(float(bm["range_4"]))
        sens[ds] = sum(vals) / len(vals)
    def gains(c):
        return [rows[cell_id(c, d)]["mean"] if cell_id(c, d) in rows else 0.0
                for d in DATASETS]
    s4 = {"sensitivity_mean": sens,
          "spearman_R1": cn.spearman([sens[d] for d in DATASETS], gains("R1_analytic_head")),
          "spearman_R2": cn.spearman([sens[d] for d in DATASETS], gains("R2_pca16_readout")),
          "note": "descriptive; 4 top-level units; no n=88 inference"}
    outd = {"cells": rows, "holm_family_24": holm, "r2_minus_r1": r2r1, "s4": s4}
    (Path(ex) / "aggregate.json").write_text(json.dumps(outd, indent=1), encoding="utf-8")
    return {"cells": len(rows), "holm_family": holm["family_size"]}


# ---------------------------------------------------------------- deps assembly
def production_gate(exec_dir: Path):
    return gate_mod.Gate(Path(exec_dir), S1)      # v5: unified enumeration


def production_deps(exec_dir: Path, *, gate_factory=None, ctx_factory=None,
                    device: str | None = None, allow_cpu: bool = True) -> Deps:
    from trunk import s3_prod_fix6 as prod
    dev = torch_device = device or prod.resolve_device(allow_cpu=False).type
    gate_factory = gate_factory or (lambda: production_gate(exec_dir))
    ctx_cache = {}

    def ctx_of(dataset):
        if dataset not in ctx_cache:
            ctx_cache[dataset] = (ctx_factory(dataset) if ctx_factory
                                  else prod.TrainCtx(dataset))
        return ctx_cache[dataset]

    def build(candidate, dataset, seed, out_dir):
        import torch
        d = torch.device(torch_device)
        if candidate in READOUT:
            info = prod.readout_val_macro(candidate, dataset, seed)
            if out_dir is not None:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / "readout_asset.json").write_text(
                    json.dumps(info, indent=1), encoding="utf-8")
            return {"readout": True, "val_macro": info["val_macro"],
                    "asset_refs": info["asset_refs"], "n_val_cum": info["n_val_cum"]}
        return prod.build_candidate(candidate, ctx_of(dataset), seed, d)

    def train(model, dataset, seed, counters):
        import torch
        return prod.train_task1(model, ctx_of(dataset), seed, counters,
                                torch.device(torch_device))

    def freeze(model, counters):
        return prod.freeze_all(model, counters)

    def fit_head(model, dataset, seed, out_dir, counters, candidate, frozen_sig=None):
        import torch
        return prod.run_boundaries(model, ctx_of(dataset), seed=seed, out_dir=out_dir,
                                   counters=counters, candidate=candidate,
                                   device=torch.device(torch_device),
                                   frozen_baseline=frozen_sig)

    def val_from_result(candidate, dataset, per_task):
        return prod.final_val_macro(per_task, dataset)

    def base_val(dataset, seed):
        import torch
        return prod.s1_baseline_val_macro(dataset, seed, TASK_COUNTS[dataset],
                                          torch.device(torch_device))

    def test_macro(candidate, dataset, seed, seed_result, out_dir=None, counters=None):
        import torch
        return prod.test_macro_from_seed_result(candidate, dataset, seed, seed_result,
                                                torch.device(torch_device),
                                                out_dir=out_dir, counters=counters)

    def init_snapshot(candidate, dataset, seed):
        ctx = ctx_of(dataset)
        return prod.common_param_snapshot(candidate, ctx.input_channels, ctx.n_classes, seed)

    def budget(out_dir, counters):
        import torch
        mbytes = None
        try:
            info = torch.cuda.max_memory_allocated(torch.device(torch_device))
            reserved = torch.cuda.max_memory_reserved(torch.device(torch_device))
            mbytes = {"gpu_peak_allocated_bytes": int(info),
                      "gpu_peak_reserved_bytes": int(reserved)}
        except Exception:
            mbytes = {"gpu_peak": "NOT_MEASURED"}
        rss = None
        try:
            from trunk import run_baseline_fix2 as rb1
            rss = rb1.process_memory_bytes()
        except Exception:
            rss = None
        # parameter counts / bytes from the artifacts actually written for this seed
        param_counts, persist_bytes, replay = {}, {}, None
        try:
            d = Path(out_dir)
            mp = d / "model.pt"
            if mp.exists():
                ck = torch.load(mp, map_location="cpu", weights_only=False)
                sd = ck.get("model_state_dict", {})
                n = sum(v.numel() for v in sd.values())
                b = sum(v.numel() * v.element_size() for v in sd.values())
                param_counts["model"] = int(n); persist_bytes["model"] = int(b)
            heads = sorted(d.glob("task*_head.pt"))
            if heads:
                hn = hb = 0
                for hp in heads:
                    sd = torch.load(hp, map_location="cpu", weights_only=False)
                    hn += sum(v.numel() for v in sd.values())
                    hb += sum(v.numel() * v.element_size() for v in sd.values())
                param_counts["heads"] = int(hn); persist_bytes["heads"] = int(hb)
            pkgs = sorted(d.glob("state_package_task*.npz"))
            pb = sum(p.stat().st_size for p in pkgs)
            persist_bytes["state_packages_serialized"] = int(pb)
            replay = 0     # no real replay buffer in this protocol
        except Exception as e:
            param_counts["error"] = str(e)[:80]
        return {"gpu": mbytes, "rss": rss if rss else "NOT_MEASURED",
                "parameter_counts": param_counts, "persistent_bytes": persist_bytes,
                "real_replay_samples": replay,
                "note": "gpu peak is reset per seed before that seed's run; "
                        "rss NOT_MEASURED when the platform counter is unavailable"}

    return Deps(build=build, train_task1=train, freeze=freeze, fit_head=fit_head,
                val_macro_from_result=val_from_result, baseline_val_macro=base_val,
                test_macro=test_macro, baseline_test_macro=None,
                gate_factory=gate_factory, device=torch_device, allow_cpu=allow_cpu,
                attach_hooks=prod.attach_hooks, detach_hooks=prod.detach_hooks,
                init_snapshot=init_snapshot, budget=budget,
                test_accepts_context=True, test_counters={})


def main(argv=None, deps: Deps | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-dir", required=True)
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--run", nargs=3, metavar=("CANDIDATE", "DATASET", "SEED"))
    ap.add_argument("--screen", nargs=2, metavar=("CANDIDATE", "DATASET"))
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args(argv)
    ex = Path(a.exec_dir)
    if deps is None:
        deps = production_deps(ex)          # production Gate bound to --exec-dir + ROOT
    try:
        if a.lock:
            print(json.dumps(cmd_lock(ex, deps)))
        elif a.run:
            print(json.dumps(cmd_run(ex, deps, a.run[0], a.run[1], int(a.run[2]))))
        elif a.screen:
            print(json.dumps(cmd_screen(ex, deps, a.screen[0], a.screen[1])))
        elif a.expand:
            print(json.dumps(cmd_expand(ex, deps)))
        elif a.seal:
            print(json.dumps(cmd_seal(ex, deps)))
        elif a.evaluate:
            print(json.dumps(cmd_evaluate(ex, deps)))
        elif a.aggregate:
            print(json.dumps(cmd_aggregate(ex, deps)))
        else:
            ap.error("no command")
    except (CLIError, gate_mod.GateError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
