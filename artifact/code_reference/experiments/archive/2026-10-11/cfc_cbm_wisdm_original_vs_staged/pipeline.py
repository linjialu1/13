"""Multi-seed core CI-CBM adaptation to CfC and UCI-HAR."""
from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".local_packages"))
sys.path.insert(0, str(ROOT))
from src.legacy import benchmark as b  # noqa: E402
from src.legacy.cicbm import benchmark as cbm  # noqa: E402

SEEDS = [11, 22, 33]
SEED = 11
MAX_EPOCHS = 12
MIN_EPOCHS = 3
PATIENCE = 2
MEMORY_PER_CLASS = 20
CURRENT_BATCH = 96
REPLAY_BATCH = 32
BALANCED_CBM_EXAMPLES_PER_CLASS = 128
EVAL_BATCH = 256
LR = 1e-3


def load_raw_uci_har():
    folder = ROOT / "data" / "UCI HAR Dataset"
    channels = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y",
                "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
    raw = []
    for split in ("train", "test"):
        sub = folder / split / "Inertial Signals"
        raw.append(np.stack([np.loadtxt(sub / f"{name}_{split}.txt", dtype=np.float32)
                             for name in channels], axis=-1))
    return raw[0], raw[1]


def train_cfctask(model, optimizer, x, y, train_idx, val_idx, seen, device, seed):
    best_key, best_state, best_opt = (-1.0, -float("inf")), None, None
    best_acc, best_loss, stagnant, trained = -1.0, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        loader = b.make_loader(x, y, train_idx, 128, seed + epoch)
        model.train()
        for xx, yy in loader:
            xx, yy = xx.to(device), yy.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx)[:, seen]
            loss = nn.functional.cross_entropy(logits, b.labels_to_local(yy, seen, device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        va, vl = b.evaluate_validation(model, x, y, val_idx, seen[-2:], device, EVAL_BATCH)
        trained = epoch
        key = (va, -vl)
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = copy.deepcopy(optimizer.state_dict())
        if va - best_acc >= 0.005 or best_loss - vl >= 0.001:
            best_acc, best_loss, stagnant = max(best_acc, va), min(best_loss, vl), 0
        else:
            stagnant += 1
        print(f"CfC task={len(seen)//2} epoch={epoch}/{MAX_EPOCHS} val={va:.3f}", flush=True)
        if epoch >= MIN_EPOCHS and stagnant >= PATIENCE:
            break
    model.load_state_dict(best_state)
    optimizer.load_state_dict(best_opt)
    return trained


def fit_replay_baseline(x, y, xt, yt, raw_train, pool, device):
    b.set_seed(SEED)
    model = b.SequenceClassifier("CfC", x.shape[-1], 48, len(b.CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    seen, memory, learned_acc, rows = [], {}, {}, []
    for task_i, new_classes in enumerate(b.TASKS, start=1):
        old = list(seen)
        seen.extend(new_classes)
        train_idx, val_idx = b.split_task_train_validation(y, pool, new_classes, 0.2, 20261004 + task_i*100)
        best_key, best_state, best_opt = (-1.0, -float("inf")), None, None
        best_acc, best_loss, stagnant, trained = -1.0, float("inf"), 0, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            loader = b.make_loader(x, y, train_idx, CURRENT_BATCH if memory else 128,
                                   SEED + task_i*100_000 + epoch)
            model.train()
            rr = np.random.default_rng(SEED + task_i*200_000 + epoch)
            for xx, yy in loader:
                xx, yy = xx.to(device), yy.to(device)
                if memory:
                    mx = np.concatenate([memory[c][0] for c in old], axis=0)
                    my = np.concatenate([memory[c][1] for c in old], axis=0)
                    pick = rr.integers(0, len(my), size=min(REPLAY_BATCH, len(xx)))
                    xx = torch.cat([xx, torch.from_numpy(mx[pick]).to(device)])
                    yy = torch.cat([yy, torch.from_numpy(my[pick]).long().to(device)])
                optimizer.zero_grad(set_to_none=True)
                logits = model(xx)[:, seen]
                loss = nn.functional.cross_entropy(logits, b.labels_to_local(yy, seen, device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            va, vl = b.evaluate_validation(model, x, y, val_idx, new_classes, device, EVAL_BATCH)
            trained = epoch
            key = (va, -vl)
            if key > best_key:
                best_key = key
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_opt = copy.deepcopy(optimizer.state_dict())
            if va - best_acc >= 0.005 or best_loss - vl >= 0.001:
                best_acc, best_loss, stagnant = max(best_acc, va), min(best_loss, vl), 0
            else:
                stagnant += 1
            if epoch >= MIN_EPOCHS and stagnant >= PATIENCE:
                break
        model.load_state_dict(best_state)
        optimizer.load_state_dict(best_opt)
        per = b.evaluate_test(model, xt, yt, seen, device, EVAL_BATCH)
        for cls in new_classes:
            learned_acc[cls] = per[cls]
        rows.append(make_row("CfC_replay20", task_i, trained, old, new_classes, seen, per, learned_acc))
        if task_i < len(b.TASKS):
            for cls in new_classes:
                candidates = train_idx[y[train_idx] == cls].copy()
                rng = np.random.default_rng(SEED + task_i*1000 + cls)
                rng.shuffle(candidates)
                keep = candidates[:min(MEMORY_PER_CLASS, len(candidates))]
                memory[cls] = (x[keep].copy(), y[keep].copy())
    return rows


def extract_cfc_latent(model, x, indices, device):
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0, len(indices), EVAL_BATCH):
            xx = torch.from_numpy(x[indices[start:start+EVAL_BATCH]]).to(device)
            z, _ = model.representation_and_logits(xx)
            out.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def make_joint(model, x, raw, indices, device, signal_stats):
    latent = extract_cfc_latent(model, x, indices, device)
    signal = cbm.raw_concept_features(raw[indices])
    mean, sd = signal_stats
    signal = np.clip((signal-mean)/(sd+1e-6), -6, 6).astype(np.float32)
    return np.concatenate([latent, signal], axis=1).astype(np.float32)


def balanced_indices(y, available_by_class, seen, seed):
    rng = np.random.default_rng(seed)
    selected=[]
    for cls in seen:
        options = available_by_class[cls]
        selected.extend(rng.choice(options, size=BALANCED_CBM_EXAMPLES_PER_CLASS,
                                   replace=len(options) < BALANCED_CBM_EXAMPLES_PER_CLASS).tolist())
    return np.asarray(selected, dtype=np.int64)


def fit_projector_cubic(z, p, teacher, old_count, cfg, device, previous=None):
    """Paper-style cubic normalized-cosine concept alignment/distillation."""
    layer = nn.Linear(z.shape[1], p.shape[1], bias=False).to(device)
    if previous is not None:
        with torch.no_grad():
            layer.weight[:old_count].copy_(previous.weight[:old_count])
    opt = torch.optim.Adam(layer.parameters(), lr=cfg["concept_projector_lr"])
    zz, pp = torch.from_numpy(z).to(device), torch.from_numpy(p).to(device)
    tt = torch.from_numpy(teacher).to(device) if teacher is not None else None

    def norm_cubic(v):
        v = (v - v.mean(0, keepdim=True)) / (v.std(0, keepdim=True, unbiased=False) + 1e-5)
        v = v.clamp(-5, 5).pow(3)
        return v / (v.norm(dim=0, keepdim=True) + 1e-6)

    for _ in range(cfg["concept_projector_epochs"]):
        opt.zero_grad(set_to_none=True)
        q = torch.sigmoid(layer(zz))
        loss = -(norm_cubic(q) * norm_cubic(pp)).sum(0).mean()
        if tt is not None and old_count:
            loss -= cfg["concept_distillation_weight"] * (
                norm_cubic(q[:, :old_count]) * norm_cubic(tt)
            ).sum(0).mean()
        loss.backward()
        opt.step()
    return layer


def joint_refine(projector, predictor, previous, z_fit, targets, all_z, labels, seen, old_count, cfg, device, kd_on_pseudo, use_distillation=True, staged_training=False):
    z = torch.from_numpy(z_fit).to(device)
    target = torch.from_numpy(targets).to(device)
    joint = torch.from_numpy(all_z).to(device)
    # Classifier columns are ordered by `seen`, not necessarily raw dataset IDs.
    yy = torch.from_numpy(cbm.seen_index(labels, seen)).long().to(device)
    kd_z = joint if kd_on_pseudo else z
    if use_distillation:
        with torch.no_grad():
            teacher = torch.sigmoid(previous(kd_z)).detach()
    opt = torch.optim.Adam(list(projector.parameters()) + list(predictor.parameters()), lr=0.001)
    history = []
    for epoch in range(20):
        opt.zero_grad(set_to_none=True)
        concepts = torch.sigmoid(projector(joint))
        ce = nn.functional.cross_entropy(predictor(concepts), yy)
        align = cbm.alignment_loss(torch.sigmoid(projector(z)), target)
        if use_distillation:
            q_kd = concepts[:, :old_count] if kd_on_pseudo else torch.sigmoid(projector(z))[:, :old_count]
            kd = cbm.alignment_loss(q_kd, teacher)
        else:
            kd = torch.zeros((), device=device)
        sparse = cfg['sparsity_lambda'] * predictor.weight.abs().sum()
        loss = ce + align + kd + sparse
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite joint loss')
        loss.backward()
        old_rows_frozen = staged_training and epoch < 10
        if old_rows_frozen and projector.weight.grad is not None:
            projector.weight.grad[:old_count].zero_()
        nn.utils.clip_grad_norm_(list(projector.parameters()) + list(predictor.parameters()), 5.0)
        opt.step()
        history.append({'epoch':epoch+1,'classification_loss':float(ce.detach()),'alignment_loss':float(align.detach()),'distillation_loss':float(kd.detach()),'old_rows_frozen':bool(old_rows_frozen)})
    return history


def fit_cbm_condition(x, y, xt, yt, raw, raw_test, pool, device,
                      condition='CBM_original', warmstart=False, joint_training=False,
                      kd_on_pseudo=False, head_only=False, use_distillation=True, staged_training=False):
    b.set_seed(SEED)
    model = b.SequenceClassifier("CfC", x.shape[-1], 48, len(b.CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    seen, profiles, class_means, learned_acc, rows = [], {}, {}, {}, []
    projector = predictor = None
    signal_stats = None
    for task_i, new_classes in enumerate(b.TASKS, start=1):
        old = list(seen)
        seen.extend(new_classes)
        train_idx, val_idx = b.split_task_train_validation(y, pool, new_classes, 0.2, 20261004 + task_i*100)
        if task_i == 1:
            trained = train_cfctask(model, optimizer, x, y, train_idx, val_idx, seen, device,
                                    SEED + task_i*100_000)
            signal_cur = cbm.raw_concept_features(raw[train_idx])
            signal_stats = (signal_cur.mean(0), signal_cur.std(0)+1e-3)
        else:
            trained = 0  # CBM adaptation keeps the CfC feature encoder fixed after Task 1.
        if task_i > 1:
            for p in model.parameters():
                if p is not model.head.weight and p is not model.head.bias:
                    p.requires_grad_(False)

        for cls in new_classes:
            sig = cbm.raw_concept_features(raw[train_idx[y[train_idx] == cls]])
            profiles[cls] = (sig.mean(0), sig.std(0)+1e-3)

        # Exemplar-free CBM: fit the new projector from current real examples only.
        z_fit = make_joint(model, x, raw, train_idx, device, signal_stats)
        raw_fit = cbm.raw_concept_features(raw[train_idx])
        targets = cbm.concept_targets(raw_fit, seen, profiles)
        teacher = None
        if projector is not None:
            with torch.no_grad():
                teacher = torch.sigmoid(projector(torch.from_numpy(z_fit).to(device))).cpu().numpy()[:, :8*len(old)]
        cfg = dict(cbm.CONFIG)
        cfg["concept_projector_epochs"] = 100
        cfg["classifier_epochs"] = 300
        previous = copy.deepcopy(projector).eval() if projector is not None else None
        projector = fit_projector_cubic(z_fit, targets, teacher, 8*len(old), cfg, device,
                                       previous=previous if warmstart else None)
        current_concepts = torch.sigmoid(projector(torch.from_numpy(z_fit).to(device))).detach().cpu().numpy().astype(np.float32)
        feature_list = [current_concepts]
        joint_feature_list = [z_fit]
        label_list = [y[train_idx].copy()]
        pseudo_count = 0

        # Generate old-class pseudo-concepts by shifting the nearest current-class
        # joint-feature distribution toward each stored old-class centroid.
        current_means = {}
        for cls in new_classes:
            current_means[cls] = z_fit[y[train_idx] == cls].mean(0)
        for cls in old:
            old_mean = class_means[cls]
            nearest = max(new_classes, key=lambda c: float(
                np.dot(old_mean, current_means[c]) /
                (np.linalg.norm(old_mean)*np.linalg.norm(current_means[c]) + 1e-8)
            ))
            source = z_fit[y[train_idx] == nearest]
            pseudo_z = source - current_means[nearest] + old_mean
            joint_feature_list.append(pseudo_z)
            with torch.no_grad():
                pseudo_c = torch.sigmoid(projector(torch.from_numpy(pseudo_z).to(device))).cpu().numpy().astype(np.float32)
            feature_list.append(pseudo_c)
            label_list.append(np.full(len(pseudo_c), cls, dtype=np.int64))
            pseudo_count += len(pseudo_c)
        for cls in new_classes:
            class_means[cls] = current_means[cls]

        concepts = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        predictor = cbm.fit_sparse_classifier(concepts, labels, cfg, device)
        if joint_training and old:
            history = joint_refine(projector, predictor, previous, z_fit, targets,
                                   np.concatenate(joint_feature_list), labels, seen, 8*len(old), cfg, device,kd_on_pseudo,use_distillation,staged_training)
            (OUT / f'{condition}_task{task_i}_losses.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        elif head_only and old:
            cc=torch.from_numpy(concepts).to(device); yy=torch.from_numpy(cbm.seen_index(labels,seen)).long().to(device)
            opt=torch.optim.Adam(predictor.parameters(),lr=0.001); history=[]
            for epoch in range(20):
                opt.zero_grad(set_to_none=True)
                ce=nn.functional.cross_entropy(predictor(cc),yy)
                loss=ce+cfg['sparsity_lambda']*predictor.weight.abs().sum()
                loss.backward();opt.step()
                history.append({'epoch':epoch+1,'classification_loss':float(ce.detach())})
            (OUT/f'{condition}_task{task_i}_losses.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        torch.save({'encoder':model.state_dict(),'projector':projector.state_dict(),
                    'classifier':predictor.state_dict(),'seen':seen.copy(),'seed':SEED},
                   OUT / f'{condition}_task{task_i}.pt')

        per = {}
        predictor.eval(); projector.eval(); model.eval()
        with torch.no_grad():
            for cls in seen:
                ids = np.flatnonzero(yt == cls)
                preds=[]
                for start in range(0, len(ids), EVAL_BATCH):
                    selected = ids[start:start+EVAL_BATCH]
                    latent = extract_cfc_latent(model, xt, selected, device)
                    sig = cbm.raw_concept_features(raw_test[selected])
                    sm, ss = signal_stats
                    sig = np.clip((sig-sm)/(ss+1e-6), -6, 6).astype(np.float32)
                    joint = torch.from_numpy(np.concatenate([latent, sig], axis=1)).to(device)
                    cc = torch.sigmoid(projector(joint))
                    preds.extend(predictor(cc)[:, :len(seen)].argmax(1).cpu().numpy().tolist())
                per[cls] = float(np.mean(np.asarray(preds) == cbm.seen_index(yt[ids], seen)))
        for cls in new_classes:
            learned_acc[cls] = per[cls]
        (OUT / f'{condition}_task{task_i}_per_class.json').write_text(json.dumps(per, indent=2), encoding='utf-8')
        rows.append(make_row(condition, task_i, trained, old, new_classes,
                             seen, per, learned_acc, pseudo_count))
    return rows


def make_row(condition, task, epochs, old, new, seen, per_class, learned, pseudo_count=0):
    return {
        "condition": condition, "seed": SEED, "task": task, "encoder_epochs": epochs,
        "old_class_test_macro_accuracy": float(np.mean([per_class[c] for c in old])) if old else None,
        "new_class_test_macro_accuracy": float(np.mean([per_class[c] for c in new])),
        "all_seen_test_macro_accuracy": float(np.mean([per_class[c] for c in seen])),
        "mean_old_class_forgetting_pp": float(100*np.mean([learned[c]-per_class[c] for c in old])) if old else None,
        "pseudo_features": pseudo_count,
    }


def main():
    global SEED
    cfg = {"diagnostic_split_seed": 20261005, "diagnostic_fraction_per_class": 0.1}
    x, y, xt, yt = b.load_uci_har()
    raw, raw_test = load_raw_uci_har()
    pool, _ = b.split_probe_and_task_pool(y, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in SEEDS:
        SEED = seed
        print(f"CfC + CI-CBM core adaptation | seed={SEED} | device={device}", flush=True)
        rows.extend(fit_cbm_condition(x, y, xt, yt, raw, raw_test, pool, device))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/"task_boundary_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {
        "dataset": "UCI-HAR official subject-disjoint split", "seeds": SEEDS,
        "condition": "exemplar-free CfC temporal adaptation of CI-CBM core method",
        "model": "ncps CfC mode=default, units=48, backbone_units=128, backbone_layers=1",
        "task_order": [[b.CLASSES[c] for c in task] for task in b.TASKS],
        "short_run": {"max_encoder_epochs_per_task": MAX_EPOCHS, "early_stopping_patience": PATIENCE},
        "cbm_adaptation": {
            "concepts": "eight signal descriptors per class; Gaussian class-profile targets",
            "projector_input": "CfC hidden representation concatenated with training-standardized signal descriptors",
            "classifier": "sparse linear classifier over predicted concepts",
            "encoder": "trained on Task 1, then frozen for CBM tasks 2 and 3",
            "concept_alignment_and_distillation": "cubic normalized-cosine alignment and old-concept distillation as in CI-CBM",
            "pseudo_concept_generation": "nearest new-class joint-feature centroid translation to each stored old-class centroid",
        },
        "real_replay": False,
        "old_class_memory": "feature centroids and concept-projector outputs only; no old input windows stored",
        "test_used_for_training_or_stopping": False,
        "limitation": "Not a literal paper reproduction: project data are time series, so signal-derived concepts substitute for CLIP image-text alignment.",
        "interpretation": "Three-seed exploratory temporal adaptation; not the published image benchmark reproduction.",
    }
    (OUT/"run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    task3 = [r for r in rows if r["task"] == 3]
    lines=["# CfC + CI-CBM core method: three-seed temporal adaptation", "", "UCI-HAR; seeds 11, 22, 33; no old raw windows are stored or replayed. Includes class-specific concept expansion, cubic normalized-cosine concept alignment/distillation, centroid-shifted pseudo-features/pseudo-concepts, and an L1-sparse classifier. CfC is trained on Task 1 then frozen.", "", "This is not a literal paper reproduction: this project has time-series data, not the paper's image datasets or CLIP image-text concept scores. Interpretable sensor descriptors substitute for image-text alignments.", "", "## Task 3 test accuracy by seed", "", "| Seed | Old class macro | New class macro | All seen macro |", "|---:|---:|---:|---:|"]
    for r in task3:
        lines.append(f"| {r['seed']} | {100*r['old_class_test_macro_accuracy']:.2f}% | {100*r['new_class_test_macro_accuracy']:.2f}% | {100*r['all_seen_test_macro_accuracy']:.2f}% |")
    keys=["old_class_test_macro_accuracy","new_class_test_macro_accuracy","all_seen_test_macro_accuracy"]
    lines += ["", "## Mean ± sample SD across seeds", "", "| Old class macro | New class macro | All seen macro |", "|---:|---:|---:|"]
    lines.append("| " + " | ".join(f"{100*np.mean([r[k] for r in task3]):.2f} ± {100*np.std([r[k] for r in task3], ddof=1):.2f}%" for k in keys) + " |")
    baseline_path = OUT.parent / "cfc_replay_validation" / "task_boundary_metrics.csv"
    if baseline_path.exists():
        with baseline_path.open("r", newline="", encoding="utf-8-sig") as f:
            baseline = [r for r in csv.DictReader(f) if r["condition"] == "CfC_no_replay" and int(r["task"]) == 3]
        if baseline:
            lines += ["", "## Context versus the existing no-replay CfC run", "", "| Condition | Old class macro | New class macro | All seen macro |", "|---|---:|---:|---:|"]
            for label, group in [("CfC no-replay classifier baseline", baseline), ("CfC + CI-CBM adaptation", task3)]:
                means = [100*np.mean([float(r[k]) for r in group]) for k in keys]
                lines.append(f"| {label} | " + " | ".join(f"{v:.2f}%" for v in means) + " |")
            lines.append("")
            lines.append("This is a descriptive pipeline comparison, not an isolated CBM effect: the CBM condition freezes the CfC encoder after Task 1 and changes the readout/training objective, while the plain CfC baseline continues end-to-end updates.")
    lines += ["", "Test data are evaluated only at task boundaries and are never used for training or early stopping. Three seeds remain a small exploratory sample."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit('Use the source ablation runner.')
