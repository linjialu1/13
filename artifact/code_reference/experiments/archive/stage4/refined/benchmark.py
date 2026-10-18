"""CI-CBM-style incremental concept bottleneck for UCI-HAR and a liquid encoder."""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent
DATA = ROOT / "data" / "UCI HAR Dataset"
CONFIG = json.loads((HERE / "run_config.json").read_text(encoding="utf-8"))
CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
TASKS = [[0, 1], [2, 3], [4, 5]]
CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
ROWS: list[dict[str, Any]] = []
CONCEPT_ROWS: list[dict[str, Any]] = []
PSEUDO_ROWS: list[dict[str, Any]] = []
EXPLANATION_ROWS: list[dict[str, Any]] = []


class LiquidCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.logit_alpha = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.logit_alpha)
        h = x.new_zeros((x.shape[0], self.logit_alpha.numel()))
        u = self.input(x)
        for t in range(x.shape[1]):
            proposal = torch.tanh(u[:, t] + self.recurrent(h))
            h = h + alpha * (proposal - h)
        return h


class LNN(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.encoder = LiquidCell(9, hidden)
        self.head = nn.Linear(hidden, len(CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return h, self.head(h)


def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Expected extracted UCI-HAR data at {DATA}")
    xs, raw_xs, ys = [], [], []
    for split in ("train", "test"):
        folder = DATA / split / "Inertial Signals"
        signals = [np.loadtxt(folder / f"{name}_{split}.txt", dtype=np.float32) for name in CHANNELS]
        raw = np.stack(signals, axis=-1)
        raw_xs.append(raw)
        xs.append(raw.copy())
        ys.append(np.loadtxt(DATA / split / f"y_{split}.txt", dtype=np.int64) - 1)
    xtr, xte = xs
    rawtr, rawte = raw_xs
    ytr, yte = ys
    mean = xtr.mean((0, 1), keepdims=True)
    std = xtr.std((0, 1), keepdims=True) + 1e-6
    return ((xtr - mean) / std).astype(np.float32), ytr, ((xte - mean) / std).astype(np.float32), yte, rawtr.astype(np.float32), rawte.astype(np.float32)


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def raw_concept_features(x: np.ndarray) -> np.ndarray:
    """Eight physically grounded descriptors from unstandardized UCI-HAR windows."""
    eps = 1e-8
    acc = x[:, :, :3]
    gyro = x[:, :, 3:6]
    total_acc = x[:, :, 6:9]
    dyn_norm = np.sqrt(np.sum(acc ** 2, axis=2) + eps)
    acc_rms = np.sqrt(np.mean(dyn_norm ** 2, axis=1) + eps)
    gyro_rms = np.sqrt(np.mean(np.sum(gyro ** 2, axis=2), axis=1) + eps)
    axis_energy = np.mean(acc ** 2, axis=1)
    axis_share = axis_energy / (axis_energy.sum(axis=1, keepdims=True) + eps)
    centered = dyn_norm - dyn_norm.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    freq = np.fft.rfftfreq(x.shape[1], d=1.0 / 50.0)
    spectral_centroid = (spectrum * freq[None, :]).sum(axis=1) / (spectrum.sum(axis=1) + eps)
    ac = np.fft.irfft(spectrum, n=x.shape[1], axis=1)
    ac = ac / (ac[:, :1] + eps)
    periodicity = np.max(ac[:, 2:51], axis=1)
    gravity = total_acc - acc
    gravity_change = np.mean(np.abs(np.diff(gravity, axis=1)), axis=(1, 2))
    impulsiveness = np.max(dyn_norm, axis=1) / (acc_rms + eps)
    return np.stack([
        np.log(acc_rms + eps), np.log(gyro_rms + eps),
        axis_share[:, 0], axis_share[:, 1], axis_share[:, 2],
        spectral_centroid, periodicity, np.log(gravity_change + eps) + np.log(impulsiveness + eps),
    ], axis=1).astype(np.float32)


CONCEPT_NAMES = [
    "动态加速度均方根", "角速度均方根", "传感器x轴加速度能量占比", "传感器y轴加速度能量占比",
    "传感器z轴加速度能量占比", "加速度幅值频谱质心", "加速度幅值周期性", "姿态变化与冲击性",
]


def class_concepts(class_id: int, feature_names: list[str]) -> list[str]:
    name = CLASSES[class_id].lower().replace("_", " ")
    return [f"{name} signal pattern matching its {feature} profile" for feature in feature_names]


def fit_quantiles(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(z, 0.1, axis=0)
    hi = np.quantile(z, 0.9, axis=0)
    scale = np.std(z, axis=0) + 1e-6
    return (z.mean(0), scale)


def make_concept_targets(x: np.ndarray, train_reference: np.ndarray) -> tuple[np.ndarray, list[str]]:
    raw = raw_concept_features(x)
    ref = raw_concept_features(train_reference)
    mu, sd = fit_quantiles(ref)
    # Standardized continuous activations are the temporal substitute for image-text scores.
    target = np.clip((raw - mu) / sd, -3, 3).astype(np.float32)
    return target, CONCEPT_NAMES


def alignment_loss(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    def normalize(v):
        v = (v - v.mean(0, keepdim=True)) / (v.std(0, keepdim=True, unbiased=False) + 1e-5)
        v = v.clamp(-5, 5).pow(3)
        return v / (v.norm(dim=0, keepdim=True) + 1e-6)
    return -(normalize(q) * normalize(p)).sum(0).mean()


def concept_targets(raw_features: np.ndarray, seen: list[int],
                    profiles: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    targets=[]
    for cls in seen:
        mean,sd=profiles[cls]
        z=np.clip((raw_features-mean)/(sd+1e-6),-8,8)
        targets.append(np.exp(-0.5*z**2).astype(np.float32))
    return np.concatenate(targets,axis=1)


def evaluate_logits(model: LNN, x: np.ndarray, y: np.ndarray, seen: list[int], device) -> tuple[float, dict[int, float]]:
    model.eval(); per = {}
    with torch.no_grad():
        for c in seen:
            ix = np.flatnonzero(y == c)
            if not len(ix): continue
            pred = []
            for s in range(0, len(ix), 256):
                xx = torch.from_numpy(x[ix[s:s+256]]).to(device)
                pred.extend(model(xx)[1][:, seen].argmax(1).cpu().numpy().tolist())
            per[c] = float(np.mean(np.asarray(pred) == seen_index(y[ix], seen)))
    return float(np.mean(list(per.values()))) if per else float("nan"), per


def seen_index(y: np.ndarray, seen: list[int]) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(seen)}
    return np.asarray([lookup[int(v)] for v in y], dtype=np.int64)


def extract(model: LNN, x: np.ndarray, indices: np.ndarray, device) -> np.ndarray:
    model.eval(); out=[]
    with torch.no_grad():
        for s in range(0, len(indices), 256):
            xx = torch.from_numpy(x[indices[s:s+256]]).to(device)
            out.append(model(xx)[0].cpu().numpy())
    return np.concatenate(out, axis=0)


def extract_joint(model: LNN, x: np.ndarray, raw_x: np.ndarray, indices: np.ndarray, device,
                  signal_stats: tuple[np.ndarray,np.ndarray]) -> np.ndarray:
    latent=extract(model,x,indices,device)
    signal=raw_concept_features(raw_x[indices])
    mean,sd=signal_stats
    signal=np.clip((signal-mean)/(sd+1e-6),-6,6).astype(np.float32)
    return np.concatenate([latent,signal],axis=1).astype(np.float32)


def train_encoder_task(model: LNN, x: np.ndarray, y: np.ndarray, train_ix: np.ndarray,
                       val_ix: np.ndarray, seen: list[int], config: dict, device, seed: int, condition: str):
    for p in model.parameters(): p.requires_grad_(True)
    if len(seen) > len(TASKS[0]):
        for p in model.encoder.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"])
    best = -1.; best_state = copy.deepcopy(model.state_dict()); stale = 0; best_loss = float("inf")
    generator = torch.Generator(device="cpu").manual_seed(seed + len(seen)*999)
    ds = TensorDataset(torch.from_numpy(x[train_ix]), torch.from_numpy(y[train_ix]).long())
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=True, generator=generator)
    start_time = time.time(); used_epochs=0
    for epoch in range(config["max_epochs_per_task"]):
        model.train(); losses=[]
        for xx, yy in loader:
            xx=xx.to(device); yy=yy.to(device)
            local=torch.as_tensor(seen, device=device)
            mapping=torch.full((len(CLASSES),), -1, device=device, dtype=torch.long); mapping[local]=torch.arange(len(seen),device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(xx)[1][:,seen]
            loss=nn.functional.cross_entropy(logits, mapping[yy]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); val_losses=[]; correct=[]
        with torch.no_grad():
            for s in range(0,len(val_ix),256):
                ids=val_ix[s:s+256]; xx=torch.from_numpy(x[ids]).to(device); yy=torch.from_numpy(y[ids]).to(device)
                logits=model(xx)[1][:,seen]; target=torch.as_tensor(seen_index(y[ids],seen),device=device)
                val_losses.append(float(nn.functional.cross_entropy(logits,target)))
                correct.extend((logits.argmax(1)==target).cpu().numpy().tolist())
        va=float(np.mean(correct)); vl=float(np.mean(val_losses)); used_epochs=epoch+1
        ROWS.append({"condition":condition,"seed":seed,"task":len(seen)//2,"epoch":epoch+1,"train_loss":np.mean(losses),"val_accuracy":va,"val_loss":vl})
        if va > best + config["min_delta_macro_accuracy"] or (abs(va-best)<1e-12 and vl < best_loss-config["min_delta_validation_loss"]):
            best=va; best_loss=vl; best_state=copy.deepcopy(model.state_dict()); stale=0
        elif epoch+1 >= config["min_epochs_per_task"]: stale+=1
        if epoch+1 >= config["min_epochs_per_task"] and stale>=config["early_stopping_patience"]: break
    model.load_state_dict(best_state)
    return used_epochs, time.time()-start_time


def fit_projector(z: np.ndarray, p: np.ndarray, old_teacher: np.ndarray | None,
                  old_count: int, config: dict, device) -> nn.Linear:
    projector=nn.Linear(z.shape[1],p.shape[1],bias=False).to(device)
    opt=torch.optim.Adam(projector.parameters(),lr=config["concept_projector_lr"])
    zz=torch.from_numpy(z).to(device); pp=torch.from_numpy(p).to(device)
    tt=torch.from_numpy(old_teacher).to(device) if old_teacher is not None else None
    for _ in range(config["concept_projector_epochs"]):
        opt.zero_grad(set_to_none=True); q=torch.sigmoid(projector(zz))
        loss=nn.functional.mse_loss(q,pp)
        if tt is not None and old_count:
            loss=loss+config["concept_distillation_weight"]*nn.functional.mse_loss(q[:,:old_count],tt)
        loss.backward(); opt.step()
    return projector


def fit_sparse_classifier(c: np.ndarray, y: np.ndarray, config: dict, device) -> nn.Linear:
    nclass=int(y.max())+1
    layer=nn.Linear(c.shape[1],nclass).to(device)
    opt=torch.optim.Adam(layer.parameters(),lr=config["classifier_lr"])
    cc=torch.from_numpy(c).to(device); yy=torch.from_numpy(y).long().to(device)
    for _ in range(config["classifier_epochs"]):
        opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(layer(cc),yy)+config["sparsity_lambda"]*layer.weight.abs().sum()
        loss.backward(); opt.step()
    return layer


def concept_pipeline(model: LNN, x: np.ndarray, y: np.ndarray, train_ix: np.ndarray,
                     raw_x: np.ndarray,
                     seen: list[int], prev_concept_names: list[str], prev_projector: nn.Linear | None,
                     class_means: dict[int,np.ndarray], concept_profiles: dict[int,tuple[np.ndarray,np.ndarray]], config: dict, device, seed: int,
                     condition: str, signal_stats: tuple[np.ndarray,np.ndarray] | None):
    # Only train indices for current phase produce activations. Concept vocabulary is class-incremental.
    new_classes=TASKS[(len(seen)//2)-1]
    names=list(prev_concept_names)
    for cls in new_classes:
        names.extend(class_concepts(cls,CONCEPT_NAMES))
    concept_raw=raw_concept_features(raw_x[train_ix])
    if signal_stats is None:
        signal_stats=(concept_raw.mean(0),concept_raw.std(0)+1e-3)
    for cls in new_classes:
        raw=concept_raw[y[train_ix]==cls]
        concept_profiles[cls]=(raw.mean(0),raw.std(0)+1e-3)
    # Approximate signal-side concept activations for every current example and every seen
    # class-conditioned concept description, including old concepts on new-task examples.
    p=concept_targets(concept_raw,seen,concept_profiles)
    # For old concepts on new examples, cache teacher outputs before adapting the projector.
    teacher=None
    if prev_projector is not None:
        with torch.no_grad(): teacher=torch.sigmoid(prev_projector(torch.from_numpy(extract_joint(model,x,raw_x,train_ix,device,signal_stats)).to(device))).cpu().numpy()[:,:len(prev_concept_names)]
    z=extract_joint(model,x,raw_x,train_ix,device,signal_stats)
    projector=fit_projector(z,p,teacher,len(prev_concept_names),config,device)
    # Preserve old centroids from prior phases; only current classes have accessible raw examples.
    for cls in new_classes:
        current_ids=train_ix[y[train_ix]==cls]
        class_means[cls]=extract_joint(model,x,raw_x,current_ids,device,signal_stats).mean(0)
    # Training-only concept fidelity diagnostic for the newly fitted concept projector.
    q=torch.sigmoid(projector(torch.from_numpy(z).to(device))).detach().cpu().numpy()
    corr=[]
    for j in range(p.shape[1]):
        if np.std(p[:,j])>1e-8 and np.std(q[:,j])>1e-8:
            corr.append(float(np.corrcoef(p[:,j],q[:,j])[0,1]))
    CONCEPT_ROWS.append({"condition":condition,"seed":seed,"task":len(seen)//2,"diagnostic":"train_activation_pearson","concept":"ALL","mean_activation":float(np.mean(corr)) if corr else "","abs_classifier_weight_mean":""})
    return names,projector,class_means,concept_profiles,signal_stats


def get_concepts(model: LNN, projector: nn.Linear, x: np.ndarray, raw_x: np.ndarray,
                 indices: np.ndarray, device, signal_stats: tuple[np.ndarray,np.ndarray]) -> np.ndarray:
    z=extract_joint(model,x,raw_x,indices,device,signal_stats)
    with torch.no_grad(): return torch.sigmoid(projector(torch.from_numpy(z).to(device))).cpu().numpy()


def run_cbm(condition: str, seed: int, xtr: np.ndarray,ytr: np.ndarray,xte: np.ndarray,yte: np.ndarray,
            rawtr: np.ndarray,rawte: np.ndarray,config:dict,device):
    seed_all(seed); model=LNN(config["hidden_size"]).to(device)
    seen=[]; class_means={}; concept_profiles={}; concept_names=[]; projector=None; predictor=None; best_by_class={}; signal_stats=None
    # fixed per-task stratified split from official train only
    for ti,task in enumerate(TASKS):
        seen.extend(task); idx=np.flatnonzero(np.isin(ytr,task)); rng=np.random.default_rng(seed+ti*100)
        train_parts=[]; val_parts=[]
        for c in task:
            ci=idx[ytr[idx]==c].copy(); rng.shuffle(ci); nv=max(1,round(len(ci)*config["validation_fraction"]))
            val_parts.extend(ci[:nv]); train_parts.extend(ci[nv:])
        train_ix=np.asarray(train_parts); val_ix=np.asarray(val_parts)
        epochs,elapsed=train_encoder_task(model,xtr,ytr,train_ix,np.asarray(val_ix),seen,config,device,seed,condition)
        # freeze encoder after first task, preserving feature coordinate system for centroid translation
        if condition!="LNN_SGD":
            for p in model.encoder.parameters(): p.requires_grad_(False)
            names,projector,class_means,concept_profiles,signal_stats=concept_pipeline(model,xtr,ytr,train_ix,rawtr,seen,concept_names,projector,class_means,concept_profiles,config,device,seed,condition,signal_stats)
            concept_names=names
            new_c=get_concepts(model,projector,xtr,rawtr,train_ix,device,signal_stats)
            y_seen=ytr[train_ix]
            feature_list=[new_c]; label_list=[y_seen]
            pseudo_n=0
            if ti>0 and condition=="CI_CBM_real_plus_pseudo":
                z_new=extract_joint(model,xtr,rawtr,train_ix,device,signal_stats)
                for old in seen[:-len(task)]:
                    new_cls=max(task,key=lambda c: float(np.dot(class_means[old],class_means[c])/(np.linalg.norm(class_means[old])*np.linalg.norm(class_means[c])+1e-8)))
                    src=z_new[y_seen==new_cls]
                    rng=np.random.default_rng(seed+ti*1000+old)
                    if config["pseudo_samples_per_old_class"] > 0 and len(src)>config["pseudo_samples_per_old_class"]:
                        src=src[rng.choice(len(src),config["pseudo_samples_per_old_class"],replace=False)]
                    pseudo=src-class_means[new_cls]+class_means[old]
                    with torch.no_grad(): pc=projector(torch.from_numpy(pseudo).float().to(device)).cpu().numpy()
                    feature_list.append(pc); label_list.append(np.full(len(pc),old,np.int64)); pseudo_n+=len(pc)
                    old_test_ids=np.flatnonzero(yte==old)
                    real_old_c=get_concepts(model,projector,xte,rawte,old_test_ids,device,signal_stats)
                    pseudo_cos=float(np.dot(pc.mean(0),real_old_c.mean(0))/(np.linalg.norm(pc.mean(0))*np.linalg.norm(real_old_c.mean(0))+1e-8))
                    pseudo_rmse=float(np.sqrt(np.mean((pc.mean(0)-real_old_c.mean(0))**2)))
                    PSEUDO_ROWS.append({"seed":seed,"task":ti+1,"old_class":CLASSES[old],"nearest_new_class":CLASSES[new_cls],"source_count":int(np.sum(y_seen==new_cls)),"pseudo_count":len(pc),"centroid_cosine":float(np.dot(class_means[old],class_means[new_cls])/(np.linalg.norm(class_means[old])*np.linalg.norm(class_means[new_cls])+1e-8)),"pseudo_vs_real_old_concept_mean_cosine":pseudo_cos,"pseudo_vs_real_old_concept_mean_rmse":pseudo_rmse})
            train_c=np.concatenate(feature_list); train_y=np.concatenate(label_list)
            predictor=fit_sparse_classifier(train_c,train_y,config,device)
            with torch.no_grad():
                fit_pred=predictor(torch.from_numpy(train_c).to(device))[:,:len(seen)].argmax(1).cpu().numpy()
            fit_per_class={int(c):float(np.mean(fit_pred[train_y==c]==c)) for c in seen if np.any(train_y==c)}
            # Evaluate learned real concept activations and sparse rule on current held-out validation.
            val_c=get_concepts(model,projector,xtr,rawtr,val_ix,device,signal_stats)
            with torch.no_grad(): val_pred=predictor(torch.from_numpy(val_c).to(device))[:,:len(seen)].argmax(1).cpu().numpy()
            val_target=seen_index(ytr[val_ix],seen)
            # Store per-concept learning diagnostics: real targets are deterministic descriptors; scores predict activation.
            for j,n in enumerate(concept_names):
                CONCEPT_ROWS.append({"condition":condition,"seed":seed,"task":ti+1,"diagnostic":"concept_summary","concept":n,"mean_activation":float(np.mean(new_c[:,j])) if j<new_c.shape[1] else "","abs_classifier_weight_mean":float(np.mean(np.abs(predictor.weight.detach().cpu().numpy()[:,j]))) if j<predictor.weight.shape[1] else ""})
        else:
            pseudo_n=0
        # Task-boundary official test evaluation only.
        if condition=="LNN_SGD":
            model.eval(); correct={}
            with torch.no_grad():
                for c in seen:
                    ids=np.flatnonzero(yte==c); logits=[]
                    for s in range(0,len(ids),256): logits.append(model(torch.from_numpy(xte[ids[s:s+256]]).to(device))[1][:,seen].cpu())
                    pred=torch.cat(logits).argmax(1).numpy(); correct[c]=float(np.mean(pred==seen_index(yte[ids],seen)))
        else:
            model.eval(); correct={}
            with torch.no_grad():
                for c in seen:
                    ids=np.flatnonzero(yte==c); cv=get_concepts(model,projector,xte,rawte,ids,device,signal_stats)
                    pred=predictor(torch.from_numpy(cv).to(device))[:,:len(seen)].argmax(1).cpu().numpy()
                    correct[c]=float(np.mean(pred==seen_index(yte[ids],seen)))
            # Independent held-out concept fidelity and local concept-ablation faithfulness.
            test_ids=np.flatnonzero(np.isin(yte,seen))
            test_c=get_concepts(model,projector,xte,rawte,test_ids,device,signal_stats)
            test_target=concept_targets(raw_concept_features(rawte[test_ids]),seen,concept_profiles)
            dim_corr=[float(np.corrcoef(test_c[:,j],test_target[:,j])[0,1])
                      for j in range(test_c.shape[1])
                      if np.std(test_c[:,j])>1e-8 and np.std(test_target[:,j])>1e-8]
            test_rmse=float(np.sqrt(np.mean((test_c-test_target)**2)))
            with torch.no_grad():
                cc=torch.from_numpy(test_c).to(device)
                logits=predictor(cc)[:,:len(seen)]
                pred=logits.argmax(1)
                probs=torch.softmax(logits,dim=1)
                rows=torch.arange(len(test_ids),device=device)
                contribution=cc*predictor.weight[:len(seen)][pred]
                top_idx=contribution.abs().argmax(1)
                rng_exp=torch.Generator(device="cpu").manual_seed(seed+ti*271)
                random_idx=torch.randint(cc.shape[1],(len(test_ids),),generator=rng_exp).to(device)
                cc_top=cc.clone(); cc_rand=cc.clone()
                cc_top[rows,top_idx]=0.5; cc_rand[rows,random_idx]=0.5
                p_top=torch.softmax(predictor(cc_top)[:,:len(seen)],dim=1)
                p_rand=torch.softmax(predictor(cc_rand)[:,:len(seen)],dim=1)
                top_conf_drop=float((probs.max(1).values-p_top[rows,pred]).mean().cpu())
                random_conf_drop=float((probs.max(1).values-p_rand[rows,pred]).mean().cpu())
                top_flip=float((p_top.argmax(1)!=pred).float().mean().cpu())
                random_flip=float((p_rand.argmax(1)!=pred).float().mean().cpu())
            EXPLANATION_ROWS.append({"condition":condition,"seed":seed,"task":ti+1,
                "test_concept_pearson_mean":float(np.mean(dim_corr)) if dim_corr else "",
                "test_concept_rmse":test_rmse,"top_concept_confidence_drop":top_conf_drop,
                "random_concept_confidence_drop":random_conf_drop,"top_concept_prediction_flip_rate":top_flip,
                "random_concept_prediction_flip_rate":random_flip})
        for c,v in correct.items(): best_by_class[c]=max(best_by_class.get(c,0.),v)
        old=seen[:-len(task)]
        forget=float(np.mean([best_by_class[c]-correct[c] for c in old])) if old else 0.
        ROWS.append({"condition":condition,"seed":seed,"task":ti+1,"epochs":epochs,"seconds":elapsed,"overall_macro_accuracy":float(np.mean(list(correct.values()))),"new_class_macro_accuracy":float(np.mean([correct[c] for c in task])),"old_class_macro_accuracy":float(np.mean([correct[c] for c in old])) if old else "","forgetting":forget,"train_balanced_accuracy":float(np.mean(list(fit_per_class.values()))) if condition!="LNN_SGD" else "","train_per_class_accuracy_json":json.dumps({CLASSES[c]:v for c,v in fit_per_class.items()},ensure_ascii=False) if condition!="LNN_SGD" else "","seen_class_accuracy_json":json.dumps({CLASSES[c]:v for c,v in correct.items()},ensure_ascii=False),"concept_count":len(concept_names),"pseudo_feature_count":pseudo_n})
        print(f"{condition} seed={seed} task={ti+1} acc={np.mean(list(correct.values())):.3f} forget={forget:.3f} concepts={len(concept_names)} pseudo={pseudo_n}",flush=True)


def write_csv(name: str, rows: list[dict[str,Any]]):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/name).open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def main():
    xtr,ytr,xte,yte,rawtr,rawte=load_data(); device=torch.device("cuda" if torch.cuda.is_available() and CONFIG["device"]=="cuda_if_available" else "cpu")
    print(f"UCI-HAR train={len(ytr)} test={len(yte)} device={device}",flush=True)
    for seed in CONFIG["seeds_to_run"]:
        for condition in CONFIG["conditions_to_run"]:
            run_cbm(condition,int(seed),xtr,ytr,xte,yte,rawtr,rawte,CONFIG,device)
            write_csv("task_metrics.csv",ROWS); write_csv("epoch_validation.csv",[r for r in ROWS if "epoch" in r]); write_csv("concept_diagnostics.csv",CONCEPT_ROWS); write_csv("pseudo_feature_diagnostics.csv",PSEUDO_ROWS); write_csv("explanation_diagnostics.csv",EXPLANATION_ROWS)
    metadata={"dataset":"UCI-HAR","device":str(device),"config":CONFIG,"task_order":[[CLASSES[c] for c in t] for t in TASKS],"official_test_used_for_training":False,"encoder_frozen_after_task1_for_cbm":True,"concept_method":"class-conditioned Gaussian matches over eight descriptors computed from unstandardized training signals; no external text-image model","concept_metrics":"held-out test Pearson correlation and RMSE against signal-derived targets; evaluation-only","explanation_metrics":"top absolute class-concept contribution versus random concept ablation; evaluation-only","pseudo_concept_method":"CI-CBM centroid-translated pseudo-features projected through current concept layer"}
    (HERE/"run_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    # Compact task-boundary accuracy plot.
    fig,ax=plt.subplots(figsize=(8,5))
    for cond in CONFIG["conditions_to_run"]:
        vals=[]
        for task in (1,2,3):
            rows=[r for r in ROWS if r.get("condition")==cond and r.get("task")==task and r.get("seed") in CONFIG["seeds_to_run"] and "overall_macro_accuracy" in r]
            vals.append(100*np.mean([r["overall_macro_accuracy"] for r in rows]) if rows else np.nan)
        ax.plot([1,2,3],vals,marker="o",label=cond)
    ax.set(xticks=[1,2,3],xlabel="Task boundary",ylabel="Seen-class macro accuracy (%)",title="Stage 4: CI-CBM-style LNN concept bottleneck")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(HERE/"task_boundary_accuracy.png",dpi=160); plt.close(fig)
    report=["# Stage 4 initial run", "",f"- Device: {device}",f"- Seeds: {CONFIG['seeds_to_run']}","- Test data did not participate in fitting or stopping.","- CI-CBM was adapted to computed temporal concepts; this is not a literal SigLIP reproduction.","", "## Task-boundary macro accuracy and forgetting", "", "| Condition | Task | Seen-class macro accuracy | New-class macro accuracy | Old-class forgetting | Pseudo features |", "|---|---:|---:|---:|---:|---:|"]
    for r in ROWS:
        if "overall_macro_accuracy" in r:
            report.append(f"| {r['condition']} | {r['task']} | {100*r['overall_macro_accuracy']:.2f}% | {100*r['new_class_macro_accuracy']:.2f}% | {100*r['forgetting']:.2f} pp | {r['pseudo_feature_count']} |")
    report += ["", "## Interpretation boundary", "", "This is a one-seed pipeline smoke run, not confirmatory evidence. The pseudo-feature condition tests CI-CBM-style centroid translation in a frozen LNN representation space. Signal descriptors replace image-text activations and are a temporal adaptation.", "", "See `task_metrics.csv`, `epoch_validation.csv`, `concept_diagnostics.csv`, `pseudo_feature_diagnostics.csv`, and `run_metadata.json`."]
    (HERE/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"Finished. Review {HERE/'REPORT.md'}",flush=True)


if __name__=="__main__": main()
