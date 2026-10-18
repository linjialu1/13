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


# Lightweight scalar temporal attention over all CfC sequence states.
# The zero initialization starts as uniform temporal pooling and lets Task 1
# learn which time steps matter. Existing task-boundary logic freezes it
# together with the CfC encoder after Task 1.
_BaseSequenceClassifier = b.SequenceClassifier


class TemporalAttentionSequenceClassifier(_BaseSequenceClassifier):
    def __init__(self, kind, input_size, hidden_size, n_classes):
        super().__init__(kind, input_size, hidden_size, n_classes)
        self.temporal_attention = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.temporal_attention.weight)

    def representation_and_logits(self, x):
        if self.kind != "CfC":
            return super().representation_and_logits(x)
        sequence, _ = self.encoder(x)
        scores = self.temporal_attention(torch.tanh(sequence)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        representation = torch.sum(weights.unsqueeze(-1) * sequence, dim=1)
        return representation, self.head(representation)


class CfCGateChangePoolSequenceClassifier(_BaseSequenceClassifier):
    """Pool CfC states using changes in the cell's actual interpolation gate."""
    gate_temperature = 0.5

    def representation_and_logits(self, x):
        if self.kind != "CfC":
            return super().representation_and_logits(x)
        sequence, _ = self.encoder(x)
        cell = self.encoder.rnn_cell
        previous = torch.cat([torch.zeros_like(sequence[:, :1]), sequence[:, :-1]], dim=1)
        joint = torch.cat([x, previous], dim=-1).reshape(-1, x.shape[-1] + sequence.shape[-1])
        if cell.backbone_layers > 0:
            joint = cell.backbone(joint)
        gate = torch.sigmoid(cell.time_a(joint) + cell.time_b(joint))
        gate = gate.reshape(sequence.shape[0], sequence.shape[1], sequence.shape[2])
        gate_change = torch.sqrt(torch.mean((gate[:, 1:] - gate[:, :-1]) ** 2, dim=-1) + 1e-12)
        gate_change = torch.cat([gate_change[:, :1], gate_change], dim=1)
        scores = torch.log(gate_change + 1e-6) / self.gate_temperature
        weights = torch.softmax(scores, dim=1)
        representation = torch.sum(weights.unsqueeze(-1) * sequence, dim=1)
        return representation, self.head(representation)


class CfCLiquidAttentionSequenceClassifier(_BaseSequenceClassifier):
    """Blend learned temporal attention with the CfC cell's own gate dynamics.

    The bounded gate-prior coefficient starts at 0.1, so the learned attention
    remains the main readout while CfC interpolation-gate changes provide a
    small, trainable temporal prior.
    """

    max_gate_prior_weight = 0.5

    def __init__(self, kind, input_size, hidden_size, n_classes):
        super().__init__(kind, input_size, hidden_size, n_classes)
        self.temporal_attention = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.temporal_attention.weight)
        # 0.5 * sigmoid(logit) = 0.1 initially.
        self.gate_prior_logit = nn.Parameter(torch.tensor(-1.38629436))

    def gate_prior_weight(self):
        return self.max_gate_prior_weight * torch.sigmoid(self.gate_prior_logit)

    def representation_and_logits(self, x):
        if self.kind != "CfC":
            return super().representation_and_logits(x)
        sequence, _ = self.encoder(x)
        attention_scores = self.temporal_attention(torch.tanh(sequence)).squeeze(-1)

        cell = self.encoder.rnn_cell
        previous = torch.cat([torch.zeros_like(sequence[:, :1]), sequence[:, :-1]], dim=1)
        joint = torch.cat([x, previous], dim=-1).reshape(-1, x.shape[-1] + sequence.shape[-1])
        if cell.backbone_layers > 0:
            joint = cell.backbone(joint)
        # CfC defaults to unit time spans in this benchmark (timespans=None).
        interpolation_gate = torch.sigmoid(cell.time_a(joint) + cell.time_b(joint))
        interpolation_gate = interpolation_gate.reshape_as(sequence)
        change = torch.sqrt(torch.mean(
            (interpolation_gate[:, 1:] - interpolation_gate[:, :-1]) ** 2,
            dim=-1) + 1e-12)
        change = torch.cat([change[:, :1], change], dim=1)
        gate_scores = torch.log(change + 1e-6)
        gate_scores = (gate_scores - gate_scores.mean(dim=1, keepdim=True)) / (
            gate_scores.std(dim=1, keepdim=True, unbiased=False) + 1e-5)
        gate_scores = gate_scores.clamp(-3.0, 3.0)
        scores = attention_scores + self.gate_prior_weight() * gate_scores
        weights = torch.softmax(scores, dim=1)
        representation = torch.sum(weights.unsqueeze(-1) * sequence, dim=1)
        return representation, self.head(representation)


class CfCNeuronwiseLiquidAttentionSequenceClassifier(_BaseSequenceClassifier):
    """Fuse learned attention with per-neuron CfC gate-change pooling.

    Each hidden neuron keeps its own temporal weights, so neurons with different
    learned CfC time scales can emphasize different portions of a sequence.
    """

    max_gate_prior_weight = 0.5
    gate_temperature = 0.5

    def __init__(self, kind, input_size, hidden_size, n_classes):
        super().__init__(kind, input_size, hidden_size, n_classes)
        self.temporal_attention = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.temporal_attention.weight)
        self.gate_prior_logit = nn.Parameter(torch.tensor(-1.38629436))

    def gate_prior_weight(self):
        return self.max_gate_prior_weight * torch.sigmoid(self.gate_prior_logit)

    def representation_and_logits(self, x):
        if self.kind != "CfC":
            return super().representation_and_logits(x)
        sequence, _ = self.encoder(x)
        attention_scores = self.temporal_attention(torch.tanh(sequence)).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_representation = torch.sum(attention_weights.unsqueeze(-1) * sequence, dim=1)

        cell = self.encoder.rnn_cell
        previous = torch.cat([torch.zeros_like(sequence[:, :1]), sequence[:, :-1]], dim=1)
        joint = torch.cat([x, previous], dim=-1).reshape(-1, x.shape[-1] + sequence.shape[-1])
        if cell.backbone_layers > 0:
            joint = cell.backbone(joint)
        interpolation_gate = torch.sigmoid(cell.time_a(joint) + cell.time_b(joint))
        interpolation_gate = interpolation_gate.reshape_as(sequence)
        change = torch.sqrt((interpolation_gate[:, 1:] - interpolation_gate[:, :-1]).square() + 1e-12)
        change = torch.cat([change[:, :1], change], dim=1)
        gate_scores = torch.log(change + 1e-6) / self.gate_temperature
        gate_weights = torch.softmax(gate_scores, dim=1)
        liquid_representation = torch.sum(gate_weights * sequence, dim=1)

        beta = self.gate_prior_weight()
        representation = (1.0 - beta) * attention_representation + beta * liquid_representation
        return representation, self.head(representation)


b.SequenceClassifier = TemporalAttentionSequenceClassifier

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
TRAJECTORY_MODE_COUNT = 2
TRAJECTORY_PROTOTYPE_BINS = 16


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


def temporal_signature(sequence):
    anchors = torch.linspace(0, sequence.shape[1] - 1, 8, device=sequence.device).long()
    sampled = sequence[:, anchors, :]
    delta = sampled[:, 1:, :] - sampled[:, :-1, :]
    return torch.cat([sampled.flatten(1), delta.flatten(1)], dim=1)


@torch.no_grad()
def make_temporal_relation_target(model, x, memory_idx, device):
    xx = torch.from_numpy(x[memory_idx]).to(device)
    model.eval(); sequence, _ = model.encoder(xx)
    signature = temporal_signature(sequence)
    distances = torch.pdist(signature, p=2)
    scale = distances.median().clamp_min(1e-6)
    return {"values": (distances / scale).detach(), "scale": scale.detach()}


def temporal_relation_loss(model, x, memory_idx, target, device):
    xx = torch.from_numpy(x[memory_idx]).to(device)
    sequence, _ = model.encoder(xx)
    signature = temporal_signature(sequence)
    distances = torch.pdist(signature, p=2) / target["scale"]
    return torch.mean((distances - target["values"]) ** 2)


def forward_with_cfc_gate_sequence(model, batch):
    """Return classifier logits and the CfC cell's per-time interpolation gates."""
    gates = []
    cell = model.encoder.rnn_cell
    handle = cell.sigmoid.register_forward_hook(
        lambda _module, _inputs, output: gates.append(output))
    try:
        logits = model(batch)
    finally:
        handle.remove()
    if not gates:
        raise RuntimeError("CfC interpolation gate hook did not capture any timesteps")
    return logits, torch.stack(gates, dim=1)


def train_cfc_with_gate_replay(model, optimizer, x, y, train_idx, val_idx,
                               memory_idx, seen, device, seed, gate_weight=10.0):
    """Replay old windows and distill their teacher CfC gate trajectories."""
    old_memory_idx = np.concatenate([memory_idx[c] for c in sorted(memory_idx)])
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    best_key, best_state, best_opt = (-1.0, -float("inf")), None, None
    best_acc, best_loss, stagnant, trained = -1.0, float("inf"), 0, 0
    gate_values = []
    for epoch in range(1, MAX_EPOCHS + 1):
        loader = b.make_loader(x, y, train_idx, CURRENT_BATCH, seed + epoch)
        rng = np.random.default_rng(seed + epoch * 13)
        model.train(); gate_values = []
        for xx, yy in loader:
            xx, yy = xx.to(device), yy.to(device)
            n_replay = min(REPLAY_BATCH, len(old_memory_idx))
            replay_idx = rng.choice(old_memory_idx, size=n_replay,
                                    replace=len(old_memory_idx) < n_replay)
            replay_x = torch.from_numpy(x[replay_idx]).to(device)
            replay_y = torch.from_numpy(y[replay_idx]).long().to(device)
            batch_x = torch.cat([xx, replay_x], dim=0)
            batch_y = torch.cat([yy, replay_y], dim=0)
            optimizer.zero_grad(set_to_none=True)
            logits, student_gates = forward_with_cfc_gate_sequence(model, batch_x)
            ce = nn.functional.cross_entropy(logits[:, seen],
                                             b.labels_to_local(batch_y, seen, device))
            with torch.no_grad():
                _, teacher_gates = forward_with_cfc_gate_sequence(teacher, replay_x)
            student_replay_gates = student_gates[len(xx):]
            gate_loss = nn.functional.mse_loss(student_replay_gates, teacher_gates)
            loss = ce + gate_weight * gate_loss
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            gate_values.append(float(gate_loss.detach()))
        va, vl = b.evaluate_validation(model, x, y, val_idx, seen[-2:], device, EVAL_BATCH)
        trained = epoch; key = (va, -vl)
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = copy.deepcopy(optimizer.state_dict())
        if va - best_acc >= 0.005 or best_loss - vl >= 0.001:
            best_acc, best_loss, stagnant = max(best_acc, va), min(best_loss, vl), 0
        else:
            stagnant += 1
        print(f"CfC gate-distill task={len(seen)//2} epoch={epoch}/{MAX_EPOCHS} "
              f"val={va:.3f} gate_mse={np.mean(gate_values):.5f}", flush=True)
        if epoch >= MIN_EPOCHS and stagnant >= PATIENCE:
            break
    model.load_state_dict(best_state); optimizer.load_state_dict(best_opt)
    return trained, float(np.mean(gate_values))


def train_cfc_with_relation_replay(model, optimizer, x, y, train_idx, val_idx,
                                   memory_idx, seen, device, seed,
                                   relation_weight=0.5):
    """Update CfC on current+real replay while preserving old memory relations."""
    old_memory_idx = np.concatenate([memory_idx[c] for c in sorted(memory_idx)])
    relation_target = make_temporal_relation_target(model, x, old_memory_idx, device)
    best_key, best_state, best_opt = (-1.0, -float("inf")), None, None
    best_acc, best_loss, stagnant, trained = -1.0, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        loader = b.make_loader(x, y, train_idx, CURRENT_BATCH, seed + epoch)
        rng = np.random.default_rng(seed + epoch * 13)
        model.train(); rel_values = []
        for xx, yy in loader:
            xx, yy = xx.to(device), yy.to(device)
            n_replay = min(REPLAY_BATCH, len(old_memory_idx))
            replay_idx = rng.choice(old_memory_idx, size=n_replay,
                                    replace=len(old_memory_idx) < n_replay)
            replay_x = torch.from_numpy(x[replay_idx]).to(device)
            replay_y = torch.from_numpy(y[replay_idx]).long().to(device)
            batch_x = torch.cat([xx, replay_x], dim=0)
            batch_y = torch.cat([yy, replay_y], dim=0)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)[:, seen]
            ce = nn.functional.cross_entropy(logits, b.labels_to_local(batch_y, seen, device))
            relation = temporal_relation_loss(model, x, old_memory_idx, relation_target, device)
            loss = ce + relation_weight * relation
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            rel_values.append(float(relation.detach()))
        va, vl = b.evaluate_validation(model, x, y, val_idx, seen[-2:], device, EVAL_BATCH)
        trained = epoch; key = (va, -vl)
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = copy.deepcopy(optimizer.state_dict())
        if va - best_acc >= 0.005 or best_loss - vl >= 0.001:
            best_acc, best_loss, stagnant = max(best_acc, va), min(best_loss, vl), 0
        else:
            stagnant += 1
        print(f"CfC relational task={len(seen)//2} epoch={epoch}/{MAX_EPOCHS} val={va:.3f} rel={np.mean(rel_values):.4f}", flush=True)
        if epoch >= MIN_EPOCHS and stagnant >= PATIENCE:
            break
    model.load_state_dict(best_state); optimizer.load_state_dict(best_opt)
    return trained, float(np.mean(rel_values))


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


def extract_cfc_sequence(model, x, indices, device):
    """Return per-time-step CfC states for trajectory-aware prototype shifts."""
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0, len(indices), EVAL_BATCH):
            xx = torch.from_numpy(x[indices[start:start+EVAL_BATCH]]).to(device)
            sequence, _ = model.encoder(xx)
            out.append(sequence.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def pool_cfc_sequence(model, sequence, device):
    """Apply the model's current temporal readout to a NumPy state sequence."""
    states = torch.from_numpy(np.asarray(sequence, dtype=np.float32)).to(device)
    with torch.no_grad():
        if hasattr(model, "temporal_attention"):
            scores = model.temporal_attention(torch.tanh(states)).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(weights.unsqueeze(-1) * states, dim=1)
        else:
            pooled = states.mean(dim=1)
    return pooled.cpu().numpy().astype(np.float32)


def compress_trajectory(sequence, bins=TRAJECTORY_PROTOTYPE_BINS):
    """Average-pool [N,T,H] states to a compact fixed-length trajectory."""
    sequence = np.asarray(sequence, dtype=np.float32)
    n, steps, hidden = sequence.shape
    edges = np.linspace(0, steps, bins + 1, dtype=np.int64)
    return np.stack([sequence[:, edges[i]:edges[i+1]].mean(axis=1)
                     for i in range(bins)], axis=1).reshape(n, bins, hidden)


def expand_trajectory(prototype, steps):
    """Linearly restore a compressed [B,H] prototype to [T,H]."""
    prototype = np.asarray(prototype, dtype=np.float32)
    old_t = np.linspace(0.0, 1.0, len(prototype))
    new_t = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(new_t, old_t, prototype[:, h])
                     for h in range(prototype.shape[1])], axis=1).astype(np.float32)


def cluster_trajectory_modes(sequence, seed, modes=TRAJECTORY_MODE_COUNT):
    """Small deterministic k-means over compressed full trajectories."""
    compressed = compress_trajectory(sequence)
    features = compressed.reshape(len(compressed), -1)
    k = min(modes, len(features))
    rng = np.random.default_rng(seed)
    centers = [features[int(rng.integers(len(features)))]]
    while len(centers) < k:
        distance = np.min(((features[:, None] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(features[int(np.argmax(distance))])
    centers = np.asarray(centers, dtype=np.float32)
    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(20):
        distance = ((features[:, None] - centers[None]) ** 2).sum(-1)
        new_labels = distance.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _:
            break
        labels = new_labels
        for mode in range(k):
            selected = features[labels == mode]
            if len(selected):
                centers[mode] = selected.mean(axis=0)
    proto = np.stack([compressed[labels == mode].mean(axis=0) for mode in range(k)])
    return labels, proto.astype(np.float16)


def pooled_trajectory_cosine(model, a, b, device, steps):
    pa = pool_cfc_sequence(model, expand_trajectory(a, steps)[None], device)[0]
    pb = pool_cfc_sequence(model, expand_trajectory(b, steps)[None], device)[0]
    return float(np.dot(pa, pb) / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-8))


def make_joint(model, x, raw, indices, device, signal_stats):
    latent = extract_cfc_latent(model, x, indices, device)
    signal = cbm.raw_concept_features(raw[indices])
    mean, sd = signal_stats
    signal = np.clip((signal-mean)/(sd+1e-6), -6, 6).astype(np.float32)
    return np.concatenate([latent, signal], axis=1).astype(np.float32)


def fit_pooled_prototypes(features, count, seed):
    """Deterministic small k-means summary of one class in pooled joint space."""
    features = np.asarray(features, dtype=np.float32)
    count = min(max(1, int(count)), len(features))
    if count == 1:
        return features.mean(0, keepdims=True)
    rng = np.random.default_rng(seed)
    centers = [features[int(rng.integers(len(features)))]]
    for _ in range(1, count):
        d2 = np.min(np.stack([((features-c)**2).sum(1) for c in centers]), axis=0)
        probs = d2 / max(float(d2.sum()), 1e-12)
        centers.append(features[int(rng.choice(len(features), p=probs))])
    centers = np.stack(centers).astype(np.float32)
    for _ in range(30):
        distances = ((features[:, None, :] - centers[None, :, :])**2).sum(2)
        labels = distances.argmin(1)
        updated = centers.copy()
        for k in range(count):
            if np.any(labels == k):
                updated[k] = features[labels == k].mean(0)
        if np.allclose(updated, centers, atol=1e-5):
            break
        centers = updated
    return centers.astype(np.float32)


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


def estimate_attention_fisher(model, x, y, indices, classes, device, seed,
                              samples_per_class=64):
    """Estimate diagonal Fisher for the temporal-attention vector on one task."""
    rng = np.random.default_rng(seed)
    selected = []
    for cls in classes:
        options = indices[y[indices] == cls]
        count = min(samples_per_class, len(options))
        selected.extend(rng.choice(options, size=count, replace=False).tolist())
    theta = model.temporal_attention.weight
    fisher = torch.zeros_like(theta, memory_format=torch.preserve_format)
    class_to_local = {int(cls): i for i, cls in enumerate(classes)}
    model.eval()
    for idx in selected:
        model.zero_grad(set_to_none=True)
        xx = torch.from_numpy(x[idx:idx+1]).to(device)
        logits = model(xx)[:, classes]
        target = torch.tensor([class_to_local[int(y[idx])]], device=device)
        loss = nn.functional.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, theta, retain_graph=False)[0]
        fisher.add_(grad.detach().square())
    if selected:
        fisher.div_(len(selected))
    return fisher.detach()


def adapt_temporal_attention(model, x, y, train_idx, val_idx, new_classes,
                             device, seed, anchor=None, fisher=None,
                             fisher_weight=0.0, attention_distill_weight=0.0,
                             epochs=12):
    """Update only attention and current-class readout rows on the new task."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.temporal_attention.weight.requires_grad_(True)
    model.head.weight.requires_grad_(True)
    if model.head.bias is not None:
        model.head.bias.requires_grad_(True)
    trainable = [model.temporal_attention.weight, model.head.weight]
    if model.head.bias is not None:
        trainable.append(model.head.bias)
    optimizer = torch.optim.Adam(trainable, lr=LR)
    class_to_local = {int(cls): i for i, cls in enumerate(new_classes)}
    best_key, best_state, best_acc, best_loss, stagnant = (-1.0, -float("inf")), None, -1.0, float("inf"), 0
    theta_anchor = anchor.to(device) if anchor is not None else None
    fisher_weight_tensor = None
    if fisher is not None:
        fisher_weight_tensor = fisher.to(device)
        fisher_weight_tensor = fisher_weight_tensor / fisher_weight_tensor.mean().clamp_min(1e-12)
    for epoch in range(1, epochs + 1):
        model.train()
        loader = b.make_loader(x, y, train_idx, 128, seed + epoch)
        for xx, yy in loader:
            xx, yy = xx.to(device), yy.to(device)
            local = torch.tensor([class_to_local[int(v)] for v in yy.detach().cpu().tolist()], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx)[:, new_classes]
            loss = nn.functional.cross_entropy(logits, local)
            if anchor is not None and attention_distill_weight > 0:
                # TTD-inspired temporal attention alignment: keep the new
                # attention distribution close to the frozen previous model
                # on each current-task sequence while adapting the readout.
                with torch.no_grad():
                    states, _ = model.encoder(xx)
                    teacher_scores = torch.nn.functional.linear(
                        torch.tanh(states), theta_anchor)
                    teacher_weights = torch.softmax(teacher_scores.squeeze(-1), dim=1)
                student_scores = model.temporal_attention(torch.tanh(states)).squeeze(-1)
                student_log_weights = torch.log_softmax(student_scores, dim=1)
                attention_kd = torch.nn.functional.kl_div(
                    student_log_weights, teacher_weights, reduction="batchmean")
                loss = loss + attention_distill_weight * attention_kd
            if theta_anchor is not None and fisher_weight_tensor is not None and fisher_weight > 0:
                delta = model.temporal_attention.weight - theta_anchor
                loss = loss + fisher_weight * torch.mean(fisher_weight_tensor * delta.square())
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()

        model.eval()
        correct = count = 0
        loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(val_idx), EVAL_BATCH):
                ids = val_idx[start:start+EVAL_BATCH]
                xx = torch.from_numpy(x[ids]).to(device)
                yy = y[ids]
                logits = model(xx)[:, new_classes]
                local = torch.as_tensor([class_to_local[int(v)] for v in yy], device=device)
                loss_sum += float(nn.functional.cross_entropy(logits, local, reduction="sum"))
                correct += int((logits.argmax(1) == local).sum())
                count += len(ids)
        val_acc, val_loss = correct / max(1, count), loss_sum / max(1, count)
        key = (val_acc, -val_loss)
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if val_acc - best_acc >= 0.005 or best_loss - val_loss >= 0.001:
            best_acc, best_loss, stagnant = max(best_acc, val_acc), min(best_loss, val_loss), 0
        else:
            stagnant += 1
        if epoch >= 3 and stagnant >= 2:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return epoch


def refine_with_cfc_pair_margins(predictor, concepts, labels, seen, old_classes,
                                new_classes, old_means, current_means, hidden_size,
                                device, margin_weight=0.2, margin=0.5, top_pairs=2):
    """Refine CBM logits on the most similar old/new CfC latent class pairs."""
    candidates = []
    for old_cls in old_classes:
        old_z = old_means[old_cls][:hidden_size]
        for new_cls in new_classes:
            new_z = current_means[new_cls][:hidden_size]
            cosine = float(np.dot(old_z, new_z) /
                           (np.linalg.norm(old_z) * np.linalg.norm(new_z) + 1e-8))
            candidates.append((old_cls, new_cls, cosine))
    selected = sorted(candidates, key=lambda row: row[2], reverse=True)[:top_pairs]
    yy = torch.from_numpy(cbm.seen_index(labels, seen)).long().to(device)
    cc = torch.from_numpy(concepts).to(device)
    pair_indices = [(seen.index(a), seen.index(b), max(0.05, score))
                    for a, b, score in selected]
    weight_sum = sum(p[2] for p in pair_indices)
    opt = torch.optim.Adam(predictor.parameters(), lr=0.001)
    history = []
    for epoch in range(40):
        opt.zero_grad(set_to_none=True)
        logits = predictor(cc)
        ce = nn.functional.cross_entropy(logits, yy)
        terms, weights = [], []
        for a, b, score in pair_indices:
            mask_a, mask_b = yy == a, yy == b
            if mask_a.any() and mask_b.any():
                loss_a = torch.relu(margin - (logits[mask_a, a] - logits[mask_a, b])).mean()
                loss_b = torch.relu(margin - (logits[mask_b, b] - logits[mask_b, a])).mean()
                terms.append(0.5 * (loss_a + loss_b)); weights.append(score / weight_sum)
        pair_loss = sum(w * term for w, term in zip(weights, terms)) if terms else torch.zeros((), device=device)
        sparse = cbm.CONFIG['sparsity_lambda'] * predictor.weight.abs().sum()
        loss = ce + margin_weight * pair_loss + sparse
        loss.backward(); opt.step()
        history.append({"epoch": epoch + 1, "ce": float(ce.detach()),
                        "pair_margin_loss": float(pair_loss.detach()),
                        "total_loss": float(loss.detach())})
    pair_info = [{"old_class": int(a), "new_class": int(b), "latent_cosine": float(score)}
                 for a, b, score in selected]
    return predictor, pair_info, history


def joint_refine(projector, predictor, previous, z_fit, targets, all_z, labels, seen, old_count, cfg, device, kd_on_pseudo, use_distillation=True, staged_training=False, pair_margin_pairs=None, pair_margin_weight=0.0, pair_margin=0.5):
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
        margin_terms, margin_weights = [], []
        if pair_margin_pairs and pair_margin_weight > 0:
            for old_cls, new_cls, similarity in pair_margin_pairs:
                a, b = seen.index(old_cls), seen.index(new_cls)
                mask_a, mask_b = yy == a, yy == b
                if mask_a.any() and mask_b.any():
                    loss_a = torch.relu(pair_margin - (predictor(concepts[mask_a])[:, a] - predictor(concepts[mask_a])[:, b])).mean()
                    loss_b = torch.relu(pair_margin - (predictor(concepts[mask_b])[:, b] - predictor(concepts[mask_b])[:, a])).mean()
                    margin_terms.append(0.5 * (loss_a + loss_b))
                    margin_weights.append(max(0.05, float(similarity)))
        if margin_terms:
            norm = sum(margin_weights)
            pair_loss = sum((w / norm) * term for w, term in zip(margin_weights, margin_terms))
        else:
            pair_loss = torch.zeros((), device=device)
        loss = ce + align + kd + sparse + pair_margin_weight * pair_loss
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite joint loss')
        loss.backward()
        old_rows_frozen = staged_training and epoch < 10
        if old_rows_frozen and projector.weight.grad is not None:
            projector.weight.grad[:old_count].zero_()
        nn.utils.clip_grad_norm_(list(projector.parameters()) + list(predictor.parameters()), 5.0)
        opt.step()
        history.append({'epoch':epoch+1,'classification_loss':float(ce.detach()),'alignment_loss':float(align.detach()),'distillation_loss':float(kd.detach()),'pair_margin_loss':float(pair_loss.detach()),'old_rows_frozen':bool(old_rows_frozen)})
    return history


def fit_cbm_condition(x, y, xt, yt, raw, raw_test, pool, device,
                      condition='CBM_original', warmstart=False, joint_training=False,
                      kd_on_pseudo=False, head_only=False, use_distillation=True, staged_training=False,
                      use_concept_bottleneck=True, use_pseudo_features=True,
                      use_concept_distillation=True, pseudo_generation="pooled_shift",
                      cfc_pair_margin_weight=0.0, liquid_relation_replay=False,
                      relation_weight=0.5, replay_memory_per_class=17,
                      liquid_gate_replay=False, gate_replay_weight=10.0,
                      continual_attention=False, attention_fisher_weight=0.0,
                      attention_distill_weight=0.0,
                      attention_fisher_samples_per_class=64,
                        pseudo_prototype_count=1, pseudo_prototype_count_by_class=None,
                        pseudo_source_class_by_class=None,
                        recenter_pseudo_prototypes=False,
                        diagnostic_probe_indices=None):
    b.set_seed(SEED)
    model = b.SequenceClassifier("CfC", x.shape[-1], 48, len(b.CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    seen, profiles, class_means, class_stds, trajectory_means, trajectory_modes, learned_acc, rows = [], {}, {}, {}, {}, {}, {}, []
    pooled_prototypes = {}
    memory_idx = {}
    projector = predictor = None
    signal_stats = None
    attention_anchor = attention_fisher = None
    for task_i, new_classes in enumerate(b.TASKS, start=1):
        gate_loss_mean = None
        old = list(seen)
        seen.extend(new_classes)
        train_idx, val_idx = b.split_task_train_validation(y, pool, new_classes, 0.2, 20261004 + task_i*100)
        if task_i == 1:
            trained = train_cfctask(model, optimizer, x, y, train_idx, val_idx, seen, device,
                                    SEED + task_i*100_000)
            signal_cur = cbm.raw_concept_features(raw[train_idx])
            signal_stats = (signal_cur.mean(0), signal_cur.std(0)+1e-3)
            if (continual_attention or attention_distill_weight > 0) and hasattr(model, "temporal_attention"):
                attention_anchor = model.temporal_attention.weight.detach().cpu().clone()
                if continual_attention:
                    attention_fisher = estimate_attention_fisher(
                        model, x, y, train_idx, new_classes, device,
                        SEED + 700_000 + task_i,
                        samples_per_class=attention_fisher_samples_per_class)
            relation_loss_mean = None
        elif liquid_relation_replay:
            trained, relation_loss_mean = train_cfc_with_relation_replay(
                model, optimizer, x, y, train_idx, val_idx, memory_idx, seen, device,
                SEED + task_i*100_000, relation_weight=relation_weight)
        elif liquid_gate_replay:
            trained, gate_loss_mean = train_cfc_with_gate_replay(
                model, optimizer, x, y, train_idx, val_idx, memory_idx, seen, device,
                SEED + task_i*100_000, gate_weight=gate_replay_weight)
            relation_loss_mean = None
        else:
            trained = 0  # CBM adaptation keeps the CfC feature encoder fixed after Task 1.
            relation_loss_mean = None
        has_real_replay = liquid_relation_replay or liquid_gate_replay
        if task_i > 1 and not has_real_replay:
            for p in model.parameters():
                if p is not model.head.weight and p is not model.head.bias:
                    p.requires_grad_(False)
        if task_i > 1 and (continual_attention or attention_distill_weight > 0) and hasattr(model, "temporal_attention"):
            adapt_temporal_attention(
                model, x, y, train_idx, val_idx, new_classes, device,
                SEED + task_i*700_000,
                anchor=attention_anchor,
                fisher=attention_fisher,
                fisher_weight=attention_fisher_weight,
                attention_distill_weight=attention_distill_weight)
            task_fisher = estimate_attention_fisher(
                model, x, y, train_idx, new_classes, device,
                SEED + 800_000 + task_i,
                samples_per_class=attention_fisher_samples_per_class)
            if attention_fisher is None:
                attention_fisher = task_fisher
            else:
                attention_fisher = attention_fisher + task_fisher
            attention_anchor = model.temporal_attention.weight.detach().cpu().clone()

        for cls in new_classes:
            sig = cbm.raw_concept_features(raw[train_idx[y[train_idx] == cls]])
            profiles[cls] = (sig.mean(0), sig.std(0)+1e-3)

        # In liquid-replay mode, fit concepts using current examples plus the
        # same real exemplars that trained the CfC backbone.
        cbm_fit_idx = train_idx
        if has_real_replay and old:
            cbm_fit_idx = np.concatenate([*([memory_idx[c] for c in sorted(old)]), train_idx])
        fit_y = y[cbm_fit_idx]
        z_fit = make_joint(model, x, raw, cbm_fit_idx, device, signal_stats)
        raw_fit = cbm.raw_concept_features(raw[cbm_fit_idx])
        targets = cbm.concept_targets(raw_fit, seen, profiles)
        teacher = None
        if projector is not None and use_concept_distillation:
            with torch.no_grad():
                teacher = torch.sigmoid(projector(torch.from_numpy(z_fit).to(device))).cpu().numpy()[:, :8*len(old)]
        cfg = dict(cbm.CONFIG)
        cfg["concept_projector_epochs"] = 100
        cfg["classifier_epochs"] = 300
        previous = copy.deepcopy(projector).eval() if projector is not None else None
        if use_concept_bottleneck:
            projector = fit_projector_cubic(z_fit, targets, teacher, 8*len(old), cfg, device,
                                           previous=previous if warmstart else None)
            current_concepts = torch.sigmoid(projector(torch.from_numpy(z_fit).to(device))).detach().cpu().numpy().astype(np.float32)
        else:
            projector = None
            current_concepts = z_fit.copy()
        feature_list = [current_concepts]
        joint_feature_list = [z_fit]
        label_list = [fit_y.copy()]
        pseudo_count = 0

        # Generate old-class pseudo-concepts by shifting the nearest current-class
        # joint-feature distribution toward each stored old-class centroid.
        current_means = {}
        current_trajectory_means = {}
        train_sequence = None
        if pseudo_generation in {"trajectory_shift", "trajectory_mixture", "trajectory_mixture_guarded"}:
            train_sequence = extract_cfc_sequence(model, x, train_idx, device)
        current_mode_labels, current_modes = {}, {}
        for cls in new_classes:
            current_means[cls] = z_fit[fit_y == cls].mean(0)
            if train_sequence is not None:
                cls_sequence = train_sequence[y[train_idx] == cls]
                if pseudo_generation == "trajectory_shift":
                    current_trajectory_means[cls] = cls_sequence.mean(0)
                else:
                    current_mode_labels[cls], current_modes[cls] = cluster_trajectory_modes(
                        cls_sequence, SEED + task_i * 10_000 + cls)
        if has_real_replay and old:
            for cls in old:
                mask_old = fit_y == cls
                if mask_old.any():
                    class_means[cls] = z_fit[mask_old].mean(0)
                    class_stds[cls] = z_fit[mask_old].std(0) + 1e-3
        for cls in old if use_pseudo_features else []:
            old_mean = class_means[cls]
            nearest = max(new_classes, key=lambda c: float(
                np.dot(old_mean, current_means[c]) /
                (np.linalg.norm(old_mean)*np.linalg.norm(current_means[c]) + 1e-8)
            ))
            forced_source = (pseudo_source_class_by_class or {}).get(int(cls))
            if forced_source is not None and int(forced_source) in new_classes:
                nearest = int(forced_source)
            source_mask = fit_y == nearest
            source = z_fit[source_mask]
            if pseudo_generation == "trajectory_shift":
                # Shift each time step in CfC state space, then use the learned
                # temporal readout. Sensor descriptors retain the original
                # centroid-shift rule, so only the liquid trajectory changes.
                source_sequence = train_sequence[source_mask]
                pseudo_sequence = (source_sequence - current_trajectory_means[nearest]
                                   + trajectory_means[cls])
                pseudo_latent = pool_cfc_sequence(model, pseudo_sequence, device)
                hidden_size = pseudo_latent.shape[1]
                pseudo_sensor = (source[:, hidden_size:] - current_means[nearest][hidden_size:]
                                 + old_mean[hidden_size:])
                pseudo_z = np.concatenate([pseudo_latent, pseudo_sensor], axis=1).astype(np.float32)
            elif pseudo_generation in {"trajectory_mixture", "trajectory_mixture_guarded"}:
                old_modes = trajectory_modes[cls]
                if pseudo_generation == "trajectory_mixture_guarded":
                    candidates = []
                    for source_cls in new_classes:
                        class_count = int(np.sum(y[train_idx] == source_cls))
                        for source_mode in range(len(current_modes[source_cls])):
                            source_rows = np.flatnonzero(current_mode_labels[source_cls] == source_mode)
                            min_count = max(32, int(np.ceil(0.2 * class_count)))
                            if len(source_rows) < min_count:
                                continue
                            score = pooled_trajectory_cosine(
                                model, old_modes[0], current_modes[source_cls][source_mode], device,
                                train_sequence.shape[1])
                            candidates.append((source_cls, source_mode, len(source_rows), score))
                    # Each old mode must use a distinct, adequately populated
                    # current mode. Otherwise retain the original pooled shift.
                    assignment = None
                    if len(candidates) >= len(old_modes):
                        import itertools
                        best_score = -float("inf")
                        for choice in itertools.permutations(candidates, len(old_modes)):
                            score = sum(pooled_trajectory_cosine(
                                model, old_mode, current_modes[item[0]][item[1]], device,
                                train_sequence.shape[1]) for old_mode, item in zip(old_modes, choice))
                            if score > best_score:
                                assignment, best_score = choice, score
                    if assignment is None:
                        pseudo_z = source - current_means[nearest] + old_mean
                    else:
                        mode_outputs = []
                        class_count = int(np.sum(y[train_idx] == nearest))
                        quota = max(1, int(np.ceil(class_count / len(old_modes))))
                        rng = np.random.default_rng(SEED + task_i * 100_000 + cls)
                        for old_mode, item in zip(old_modes, assignment):
                            source_cls, source_mode = item[0], item[1]
                            class_rows = np.flatnonzero(y[train_idx] == source_cls)
                            source_rows = np.flatnonzero(current_mode_labels[source_cls] == source_mode)
                            chosen = rng.choice(source_rows, size=quota, replace=len(source_rows) < quota)
                            src = z_fit[class_rows][chosen]
                            src_seq = train_sequence[class_rows][chosen]
                            current_proto = expand_trajectory(current_modes[source_cls][source_mode], src_seq.shape[1])
                            old_proto = expand_trajectory(old_mode, src_seq.shape[1])
                            shifted_seq = src_seq - current_proto[None] + old_proto[None]
                            latent = pool_cfc_sequence(model, shifted_seq, device)
                            hidden_size = latent.shape[1]
                            sensor = (src[:, hidden_size:] - current_means[source_cls][hidden_size:]
                                      + old_mean[hidden_size:])
                            mode_pseudo = np.concatenate([latent, sensor], axis=1).astype(np.float32)
                            mode_outputs.append(mode_pseudo)
                        pseudo_z = np.concatenate(mode_outputs, axis=0)
                else:
                    # Match each stored old-class trajectory mode to the closest
                    # current-class mode in the model's actual temporal readout space.
                    for old_mode in old_modes:
                        pair = max(((c, mode) for c in new_classes
                                    for mode in range(len(current_modes[c]))),
                                   key=lambda cm: pooled_trajectory_cosine(
                                       model, old_mode, current_modes[cm[0]][cm[1]], device,
                                       train_sequence.shape[1]))
                        source_cls, source_mode = pair
                        source_rows = np.flatnonzero(current_mode_labels[source_cls] == source_mode)
                        if len(source_rows) == 0:
                            source_rows = np.flatnonzero(y[train_idx] == source_cls)
                        src = z_fit[y[train_idx] == source_cls][source_rows]
                        src_seq = train_sequence[y[train_idx] == source_cls][source_rows]
                        current_proto = expand_trajectory(current_modes[source_cls][source_mode], src_seq.shape[1])
                        old_proto = expand_trajectory(old_mode, src_seq.shape[1])
                        shifted_seq = src_seq - current_proto[None] + old_proto[None]
                        latent = pool_cfc_sequence(model, shifted_seq, device)
                        hidden_size = latent.shape[1]
                        sensor = (src[:, hidden_size:] - current_means[source_cls][hidden_size:]
                                  + old_mean[hidden_size:])
                        mode_pseudo = np.concatenate([latent, sensor], axis=1).astype(np.float32)
                        joint_feature_list.append(mode_pseudo)
                        if projector is None:
                            mode_concepts = mode_pseudo.copy()
                        else:
                            with torch.no_grad():
                                mode_concepts = torch.sigmoid(
                                    projector(torch.from_numpy(mode_pseudo).to(device))
                                ).cpu().numpy().astype(np.float32)
                        feature_list.append(mode_concepts)
                        label_list.append(np.full(len(mode_concepts), cls, dtype=np.int64))
                        pseudo_count += len(mode_concepts)
                    continue
            elif pseudo_generation == "pooled_multi_proto":
                centers = pooled_prototypes.get(cls)
                if centers is None:
                    centers = class_means[cls][None, :]
                if recenter_pseudo_prototypes and len(centers) > 1:
                    centers = (centers - centers.mean(axis=0, keepdims=True)
                               + class_means[cls][None, :])
                rng = np.random.default_rng(SEED + task_i * 100_000 + cls)
                order = rng.permutation(len(source))
                chunks = np.array_split(order, len(centers))
                pseudo_z = np.concatenate([
                    source[chunk] - current_means[nearest] + center
                    for center, chunk in zip(centers, chunks) if len(chunk)
                ], axis=0).astype(np.float32)
            elif pseudo_generation == "pooled_shift_diag_var":
                # Match each feature dimension's spread to the stored old class
                # while retaining the current-class sample directions.
                source_std = source.std(0) + 1e-3
                scale = np.clip(class_stds[cls] / source_std, 0.5, 2.0)
                pseudo_z = (source - current_means[nearest]) * scale + old_mean
            else:
                pseudo_z = source - current_means[nearest] + old_mean
            joint_feature_list.append(pseudo_z)
            if projector is None:
                pseudo_c = pseudo_z.copy()
            else:
                with torch.no_grad():
                    pseudo_c = torch.sigmoid(projector(torch.from_numpy(pseudo_z).to(device))).cpu().numpy().astype(np.float32)
            feature_list.append(pseudo_c)
            label_list.append(np.full(len(pseudo_c), cls, dtype=np.int64))
            pseudo_count += len(pseudo_c)
        for cls in new_classes:
            class_means[cls] = current_means[cls]
            class_stds[cls] = z_fit[fit_y == cls].std(0) + 1e-3
            class_proto_count = int((pseudo_prototype_count_by_class or {}).get(
                int(cls), pseudo_prototype_count))
            if class_proto_count > 1:
                pooled_prototypes[cls] = fit_pooled_prototypes(
                    z_fit[fit_y == cls], class_proto_count,
                    SEED + task_i * 10_000 + cls)
            if train_sequence is not None:
                if pseudo_generation == "trajectory_shift":
                    trajectory_means[cls] = current_trajectory_means[cls]
                elif pseudo_generation in {"trajectory_mixture", "trajectory_mixture_guarded"}:
                    trajectory_modes[cls] = current_modes[cls]

        concepts = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        predictor = cbm.fit_sparse_classifier(concepts, labels, cfg, device)
        if old and cfc_pair_margin_weight > 0:
            predictor, pair_info, margin_history = refine_with_cfc_pair_margins(
                predictor, concepts, labels, seen, old, new_classes, class_means,
                current_means, z_fit.shape[1] - 8, device,
                margin_weight=cfc_pair_margin_weight)
            (OUT / f'{condition}_task{task_i}_pair_margins.json').write_text(
                json.dumps({"selected_pairs": pair_info, "history": margin_history}, indent=2),
                encoding='utf-8')
        if joint_training and old:
            pair_margin_pairs = []
            if cfc_pair_margin_weight > 0:
                hidden_size = z_fit.shape[1] - 8
                candidates = []
                for old_cls in old:
                    old_z = class_means[old_cls][:hidden_size]
                    for new_cls in new_classes:
                        new_z = current_means[new_cls][:hidden_size]
                        similarity = float(np.dot(old_z, new_z) /
                                           (np.linalg.norm(old_z) * np.linalg.norm(new_z) + 1e-8))
                        candidates.append((old_cls, new_cls, similarity))
                pair_margin_pairs = sorted(candidates, key=lambda item: item[2], reverse=True)[:2]
                (OUT / f'{condition}_task{task_i}_pair_margins.json').write_text(
                    json.dumps({"selected_pairs": [
                        {"old_class": int(a), "new_class": int(b), "latent_cosine": float(s)}
                        for a, b, s in pair_margin_pairs]}, indent=2), encoding='utf-8')
            history = joint_refine(projector, predictor, previous, z_fit, targets,
                                   np.concatenate(joint_feature_list), labels, seen, 8*len(old), cfg, device,kd_on_pseudo,use_distillation,staged_training,
                                   pair_margin_pairs=pair_margin_pairs,
                                   pair_margin_weight=cfc_pair_margin_weight)
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
        if has_real_replay:
            for cls in new_classes:
                candidates = train_idx[y[train_idx] == cls].copy()
                rng = np.random.default_rng(SEED + task_i * 1000 + cls)
                rng.shuffle(candidates)
                memory_idx[cls] = candidates[:min(replay_memory_per_class, len(candidates))]
        torch.save({'encoder':model.state_dict(),'projector':projector.state_dict() if projector is not None else None,
                    'classifier':predictor.state_dict(),'seen':seen.copy(),'seed':SEED,
                    'pooled_prototypes': {int(c): v.copy() for c, v in pooled_prototypes.items()},
                    'memory_indices': {int(c): ix.copy() for c, ix in memory_idx.items()} if has_real_replay else None},
                   OUT / f'{condition}_task{task_i}.pt')

        per = {}
        predictor.eval(); model.eval()
        if projector is not None:
            projector.eval()
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
                    cc = torch.sigmoid(projector(joint)) if projector is not None else joint
                    preds.extend(predictor(cc)[:, :len(seen)].argmax(1).cpu().numpy().tolist())
                per[cls] = float(np.mean(np.asarray(preds) == cbm.seen_index(yt[ids], seen)))
        for cls in new_classes:
            learned_acc[cls] = per[cls]
        (OUT / f'{condition}_task{task_i}_per_class.json').write_text(json.dumps(per, indent=2), encoding='utf-8')
        row = make_row(condition, task_i, trained, old, new_classes,
                       seen, per, learned_acc, pseudo_count)
        row['pseudo_prototype_count'] = int(pseudo_prototype_count)
        row['prototype_memory_bytes'] = int(sum(v.nbytes for v in pooled_prototypes.values()))
        if diagnostic_probe_indices is not None:
            eval_indices = {c: (diagnostic_probe_indices[y[diagnostic_probe_indices] == c]
                                if c in old else val_idx[y[val_idx] == c]) for c in seen}
            val_per = {}
            val_confusion = np.zeros((len(seen), len(seen)), dtype=np.int64)
            margin_sums = np.zeros(len(seen), dtype=np.float64)
            margin_counts = np.zeros(len(seen), dtype=np.int64)
            with torch.no_grad():
                for cls, ids in eval_indices.items():
                    class_idx = seen.index(cls)
                    preds = []
                    for start in range(0, len(ids), EVAL_BATCH):
                        selected = ids[start:start+EVAL_BATCH]
                        latent = extract_cfc_latent(model, x, selected, device)
                        sig = cbm.raw_concept_features(raw[selected])
                        sm, ss = signal_stats
                        sig = np.clip((sig-sm)/(ss+1e-6), -6, 6).astype(np.float32)
                        joint = torch.from_numpy(np.concatenate([latent, sig], axis=1)).to(device)
                        cc = torch.sigmoid(projector(joint))
                        logits = predictor(cc)[:, :len(seen)]
                        predicted = logits.argmax(1).cpu().numpy()
                        target = cbm.seen_index(y[selected], seen)
                        preds.extend(predicted.tolist())
                        np.add.at(val_confusion, (target, predicted), 1)
                        if len(seen) > 1:
                            target_scores = logits.gather(1, torch.as_tensor(target, device=device).view(-1, 1)).squeeze(1)
                            masked = logits.clone()
                            masked.scatter_(1, torch.as_tensor(target, device=device).view(-1, 1), -torch.inf)
                            margins = (target_scores - masked.max(1).values).cpu().numpy()
                            margin_sums[class_idx] += float(margins.sum())
                            margin_counts[class_idx] += len(margins)
                    val_per[cls] = float(np.mean(np.asarray(preds) == cbm.seen_index(y[ids], seen)))
            row['validation_per_class_accuracy'] = json.dumps({
                str(int(cls)): float(value) for cls, value in val_per.items()
            })
            row['validation_confusion_counts'] = json.dumps({
                str(int(true_cls)): {str(int(pred_cls)): int(val_confusion[i, j])
                                     for j, pred_cls in enumerate(seen)}
                for i, true_cls in enumerate(seen)
            })
            row['validation_true_vs_best_other_logit_margin'] = json.dumps({
                str(int(cls)): float(margin_sums[i] / margin_counts[i]) if margin_counts[i] else None
                for i, cls in enumerate(seen)
            })
            row['validation_old_class_macro_accuracy'] = float(np.mean([val_per[c] for c in old])) if old else None
            row['validation_new_class_macro_accuracy'] = float(np.mean([val_per[c] for c in new_classes]))
            row['validation_all_seen_macro_accuracy'] = float(np.mean(list(val_per.values())))
        if liquid_relation_replay:
            row["real_replay_examples"] = int(sum(len(ix) for ix in memory_idx.values()))
            row["real_replay_bytes"] = int(sum(x[ix].nbytes + y[ix].nbytes for ix in memory_idx.values()))
            row["temporal_relation_loss_last_epoch"] = relation_loss_mean
            n_old_memory = sum(len(memory_idx[c]) for c in old)
            row["temporal_relation_target_bytes_next_task"] = (
                int(n_old_memory * (n_old_memory - 1) + 8)
                if task_i < len(b.TASKS) and n_old_memory > 1 else 0)
        if liquid_gate_replay:
            row["real_replay_examples"] = int(sum(len(ix) for ix in memory_idx.values()))
            row["real_replay_bytes"] = int(sum(x[ix].nbytes + y[ix].nbytes for ix in memory_idx.values()))
            row["cfc_gate_distill_loss_last_epoch"] = gate_loss_mean
        rows.append(row)
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
