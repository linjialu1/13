"""Multi-seed, multi-dataset benchmark for class-incremental time-series learning.

Datasets: UCI-HAR, WISDM 2019, and UWaveGestureLibrary.
Models: custom LNN, ncps LTC/CfC, LSTM, and GRU.
Conditions: class-incremental learning without replay and joint-training control.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "experiments" / "baselines"

# Allow running the benchmark before ncps has been installed into a locked-down
# Conda environment. Normal users should install requirements.txt as usual.
LOCAL_PACKAGES = ROOT / ".local_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
try:
    from ncps.torch import CfC, LTC
except ImportError:
    CfC = LTC = None

WISDM_URL = "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
UWAVE_TRAIN_URL = "https://zenodo.org/api/records/11206282/files/UWaveGestureLibrary_TRAIN.ts/content"
UWAVE_TEST_URL = "https://zenodo.org/api/records/11206282/files/UWaveGestureLibrary_TEST.ts/content"
UCI_CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
WISDM_CLASSES = ["walking", "jogging", "stairs", "sitting", "standing", "typing"]
UCI_TASKS = [[0, 1], [2, 3], [4, 5]]
WISDM_TASKS = [[0, 1], [2, 3], [4, 5]]
UWAVE_CLASSES = [f"gesture_{i}" for i in range(1, 9)]
SEEDS = [11, 22, 33]


def load_uwave_file(path: Path, url: str) -> tuple[np.ndarray, list[str]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading UWaveGestureLibrary: {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    text = path.read_text(encoding="utf-8", errors="replace")
    body = text.split("@data", 1)[1]
    cases, labels = [], []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        label = parts[-1].strip()
        dims = [np.asarray([float(v) for v in part.split(",") if v], dtype=np.float32) for part in parts[:-1]]
        if len(dims) != 3 or len({len(d) for d in dims}) != 1:
            raise ValueError(f"Unexpected UWave .ts row in {path.name}: {line[:120]}")
        cases.append(np.stack(dims, axis=-1))
        labels.append(label)
    return np.asarray(cases, dtype=np.float32), labels


def load_uwave() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    cache = DATA_DIR / "uwave_gesture_library.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x_train"], z["y_train"], z["x_test"], z["y_test"], json.loads(str(z["meta"]))
    x_train, y_train_labels = load_uwave_file(DATA_DIR / "UWaveGestureLibrary_TRAIN.ts", UWAVE_TRAIN_URL)
    x_test, y_test_labels = load_uwave_file(DATA_DIR / "UWaveGestureLibrary_TEST.ts", UWAVE_TEST_URL)
    labels = sorted(set(y_train_labels) | set(y_test_labels), key=lambda value: int(float(value)))
    lookup = {label: i for i, label in enumerate(labels)}
    y_train = np.asarray([lookup[label] for label in y_train_labels], dtype=np.int64)
    y_test = np.asarray([lookup[label] for label in y_test_labels], dtype=np.int64)
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    meta = {"classes": [f"gesture_{int(float(label))}" for label in labels], "train_windows": len(y_train),
            "test_windows": len(y_test), "channels": 3, "series_length": int(x_train.shape[1]),
            "source": "UEA Time Series Classification Archive via Zenodo record 11206282",
            "original_train_test_split": True}
    np.savez_compressed(cache, x_train=x_train.astype(np.float32), y_train=y_train,
                        x_test=x_test.astype(np.float32), y_test=y_test, meta=json.dumps(meta))
    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test, meta


def task_orders(tasks: list[list[int]], n_orders: int, seed: int) -> list[tuple[str, list[list[int]]]]:
    """Fixed, reproducible order sensitivity checks; task composition also varies in the last order."""
    orders = [("canonical", [list(task) for task in tasks])]
    if n_orders >= 2:
        orders.append(("reverse_task_order", [list(task) for task in reversed(tasks)]))
    if n_orders >= 3:
        rng = np.random.default_rng(seed)
        classes = np.asarray([c for task in tasks for c in task])
        permuted = rng.permutation(classes).tolist()
        orders.append(("permuted_class_pairs", [permuted[i:i + len(tasks[0])] for i in range(0, len(permuted), len(tasks[0]))]))
    for idx in range(3, n_orders):
        rng = np.random.default_rng(seed + idx)
        shuffled = [list(task) for task in tasks]
        rng.shuffle(shuffled)
        orders.append((f"shuffled_{idx + 1}", shuffled))
    return orders[:n_orders]


def load_uci_har() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    extracted = DATA_DIR / "UCI HAR Dataset"
    if not extracted.exists():
        archive = DATA_DIR / "uci_har.zip"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
            print(f"Downloading UCI-HAR: {url}", flush=True)
            urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_DIR)
    xs, ys = [], []
    signals = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
    for split in ("train", "test"):
        folder = extracted / split / "Inertial Signals"
        channels = [np.loadtxt(folder / f"{name}_{split}.txt", dtype=np.float32) for name in signals]
        xs.append(np.stack(channels, axis=-1))
        ys.append(np.loadtxt(extracted / split / f"y_{split}.txt", dtype=np.int64) - 1)
    x_train, x_test = xs
    y_train, y_test = ys
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((x_train - mean) / std).astype(np.float32), y_train, ((x_test - mean) / std).astype(np.float32), y_test


def wisdm_members(outer: zipfile.ZipFile) -> tuple[zipfile.ZipFile, io.BytesIO]:
    """Return a nested WISDM archive backed by memory (without writing another 280 MB copy)."""
    member = next(n for n in outer.namelist() if n.lower().endswith("wisdm-dataset.zip"))
    data = io.BytesIO(outer.read(member))
    return zipfile.ZipFile(data), data


def make_windows(segment: list[list[float]], length: int = 128, stride: int = 64) -> list[np.ndarray]:
    a = np.asarray(segment, dtype=np.float32)
    return [a[i:i + length] for i in range(0, len(a) - length + 1, stride)]


def load_wisdm() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Window phone-accelerometer records, holding out entire participants."""
    cache = DATA_DIR / "wisdm_har_windows.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x_train"], z["y_train"], z["x_test"], z["y_test"], json.loads(str(z["meta"]))
    archive = DATA_DIR / "wisdm.zip"
    if not archive.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading WISDM (~296 MB): {WISDM_URL}", flush=True)
        urllib.request.urlretrieve(WISDM_URL, archive)

    class_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    class_names = ["walking", "jogging", "stairs", "sitting", "standing", "typing"]
    train_x, train_y, test_x, test_y = [], [], [], []
    train_subjects, test_subjects = [], []
    count_rows = 0
    with zipfile.ZipFile(archive) as outer:
        nested, _buffer = wisdm_members(outer)
        members = sorted(n for n in nested.namelist() if "/raw/phone/accel/data_" in n and n.endswith("_accel_phone.txt"))
        subjects = sorted(int(Path(n).name.split("_")[1]) for n in members)
        test_ids = set(subjects[::5])  # fixed subject-disjoint 20% test split
        print(f"Parsing WISDM phone accelerometer for {len(subjects)} participants; held out {len(test_ids)}", flush=True)
        for member in members:
            subject = int(Path(member).name.split("_")[1])
            is_test = subject in test_ids
            segments: list[tuple[int, list[list[float]]]] = []
            current_code, current = None, []
            with nested.open(member) as raw:
                for raw_line in raw:
                    line = raw_line.decode("utf-8", errors="ignore").strip().rstrip(";")
                    fields = line.split(",")
                    if len(fields) < 6:
                        continue
                    code = fields[1].strip()
                    if code != current_code:
                        if current_code in class_index and current:
                            segments.append((class_index[current_code], current))
                        current_code, current = code, []
                    count_rows += 1
                    if code in class_index:
                        try:
                            current.append([float(fields[3]), float(fields[4]), float(fields[5])])
                        except ValueError:
                            pass
            if current_code in class_index and current:
                segments.append((class_index[current_code], current))
            for cls, segment in segments:
                windows = make_windows(segment)
                if not windows:
                    continue
                target_x, target_y = (test_x, test_y) if is_test else (train_x, train_y)
                target_x.extend(windows)
                target_y.extend([cls] * len(windows))
            (test_subjects if is_test else train_subjects).append(subject)
    if not train_x or not test_x:
        raise RuntimeError("WISDM parsing produced no windows. Check the downloaded archive format.")
    rng = np.random.default_rng(2026)
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64)
    perm = rng.permutation(len(train_y))
    train_x, train_y = train_x[perm], train_y[perm]
    x_test, y_test = np.asarray(test_x, np.float32), np.asarray(test_y, np.int64)
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    x_test = (x_test - mean) / std
    meta = {"classes": class_names, "train_subjects": train_subjects, "test_subjects": test_subjects,
            "raw_rows_scanned": count_rows, "train_windows": len(train_y), "test_windows": len(y_test),
            "window_length": 128, "stride": 64, "sample_rate_hz": 20, "sensor": "phone accelerometer (x,y,z)"}
    np.savez_compressed(cache, x_train=train_x, y_train=train_y, x_test=x_test, y_test=y_test, meta=json.dumps(meta))
    return train_x, train_y, x_test, y_test, meta


class LiquidCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.logit_alpha = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        alpha = torch.sigmoid(self.logit_alpha)
        h = x.new_zeros((batch, self.logit_alpha.numel()))
        input_drive = self.input(x)
        for t in range(steps):
            proposal = torch.tanh(input_drive[:, t] + self.recurrent(h))
            h = h + alpha * (proposal - h)
        return h, alpha


class SequenceClassifier(nn.Module):
    def __init__(self, kind: str, input_size: int, hidden_size: int, n_classes: int):
        super().__init__()
        self.kind = kind
        if kind == "LNN":
            self.encoder = LiquidCell(input_size, hidden_size)
        elif kind in {"LTC", "CfC"}:
            if CfC is None or LTC is None:
                raise ImportError("The LTC/CfC models require ncps. Install it with: python -m pip install ncps")
            cell = LTC if kind == "LTC" else CfC
            self.encoder = cell(input_size, hidden_size, batch_first=True, return_sequences=True)
        elif kind == "LSTM":
            self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif kind == "GRU":
            self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "LNN":
            h, _ = self.encoder(x)
        elif self.kind in {"LTC", "CfC"}:
            sequence, _ = self.encoder(x)
            h = sequence[:, -1]
        else:
            _, state = self.encoder(x)
            h = state[0][-1] if self.kind == "LSTM" else state[-1]
        return self.head(h)

    def liquid_tau(self) -> np.ndarray | None:
        if self.kind != "LNN":
            return None
        alpha = torch.sigmoid(self.encoder.logit_alpha).detach().cpu().numpy()
        return (1 / np.clip(alpha, 1e-5, None)).copy()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tx = torch.from_numpy(x[indices])
    ty = torch.from_numpy(y[indices]).long()
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      pin_memory=torch.cuda.is_available())


def split_train_validation(y: np.ndarray, eligible: np.ndarray, fraction: float,
                           split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a stable stratified split using training labels only."""
    train_parts, validation_parts = [], []
    for cls in np.unique(y[eligible]):
        class_indices = eligible[y[eligible] == cls].copy()
        rng = np.random.default_rng(split_seed + int(cls))
        rng.shuffle(class_indices)
        if len(class_indices) < 2:
            raise ValueError(f"Class {cls} needs at least two training examples for train/validation splitting")
        n_validation = min(max(int(round(len(class_indices) * fraction)), 1), len(class_indices) - 1)
        validation_parts.append(class_indices[:n_validation])
        train_parts.append(class_indices[n_validation:])
    return np.concatenate(train_parts), np.concatenate(validation_parts)


@torch.no_grad()
def evaluate(model: SequenceClassifier, x: np.ndarray, y: np.ndarray, seen: list[int], device: torch.device,
             batch_size: int = 256) -> tuple[float, list[float]]:
    model.eval()
    per_class = []
    lookup = torch.as_tensor(seen, device=device)
    for cls in seen:
        idx = np.flatnonzero(y == cls)
        preds = []
        for start in range(0, len(idx), batch_size):
            logits = model(torch.from_numpy(x[idx[start:start + batch_size]]).to(device))[:, lookup]
            preds.extend(lookup[logits.argmax(1)].cpu().numpy().tolist())
        per_class.append(float(np.mean(np.asarray(preds) == cls)))
    idx = np.flatnonzero(np.isin(y, seen))
    correct = total = 0
    for start in range(0, len(idx), batch_size):
        yy = y[idx[start:start + batch_size]]
        logits = model(torch.from_numpy(x[idx[start:start + batch_size]]).to(device))[:, lookup]
        pred = lookup[logits.argmax(1)].cpu().numpy()
        correct += int((pred == yy).sum()); total += len(yy)
    return correct / max(total, 1), per_class


@torch.no_grad()
def evaluate_validation(model: SequenceClassifier, x: np.ndarray, y: np.ndarray, indices: np.ndarray,
                        seen: list[int], device: torch.device, batch_size: int = 256) -> tuple[float, float]:
    """Return macro accuracy and cross-entropy on the current task's validation split."""
    model.eval()
    lookup = torch.full((model.head.out_features,), -1, dtype=torch.long, device=device)
    lookup[torch.as_tensor(seen, device=device)] = torch.arange(len(seen), device=device)
    class_correct = []
    total_loss, total_count = 0.0, 0
    for cls in seen:
        class_indices = indices[y[indices] == cls]
        if len(class_indices) == 0:
            continue
        correct = 0
        for start in range(0, len(class_indices), batch_size):
            batch_indices = class_indices[start:start + batch_size]
            yy = torch.from_numpy(y[batch_indices]).long().to(device)
            logits = model(torch.from_numpy(x[batch_indices]).to(device))[:, seen]
            mapped = lookup[yy]
            loss = nn.functional.cross_entropy(logits, mapped, reduction="sum")
            pred = logits.argmax(1)
            correct += int((pred == mapped).sum().item())
            total_loss += float(loss.item())
            total_count += len(batch_indices)
        class_correct.append(correct / len(class_indices))
    return float(np.mean(class_correct)), total_loss / max(total_count, 1)


def train_epoch(model: SequenceClassifier, loader: DataLoader, seen: list[int], optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    model.train()
    lookup = torch.full((model.head.out_features,), -1, dtype=torch.long, device=device)
    lookup[torch.as_tensor(seen, device=device)] = torch.arange(len(seen), device=device)
    losses = []
    for xx, yy in loader:
        xx, yy = xx.to(device, non_blocking=True), yy.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xx)[:, seen]
        loss = nn.functional.cross_entropy(logits, lookup[yy])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def run_condition(kind: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                  classes: list[str], tasks: list[list[int]], seed: int, min_epochs: int, max_epochs: int,
                  patience: int, min_delta_accuracy: float, min_delta_loss: float,
                  validation_fraction: float, hidden: int, batch_size: int, lr: float, device: torch.device,
                  condition: str, order_name: str = "canonical") -> tuple[list[dict], dict]:
    set_seed(seed)
    model = SequenceClassifier(kind, x_train.shape[-1], hidden, len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = []
    best = np.zeros(len(classes), dtype=np.float64)
    seen: list[int] = []
    stage_order = tasks if condition == "incremental" else [list(range(len(classes)))]
    tau_start = model.liquid_tau()
    task_best: list[float] = []
    task_learned_accuracy: list[float] = []
    for stage_i, new_classes in enumerate(stage_order):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stage_started = time.perf_counter()
        if condition == "incremental":
            seen.extend(new_classes)
            eligible = np.flatnonzero(np.isin(y_train, new_classes))
            active = list(seen)
            stage_name = f"Task {stage_i + 1}"
        else:
            seen = list(range(len(classes)))
            eligible = np.arange(len(y_train))
            active = seen
            stage_name = "Joint"
        train_indices, validation_indices = split_train_validation(
            y_train, eligible, validation_fraction, split_seed=20261004)
        best_validation_accuracy = -float("inf")
        best_validation_loss = float("inf")
        stagnant_epochs = 0
        epochs_trained = 0
        stop_reason = "max_epochs_reached"
        validation_accuracy = validation_loss = float("nan")
        for epoch in range(1, max_epochs + 1):
            loader = make_loader(x_train, y_train, train_indices, batch_size, shuffle=True)
            loss = train_epoch(model, loader, active, optimizer, device)
            validation_accuracy, validation_loss = evaluate_validation(
                model, x_train, y_train, validation_indices, active, device)
            epochs_trained = epoch
            accuracy_improved = validation_accuracy - best_validation_accuracy >= min_delta_accuracy
            loss_improved = best_validation_loss - validation_loss >= min_delta_loss
            if accuracy_improved or loss_improved:
                best_validation_accuracy = max(best_validation_accuracy, validation_accuracy)
                best_validation_loss = min(best_validation_loss, validation_loss)
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
            if epoch >= min_epochs and stagnant_epochs >= patience:
                stop_reason = "validation_plateau"
                break
        convergence_status = ("converged" if stop_reason == "validation_plateau" else
                              "not_confirmed_at_epoch_limit")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        accuracy, per_class = evaluate(model, x_test, y_test, seen, device)
        for cls, value in zip(seen, per_class):
            best[cls] = max(best[cls], value)
        old = seen[:-len(new_classes)] if condition == "incremental" else []
        forgetting = float(np.mean([best[c] - per_class[seen.index(c)] for c in old])) if old else 0.0
        current_task_accuracy = float(np.mean([per_class[seen.index(c)] for c in new_classes])) if condition == "incremental" else float("nan")
        prior_task_drops = [task_best[i] - float(np.mean([per_class[seen.index(c)] for c in stage_order[i]]))
                            for i in range(stage_i)] if condition == "incremental" else []
        bwt = float(np.mean([-drop for drop in prior_task_drops])) if prior_task_drops else 0.0
        if condition == "incremental":
            task_best.append(current_task_accuracy)
            task_learned_accuracy.append(current_task_accuracy)
        tau = model.liquid_tau()
        row = {"dataset": "", "model": kind, "condition": condition, "seed": seed, "stage": stage_name,
               "order": order_name,
               "seen_classes": len(seen), "accuracy": accuracy, "macro_accuracy": float(np.mean(per_class)),
               "forgetting": forgetting, "bwt": bwt, "current_task_accuracy": current_task_accuracy if condition == "incremental" else "",
               "training_seconds": time.perf_counter() - stage_started, "loss": loss,
               "epochs_trained": epochs_trained, "stop_reason": stop_reason,
               "convergence_status": convergence_status,
               "validation_macro_accuracy": validation_accuracy, "validation_loss": validation_loss,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "mean_tau": float(np.mean(tau)) if tau is not None else "",
               "tau_drift_l2": float(np.linalg.norm(tau - tau_start)) if tau is not None else ""}
        for i, name in enumerate(classes):
            row[f"acc_{name}"] = per_class[seen.index(i)] if i in seen else ""
        if condition == "incremental":
            for task_i, task_classes in enumerate(stage_order):
                row[f"task_{task_i + 1}_accuracy"] = (float(np.mean([per_class[seen.index(c)] for c in task_classes]))
                                                       if all(c in seen for c in task_classes) else "")
            for task_i, score in enumerate(task_learned_accuracy):
                row[f"task_{task_i + 1}_accuracy_when_learned"] = score
        results.append(row)
        print(f"  {kind:<4} seed={seed} {stage_name:<6} accuracy={accuracy*100:5.1f}% "
              f"old-class forgetting={forgetting*100:5.1f} pp loss={loss:.4f}", flush=True)
    extra = {"tau_start": tau_start.tolist() if tau_start is not None else None,
             "tau_final": tau.tolist() if tau is not None else None}
    return results, extra


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["dataset"], r["model"], r["condition"], r.get("order", "canonical"), r["stage"]) for r in rows})
    summaries = []
    for dataset, model, condition, order, stage in keys:
        group = [r for r in rows if (r["dataset"], r["model"], r["condition"], r.get("order", "canonical"), r["stage"]) == (dataset, model, condition, order, stage)]
        row = {"dataset": dataset, "model": model, "condition": condition, "order": order, "stage": stage, "n_seeds": len(group)}
        metric_keys = {"accuracy", "macro_accuracy", "forgetting", "bwt", "current_task_accuracy",
                       "training_seconds", "loss", "epochs_trained", "validation_macro_accuracy",
                       "validation_loss", "mean_tau", "tau_drift_l2", "parameter_count"}
        metric_keys.update(k for r in group for k in r if k.startswith("task_") and "accuracy" in k)
        for key in sorted(metric_keys):
            values = np.asarray([float(r[key]) for r in group if r[key] != ""], dtype=float)
            row[f"{key}_mean"] = float(values.mean()) if len(values) else ""
            row[f"{key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else "")
        summaries.append(row)
    return summaries


def paired_joint_differences(rows: list[dict]) -> list[dict]:
    """Per-seed joint minus final incremental macro accuracy, in percentage points."""
    by_seed: dict[tuple[str, str, int, str], dict[str, float]] = {}
    joint_scores: dict[tuple[str, str, int], float] = {}
    for row in rows:
        key = (row["dataset"], row["model"], int(row["seed"]), row.get("order", "canonical"))
        if row["condition"] == "incremental" and row["stage"].startswith("Task "):
            cell = by_seed.setdefault(key, {})
            cell["incremental"] = float(row["macro_accuracy"])
        elif row["condition"] == "joint":
            joint_scores[key[:3]] = float(row["macro_accuracy"])
    return [{"dataset": dataset, "model": model, "seed": seed, "order": order,
             "joint_minus_incremental_macro_accuracy_pp": 100 * (joint_scores[(dataset, model, seed)] - v["incremental"])}
            for (dataset, model, seed, order), v in sorted(by_seed.items())
            if "incremental" in v and (dataset, model, seed) in joint_scores]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def plot_results(rows: list[dict], datasets: list[str], classes_by_dataset: dict[str, list[str]]) -> None:
    models = [m for m in ("LNN", "LTC", "CfC", "LSTM", "GRU") if any(r["model"] == m for r in rows)]
    colors = {"LNN": "#2563eb", "LTC": "#7c3aed", "CfC": "#db2777", "LSTM": "#f97316", "GRU": "#16a34a"}
    fig, axes = plt.subplots(len(datasets), 4, figsize=(21, 5.2 * len(datasets)), squeeze=False, constrained_layout=True)
    for di, dataset in enumerate(datasets):
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        canonical = [r for r in dataset_rows if r.get("order") == "canonical"]
        stages = sorted({r["stage"] for r in canonical if r["condition"] == "incremental"}, key=lambda s: int(s.split()[-1]))
        ax = axes[di, 0]
        for model in models:
            pts = [r for r in canonical if r["model"] == model and r["condition"] == "incremental"]
            means = [100 * np.mean([r["macro_accuracy"] for r in pts if r["stage"] == s]) for s in stages]
            stds = [100 * np.std([r["macro_accuracy"] for r in pts if r["stage"] == s], ddof=1) if len([r for r in pts if r["stage"] == s]) > 1 else 0 for s in stages]
            if means:
                ax.errorbar(range(1, len(stages)+1), means, yerr=stds, marker="o", capsize=4, linewidth=2, label=model, color=colors[model])
        ax.set(title=f"{dataset}: accuracy (canonical order)", xlabel="Learning stage", ylabel="Macro accuracy (%)", xticks=range(1, len(stages)+1), ylim=(0, 100)); ax.grid(alpha=.25); ax.legend(fontsize=8)

        ax = axes[di, 1]
        for model in models:
            pts = [r for r in canonical if r["model"] == model and r["condition"] == "incremental"]
            means = [100 * np.mean([r["forgetting"] for r in pts if r["stage"] == s]) for s in stages]
            stds = [100 * np.std([r["forgetting"] for r in pts if r["stage"] == s], ddof=1) if len([r for r in pts if r["stage"] == s]) > 1 else 0 for s in stages]
            if means:
                ax.errorbar(range(1, len(stages)+1), means, yerr=stds, marker="o", capsize=4, linewidth=2, label=model, color=colors[model])
        ax.set(title=f"{dataset}: old-class forgetting", xlabel="After learning task", ylabel="Mean accuracy drop (pp)", xticks=range(1, len(stages)+1)); ax.grid(alpha=.25); ax.legend(fontsize=8)

        ax = axes[di, 2]
        for i, model in enumerate(models):
            inc = [r["macro_accuracy"] * 100 for r in canonical if r["model"] == model and r["condition"] == "incremental" and r["stage"] == stages[-1]] if stages else []
            joint = [r["macro_accuracy"] * 100 for r in dataset_rows if r["model"] == model and r["condition"] == "joint"]
            if inc:
                ax.scatter(np.full(len(inc), i - .12), inc, color=colors[model], alpha=.7)
            if joint:
                ax.scatter(np.full(len(joint), i + .12), joint, facecolors="none", edgecolors=colors[model], alpha=.9)
            if inc and joint:
                ax.plot([i - .12, i + .12], [np.mean(inc), np.mean(joint)], color=colors[model], alpha=.5)
        ax.set(title=f"{dataset}: sequential vs joint", xlabel="Filled = incremental; hollow = joint", ylabel="Macro accuracy (%)", xticks=range(len(models)), xticklabels=models, ylim=(0, 100)); ax.grid(axis="y", alpha=.25); ax.tick_params(axis="x", labelrotation=25)

        ax = axes[di, 3]
        for model in models:
            pts = [r for r in canonical if r["model"] == model and r["condition"] == "incremental"]
            means = [np.mean([r["training_seconds"] for r in pts if r["stage"] == s]) for s in stages]
            if means:
                ax.plot(range(1, len(stages)+1), means, marker="o", label=model, color=colors[model])
        ax.set(title=f"{dataset}: training time per task", xlabel="Learning stage", ylabel="Seconds", xticks=range(1, len(stages)+1)); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Class-incremental benchmark: mean ± SD across seeds (canonical task order)", fontsize=16, fontweight="bold")
    path = OUT_DIR / "benchmark_dashboard.png"
    fig.savefig(path, dpi=180); plt.close(fig)

    for dataset in datasets:
        classes = classes_by_dataset[dataset]
        liquid_models = [m for m in ("LNN", "LTC", "CfC") if any(r["dataset"] == dataset and r["model"] == m for r in rows)]
        if not liquid_models:
            continue
        fig, axs = plt.subplots(1, len(liquid_models), figsize=(6 * len(liquid_models), max(4.5, len(classes) * .45)), squeeze=False, constrained_layout=True)
        for ax, model in zip(axs[0], liquid_models):
            stages = sorted({r["stage"] for r in rows if r["dataset"] == dataset and r["model"] == model and r["condition"] == "incremental" and r.get("order") == "canonical"}, key=lambda s: int(s.split()[-1]))
            acc = np.full((len(classes), len(stages)), np.nan)
            for stage_i, stage in enumerate(stages):
                group = [r for r in rows if r["dataset"] == dataset and r["model"] == model and r["condition"] == "incremental" and r.get("order") == "canonical" and r["stage"] == stage]
                for c, name in enumerate(classes):
                    class_key = f"{dataset.lower().replace('-', '_')}_acc_{name}"
                    vals = [float(r[class_key]) for r in group if r[class_key] != ""]
                    if vals:
                        acc[c, stage_i] = np.mean(vals) * 100
            im = ax.imshow(acc, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
            ax.set(title=f"{dataset}: {model}", xlabel="Learning stage", ylabel="Class", xticks=range(len(stages)), xticklabels=stages, yticks=range(len(classes)), yticklabels=classes)
            ax.tick_params(axis="x", labelrotation=25)
            for i in range(len(classes)):
                for j in range(len(stages)):
                    if not np.isnan(acc[i, j]):
                        ax.text(j, i, f"{acc[i,j]:.1f}%", ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(im, ax=axs.ravel().tolist(), label="Accuracy (%)")
        fig.savefig(OUT_DIR / f"{dataset.lower().replace('-', '_')}_liquid_models_class_accuracy.png", dpi=180); plt.close(fig)

    order_names = sorted({r.get("order", "canonical") for r in rows if r["condition"] == "incremental"})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(15, 4.5 * len(datasets)), squeeze=False, constrained_layout=True)
    for di, dataset in enumerate(datasets):
        drows = [r for r in rows if r["dataset"] == dataset and r["condition"] == "incremental"]
        final_stage = max((r["stage"] for r in drows), key=lambda s: int(s.split()[-1])) if drows else ""
        for ax, metric, title in ((axes[di, 0], "macro_accuracy", "Final average accuracy"), (axes[di, 1], "forgetting", "Final forgetting of old classes")):
            width = .8 / max(1, len(models))
            for mi, model in enumerate(models):
                means, errors = [], []
                for order in order_names:
                    vals = [float(r[metric]) * 100 for r in drows if r["model"] == model and r.get("order") == order and r["stage"] == final_stage]
                    means.append(float(np.mean(vals)) if vals else 0.0)
                    errors.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0)
                positions = np.arange(len(order_names)) - .4 + width * (mi + .5)
                if any(r["model"] == model for r in drows):
                    ax.bar(positions, means, width=width, yerr=errors, capsize=2, label=model, color=colors[model])
            ax.set(title=f"{dataset}: {title} by task order", ylabel="Percent (%)", xticks=range(len(order_names)), xticklabels=order_names); ax.grid(axis="y", alpha=.25)
            if di == 0 and metric == "macro_accuracy": ax.legend(ncol=len(models), fontsize=8)
    fig.suptitle("Sensitivity to task order (mean ± SD across seeds)", fontsize=15, fontweight="bold")
    fig.savefig(OUT_DIR / "task_order_sensitivity.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 3.7 * len(datasets)), squeeze=False, constrained_layout=True)
    for di, dataset in enumerate(datasets):
        ax = axes[di, 0]
        drows = [r for r in rows if r["dataset"] == dataset and r["condition"] == "incremental" and r.get("order") == "canonical"]
        stages = sorted({r["stage"] for r in drows}, key=lambda s: int(s.split()[-1]))
        for model in models:
            pts = [r for r in drows if r["model"] == model]
            means = [100 * np.mean([r["bwt"] for r in pts if r["stage"] == s]) for s in stages]
            stds = [100 * np.std([r["bwt"] for r in pts if r["stage"] == s], ddof=1) if len([r for r in pts if r["stage"] == s]) > 1 else 0 for s in stages]
            if means:
                ax.errorbar(range(1, len(stages)+1), means, yerr=stds, marker="o", capsize=3, label=model, color=colors[model])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=f"{dataset}: backward transfer (canonical order)", xlabel="After learning task", ylabel="BWT (percentage points)", xticks=range(1, len(stages)+1)); ax.grid(alpha=.25); ax.legend(ncol=len(models), fontsize=8)
    fig.savefig(OUT_DIR / "backward_transfer.png", dpi=180); plt.close(fig)


def write_experiment_report(rows: list[dict], datasets: list[str], models: list[str], seeds: list[int],
                            orders: int, min_epochs: int, max_epochs: int, patience: int,
                            min_delta_accuracy: float, min_delta_loss: float,
                            validation_fraction: float, hidden: int, device: torch.device,
                            dataset_meta: dict) -> None:
    lines = ["# 第二阶段机制筛查实验报告", "",
             "该报告由当前运行的 seed-level CSV 自动汇总。数值为均值 ± 样本标准差；这是观察性基准，不是对所有 LNN 的普遍性证明。", "",
             "## 运行配置", "",
             f"- 数据集：{', '.join(datasets)}", f"- 模型：{', '.join(models)}", f"- 种子：{seeds}",
             f"- 任务顺序：{orders} 种；每阶段 {min_epochs}–{max_epochs} epochs；验证集比例 {validation_fraction:.0%}；hidden size {hidden}；设备 `{device}`",
             f"- 早停：至少 {min_epochs} 轮；验证宏准确率改善 < {min_delta_accuracy*100:.2f} 个百分点且损失改善 < {min_delta_loss:g}，连续 {patience} 轮停止。",
             "- `not_confirmed_at_epoch_limit` 表示到达轮数上限但早停条件未触发；该结果不视为已确认收敛。", "",
             "## 数据概况", ""]
    for dataset in datasets:
        meta = dataset_meta[dataset]
        lines.append(f"- **{dataset}**：训练样本 {meta.get('train_windows', '?'):,}，测试样本 {meta.get('test_windows', '?'):,}；顺序：" + ", ".join(item["name"] for item in meta.get("task_orders", [])))
    lines.extend(["", "## Canonical 顺序最终结果", "",
                  "准确率与遗忘均以百分点表示；遗忘是历史最佳表现相对当前的下降。BWT 为平均当前任务表现减去旧任务刚学完时的表现，负值表示负向后向迁移。", "",
                  "| 数据集 | 模型 | 最终宏准确率 | 最终旧类遗忘 | 最终 BWT | 参数量 |", "|---|---|---:|---:|---:|---:|"])
    for dataset in datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset and r["condition"] == "incremental" and r.get("order") == "canonical"]
        final_stage = max((r["stage"] for r in ds_rows), key=lambda s: int(s.split()[-1]))
        for model in models:
            group = [r for r in ds_rows if r["model"] == model and r["stage"] == final_stage]
            if not group:
                continue
            def fmt(key: str, scale: float = 100) -> str:
                values = np.asarray([float(r[key]) * scale for r in group], dtype=float)
                return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}" if len(values) > 1 else f"{values.mean():.2f}"
            lines.append(f"| {dataset} | {model} | {fmt('macro_accuracy')}% | {fmt('forgetting')} | {fmt('bwt')} | {np.mean([r['parameter_count'] for r in group]):.0f} |")
    lines.extend(["", "## 逐种子结果与收敛状态", "",
                  "以下列出 canonical 顺序每个模型最终任务的逐种子宏准确率。三次结果方向不一致时应视为筛查信号尚不稳定；`seed_level_results.csv` 保留所有任务的逐种子指标。", "",
                  "| 数据集 | 模型 | 种子 | 最终宏准确率 | 实际轮数 | 停止原因 | 收敛状态 |",
                  "|---|---|---:|---:|---:|---|---|"])
    for dataset in datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset and r["condition"] == "incremental" and r.get("order") == "canonical"]
        final_stage = max((r["stage"] for r in ds_rows), key=lambda s: int(s.split()[-1]))
        for model in models:
            for row in sorted((r for r in ds_rows if r["model"] == model and r["stage"] == final_stage),
                              key=lambda r: int(r["seed"])):
                lines.append(f"| {dataset} | {model} | {row['seed']} | {float(row['macro_accuracy'])*100:.2f}% | "
                             f"{row['epochs_trained']} | {row['stop_reason']} | {row['convergence_status']} |")
    lines.extend(["", "## 任务顺序敏感性", ""])
    for dataset in datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset and r["condition"] == "incremental"]
        final_stage = max((r["stage"] for r in ds_rows), key=lambda s: int(s.split()[-1]))
        lines.append(f"### {dataset}")
        for model in models:
            results = []
            for order in sorted({r.get("order", "canonical") for r in ds_rows}):
                values = [100 * float(r["macro_accuracy"]) for r in ds_rows if r["model"] == model and r.get("order") == order and r["stage"] == final_stage]
                if values:
                    results.append(f"{order}: {np.mean(values):.1f}%")
            if results:
                lines.append(f"- **{model}** — " + "; ".join(results))
        lines.append("")
    lines.extend(["## 联合训练对照诊断", ""])
    for dataset in datasets:
        meta = dataset_meta[dataset]
        n_classes = len(meta.get("classes", []))
        chance = 1 / n_classes if n_classes else 0
        joint = [float(r["macro_accuracy"]) for r in rows if r["dataset"] == dataset and r["condition"] == "joint"]
        if joint:
            joint_mean = float(np.mean(joint))
            lines.append(f"- **{dataset}**：所有模型/种子的联合训练宏准确率均值 {joint_mean*100:.1f}%；均匀随机猜测基线为 {chance*100:.1f}%。")
            if joint_mean <= chance + .05:
                lines.append("  - 联合训练对照接近随机水平；该配置尚未证明模型充分学会任务，因此此数据集上的遗忘数值只能视为训练不足下的探索结果，不能单独作为灾难性遗忘证据。")
    lines.extend(["", "## 先前 UCI-HAR/WISDM 基线（独立实验配置）", "",
                  "归档于 `experiments/archive/stage1/outputs/benchmark/`：自定义简化 LNN、LSTM、GRU，5 个种子、每任务 5 轮、固定任务顺序。没有标准 LTC/CfC，因此不与本次结果合并。", "",
                  "| 数据集 | 自定义 LNN Task 2 遗忘 | Task 3 遗忘 | 最终增量宏准确率 | 联合训练宏准确率 |",
                  "|---|---:|---:|---:|---:|",
                  "| UCI-HAR | 78.9 ± 3.2 pp | 86.4 ± 2.6 pp | 33.94 ± 1.13% | 68.65 ± 2.75% |",
                  "| WISDM | 74.7 ± 4.4 pp | 75.4 ± 1.5 pp | 25.68 ± 0.98% | 46.63 ± 1.65% |", "",
                  "这两组的联合训练表现高于 UWave 短跑，但任务顺序只有一种，液态模型也不是标准 LTC/CfC。"])
    lines.extend(["", "## 结论限制", "",
                  "- LNN 是原有自定义简化液态单元；LTC 和 CfC 使用 ncps 的 PyTorch 实现。它们是相关模型，不应混称成单一架构。",
                  f"- {orders} 种任务顺序与 {len(seeds)} 个训练种子可以检查部分顺序/初始化敏感性，但数据集数和受试者划分仍有限。训练种子不是新的独立受试者样本。",
                  "- 任务顺序、无回放协议下观察到遗忘，不等于所有持续学习方法下都必然遗忘。机制因果解释仍需后续漂移分析和干预实验。", "",
                  "## 数据与软件来源", "",
                  "- [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)",
                  "- [WISDM 2019](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)",
                  "- [UWaveGestureLibrary (Zenodo)](https://zenodo.org/records/11206282)",
                  "- [ncps LTC/CfC reference implementation](https://github.com/mlech26l/ncps)", ""])
    (OUT_DIR / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUT_DIR
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=["UCI-HAR", "WISDM", "UWave"], default=["UCI-HAR", "WISDM", "UWave"])
    p.add_argument("--task-orders", type=int, choices=range(1, 10), default=3)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--min-epochs", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta-accuracy", type=float, default=0.005)
    p.add_argument("--min-delta-loss", type=float, default=0.001)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--models", nargs="+", choices=["LNN", "LTC", "CfC", "LSTM", "GRU"], default=["LNN", "LTC", "CfC", "LSTM", "GRU"])
    args = p.parse_args()
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs:
        p.error("--max-epochs must be greater than or equal to --min-epochs, and --min-epochs must be positive")
    if args.patience < 1 or not 0 < args.validation_fraction < 1:
        p.error("--patience must be positive and --validation-fraction must be between 0 and 1")
    if args.seeds != SEEDS:
        p.error("Stage 2 uses the fixed paired seeds 11, 22, and 33; omit --seeds to use them")
    OUT_DIR = Path(args.output_dir).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ("LTC" in args.models or "CfC" in args.models) and CfC is None:
        raise SystemExit("LTC/CfC require ncps. Install dependencies with `python -m pip install -r requirements.txt`.")
    print(f"Device: {device}; seeds: {args.seeds}; task orders: {args.task_orders}; "
          f"epochs/task: {args.min_epochs}-{args.max_epochs} (patience {args.patience}); torch: {torch.__version__}", flush=True)
    all_rows, dataset_meta, tau_records = [], {}, []
    classes_by_dataset = {"UCI-HAR": UCI_CLASSES, "WISDM": WISDM_CLASSES, "UWave": UWAVE_CLASSES}
    tasks_by_dataset = {"UCI-HAR": UCI_TASKS, "WISDM": WISDM_TASKS, "UWave": [[i, i + 1] for i in range(0, 8, 2)]}
    for dataset in args.datasets:
        print(f"\n=== Dataset: {dataset} ===", flush=True)
        if dataset == "UCI-HAR":
            x_train, y_train, x_test, y_test = load_uci_har()
            meta = {"classes": UCI_CLASSES, "train_windows": len(y_train), "test_windows": len(y_test), "official_subject_disjoint_split": True}
        elif dataset == "WISDM":
            x_train, y_train, x_test, y_test, meta = load_wisdm()
        else:
            x_train, y_train, x_test, y_test, meta = load_uwave()
        orders = task_orders(tasks_by_dataset[dataset], args.task_orders, 20261004)
        meta["task_orders"] = [{"name": name, "tasks": [[classes_by_dataset[dataset][c] for c in task] for task in order]} for name, order in orders]
        dataset_meta[dataset] = meta
        print(f"windows: train={len(y_train):,} test={len(y_test):,}; channels={x_train.shape[-1]} length={x_train.shape[1]}", flush=True)
        for seed in args.seeds:
            for model_kind in args.models:
                for condition in ("incremental", "joint"):
                    selected_orders = orders if condition == "incremental" else [("joint", tasks_by_dataset[dataset])]
                    for order_name, selected_tasks in selected_orders:
                        print(f"\n{dataset} | {model_kind} | {condition} | {order_name} | seed {seed}", flush=True)
                        rows, extra = run_condition(model_kind, x_train, y_train, x_test, y_test,
                                                    classes_by_dataset[dataset], selected_tasks, seed,
                                                    args.min_epochs, args.max_epochs, args.patience,
                                                    args.min_delta_accuracy, args.min_delta_loss,
                                                    args.validation_fraction, args.hidden, args.batch_size,
                                                    args.lr, device, condition, order_name)
                        for row in rows:
                            row["dataset"] = dataset
                            prefix = dataset.lower().replace("-", "_")
                            for name in classes_by_dataset[dataset]:
                                row[f"{prefix}_acc_{name}"] = row.pop(f"acc_{name}")
                            row["train_windows"] = len(y_train)
                            row["test_windows"] = len(y_test)
                            row["hidden_size"] = args.hidden
                            row["min_epochs_per_stage"] = args.min_epochs
                            row["max_epochs_per_stage"] = args.max_epochs
                            row["task_order_count"] = args.task_orders
                            all_rows.append(row)
                        # Persist completed configurations so a long GPU run is recoverable.
                        write_csv(OUT_DIR / "seed_level_results.csv", all_rows)
                        if model_kind == "LNN":
                            tau_records.append({"dataset": dataset, "condition": condition, "order": order_name, "seed": seed, **extra})
    summary = summarize(all_rows)
    write_csv(OUT_DIR / "seed_level_results.csv", all_rows)
    write_csv(OUT_DIR / "summary_mean_sd.csv", summary)
    write_csv(OUT_DIR / "paired_joint_differences.csv", paired_joint_differences(all_rows))
    with (OUT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"datasets": args.datasets, "models": args.models, "seeds": args.seeds, "task_orders": args.task_orders,
                   "min_epochs_per_task": args.min_epochs, "max_epochs_per_task": args.max_epochs,
                   "early_stopping_patience": args.patience,
                   "min_delta_macro_accuracy": args.min_delta_accuracy,
                   "min_delta_validation_loss": args.min_delta_loss,
                   "validation_fraction_from_training_data": args.validation_fraction,
                   "validation_split_seed": 20261004,
                   "test_data_used_for_early_stopping": False,
                   "hidden_size": args.hidden,
                   "batch_size": args.batch_size, "learning_rate": args.lr,
                   "device": str(device), "incremental_replay": False,
                   "evaluation": "macro and micro accuracy over all seen classes; task ID not given; BWT by task group",
                   "liquid_implementation": "LNN custom baseline; LTC and CfC from ncps 1.0.1"},
                  f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "liquid_time_constants.json").open("w", encoding="utf-8") as f:
        json.dump(tau_records, f, indent=2)
    plot_results(all_rows, args.datasets, classes_by_dataset)
    write_experiment_report(all_rows, args.datasets, args.models, args.seeds, args.task_orders,
                            args.min_epochs, args.max_epochs, args.patience,
                            args.min_delta_accuracy, args.min_delta_loss,
                            args.validation_fraction, args.hidden, device, dataset_meta)
    print(f"\nSaved benchmark outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
