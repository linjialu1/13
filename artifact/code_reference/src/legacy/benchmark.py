"""Stage 3: a small LNN mechanism screen against a GRU reference.

Run this file directly from PyCharm. It reuses the Phase 1/2 UCI-HAR official
train/test split, reserves training-only validation and representation probes,
and writes incremental CSV outputs under this directory. It does not start a
run on import.
"""

from __future__ import annotations

import csv
import copy
import json
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

try:
    from ncps.torch import CfC
except ImportError:
    CfC = None


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "experiments" / "archive" / "2026-10-05" / "00_uci_har_stage3_baseline"
CONFIG_PATH = OUT_DIR / "run_config.json"
CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
TASKS = [[0, 1], [2, 3], [4, 5]]


class LiquidCell(nn.Module):
    """The same lightweight LNN cell used in the Phase 1/2 benchmark."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.logit_alpha = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, steps, _ = x.shape
        alpha = torch.sigmoid(self.logit_alpha)
        h = x.new_zeros((x.shape[0], self.logit_alpha.numel()))
        input_drive = self.input(x)
        for t in range(steps):
            proposal = torch.tanh(input_drive[:, t] + self.recurrent(h))
            h = h + alpha * (proposal - h)
        return h


class SequenceClassifier(nn.Module):
    def __init__(self, kind: str, input_size: int, hidden_size: int, n_classes: int):
        super().__init__()
        self.kind = kind
        if kind == "LNN":
            self.encoder = LiquidCell(input_size, hidden_size)
        elif kind == "CfC":
            if CfC is None:
                raise ImportError("CfC requires ncps==1.0.1; install the project-pinned dependency.")
            self.encoder = CfC(input_size, hidden_size, batch_first=True, return_sequences=True)
        elif kind == "GRU":
            self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unsupported model: {kind}")
        self.head = nn.Linear(hidden_size, n_classes)

    def representation_and_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "LNN":
            representation = self.encoder(x)
        elif self.kind == "CfC":
            sequence, _ = self.encoder(x)
            representation = sequence[:, -1]
        else:
            _, state = self.encoder(x)
            representation = state[-1]
        return representation, self.head(representation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.representation_and_logits(x)[1]


def load_uci_har() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Phase 1/2's official subject-disjoint split and normalization."""
    extracted = DATA_DIR / "UCI HAR Dataset"
    if not extracted.exists():
        raise FileNotFoundError(
            f"UCI-HAR is not extracted at {extracted}. Use the existing Phase 1/2 data folder."
        )
    signals = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for split in ("train", "test"):
        folder = extracted / split / "Inertial Signals"
        channels = [np.loadtxt(folder / f"{name}_{split}.txt", dtype=np.float32) for name in signals]
        xs.append(np.stack(channels, axis=-1))
        ys.append(np.loadtxt(extracted / split / f"y_{split}.txt", dtype=np.int64) - 1)
    x_train, x_test = xs
    y_train, y_test = ys
    # Match Phase 1/2: training statistics only; never fit preprocessing on test.
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((x_train - mean) / std).astype(np.float32), y_train, \
        ((x_test - mean) / std).astype(np.float32), y_test


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_probe_and_task_pool(y: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Reserve fixed, stratified train-only samples for representation diagnostics."""
    rng_seed = int(config["diagnostic_split_seed"])
    fraction = float(config["diagnostic_fraction_per_class"])
    probe_parts: list[np.ndarray] = []
    pool_parts: list[np.ndarray] = []
    for cls in range(len(CLASSES)):
        class_indices = np.flatnonzero(y == cls).copy()
        rng = np.random.default_rng(rng_seed + cls)
        rng.shuffle(class_indices)
        n_probe = max(1, int(round(len(class_indices) * fraction)))
        if n_probe >= len(class_indices) - 1:
            raise ValueError(f"Not enough training examples in class {CLASSES[cls]} for probe split")
        probe_parts.append(class_indices[:n_probe])
        pool_parts.append(class_indices[n_probe:])
    return np.concatenate(pool_parts), np.concatenate(probe_parts)


def split_task_train_validation(y: np.ndarray, pool_indices: np.ndarray, classes: list[int],
                                fraction: float, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for cls in classes:
        indices = pool_indices[y[pool_indices] == cls].copy()
        rng = np.random.default_rng(split_seed + cls)
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(f"Class {CLASSES[cls]} needs at least two examples for validation splitting")
        n_validation = min(max(int(round(len(indices) * fraction)), 1), len(indices) - 1)
        validation_parts.append(indices[:n_validation])
        train_parts.append(indices[n_validation:])
    return np.concatenate(train_parts), np.concatenate(validation_parts)


def make_loader(x: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int,
                generator_seed: int) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(generator_seed)
    dataset = TensorDataset(torch.from_numpy(x[indices]), torch.from_numpy(y[indices]).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      pin_memory=torch.cuda.is_available(), generator=generator)


def labels_to_local(y: torch.Tensor, active_classes: list[int], device: torch.device) -> torch.Tensor:
    lookup = torch.full((len(CLASSES),), -1, dtype=torch.long, device=device)
    lookup[torch.as_tensor(active_classes, dtype=torch.long, device=device)] = torch.arange(
        len(active_classes), dtype=torch.long, device=device
    )
    return lookup[y]


def evaluate_validation(model: SequenceClassifier, x: np.ndarray, y: np.ndarray,
                        indices: np.ndarray, new_classes: list[int], device: torch.device,
                        batch_size: int) -> tuple[float, float]:
    """Early-stop only on the current task's held-out training examples."""
    model.eval()
    class_accuracies: list[float] = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for cls in new_classes:
            class_indices = indices[y[indices] == cls]
            if not len(class_indices):
                continue
            correct = 0
            for start in range(0, len(class_indices), batch_size):
                batch = class_indices[start:start + batch_size]
                xx = torch.from_numpy(x[batch]).to(device)
                yy = torch.from_numpy(y[batch]).long().to(device)
                logits = model(xx)[:, new_classes]
                mapped = labels_to_local(yy, new_classes, device)
                total_loss += float(nn.functional.cross_entropy(logits, mapped, reduction="sum").item())
                total_count += len(batch)
                correct += int((logits.argmax(1) == mapped).sum().item())
            class_accuracies.append(correct / len(class_indices))
    return float(np.mean(class_accuracies)), total_loss / max(total_count, 1)


def train_epoch(model: SequenceClassifier, loader: DataLoader, active_classes: list[int],
                optimizer: torch.optim.Optimizer, device: torch.device,
                frozen_cfc_recurrent: tuple[torch.Tensor, torch.Tensor, int] | None = None) -> float:
    model.train()
    losses: list[float] = []
    for xx, yy in loader:
        xx = xx.to(device, non_blocking=True)
        yy = yy.to(device, non_blocking=True)
        mapped = labels_to_local(yy, active_classes, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xx)[:, active_classes]
        loss = nn.functional.cross_entropy(logits, mapped)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if frozen_cfc_recurrent is not None:
            recurrent_weight, recurrent_anchor, input_size = frozen_cfc_recurrent
            with torch.no_grad():
                recurrent_weight[:, input_size:].copy_(recurrent_anchor)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def extract_representations(model: SequenceClassifier, x: np.ndarray, indices: np.ndarray,
                            device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        xx = torch.from_numpy(x[selected]).to(device)
        representations, _ = model.representation_and_logits(xx)
        chunks.append(representations.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float64)


@torch.no_grad()
def evaluate_test(model: SequenceClassifier, x: np.ndarray, y: np.ndarray, seen: list[int],
                  device: torch.device, batch_size: int) -> dict[int, float]:
    """Called only after a task has finished; values never affect training."""
    model.eval()
    per_class: dict[int, float] = {}
    class_tensor = torch.as_tensor(seen, dtype=torch.long, device=device)
    for cls in seen:
        indices = np.flatnonzero(y == cls)
        correct = 0
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            logits = model(torch.from_numpy(x[batch]).to(device))[:, seen]
            pred = class_tensor[logits.argmax(1)].cpu().numpy()
            correct += int((pred == cls).sum())
        per_class[cls] = correct / max(len(indices), 1)
    return per_class


def parameter_groups(model: SequenceClassifier) -> dict[str, list[tuple[str, torch.Tensor]]]:
    if model.kind == "LNN":
        return {
            "liquid_dynamics": [("encoder.logit_alpha", model.encoder.logit_alpha)],
            "encoder_input": [(f"encoder.input.{name}", p) for name, p in model.encoder.input.named_parameters()],
            "encoder_recurrent": [(f"encoder.recurrent.{name}", p) for name, p in model.encoder.recurrent.named_parameters()],
            "classifier": [(f"head.{name}", p) for name, p in model.head.named_parameters()],
        }
    if model.kind == "CfC":
        # CfC concatenates x_t and h_{t-1}; split the first backbone matrix
        # into input/state blocks for precise freeze and parameter diagnostics.
        first_weight = model.encoder.rnn_cell.backbone[0].weight
        recurrent = first_weight[:, model.encoder.input_size:]
        input_block = first_weight[:, :model.encoder.input_size]
        return {
            "cfc_recurrent_block": [("encoder.rnn_cell.backbone.0.weight[hidden_state_columns]", recurrent)],
            "cfc_input_block": [("encoder.rnn_cell.backbone.0.weight[input_columns]", input_block)],
            "cfc_other_cell": [
                (f"encoder.{name}", p)
                for name, p in model.encoder.rnn_cell.named_parameters()
                if name != "backbone.0.weight"
            ],
            "classifier": [(f"head.{name}", p) for name, p in model.head.named_parameters()],
        }
    names = dict(model.encoder.named_parameters())
    input_names = ("weight_ih_l0", "bias_ih_l0", "weight_ih_l0_reverse", "bias_ih_l0_reverse")
    recurrent_names = ("weight_hh_l0", "bias_hh_l0", "weight_hh_l0_reverse", "bias_hh_l0_reverse")
    return {
        "encoder_input": [(name, names[name]) for name in input_names if name in names],
        "encoder_recurrent": [(name, names[name]) for name in recurrent_names if name in names],
        "classifier": [(f"head.{name}", p) for name, p in model.head.named_parameters()],
    }


def snapshot_parameters(model: SequenceClassifier) -> dict[str, torch.Tensor]:
    groups = parameter_groups(model)
    return {
        group: torch.cat([p.detach().reshape(-1).cpu().to(torch.float64) for _, p in params]).clone()
        for group, params in groups.items()
    }


def parameter_drift_rows(model: SequenceClassifier, condition: str, task_number: int,
                         current: dict[str, torch.Tensor], previous: dict[str, torch.Tensor] | None,
                         task1: dict[str, torch.Tensor], alpha_task1: np.ndarray | None,
                         seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, values in current.items():
        prior = previous[group] if previous is not None else values
        change = torch.linalg.vector_norm(values - prior).item()
        cumulative = torch.linalg.vector_norm(values - task1[group]).item()
        previous_norm = torch.linalg.vector_norm(prior).item()
        base_norm = torch.linalg.vector_norm(task1[group]).item()
        row: dict[str, Any] = {
            "condition": condition, "model": model.kind, "seed": seed, "task_boundary": task_number,
            "parameter_group": group, "parameter_count": int(values.numel()),
            "parameter_l2_norm": float(torch.linalg.vector_norm(values).item()),
            "change_l2_since_previous_task": float(change),
            "relative_change_since_previous_task": float(change / max(previous_norm, 1e-12)),
            "cumulative_change_l2_since_task1": float(cumulative),
            "relative_cumulative_change_since_task1": float(cumulative / max(base_norm, 1e-12)),
        }
        if group == "liquid_dynamics":
            alpha_now = torch.sigmoid(model.encoder.logit_alpha.detach()).cpu().numpy().astype(np.float64)
            row.update({
                "mean_alpha": float(alpha_now.mean()),
                "alpha_l2_change_since_task1": float(np.linalg.norm(alpha_now - alpha_task1)),
                "mean_absolute_alpha_change_since_task1": float(np.mean(np.abs(alpha_now - alpha_task1))),
            })
        rows.append(row)
    return rows


def representation_drift_rows(condition: str, model_name: str, task_number: int,
                              old_classes: list[int], probe_y: np.ndarray,
                              current_representations: np.ndarray,
                              anchors: dict[int, np.ndarray], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cls in old_classes:
        positions = np.flatnonzero(probe_y == cls)
        current = current_representations[positions]
        reference = anchors[cls]
        eps = 1e-12
        current_norm = np.linalg.norm(current, axis=1)
        reference_norm = np.linalg.norm(reference, axis=1)
        cosine = np.sum(current * reference, axis=1) / np.maximum(current_norm * reference_norm, eps)
        sample_l2 = np.linalg.norm(current - reference, axis=1)
        ref_centroid = reference.mean(axis=0)
        current_centroid = current.mean(axis=0)
        centroid_shift = np.linalg.norm(current_centroid - ref_centroid)
        rows.append({
            "condition": condition, "model": model_name, "seed": seed,
            "task_boundary": task_number, "class_id": cls, "class_name": CLASSES[cls],
            "n_probe_samples": int(len(positions)),
            "mean_sample_cosine_distance_from_first_learned": float(np.mean(1.0 - cosine)),
            "mean_sample_l2_drift": float(np.mean(sample_l2)),
            "mean_sample_relative_l2_drift": float(np.mean(sample_l2 / np.maximum(reference_norm, eps))),
            "centroid_l2_shift": float(centroid_shift),
            "centroid_relative_l2_shift": float(centroid_shift / max(np.linalg.norm(ref_centroid), eps)),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_existing_rows(filename: str) -> list[dict[str, Any]]:
    path = OUT_DIR / filename
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_existing_bundle() -> dict[str, list[dict[str, Any]]]:
    files = {
        "epochs": "epoch_validation.csv",
        "boundaries": "task_boundary_metrics.csv",
        "per_class": "per_class_test_metrics.csv",
        "representations": "old_representation_drift.csv",
        "parameters": "parameter_group_drift.csv",
    }
    return {key: load_existing_rows(filename) for key, filename in files.items()}


def train_condition(model_name: str, condition: str, x_train: np.ndarray, y_train: np.ndarray,
                    x_test: np.ndarray, y_test: np.ndarray, pool_indices: np.ndarray,
                    probe_indices: np.ndarray, config: dict[str, Any], device: torch.device,
                    bundle: dict[str, list[dict[str, Any]]]) -> None:
    seed = int(config["seed"])
    set_seed(seed)
    model = SequenceClassifier(model_name, x_train.shape[-1], int(config["hidden_size"]), len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    seen: list[int] = []
    learned_test_accuracy: dict[int, float] = {}
    representation_anchors: dict[int, np.ndarray] = {}
    previous_parameters: dict[str, torch.Tensor] | None = None
    task1_parameters: dict[str, torch.Tensor] | None = None
    alpha_task1: np.ndarray | None = None
    frozen_cfc_recurrent: tuple[torch.Tensor, torch.Tensor, int] | None = None
    cfc_recurrent_hook: torch.utils.hooks.RemovableHandle | None = None
    patience = int(config["early_stopping_patience"])
    min_epochs = int(config["min_epochs_per_task"])
    max_epochs = int(config["max_epochs_per_task"])

    for task_index, new_classes in enumerate(TASKS, start=1):
        old_classes = list(seen)
        seen.extend(new_classes)
        train_indices, validation_indices = split_task_train_validation(
            y_train, pool_indices, new_classes,
            float(config["validation_fraction_from_task_training_pool"]),
            int(config["validation_split_seed"]) + task_index * 100,
        )
        if task_index == 2 and condition == "LNN_freeze_alpha_after_task1":
            model.encoder.logit_alpha.requires_grad_(False)
        elif task_index == 2 and condition == "LNN_freeze_recurrent_after_task1":
            for parameter in model.encoder.recurrent.parameters():
                parameter.requires_grad_(False)
        elif task_index == 2 and condition == "CfC_freeze_recurrent_after_task1":
            recurrent_weight = model.encoder.rnn_cell.backbone[0].weight
            input_size = int(model.encoder.input_size)
            gradient_mask = torch.ones_like(recurrent_weight)
            gradient_mask[:, input_size:] = 0
            cfc_recurrent_hook = recurrent_weight.register_hook(
                lambda grad, mask=gradient_mask: grad * mask
            )
            recurrent_anchor = recurrent_weight[:, input_size:].detach().clone()
            frozen_cfc_recurrent = (recurrent_weight, recurrent_anchor, input_size)

        best_key = (-float("inf"), -float("inf"))
        best_state: dict[str, torch.Tensor] | None = None
        best_optimizer_state: dict[str, Any] | None = None
        best_accuracy = -float("inf")
        best_loss = float("inf")
        stagnant = 0
        epoch_rows_for_task: list[dict[str, Any]] = []
        stop_reason = "max_epochs_reached"
        started = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            loader = make_loader(
                x_train, y_train, train_indices, int(config["batch_size"]),
                generator_seed=seed + task_index * 100_000 + epoch,
            )
            train_loss = train_epoch(
                model, loader, seen, optimizer, device,
                frozen_cfc_recurrent=frozen_cfc_recurrent,
            )
            val_accuracy, val_loss = evaluate_validation(
                model, x_train, y_train, validation_indices, new_classes, device,
                int(config["evaluation_batch_size"]),
            )
            accuracy_improved = val_accuracy - best_accuracy >= float(config["min_delta_macro_accuracy"])
            loss_improved = best_loss - val_loss >= float(config["min_delta_validation_loss"])
            if accuracy_improved or loss_improved:
                best_accuracy = max(best_accuracy, val_accuracy)
                best_loss = min(best_loss, val_loss)
                stagnant = 0
            else:
                stagnant += 1
            checkpoint_key = (val_accuracy, -val_loss)
            if checkpoint_key > best_key:
                best_key = checkpoint_key
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            epoch_row = {
                "condition": condition, "model": model_name, "seed": seed,
                "task": task_index, "epoch": epoch, "train_loss": train_loss,
                "validation_new_class_macro_accuracy": val_accuracy,
                "validation_new_class_loss": val_loss,
                "accuracy_improved_by_threshold": bool(accuracy_improved),
                "loss_improved_by_threshold": bool(loss_improved),
                "stagnant_epochs": stagnant,
                "stop_reason": "in_progress",
                "convergence_status": "in_progress",
            }
            epoch_rows_for_task.append(epoch_row)
            bundle["epochs"].append(epoch_row)
            write_csv(OUT_DIR / "epoch_validation.csv", bundle["epochs"])
            print(
                f"{condition}: Task {task_index}/{len(TASKS)} epoch {epoch:02d}/{max_epochs} "
                f"val_macro_acc={val_accuracy:.4f} val_loss={val_loss:.4f}",
                flush=True,
            )
            if epoch >= min_epochs and stagnant >= patience:
                stop_reason = "validation_plateau"
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        if best_optimizer_state is not None:
            optimizer.load_state_dict(best_optimizer_state)
        epochs_trained = len(epoch_rows_for_task)
        convergence_status = "converged" if stop_reason == "validation_plateau" else "not_confirmed_at_epoch_limit"
        for row in epoch_rows_for_task:
            row["stop_reason"] = stop_reason
            row["convergence_status"] = convergence_status

        current_parameters = snapshot_parameters(model)
        if task_index == 1:
            task1_parameters = {name: values.clone() for name, values in current_parameters.items()}
            if model_name == "LNN":
                alpha_task1 = torch.sigmoid(model.encoder.logit_alpha.detach()).cpu().numpy().astype(np.float64)
        assert task1_parameters is not None
        bundle["parameters"].extend(parameter_drift_rows(
            model, condition, task_index, current_parameters, previous_parameters,
            task1_parameters, alpha_task1, seed,
        ))
        previous_parameters = {name: values.clone() for name, values in current_parameters.items()}

        probe_representations = extract_representations(
            model, x_train, probe_indices, device, int(config["evaluation_batch_size"])
        )
        probe_y = y_train[probe_indices]
        for cls in new_classes:
            positions = np.flatnonzero(probe_y == cls)
            representation_anchors[cls] = probe_representations[positions].copy()
        if old_classes:
            bundle["representations"].extend(representation_drift_rows(
                condition, model_name, task_index, old_classes, probe_y,
                probe_representations, representation_anchors, seed,
            ))

        per_class_test = evaluate_test(
            model, x_test, y_test, seen, device, int(config["evaluation_batch_size"])
        )
        for cls in new_classes:
            learned_test_accuracy[cls] = per_class_test[cls]
        old_values = [per_class_test[cls] for cls in old_classes]
        new_values = [per_class_test[cls] for cls in new_classes]
        old_forgetting = [learned_test_accuracy[cls] - per_class_test[cls] for cls in old_classes]
        bundle["boundaries"].append({
            "condition": condition, "model": model_name, "seed": seed,
            "task_boundary": task_index,
            "new_classes": ";".join(CLASSES[cls] for cls in new_classes),
            "seen_classes": ";".join(CLASSES[cls] for cls in seen),
            "epochs_trained": epochs_trained, "stop_reason": stop_reason,
            "convergence_status": convergence_status,
            "old_class_test_macro_accuracy": float(np.mean(old_values)) if old_values else "",
            "new_class_test_macro_accuracy": float(np.mean(new_values)),
            "all_seen_test_macro_accuracy": float(np.mean(list(per_class_test.values()))),
            "mean_old_class_forgetting_from_when_learned": float(np.mean(old_forgetting)) if old_forgetting else "",
            "task_training_seconds": time.perf_counter() - started,
        })
        for cls in seen:
            bundle["per_class"].append({
                "condition": condition, "model": model_name, "seed": seed,
                "task_boundary": task_index, "class_id": cls, "class_name": CLASSES[cls],
                "test_accuracy": per_class_test[cls],
                "accuracy_drop_from_when_learned": learned_test_accuracy[cls] - per_class_test[cls],
            })

        # Save at every task boundary so an interrupted desktop run leaves usable records.
        write_csv(OUT_DIR / "epoch_validation.csv", bundle["epochs"])
        write_csv(OUT_DIR / "task_boundary_metrics.csv", bundle["boundaries"])
        write_csv(OUT_DIR / "per_class_test_metrics.csv", bundle["per_class"])
        write_csv(OUT_DIR / "old_representation_drift.csv", bundle["representations"])
        write_csv(OUT_DIR / "parameter_group_drift.csv", bundle["parameters"])
        print(
            f"{condition}: Task {task_index}, epochs={epochs_trained}, {convergence_status}, "
            f"new test macro acc={np.mean(new_values):.3f}", flush=True,
        )


def write_report(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> None:
    lines = [
        "# Stage 3：LNN 机制快速筛查",
        "",
        "本报告由 benchmark.py 在训练完成后生成。单随机种子结果仅用于方向筛查，不支持统计显著性或普遍性结论。",
        "",
        "## 配置",
        "",
        f"- 数据集：UCI-HAR；种子：{config['seed']}；隐藏维度：{config['hidden_size']}",
        "- 条件：LNN 正常更新、LNN 任务 1 后冻结 logit_alpha、LNN 任务 1 后冻结循环权重、GRU 正常更新。",
        f"- 早停：至少 {config['min_epochs_per_task']} 轮，最多 {config['max_epochs_per_task']} 轮；验证宏准确率阈值 {config['min_delta_macro_accuracy']:.3f}、损失阈值 {config['min_delta_validation_loss']:.3f}，连续 {config['early_stopping_patience']} 轮均未改善时停止。",
        "- 验证集和表示诊断样本均来自官方训练集；诊断样本不训练、不参与早停。官方测试集只在每个任务训练结束后评估，不参与训练决策。",
        "",
        "## 任务边界结果",
        "",
        "| 条件 | 任务 | 训练轮数 | 收敛状态 | 旧类测试宏准确率 | 新类测试宏准确率 | 旧类遗忘 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in bundle["boundaries"]:
        old_acc = row["old_class_test_macro_accuracy"]
        old_text = "—" if old_acc == "" else f"{100 * float(old_acc):.2f}%"
        forgetting = row["mean_old_class_forgetting_from_when_learned"]
        forgetting_text = "—" if forgetting == "" else f"{100 * float(forgetting):.2f} pp"
        lines.append(
            f"| {row['condition']} | {row['task_boundary']} | {row['epochs_trained']} | {row['convergence_status']} | "
            f"{old_text} | {100 * float(row['new_class_test_macro_accuracy']):.2f}% | {forgetting_text} |"
        )
    lines += [
        "",
        "## 旧类表示漂移",
        "",
        "每行先按类别计算，再对旧类别求均值；Task 2/3 均相对该类首次学完时的表示。",
        "",
        "| 条件 | 任务边界 | 旧类别数 | 平均余弦距离 | 平均相对 L2 漂移 | 平均中心相对位移 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rep_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in bundle["representations"]:
        key = (str(row["condition"]), int(row["task_boundary"]))
        rep_groups.setdefault(key, []).append(row)
    for (condition, task), rows in sorted(rep_groups.items()):
        cosine = np.mean([float(row["mean_sample_cosine_distance_from_first_learned"]) for row in rows])
        rel_l2 = np.mean([float(row["mean_sample_relative_l2_drift"]) for row in rows])
        centroid = np.mean([float(row["centroid_relative_l2_shift"]) for row in rows])
        lines.append(f"| {condition} | {task} | {len(rows)} | {cosine:.4f} | {rel_l2:.4f} | {centroid:.4f} |")

    lines += [
        "",
        "## LNN 参数组变化",
        "",
        "相对变化以该组上一任务边界的参数 L2 范数为分母；不同模型、不同参数组不作同尺度因果比较。",
        "",
        "| 条件 | 边界 | 参数组 | 相对变化范数 | 累计相对变化（相对 Task 1） |",
        "|---|---:|---|---:|---:|",
    ]
    for row in bundle["parameters"]:
        if row["model"] != "LNN" or int(row["task_boundary"]) == 1:
            continue
        lines.append(
            f"| {row['condition']} | {row['task_boundary']} | {row['parameter_group']} | "
            f"{100 * float(row['relative_change_since_previous_task']):.2f}% | "
            f"{100 * float(row['relative_cumulative_change_since_task1']):.2f}% |"
        )

    by_condition_task = {
        (str(row["condition"]), int(row["task_boundary"])): row for row in bundle["boundaries"]
    }
    full_task3 = by_condition_task.get(("LNN_full_update", 3))
    freeze_task3 = by_condition_task.get(("LNN_freeze_alpha_after_task1", 3))
    full_task2 = by_condition_task.get(("LNN_full_update", 2))
    freeze_task2 = by_condition_task.get(("LNN_freeze_alpha_after_task1", 2))
    if full_task3 and freeze_task3 and full_task2 and freeze_task2:
        task2_gain = 100 * (
            float(freeze_task2["old_class_test_macro_accuracy"])
            - float(full_task2["old_class_test_macro_accuracy"])
        )
        task3_gain = 100 * (
            float(freeze_task3["old_class_test_macro_accuracy"])
            - float(full_task3["old_class_test_macro_accuracy"])
        )
        full_rep2 = rep_groups.get(("LNN_full_update", 2), [])
        freeze_rep2 = rep_groups.get(("LNN_freeze_alpha_after_task1", 2), [])
        full_rep3 = rep_groups.get(("LNN_full_update", 3), [])
        freeze_rep3 = rep_groups.get(("LNN_freeze_alpha_after_task1", 3), [])
        def mean_cos(rows: list[dict[str, Any]]) -> float:
            return float(np.mean([float(row["mean_sample_cosine_distance_from_first_learned"]) for row in rows]))
        if full_rep2 and freeze_rep2 and full_rep3 and freeze_rep3:
            lines += [
                "",
                "## 初步判读",
                "",
                f"- 冻结 alpha 相对 LNN 正常组的旧类测试宏准确率差：Task 2 {task2_gain:+.2f} 个百分点，Task 3 {task3_gain:+.2f} 个百分点。",
                f"- 旧类平均表示余弦距离：Task 2 正常组 {mean_cos(full_rep2):.4f}、冻结组 {mean_cos(freeze_rep2):.4f}；Task 3 正常组 {mean_cos(full_rep3):.4f}、冻结组 {mean_cos(freeze_rep3):.4f}。",
                "- 本次冻结消除了 alpha 后续变化，但没有明显减少旧类表示漂移或旧类遗忘。结合旧类测试准确率接近零，这个单一干预没有支持“alpha 漂移是主要遗忘原因”的假设。",
                "- 这不否定表示漂移/参数变化作为研究方向。LNN 正常组的循环权重和分类头在后续任务变化较大，后续可优先针对循环权重保护做一个同样小的干预；这仍需新实验验证。",
            ]
    baseline_task3 = by_condition_task.get(("LNN_full_update", 3))
    recurrent_task3 = by_condition_task.get(("LNN_freeze_recurrent_after_task1", 3))
    baseline_task2 = by_condition_task.get(("LNN_full_update", 2))
    recurrent_task2 = by_condition_task.get(("LNN_freeze_recurrent_after_task1", 2))
    if baseline_task2 and recurrent_task2 and baseline_task3 and recurrent_task3:
        task2_gain = 100 * (
            float(recurrent_task2["old_class_test_macro_accuracy"])
            - float(baseline_task2["old_class_test_macro_accuracy"])
        )
        task3_gain = 100 * (
            float(recurrent_task3["old_class_test_macro_accuracy"])
            - float(baseline_task3["old_class_test_macro_accuracy"])
        )
        full_rep2 = rep_groups.get(("LNN_full_update", 2), [])
        recurrent_rep2 = rep_groups.get(("LNN_freeze_recurrent_after_task1", 2), [])
        full_rep3 = rep_groups.get(("LNN_full_update", 3), [])
        recurrent_rep3 = rep_groups.get(("LNN_freeze_recurrent_after_task1", 3), [])
        if full_rep2 and recurrent_rep2 and full_rep3 and recurrent_rep3:
            baseline_class_task3 = {
                int(row["class_id"]): float(row["test_accuracy"])
                for row in bundle["per_class"]
                if row["condition"] == "LNN_full_update" and int(row["task_boundary"]) == 3
            }
            recurrent_class_task3 = {
                int(row["class_id"]): float(row["test_accuracy"])
                for row in bundle["per_class"]
                if row["condition"] == "LNN_freeze_recurrent_after_task1" and int(row["task_boundary"]) == 3
            }
            old_class_differences = [
                (cls, baseline_class_task3[cls], recurrent_class_task3[cls])
                for cls in range(4)
                if cls in baseline_class_task3 and cls in recurrent_class_task3
            ]
            per_class_text = "; ".join(
                f"{CLASSES[cls]} {100 * baseline:.1f}%→{100 * recurrent:.1f}%"
                for cls, baseline, recurrent in old_class_differences
            )
            lines += [
                "",
                "### 冻结循环权重的增量结果",
                "",
                f"- 相对 LNN 正常组，冻结循环权重后的旧类测试宏准确率差：Task 2 {task2_gain:+.2f} 个百分点，Task 3 {task3_gain:+.2f} 个百分点。",
                f"- 旧类平均表示余弦距离：Task 2 正常组 {mean_cos(full_rep2):.4f}、冻结循环权重组 {mean_cos(recurrent_rep2):.4f}；Task 3 正常组 {mean_cos(full_rep3):.4f}、冻结循环权重组 {mean_cos(recurrent_rep3):.4f}。",
                f"- Task 3 逐类旧类测试准确率（正常组→冻结循环权重组）：{per_class_text}。宏平均改善集中在少数类别，不能当作所有旧类都恢复。",
                "- 冻结循环权重条件下，LNN 的 alpha 仍可学习；parameter_group_drift.csv 可核对循环权重变化和 alpha 的实际变化。",
            ]
    lines += [
        "",
        "## 解读边界",
        "",
        "冻结 alpha 没有显示稳定收益；冻结循环权重则在 Task 3 降低了平均旧类表示漂移，并提高了旧类宏准确率，但 Task 2 旧类仍为 0%，Task 3 的准确率改善也集中在 WALKING_DOWNSTAIRS。当前可视为值得用额外种子复查的初步信号，不能视为普遍的因果结论。每个条件只有一个种子。GRU 是架构参照，不与 LNN 时间常数作逐参数等价比较。",
        "",
        "详见 epoch_validation.csv、task_boundary_metrics.csv、per_class_test_metrics.csv、old_representation_drift.csv 和 parameter_group_drift.csv。",
    ]
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(bundle: dict[str, list[dict[str, Any]]]) -> None:
    rows = bundle["boundaries"]
    if not rows:
        return
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        axes[0].plot([int(row["task_boundary"]) for row in subset],
                     [100 * float(row["new_class_test_macro_accuracy"]) for row in subset],
                     marker="o", label=condition)
        old = [row for row in subset if row["old_class_test_macro_accuracy"] != ""]
        if old:
            axes[1].plot([int(row["task_boundary"]) for row in old],
                         [100 * float(row["old_class_test_macro_accuracy"]) for row in old],
                         marker="o", label=condition)
    axes[0].set(title="New-class test macro accuracy", xlabel="Task boundary", ylabel="Accuracy (%)")
    axes[1].set(title="Old-class test macro accuracy", xlabel="Task boundary", ylabel="Accuracy (%)")
    for axis in axes:
        axis.set_xticks([1, 2, 3])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_boundary_accuracy.png", dpi=160)
    plt.close(fig)


def write_multi_seed_report(bundle: dict[str, list[dict[str, Any]]],
                            config: dict[str, Any]) -> None:
    """Generate per-seed results and paired intervention summaries without pooling rows."""
    condition_names = list(dict.fromkeys(str(row["condition"]) for row in bundle["boundaries"]))
    seeds = sorted({int(row["seed"]) for row in bundle["boundaries"]})
    boundary_index = {
        (str(row["condition"]), int(row["seed"]), int(row["task_boundary"])): row
        for row in bundle["boundaries"]
    }
    rep_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in bundle["representations"]:
        key = (str(row["condition"]), int(row["seed"]), int(row["task_boundary"]))
        rep_groups.setdefault(key, []).append(row)

    paired_rows: list[dict[str, Any]] = []
    comparison_specs = [
        ("LNN_full_update", "LNN_freeze_alpha_after_task1", "freeze_alpha"),
        ("LNN_full_update", "LNN_freeze_recurrent_after_task1", "freeze_recurrent"),
        ("LNN_full_update", "CfC_full_update", "CfC_vs_LNN"),
        ("CfC_full_update", "CfC_freeze_recurrent_after_task1", "CfC_freeze_recurrent"),
        ("LNN_full_update", "GRU_full_update", "GRU_vs_LNN"),
    ]
    for baseline_condition, treatment_condition, label in comparison_specs:
        for seed in seeds:
            for task in (2, 3):
                baseline = boundary_index.get((baseline_condition, seed, task))
                treatment = boundary_index.get((treatment_condition, seed, task))
                if not baseline or not treatment:
                    continue
                baseline_rep = rep_groups.get((baseline_condition, seed, task), [])
                treatment_rep = rep_groups.get((treatment_condition, seed, task), [])
                baseline_cos = float(np.mean([
                    float(row["mean_sample_cosine_distance_from_first_learned"]) for row in baseline_rep
                ])) if baseline_rep else float("nan")
                treatment_cos = float(np.mean([
                    float(row["mean_sample_cosine_distance_from_first_learned"]) for row in treatment_rep
                ])) if treatment_rep else float("nan")
                paired_rows.append({
                    "comparison": label, "seed": seed, "task_boundary": task,
                    "old_test_macro_accuracy_change_pp": 100 * (
                        float(treatment["old_class_test_macro_accuracy"])
                        - float(baseline["old_class_test_macro_accuracy"])
                    ),
                    "new_test_macro_accuracy_change_pp": 100 * (
                        float(treatment["new_class_test_macro_accuracy"])
                        - float(baseline["new_class_test_macro_accuracy"])
                    ),
                    "old_representation_cosine_distance_change": (
                        "" if label in {"CfC_vs_LNN", "GRU_vs_LNN"} else treatment_cos - baseline_cos
                    ),
                    "old_representation_cosine_distance_baseline": baseline_cos,
                    "old_representation_cosine_distance_intervention": treatment_cos,
                    "convergence_status_intervention": treatment["convergence_status"],
                    "baseline_condition": baseline_condition,
                    "comparison_condition": treatment_condition,
                })
    write_csv(OUT_DIR / "paired_intervention_by_seed.csv", paired_rows)
    summary_rows: list[dict[str, Any]] = []
    for comparison in ("freeze_alpha", "freeze_recurrent", "CfC_vs_LNN", "CfC_freeze_recurrent", "GRU_vs_LNN"):
        for task in (2, 3):
            rows = [row for row in paired_rows
                    if row["comparison"] == comparison and int(row["task_boundary"]) == task]
            if not rows:
                continue
            def mean_sd(field: str) -> tuple[float | str, float | str]:
                values = np.asarray([float(row[field]) for row in rows if row[field] != ""], dtype=np.float64)
                if not len(values):
                    return "", ""
                return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else ""
            old_mean, old_sd = mean_sd("old_test_macro_accuracy_change_pp")
            new_mean, new_sd = mean_sd("new_test_macro_accuracy_change_pp")
            rep_mean, rep_sd = mean_sd("old_representation_cosine_distance_change")
            summary_rows.append({
                "comparison": comparison, "task_boundary": task, "n_paired_seeds": len(rows),
                "old_test_macro_accuracy_change_pp_mean": old_mean,
                "old_test_macro_accuracy_change_pp_sd": old_sd,
                "new_test_macro_accuracy_change_pp_mean": new_mean,
                "new_test_macro_accuracy_change_pp_sd": new_sd,
                "old_representation_cosine_distance_change_mean": rep_mean,
                "old_representation_cosine_distance_change_sd": rep_sd,
            })
    write_csv(OUT_DIR / "paired_intervention_summary.csv", summary_rows)

    lines = [
        "# Stage 3：LNN 机制快速筛查",
        "",
        "UCI-HAR 官方受试者隔离划分；旧类不回放。保留逐种子结果并报告配对条件比较；CfC 使用 ncps 1.0.1 官方实现。",
        "",
        "## 配置",
        "",
        f"- 已有配对记录种子：{', '.join(map(str, seeds))}；本次 CfC/GRU 运行种子：{', '.join(map(str, config.get('seeds_to_run', [config['seed']]))) }；隐藏维度：{config['hidden_size']}。",
        "- 条件：LNN 正常更新、LNN Task 1 后冻结 alpha、LNN 冻结循环权重、CfC 正常更新、CfC Task 1 后冻结循环块、GRU 正常更新。CfC 冻结范围为 CfC 首层拼接映射中的隐藏状态列。",
        f"- 早停：至少 {config['min_epochs_per_task']} 轮，最多 {config['max_epochs_per_task']} 轮；验证宏准确率阈值 {config['min_delta_macro_accuracy']:.3f}、验证损失阈值 {config['min_delta_validation_loss']:.3f}，连续 {config['early_stopping_patience']} 轮两项都未达阈值时停止。",
        "- 验证集与表示诊断样本只来自官方训练集；诊断样本不参与训练或早停。官方测试集只在每个任务训练结束后评估。",
        "",
        "## 每种子任务边界结果",
        "",
        "| 条件 | 种子 | 任务 | 轮数 | 收敛状态 | 旧类测试宏准确率 | 新类测试宏准确率 | 旧类遗忘 |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in sorted(bundle["boundaries"], key=lambda r: (str(r["condition"]), int(r["seed"]), int(r["task_boundary"]))):
        old = row["old_class_test_macro_accuracy"]
        forgetting = row["mean_old_class_forgetting_from_when_learned"]
        old_text = "—" if old == "" else f"{100 * float(old):.2f}%"
        forgetting_text = "—" if forgetting == "" else f"{100 * float(forgetting):.2f} pp"
        lines.append(
            f"| {row['condition']} | {row['seed']} | {row['task_boundary']} | {row['epochs_trained']} | "
            f"{row['convergence_status']} | {old_text} | {100 * float(row['new_class_test_macro_accuracy']):.2f}% | "
            f"{forgetting_text} |"
        )

    lines += [
        "",
        "## 按种子配对的条件比较",
        "",
        "准确率差为“比较组 - 对照组”，单位为百分点；表示距离差为余弦距离之差，负值代表比较组漂移较小。CfC_freeze_recurrent 比较同架构；不同架构的潜表示漂移不直接相减，因此 CfC_vs_LNN 与 GRU_vs_LNN 只报告准确率差。单种子 SD 不适用。",
        "",
        "| 比较组 | 任务 | 配对种子数 | 旧类准确率差均值 ± SD | 新类准确率差均值 ± SD | 表示距离差均值 ± SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def format_mean_sd(mean_value: Any, sd_value: Any, precision: int) -> str:
            if mean_value == "":
                return "not compared across architectures"
            mean_text = f"{float(mean_value):+.{precision}f}"
            if sd_value == "":
                return f"{mean_text} (n=1)"
            return f"{mean_text} ± {float(sd_value):.{precision}f}"

        lines.append(
            f"| {row['comparison']} | {row['task_boundary']} | {row['n_paired_seeds']} | "
            f"{format_mean_sd(row['old_test_macro_accuracy_change_pp_mean'], row['old_test_macro_accuracy_change_pp_sd'], 2)} pp | "
            f"{format_mean_sd(row['new_test_macro_accuracy_change_pp_mean'], row['new_test_macro_accuracy_change_pp_sd'], 2)} pp | "
            f"{format_mean_sd(row['old_representation_cosine_distance_change_mean'], row['old_representation_cosine_distance_change_sd'], 4)} |"
        )

    lines += [
        "",
        "## 旧类表示漂移（每种子）",
        "",
        "| 条件 | 种子 | 任务边界 | 类别数 | 平均余弦距离 | 平均相对 L2 漂移 | 中心相对位移 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (condition, seed, task), rows in sorted(rep_groups.items()):
        cosine = np.mean([float(row["mean_sample_cosine_distance_from_first_learned"]) for row in rows])
        rel_l2 = np.mean([float(row["mean_sample_relative_l2_drift"]) for row in rows])
        centroid = np.mean([float(row["centroid_relative_l2_shift"]) for row in rows])
        lines.append(f"| {condition} | {seed} | {task} | {len(rows)} | {cosine:.4f} | {rel_l2:.4f} | {centroid:.4f} |")

    lines += [
        "",
        "## 解读边界",
        "",
        "本次 CfC 冻结组只在一个种子上筛查：冻结后循环块范数保持不变，但准确率和表示漂移未同步稳定改善，不能据此认定 CfC 的遗忘机制已被解释。GRU 是架构参照；单种子结果不支持稳定性或显著性结论。",
        "",
        "逐轮验证、每类别边界测试、分组参数范数分别见 epoch_validation.csv、per_class_test_metrics.csv、parameter_group_drift.csv；配对原始差值及汇总见 paired_intervention_by_seed.csv 和 paired_intervention_summary.csv。",
    ]
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_multi_seed_plot(bundle: dict[str, list[dict[str, Any]]]) -> None:
    conditions = list(dict.fromkeys(str(row["condition"]) for row in bundle["boundaries"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for condition in conditions:
        for axis_index, metric in enumerate(("new_class_test_macro_accuracy", "old_class_test_macro_accuracy")):
            means: list[float] = []
            errors: list[float] = []
            tasks: list[int] = []
            for task in (1, 2, 3):
                values = [
                    float(row[metric]) for row in bundle["boundaries"]
                    if row["condition"] == condition and int(row["task_boundary"]) == task
                    and row[metric] != ""
                ]
                if values:
                    tasks.append(task)
                    means.append(100 * float(np.mean(values)))
                    errors.append(100 * (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0))
            if tasks:
                axes[axis_index].errorbar(tasks, means, yerr=errors, marker="o", capsize=3, label=condition)
    axes[0].set(title="New-class test macro accuracy", xlabel="Task boundary", ylabel="Accuracy (%)")
    axes[1].set(title="Old-class test macro accuracy", xlabel="Task boundary", ylabel="Accuracy (%)")
    for axis in axes:
        axis.set_xticks([1, 2, 3])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_boundary_accuracy.png", dpi=160)
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    x_train, y_train, x_test, y_test = load_uci_har()
    pool_indices, probe_indices = split_probe_and_task_pool(y_train, config)
    probe_indices = np.sort(probe_indices)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"] == "cuda_if_available" else "cpu"
    )
    print(f"Dataset=UCI-HAR train={len(y_train)} test={len(y_test)} device={device}", flush=True)
    print(f"Train-only probe set={len(probe_indices)}; remaining task pool={len(pool_indices)}", flush=True)

    bundle = load_existing_bundle()
    all_conditions = [
        ("LNN", "LNN_full_update"),
        ("LNN", "LNN_freeze_alpha_after_task1"),
        ("GRU", "GRU_full_update"),
        ("LNN", "LNN_freeze_recurrent_after_task1"),
        ("CfC", "CfC_full_update"),
        ("CfC", "CfC_freeze_recurrent_after_task1"),
    ]
    requested = config.get("conditions_to_run", [condition for _, condition in all_conditions])
    rerun_completed = bool(config.get("rerun_completed_conditions", False))
    valid = {condition: model for model, condition in all_conditions}
    unknown = sorted(set(requested) - set(valid))
    if unknown:
        raise ValueError(f"Unknown conditions in run_config.json: {unknown}")
    seeds_to_run = [int(value) for value in config.get("seeds_to_run", [config["seed"]])]
    for seed in seeds_to_run:
        seed_config = dict(config)
        seed_config["seed"] = seed
        for condition in requested:
            model_name = valid[condition]
            boundaries = [
                row for row in bundle["boundaries"]
                if row["condition"] == condition and int(row["seed"]) == seed
            ]
            completed_tasks = {int(row["task_boundary"]) for row in boundaries}
            if completed_tasks == {1, 2, 3} and not rerun_completed:
                print(f"Skipping already complete condition: {condition}, seed={seed}", flush=True)
                continue
            if completed_tasks:
                # A partial prior attempt is discarded for this condition/seed only.
                for key in bundle:
                    bundle[key] = [
                        row for row in bundle[key]
                        if not (row.get("condition") == condition and int(row["seed"]) == seed)
                    ]
            train_condition(model_name, condition, x_train, y_train, x_test, y_test,
                            pool_indices, probe_indices, seed_config, device, bundle)

    run_metadata = dict(config)
    run_metadata.update({
        "dataset": "UCI-HAR",
        "models_and_conditions": list(dict.fromkeys(row["condition"] for row in bundle["boundaries"])),
        "seeds_present": sorted({int(row["seed"]) for row in bundle["boundaries"]}),
        "task_order": [[CLASSES[cls] for cls in task] for task in TASKS],
        "official_split": "UCI-HAR subject-disjoint official train/test split reused from Phase 1/2",
        "train_windows": int(len(y_train)), "test_windows": int(len(y_test)),
        "probe_examples": int(len(probe_indices)), "task_pool_examples": int(len(pool_indices)),
        "device_used": str(device), "test_used_for_training_or_stopping": False,
        "test_evaluation_timing": "after each task has completed training",
        "cfc_implementation": "ncps==1.0.1",
        "cfc_options": "mode=default; backbone_units=128; backbone_layers=1; timespans omitted (ncps uses one unit per sequence step)",
        "interventions": {
            "LNN_freeze_alpha_after_task1": "freeze encoder.logit_alpha after Task 1; other parameters remain trainable",
            "LNN_freeze_recurrent_after_task1": "freeze encoder.recurrent weights after Task 1; alpha, input weights, and classifier remain trainable",
            "CfC_freeze_recurrent_after_task1": "freeze only hidden-state columns of ncps CfCCell first backbone weight after Task 1; input columns, other CfC parameters, and classifier remain trainable",
        },
        "no_replay": True,
    })
    (OUT_DIR / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_multi_seed_report(bundle, run_metadata)
    write_multi_seed_plot(bundle)
    print(f"Finished. Review {OUT_DIR / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
