"""Compare original CBM with staged adaptation on the project WISDM split."""
import csv
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pipeline as p
import benchmark as wb

OUT = Path(__file__).resolve().parent
SEEDS = [11, 22, 33]
KEYS = ["old_class_test_macro_accuracy", "new_class_test_macro_accuracy", "all_seen_test_macro_accuracy"]
CONDITIONS = ["CBM_original", "staged_10_then_10"]
CLASS_NAMES = ["walking", "jogging", "stairs", "sitting", "standing", "typing"]


def load_raw_windows_and_cache():
    """Rebuild raw windows in exactly the cached loader order; assert normalization matches."""
    xtr, ytr, xte, yte, meta = wb.load_wisdm()
    archive_path = wb.DATA_DIR / "wisdm.zip"
    class_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    train_x, train_y, test_x, test_y = [], [], [], []
    with zipfile.ZipFile(archive_path) as outer:
        nested, _buffer = wb.wisdm_members(outer)
        members = sorted(n for n in nested.namelist()
                         if "/raw/phone/accel/data_" in n and n.endswith("_accel_phone.txt"))
        subjects = sorted(int(Path(n).name.split("_")[1]) for n in members)
        test_ids = set(subjects[::5])
        for member in members:
            subject = int(Path(member).name.split("_")[1])
            segments, current_code, current = [], None, []
            with nested.open(member) as f:
                for raw_line in f:
                    fields = raw_line.decode("utf-8", errors="ignore").strip().rstrip(";").split(",")
                    if len(fields) < 6:
                        continue
                    code = fields[1].strip()
                    if code != current_code:
                        if current_code in class_index and current:
                            segments.append((class_index[current_code], current))
                        current_code, current = code, []
                    if code in class_index:
                        try:
                            current.append([float(fields[3]), float(fields[4]), float(fields[5])])
                        except ValueError:
                            pass
            if current_code in class_index and current:
                segments.append((class_index[current_code], current))
            for cls, segment in segments:
                windows = wb.make_windows(segment)
                if windows:
                    dst_x, dst_y = (test_x, test_y) if subject in test_ids else (train_x, train_y)
                    dst_x.extend(windows); dst_y.extend([cls] * len(windows))
    raw_train = np.asarray(train_x, dtype=np.float32)
    raw_y = np.asarray(train_y, dtype=np.int64)
    perm = np.random.default_rng(2026).permutation(len(raw_y))
    raw_train, raw_y = raw_train[perm], raw_y[perm]
    raw_test = np.asarray(test_x, dtype=np.float32)
    raw_y_test = np.asarray(test_y, dtype=np.int64)
    mean = raw_train.mean(axis=(0, 1), keepdims=True)
    std = raw_train.std(axis=(0, 1), keepdims=True) + 1e-6
    expected_train = ((raw_train - mean) / std).astype(np.float32)
    expected_test = ((raw_test - mean) / std).astype(np.float32)
    if not (np.array_equal(raw_y, ytr) and np.array_equal(raw_y_test, yte)):
        raise RuntimeError("Raw WISDM reconstruction labels/order do not match the cached split")
    err_train = float(np.max(np.abs(expected_train - xtr)))
    err_test = float(np.max(np.abs(expected_test - xte)))
    if err_train > 2e-5 or err_test > 2e-5:
        raise RuntimeError(f"Raw WISDM reconstruction differs from cache: train={err_train}, test={err_test}")
    return xtr, ytr, xte, yte, raw_train, raw_test, meta, {"train_max_abs_error": err_train,
                                                          "test_max_abs_error": err_test}


def wisdm_sensor_concepts(x):
    """Eight train-derived descriptors for 3-axis phone accelerometer windows."""
    eps = 1e-8
    magnitude = np.sqrt(np.sum(x * x, axis=2) + eps)
    rms = np.sqrt(np.mean(magnitude * magnitude, axis=1) + eps)
    axis_energy = np.mean(x * x, axis=1)
    axis_share = axis_energy / (axis_energy.sum(axis=1, keepdims=True) + eps)
    centered = magnitude - magnitude.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    freq = np.fft.rfftfreq(x.shape[1], d=1.0 / 20.0)
    spectral_centroid = (spectrum * freq[None, :]).sum(axis=1) / (spectrum.sum(axis=1) + eps)
    ac = np.fft.irfft(spectrum, n=x.shape[1], axis=1)
    ac = ac / (ac[:, :1] + eps)
    periodicity = np.max(ac[:, 2:51], axis=1)
    jerk = np.mean(np.abs(np.diff(magnitude, axis=1)), axis=1)
    impulsiveness = np.max(magnitude, axis=1) / (rms + eps)
    return np.stack([np.log(rms + eps), axis_share[:, 0], axis_share[:, 1], axis_share[:, 2],
                     spectral_centroid, periodicity, np.log(jerk + eps), impulsiveness], axis=1).astype(np.float32)


def save_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    start = time.time()
    x, y, xt, yt, raw, raw_test, meta, cache_check = load_raw_windows_and_cache()
    # The sequence engine keeps its existing task protocol; replace only dataset labels and concepts.
    p.b.CLASSES = CLASS_NAMES
    p.b.TASKS = [[0, 1], [2, 3], [4, 5]]
    p.cbm.CLASSES = [name.upper() for name in CLASS_NAMES]
    p.cbm.TASKS = [[0, 1], [2, 3], [4, 5]]
    p.cbm.raw_concept_features = wisdm_sensor_concepts
    device = p.torch.device("cuda" if p.torch.cuda.is_available() else "cpu")
    pool, _ = p.b.split_probe_and_task_pool(y, {"diagnostic_split_seed": 20261005,
                                                 "diagnostic_fraction_per_class": 0.1})
    all_rows, per_class, checks = [], [], []
    for seed in SEEDS:
        p.SEED = seed
        seed_task1 = {}
        for condition in CONDITIONS:
            p.OUT = OUT / f"seed{seed}" / condition
            p.OUT.mkdir(parents=True, exist_ok=True)
            staged = condition == "staged_10_then_10"
            print(f"WISDM seed={seed} condition={condition}", flush=True)
            result = p.fit_cbm_condition(
                x, y, xt, yt, raw, raw_test, pool, device,
                condition=condition, warmstart=staged, joint_training=staged,
                use_distillation=False, staged_training=staged,
            )
            all_rows.extend(result)
            save_csv("task_boundary_metrics.csv", all_rows)
            task3 = json.loads((p.OUT / f"{condition}_task3_per_class.json").read_text(encoding="utf-8"))
            for cls, acc in task3.items():
                per_class.append({"seed": seed, "condition": condition,
                                  "class": CLASS_NAMES[int(cls)], "accuracy": acc})
            checkpoint = p.torch.load(p.OUT / f"{condition}_task1.pt", map_location="cpu")
            seed_task1[condition] = checkpoint["encoder"]
            print(json.dumps(result[-1]), flush=True)
        same = all(p.torch.equal(seed_task1[CONDITIONS[0]][k], seed_task1[CONDITIONS[1]][k])
                   for k in seed_task1[CONDITIONS[0]])
        checks.append({"seed": seed, "same_task1_encoder": same})
        if not same:
            raise RuntimeError(f"Original/staged Task-1 encoders differ for seed {seed}")

    save_csv("per_class_task3.csv", per_class)
    task3 = [r for r in all_rows if r["task"] == 3]
    summary = []
    for condition in CONDITIONS:
        group = [r for r in task3 if r["condition"] == condition]
        record = {"condition": condition}
        for key in KEYS:
            values = np.asarray([r[key] for r in group], dtype=float)
            record[key + "_mean"] = float(values.mean())
            record[key + "_sd"] = float(values.std(ddof=1))
        summary.append(record)
    by_seed = {s: {r["condition"]: r for r in task3 if r["seed"] == s} for s in SEEDS}
    paired = [
        {"seed": s, **{k: 100 * (by_seed[s][CONDITIONS[1]][k] - by_seed[s][CONDITIONS[0]][k])
                       for k in KEYS}}
        for s in SEEDS
    ]
    save_csv("condition_summary.csv", summary)
    save_csv("paired_deltas.csv", paired)
    (OUT / "encoder_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (OUT / "data_validation.json").write_text(json.dumps({"cache_raw_reconstruction": cache_check,
        "meta": meta, "train_class_counts": {CLASS_NAMES[i]: int(np.sum(y == i)) for i in range(6)},
        "test_class_counts": {CLASS_NAMES[i]: int(np.sum(yt == i)) for i in range(6)}},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# WISDM：CBM 原版基线 vs 分阶段训练（三种子）", "",
             "使用项目缓存的 WISDM phone accelerometer 窗口，128 点长度、64 点步长，官方项目协议按受试者隔离训练/测试。六类按项目既有顺序组成三个任务：walking/jogging、stairs/sitting、standing/typing。每类任务划分、类别顺序和编码器初始化在两条件间配对。", "",
             "WISDM 原始缓存是按训练集统计量标准化后的三轴加速度。本实验据此构造 8 个统一传感器描述特征：幅值 RMS、三轴能量占比、频谱质心、周期性、变化率和冲击性；同一特征用于两种 CBM 条件。", "",
             "| 方法 | 旧类准确率 | 新类准确率 | 全部已见类准确率 |", "|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['condition']} | " + " | ".join(f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_sd']:.2f}%" for k in KEYS) + " |")
    lines += ["", "## 分阶段相对原版的逐种子变化（百分点）", "", "| 种子 | 旧类 | 新类 | 全部已见类 |", "|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['seed']} | " + " | ".join(f"{r[k]:+.2f}" for k in KEYS) + " |")
    lines += ["", "原版条件遵循本项目此前的 CBM 原始适配流程；分阶段条件额外包含旧概念投影行热启动及 10 步保护旧行 + 10 步全行联合微调。因此这是两套方案的比较，不把总差异单独归因于阶段安排。两条件都不使用联合阶段蒸馏；投影器概念拟合沿用原有流程。", "",
              "限制：WISDM 是项目现有的六类 phone accelerometer 子集，不是 UCI 页面所有 18 类/全部手机与手表通道。CBM 的概念目标由标准化窗口构造，是项目内的信号概念适配，不是原论文图文概念复现。测试集只用于任务边界评估，不用于选配置。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "run_metadata.json").write_text(json.dumps({"seeds": SEEDS, "device": str(device),
        "dataset": "project WISDM six-class phone-accelerometer subset",
        "conditions": CONDITIONS, "task_order": [[CLASS_NAMES[i] for i in t] for t in p.b.TASKS],
        "short_run": {"max_encoder_epochs": p.MAX_EPOCHS, "early_stop_patience": p.PATIENCE},
        "staged_steps": {"old_projection_rows_frozen": 10, "all_rows_open": 10},
        "joint_stage_distillation": False, "elapsed_seconds": time.time() - start},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print((OUT / "REPORT.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
