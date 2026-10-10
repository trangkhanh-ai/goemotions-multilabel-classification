"""Xuất bảng và hình từ kết quả A/B/C đã lưu; không fit/tune hay mở nhãn test.

python -m scripts.analysis.export_project_analysis
Thiếu artifact -> status=missing và số trống, không thay bằng 0.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.goemotions import EXPECTED_ROWS, REVISION, sha256
from src.evaluation.protocols import save_json
from src.evaluation.metrics import validate_thresholds
from src.models.transformer import ARCHITECTURES, configure_console, run_folder


from src.paths import ROOT
MODES = ("fixed", "global", "tuned")


def read_optional(path, issues):
    """Giữ lỗi nguồn riêng; không biến thiếu số liệu thành một điểm số bằng 0."""
    path = Path(path)
    if not path.is_file():
        issues.append({"path": str(path), "status": "missing", "reason": "File chưa có"})
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        issues.append({"path": str(path), "status": "invalid", "reason": str(error)})
        return None


def check_metric(metric, labels, split):
    """Kiểm metric đủ nhãn, đúng split-size và giá trị hữu hạn."""
    if not isinstance(metric, dict):
        raise ValueError("Chưa có metric")
    if metric.get("n_labels") != len(labels) or metric.get("n_samples") != EXPECTED_ROWS[split]:
        raise ValueError("Metric không đúng số nhãn hoặc số mẫu split chính thức")
    for key in ("macro_f1", "micro_f1"):
        value = metric.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Metric {key} không hợp lệ")
    rows = metric.get("per_label", [])
    if len(rows) != len(labels):
        raise ValueError("Thiếu metric từng nhãn")
    for j, row in enumerate(rows):
        if row.get("label_id") != j or row.get("label") != labels[j]:
            raise ValueError("Metric per-label sai thứ tự mapping")
        for key in ("precision", "recall", "f1"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Metric nhãn {labels[j]} không hợp lệ")
    return metric


def metadata_ok(metadata, labels, method, architecture=None, seed=None):
    if not metadata or metadata.get("smoke") is not False or metadata.get("data_revision") != REVISION:
        return False
    if metadata.get("label_names") != labels:
        return False
    if method == "A":
        return metadata.get("n_train") == EXPECTED_ROWS["train"] and metadata.get("n_validation") == EXPECTED_ROWS["validation"]
    if method == "B":
        return (metadata.get("method") == "zero_shot" and metadata.get("status") == "complete"
                and metadata.get("split") == "validation" and metadata.get("sample_count") == EXPECTED_ROWS["validation"])
    return (metadata.get("method") == "C" and metadata.get("completed") is True
            and metadata.get("architecture") == architecture and metadata.get("seed") == seed
            and metadata.get("sizes") == {"train": EXPECTED_ROWS["train"], "validation": EXPECTED_ROWS["validation"]})


def frozen_protocol(folder, metadata, labels, issues):
    """B/C protocol phải thuộc đúng metadata/scores đã khóa trên validation."""
    folder = Path(folder)
    path = folder / "final_protocol.json"
    protocol = read_optional(path, issues)
    if protocol is None:
        return None
    try:
        if (protocol.get("protocol_version") != 1 or protocol.get("smoke") is not False
                or protocol.get("label_names") != labels or protocol.get("data_revision") != REVISION
                or protocol.get("selection_data") != "validation" or not protocol.get("frozen_at_utc")):
            raise ValueError("Protocol sai schema/mapping hoặc chưa khóa trên validation")
        if (protocol.get("run_metadata_sha256") != sha256(folder / "run_metadata.json")
                or protocol.get("validation_scores_sha256") != sha256(folder / "validation_scores.npz")):
            raise ValueError("Metadata/validation scores đã đổi sau khi khóa")
        checkpoint = metadata.get("checkpoint", metadata.get("model_checkpoint"))
        revision = metadata.get("model_revision", metadata.get("checkpoint_revision"))
        if protocol.get("checkpoint") != checkpoint or protocol.get("model_revision") != revision:
            raise ValueError("Protocol không thuộc checkpoint hiện tại")
        if metadata["method"] == "C" and (protocol.get("architecture") != metadata["architecture"]
                                           or protocol.get("seed") != metadata["seed"]):
            raise ValueError("Protocol không thuộc kiến trúc/seed hiện tại")
        if metadata["method"] == "zero_shot" and protocol.get("hypothesis_template") != metadata.get("hypothesis_template"):
            raise ValueError("Protocol không thuộc template hiện tại")
        if sorted(row["threshold_mode"] for row in protocol["configurations"]) != sorted(MODES):
            raise ValueError("Protocol cần đủ fixed/global/tuned không lặp")
        for row in protocol["configurations"]:
            validate_thresholds(row.get("thresholds"), len(labels))
    except (KeyError, OSError, ValueError) as error:
        issues.append({"path": str(path), "status": "invalid", "reason": str(error)})
        return None
    return protocol


def make_record(labels, split, method, variant, architecture, seed, mode, metric, source, reason="", config=None):
    record = {"method": method, "variant": variant, "architecture": architecture, "seed": seed,
              "split": split, "threshold_mode": mode, "source": str(source),
              "status": "ok" if metric is not None else "missing", "reason": "" if metric is not None else reason,
              "validation_calibrated": split == "validation" and mode != "fixed",
              "metric": None, "config_signature": config}
    if metric is not None:
        try:
            record["metric"] = check_metric(metric, labels, split)
        except ValueError as error:
            record.update(status="invalid", reason=str(error))
    return record


def collect_results(root, labels, seeds):
    """Trả record cho mọi cấu hình dự kiến, kể cả cấu hình chưa có kết quả."""
    root = Path(root)
    records, curves, issues, train_support = [], [], [], None
    base = root / "data/processed/baseline"
    a_protocol = read_optional(base / "final_protocol.json", issues)
    a_final = read_optional(base / "final/final_results.json", issues)
    a_test_ok = False
    if a_protocol and a_final:
        a_test_ok = (a_protocol.get("schema_version") == 1 and a_protocol.get("label_names") == labels
                     and a_protocol.get("data_revision") == REVISION
                     and a_final.get("data_revision") == REVISION
                     and a_final.get("protocol_sha256") == sha256(base / "final_protocol.json"))
        if not a_test_ok:
            issues.append({"path": str(base / "final/final_results.json"), "status": "invalid", "reason": "A test không khớp protocol"})
    for variant, folder in (("standard", base / "full"), ("balanced", base / "balanced/full")):
        metadata_path = folder / "validation_metrics.json"
        metadata = read_optional(metadata_path, issues)
        valid = metadata_ok(metadata, labels, "A")
        if valid and variant == "standard":
            counts = metadata.get("train_positive_counts")
            if isinstance(counts, list) and len(counts) == len(labels) and all(isinstance(x, int) and x >= 0 for x in counts):
                train_support = counts
        analysis_path = folder / "analysis_validation.json"
        analysis = read_optional(analysis_path, issues)
        analysis_valid = (valid and analysis and analysis.get("artifact_sha256") == metadata.get("artifact_sha256"))
        fields = {"fixed": "fixed_0_5", "global": "global_on_validation", "tuned": "tuned_on_validation"}
        for split in ("validation", "test"):
            for mode in MODES:
                metric, source = None, analysis_path if split == "validation" else base / "final/final_results.json"
                if split == "validation" and analysis_valid:
                    metric = analysis.get(fields[mode])
                if split == "test" and a_test_ok and valid:
                    run = a_protocol.get("runs", {}).get(variant, {})
                    if run.get("artifact_sha256") == metadata.get("artifact_sha256"):
                        choices = [row for row in a_final.get("results", []) if row.get("name") == f"{variant}_{mode}"]
                        if len(choices) == 1 and choices[0].get("protocol_sha256") == a_final["protocol_sha256"]:
                            metric = choices[0].get("metrics")
                records.append(make_record(labels, split, "A", variant, "tfidf_lr", None, mode, metric,
                                           source, "Chưa có kết quả A hợp lệ cho cấu hình này"))

    experiments = [("B", "zero_shot", "bart_mnli", None, root / "data/processed/zero_shot/full")]
    experiments += [("C", "standard", architecture, seed, run_folder(root, architecture, seed))
                    for architecture in ARCHITECTURES for seed in seeds]
    for method, variant, architecture, seed, folder in experiments:
        metadata_path = folder / "run_metadata.json"
        metadata = read_optional(metadata_path, issues)
        valid = metadata_ok(metadata, labels, method, architecture, seed)
        protocol = frozen_protocol(folder, metadata, labels, issues) if valid else None
        signature = json.dumps({"config": metadata.get("config"), "model_revision": metadata.get("model_revision")},
                               sort_keys=True, ensure_ascii=False) if valid and method == "C" else None
        fixed_path = folder / "validation_metrics.json"
        fixed_metric = read_optional(fixed_path, issues) if valid else None
        if valid and method == "B" and fixed_metric is None:
            fixed_metric = metadata.get("metrics")
        test_path = folder / ("test_results.json" if method == "C" else "test_run_metadata.json")
        test_result = read_optional(test_path, issues)
        test_ok = False
        if valid and protocol and test_result:
            try:
                expected_score_hash = (test_result.get("scores_sha256") if method == "C" else
                                       test_result.get("artifact_sha256", {}).get("test_scores.npz"))
                test_ok = (test_result.get("protocol_sha256") == sha256(folder / "final_protocol.json")
                           and expected_score_hash == sha256(folder / "test_scores.npz")
                           and test_result.get("split") == "test")
                if method == "C":
                    test_ok = test_ok and test_result.get("architecture") == architecture and test_result.get("seed") == seed
                else:
                    test_ok = test_ok and test_result.get("status") == "complete" and test_result.get("smoke") is False
            except OSError:
                test_ok = False
            if not test_ok:
                issues.append({"path": str(test_path), "status": "invalid", "reason": "Test result không khớp protocol/scores"})
        for split in ("validation", "test"):
            for mode in MODES:
                metric, source = None, fixed_path if split == "validation" and mode == "fixed" else folder / "final_protocol.json"
                if split == "validation" and valid:
                    if mode == "fixed":
                        metric = fixed_metric
                    elif protocol:
                        choices = [row for row in protocol["configurations"] if row["threshold_mode"] == mode]
                        metric = choices[0].get("validation_metrics") if len(choices) == 1 else None
                if split == "test":
                    source = test_path
                    if test_ok:
                        choices = [row for row in test_result.get("results" if method == "C" else "configurations", [])
                                   if row.get("threshold_mode") == mode]
                        metric = choices[0].get("metrics") if len(choices) == 1 else None
                records.append(make_record(labels, split, method, variant, architecture, seed, mode, metric,
                                           source, "Chưa có run full/protocol/kết quả hợp lệ", config=signature))
        if method == "C":
            history = metadata.get("history", []) if valid else []
            if not history:
                curves.append({"architecture": architecture, "seed": seed, "epoch": None,
                               "status": "missing", "reason": "Chưa có history của run full hoàn tất",
                               "train_loss": None, "validation_macro_f1": None, "validation_micro_f1": None,
                               "epoch_seconds": None, "validation_seconds": None, "selected_epoch": None})
            for row in history:
                item = {"architecture": architecture, "seed": seed, "epoch": row.get("epoch"),
                        "status": "ok", "reason": "", "train_loss": row.get("train_loss"),
                        "validation_macro_f1": row.get("macro_f1"), "validation_micro_f1": row.get("micro_f1"),
                        "epoch_seconds": row.get("epoch_seconds"), "validation_seconds": row.get("validation_seconds"),
                        "selected_epoch": metadata.get("selected_epoch")}
                values = (item["train_loss"], item["validation_macro_f1"], item["validation_micro_f1"])
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                    item.update(status="invalid", reason="History chứa giá trị thiếu/không hữu hạn")
                curves.append(item)
    return records, curves, train_support, issues


def aggregate_c(records, architecture, split, mode, expected_seeds):
    group = [row for row in records if row["method"] == "C" and row["architecture"] == architecture
             and row["split"] == split and row["threshold_mode"] == mode]
    good = [row for row in group if row["status"] == "ok"]
    complete = (len(expected_seeds) >= 3 and len(good) == len(expected_seeds)
                and sorted(row["seed"] for row in good) == sorted(expected_seeds)
                and len({row["config_signature"] for row in good}) == 1)
    output = {"architecture": architecture, "split": split, "threshold_mode": mode,
              "expected_seeds": ",".join(map(str, expected_seeds)), "n_seeds_available": len(good),
              "status": "ok" if complete else "missing", "ddof": 1,
              "reason": "" if complete else "Cần đủ >=3 cùng seed/config/revision; không bỏ run thiếu để lấy trung bình",
              "validation_calibrated": split == "validation" and mode != "fixed",
              "macro_f1_mean": None, "macro_f1_std": None, "micro_f1_mean": None, "micro_f1_std": None}
    if complete:
        for key in ("macro_f1", "micro_f1"):
            values = [row["metric"][key] for row in good]
            output[key + "_mean"] = float(np.mean(values))
            output[key + "_std"] = float(np.std(values, ddof=1))
    return output, sorted(good, key=lambda row: row["seed"]) if complete else []


def rare_comparisons(records, labels, train_support, seeds):
    if train_support is None:
        return [{"status": "missing", "reason": "Thiếu train_positive_counts A standard; không chọn nhãn hiếm bằng test"}]
    rare_ids = sorted(range(len(labels)), key=lambda j: (train_support[j], j))[:5]
    result = []
    for split in ("validation", "test"):
        a_before = next(row for row in records if row["method"] == "A" and row["variant"] == "standard"
                        and row["split"] == split and row["threshold_mode"] == "fixed")
        a_after = next(row for row in records if row["method"] == "A" and row["variant"] == "balanced"
                       and row["split"] == split and row["threshold_mode"] == "tuned")
        for j in rare_ids:
            row = {"method": "A", "architecture": "tfidf_lr", "split": split, "label": labels[j],
                   "train_support": train_support[j], "before": "standard_fixed", "after": "balanced_tuned",
                   "status": "missing", "n_seeds": None, "validation_calibrated": split == "validation"}
            if a_before["status"] == a_after["status"] == "ok":
                before, after = a_before["metric"]["per_label"][j], a_after["metric"]["per_label"][j]
                row.update(status="ok", evaluation_support=before["support"], before_f1_mean=before["f1"],
                           after_f1_mean=after["f1"], delta_f1_mean=after["f1"] - before["f1"])
            result.append(row)
        for architecture in ARCHITECTURES:
            _, before = aggregate_c(records, architecture, split, "fixed", seeds)
            _, after = aggregate_c(records, architecture, split, "tuned", seeds)
            for j in rare_ids:
                row = {"method": "C", "architecture": architecture, "split": split, "label": labels[j],
                       "train_support": train_support[j], "before": "fixed", "after": "tuned",
                       "status": "missing", "n_seeds": 0, "validation_calibrated": split == "validation"}
                if before and after and [r["seed"] for r in before] == [r["seed"] for r in after]:
                    x = np.asarray([r["metric"]["per_label"][j]["f1"] for r in before])
                    y = np.asarray([r["metric"]["per_label"][j]["f1"] for r in after])
                    row.update(status="ok", n_seeds=len(x), evaluation_support=before[0]["metric"]["per_label"][j]["support"],
                               before_f1_mean=float(x.mean()), before_f1_std=float(x.std(ddof=1)),
                               after_f1_mean=float(y.mean()), after_f1_std=float(y.std(ddof=1)),
                               delta_f1_mean=float((y - x).mean()), delta_f1_std=float((y - x).std(ddof=1)))
                result.append(row)
    return result


def export_figures(output, curves, aggregates, seeds, issues):
    """Chỉ vẽ khi đủ ba kiến trúc/seed; không dựng cột 0 cho model chưa có."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        issues.append({"path": str(output / "figures"), "status": "missing", "reason": "Chưa có matplotlib"})
        return []
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    saved = []
    good = pd.DataFrame([row for row in curves if row["status"] == "ok"])
    complete = not good.empty and all(
        sorted(good.loc[good["architecture"] == architecture, "seed"].unique().tolist()) == sorted(seeds)
        for architecture in ARCHITECTURES)
    if complete:
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for axis, architecture in zip(axes, ARCHITECTURES):
            subset = good[good["architecture"] == architecture]
            for column, label in (("validation_macro_f1", "Macro-F1"), ("validation_micro_f1", "Micro-F1")):
                grouped = subset.groupby("epoch")[column].agg(["mean", "std", "count"])
                if not (grouped["count"] == len(seeds)).all():
                    continue
                x = grouped.index.to_numpy()
                mean, std = grouped["mean"].to_numpy(), grouped["std"].to_numpy()
                axis.plot(x, mean, marker="o", label=label)
                axis.fill_between(x, mean - std, mean + std, alpha=0.15)
            axis.set(title=architecture, xlabel="Epoch", ylabel="Validation F1")
            axis.legend()
            axis.grid(alpha=0.2)
        figure.tight_layout()
        path = figures / "learning_curves_validation.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        saved.append(str(path))
    else:
        issues.append({"path": str(figures / "learning_curves_validation.png"), "status": "missing", "reason": "Cần history đủ ba C với cùng >=3 seeds"})
    for split in ("validation", "test"):
        rows = [r for r in aggregates if r["split"] == split]
        if len(rows) != 6 or any(r["status"] != "ok" for r in rows):
            issues.append({"path": str(figures / f"fixed_tuned_{split}.png"), "status": "missing", "reason": "Chưa đủ mean/std của ba C"})
            continue
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(3)
        for axis, metric in zip(axes, ("macro_f1", "micro_f1")):
            for offset, mode in ((-0.18, "fixed"), (0.18, "tuned")):
                chosen = [next(r for r in rows if r["architecture"] == architecture and r["threshold_mode"] == mode)
                          for architecture in ARCHITECTURES]
                axis.bar(x + offset, [r[metric + "_mean"] for r in chosen], width=0.36,
                         yerr=[r[metric + "_std"] for r in chosen], capsize=3, label=mode)
            axis.set_xticks(x, list(ARCHITECTURES))
            axis.set(title=metric, ylabel="F1 mean ± sample std")
            axis.legend()
            axis.grid(axis="y", alpha=0.2)
        caption = "Validation (tuned uses validation labels)" if split == "validation" else "Test (thresholds frozen on validation)"
        figure.suptitle(caption)
        figure.tight_layout()
        path = figures / f"fixed_tuned_{split}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        saved.append(str(path))
    return saved


def collect_training_costs(root, labels, seeds, issues):
    """Chi phí C lấy từ elapsed_seconds thật của các run full đã hoàn tất."""
    rows = []
    for architecture in ARCHITECTURES:
        runs = []
        for seed in seeds:
            folder = run_folder(root, architecture, seed)
            metadata = read_optional(folder / "run_metadata.json", issues)
            if not metadata_ok(metadata, labels, "C", architecture, seed):
                continue
            seconds, parameters = metadata.get("elapsed_seconds"), metadata.get("parameter_count")
            if (isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0
                    and isinstance(parameters, int) and parameters > 0):
                runs.append({"seed": seed, "seconds": seconds, "parameters": parameters})
        complete = (len(runs) == len(seeds) and len(seeds) >= 3
                    and len({run["parameters"] for run in runs}) == 1)
        row = {"architecture": architecture, "status": "ok" if complete else "missing",
               "n_seeds_available": len(runs), "n_seeds_expected": len(seeds),
               "elapsed_seconds_sum": None, "elapsed_seconds_mean": None, "parameter_count": None,
               "reason": "" if complete else "Cần đủ full seeds và elapsed_seconds/parameter_count hợp lệ",
               "scope": "elapsed_seconds per full C run: preparation, training, validation, checkpoint IO"}
        if complete:
            seconds = [run["seconds"] for run in runs]
            row.update(elapsed_seconds_sum=float(sum(seconds)), elapsed_seconds_mean=float(np.mean(seconds)),
                       parameter_count=runs[0]["parameters"])
        rows.append(row)
    return rows


def common_support(records, label_id, split):
    """Support là nhãn thật; nếu các artifact không cùng support thì không gộp."""
    values = {row["metric"]["per_label"][label_id]["support"] for row in records
              if row["split"] == split and row["status"] == "ok"}
    return next(iter(values)) if len(values) == 1 else None


def display_f1(row, prefix, *, delta=False):
    if row.get("status") != "ok" or row.get(prefix + "_mean") is None:
        return "Thiếu kết quả"
    value = row[prefix + "_mean"]
    shown = f"{value:+.4f}" if delta else f"{value:.4f}"
    deviation = row.get(prefix + "_std")
    return shown + (f" ± {deviation:.4f}" if deviation is not None else "")


def write_analysis_markdown(output, records, rare, labels, support, costs, figure_paths):
    """Bảng dùng test chỉ khi đủ mọi cặp; nếu chưa đủ, ghi rõ validation calibration."""
    output = Path(output)
    rare_ids = (sorted(range(len(labels)), key=lambda j: (support[j], j))[:5]
                if support is not None else [])
    expected = {(labels[j], architecture) for j in rare_ids for architecture in ("tfidf_lr", *ARCHITECTURES)}
    complete_test = (bool(expected) and
                     {(row.get("label"), row.get("architecture")) for row in rare
                      if row.get("split") == "test" and row.get("status") == "ok"} == expected)
    split = "test" if complete_test else "validation"
    lines = ["# Phân tích kết quả, nhãn hiếm và chi phí", "", "## 1. Năm nhãn hiếm: trước và sau cải tiến", ""]
    if split == "test":
        lines.append("**Bảng chính: TEST.** Ngưỡng của các cấu hình được khóa trên validation trước khi đánh giá test. "
                     "Bảng giữ cả mức tăng, giảm và không đổi.")
    else:
        lines.append("**Bảng tạm: VALIDATION — có calibration.** Test chưa đủ mọi cặp A/C1/C2/C3 nên chưa dùng "
                     "để đưa ra bảng so sánh cuối. Các cấu hình tuned được đo trên cùng validation đã dùng để chọn ngưỡng; "
                     "mức tăng có thể lạc quan, không được gọi là kết quả test. Các ô thiếu được ghi rõ.")
    lines += ["", "Nhãn hiếm lấy theo năm support thấp nhất trên train từ metadata A standard; hòa theo ID nhãn. "
              "A: standard fixed → balanced tuned. C1/C2/C3: cùng kiến trúc, cùng ít nhất ba seed, fixed → tuned. "
              "C báo mean ± sample std (`ddof=1`); Δ tính theo cặp seed. A là một run, không tạo std bằng 0.", "",
              "| Nhãn | Mô hình | Train + | Validation + | Test + | F1 trước | F1 sau | Δ F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    model_names = {"tfidf_lr": "A: TF-IDF + LR", "bert": "C1: BERT", "roberta": "C2: RoBERTa", "distilbert": "C3: DistilBERT"}
    for j in rare_ids:
        val_support = common_support(records, j, "validation")
        test_support = common_support(records, j, "test")
        for architecture, name in model_names.items():
            chosen = [row for row in rare if row.get("split") == split and row.get("label") == labels[j]
                      and row.get("architecture") == architecture]
            row = chosen[0] if len(chosen) == 1 else {"status": "missing"}
            cells = [labels[j], name, str(support[j]), str(val_support) if val_support is not None else "Chưa có/không khớp",
                     str(test_support) if test_support is not None else "Chưa có/không khớp",
                     display_f1(row, "before_f1"), display_f1(row, "after_f1"), display_f1(row, "delta_f1", delta=True)]
            lines.append("| " + " | ".join(cells) + " |")
    if not rare_ids:
        lines += ["", "**Thiếu support train hợp lệ:** chưa thể xác định năm nhãn hiếm; không chọn thay bằng nhãn test khó."]
    lines += ["", "Support là số câu có nhãn thật, không phải số lần model dự đoán nhãn. "
              "Δ > 0 là tăng, Δ < 0 là giảm. Bảng làm tròn bốn chữ số; số gốc và mọi mức giảm được giữ "
              "trong `rare_before_after.csv`. Các nhóm C thiếu seed/config/revision nhất quán sẽ không có mean/std.", "",
              "## 2. Chi phí thực nghiệm C", "",
              "Lấy `elapsed_seconds` từ metadata của từng run full hoàn tất. Đây là thời gian toàn run C "
              "(chuẩn bị dữ liệu/model, train, validation và ghi checkpoint), không phải riêng thời gian optimizer. "
              "Tổng và trung bình chỉ hiển thị khi đủ tập seed yêu cầu; chi phí phụ thuộc máy và cache.", "",
              "| Kiến trúc | Full seeds đủ chi phí | Số tham số | Tổng elapsed (giây) | Mean elapsed/run (giây) | Trạng thái |",
              "|---|---|---:|---:|---:|---|"]
    for row in costs:
        ok = row["status"] == "ok"
        lines.append("| " + " | ".join([row["architecture"], f"{row['n_seeds_available']}/{row['n_seeds_expected']}",
                     f"{row['parameter_count']:,}" if ok else "Thiếu kết quả",
                     f"{row['elapsed_seconds_sum']:.2f}" if ok else "Thiếu kết quả",
                     f"{row['elapsed_seconds_mean']:.2f}" if ok else "Thiếu kết quả", "Đủ" if ok else "Thiếu run/metadata"]) + " |")
    lines += ["", "Nguồn chi tiết: `training_costs.csv`. Phép đo chi phí B cần tổng thời gian suy luận có xử lý "
              "resume; không dùng thời gian của một lần tiếp tục để đại diện toàn dataset.", "",
              "## 3. Learning curves trên validation", ""]
    learning = next((Path(path) for path in figure_paths if Path(path).name == "learning_curves_validation.png"), None)
    if learning is not None and learning.is_file():
        lines += ["Mỗi đường là trung bình qua cùng tập seed; dải màu là ±1 sample std. Đây là metric validation "
                  "ở ngưỡng 0,5 theo epoch, không phải learning curve trên test.", "",
                  f"![Validation learning curves của BERT, RoBERTa và DistilBERT](<{learning.resolve().as_posix()}>)"]
    else:
        lines.append("**Chưa có hình đủ ba kiến trúc × các seed:** xem `learning_curves.csv` và `missing_artifacts.csv`; "
                     "không dựng đường giả hoặc thay model thiếu bằng điểm 0.")
    lines += ["", "## 4. Hồ sơ đối chiếu", "",
              "- `per_label.csv`: P/R/F1 và TP/FP/FN/support cho từng nhãn, split, cấu hình và seed.",
              "- `aggregate_metrics.csv`: Macro/Micro-F1 fixed/tuned mean ± sample std của ba C.",
              "- `rare_before_after.csv`: nhãn hiếm, cặp trước/sau và Δ, giữ toàn bộ tăng/giảm.",
              "- `learning_curves.csv`, `training_costs.csv`: epoch và chi phí từ metadata thật.",
              "- `missing_artifacts.csv`, `analysis_manifest.json`: trạng thái thiếu/không hợp lệ và phạm vi dữ liệu.",
              "", "Script chỉ đọc artifact và kết quả đã lưu; không huấn luyện, chọn lại ngưỡng hoặc mở nhãn test."]
    (output / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    return {"rare_primary_split": split, "rare_test_complete": complete_test}


def export_analysis(root, seeds=(42, 123, 2026), output=None, *, figures=True):
    root = Path(root).resolve()
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Cần >=3 seed khác nhau")
    labels = json.loads((root / "data/labels.json").read_text(encoding="utf-8"))
    records, curves, support, issues = collect_results(root, labels, list(seeds))
    output = Path(output) if output else root / "reports/project_results"
    output.mkdir(parents=True, exist_ok=True)
    per_label = []
    for record in records:
        metadata = {key: record[key] for key in ("method", "variant", "architecture", "seed", "split",
                                                 "threshold_mode", "status", "reason", "source", "validation_calibrated")}
        for j, label in enumerate(labels):
            item = dict(metadata, label_id=j, label=label,
                        train_support=support[j] if support is not None else None)
            item.update({key: None for key in ("support", "predicted_support", "tp", "fp", "fn", "tn",
                                               "precision", "recall", "f1")})
            if record["metric"] is not None:
                item.update(record["metric"]["per_label"][j])
            per_label.append(item)
    pd.DataFrame(per_label).to_csv(output / "per_label.csv", index=False, encoding="utf-8-sig")
    rare = rare_comparisons(records, labels, support, list(seeds))
    pd.DataFrame(rare).to_csv(output / "rare_before_after.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(output / "learning_curves.csv", index=False, encoding="utf-8-sig")
    aggregates = [aggregate_c(records, architecture, split, mode, list(seeds))[0]
                  for split in ("validation", "test") for architecture in ARCHITECTURES for mode in ("fixed", "tuned")]
    pd.DataFrame(aggregates).to_csv(output / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    saved_figures = export_figures(output, curves, aggregates, list(seeds), issues) if figures else []
    costs = collect_training_costs(root, labels, list(seeds), issues)
    pd.DataFrame(costs).to_csv(output / "training_costs.csv", index=False, encoding="utf-8-sig")
    human_summary = write_analysis_markdown(output, records, rare, labels, support, costs, saved_figures)
    issues += [{"path": row["source"], "status": row["status"], "reason": row["reason"]}
               for row in records if row["status"] != "ok"]
    missing = pd.DataFrame(issues, columns=["path", "status", "reason"]).drop_duplicates()
    missing.to_csv(output / "missing_artifacts.csv", index=False, encoding="utf-8-sig")
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "data_revision": REVISION,
                "labels": labels, "seeds": list(seeds), "ddof": 1,
                "n_configurations_ok": sum(row["status"] == "ok" for row in records),
                "n_configurations_expected": len(records), "n_missing_or_invalid_sources": len(missing),
                "figures": saved_figures,
                **human_summary,
                "rare_labels": ([labels[j] for j in sorted(range(len(labels)), key=lambda j: (support[j], j))[:5]]
                                if support is not None else None),
                "notes": ["Missing metrics stay blank; zero is reserved for measured zero.",
                          "Validation tuned metrics reuse labels used for threshold selection; do not call them test.",
                          "A improvement compares standard fixed to balanced tuned; C compares paired fixed/tuned seeds.",
                          "Negative deltas are retained. No fitting, threshold tuning, or raw test labels were read."]}
    save_json(output / "analysis_manifest.json", manifest)
    return manifest


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = export_analysis(ROOT, args.seeds, args.output, figures=not args.no_figures)
    print(f"Đã xuất {result['n_configurations_ok']}/{result['n_configurations_expected']} cấu hình có số thật; "
          f"{result['n_missing_or_invalid_sources']} nguồn thiếu/không hợp lệ.")


if __name__ == "__main__":
    main()
