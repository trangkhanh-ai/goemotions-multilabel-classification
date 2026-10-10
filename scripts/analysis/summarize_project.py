"""Gom kết quả đã chạy; không sinh dự đoán và không điền số chưa có.

python -m scripts.analysis.summarize_project --require-complete
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.datasets.goemotions import REVISION, sha256
from src.evaluation.protocols import read_json, save_json, validate_protocol
from src.models.transformer import ARCHITECTURES, METRIC_NAMES, load_transformer_run, run_folder

from src.paths import ROOT
SEEDS = (42, 123, 2026)


def collect(root=ROOT):
    root = Path(root)
    labels = read_json(root / "data/labels.json")
    records, missing, sources = [], [], []

    def record(system, seed, split, mode, metrics, path):
        if metrics["n_samples"] != {"validation": 5426, "test": 5427}[split]:
            raise ValueError("Bảng chính chỉ dùng toàn bộ split chính thức")
        records.append({"system": system, "seed": seed, "split": split,
                        "threshold_mode": mode,
                        **{key: metrics[key] for key in METRIC_NAMES},
                        "n_samples": metrics["n_samples"]})
        sources.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path)})

    for variant in ("standard", "balanced"):
        base = root / "data/processed/baseline"
        folder = base / "full" if variant == "standard" else base / "balanced/full"
        analysis = folder / "analysis_validation.json"
        if analysis.exists():
            values = read_json(analysis)
            for mode, key in (("fixed", "fixed_0_5"), ("global", "global_on_validation"),
                              ("tuned", "tuned_on_validation")):
                record("A_" + variant, 42, "validation", mode, values[key], analysis)
        else:
            missing.append("A " + variant + " validation")
    a_test = root / "data/processed/baseline/final/final_results.json"
    if a_test.exists():
        value = read_json(a_test)
        if value["protocol_sha256"] != sha256(root / "data/processed/baseline/final_protocol.json"):
            raise ValueError("A test khác protocol")
        for row in value["results"]:
            record("A_" + row["variant"], 42, "test", row["threshold_mode"], row["metrics"], a_test)
    else:
        missing.append("A test")

    b_folder = root / "data/processed/zero_shot/full"
    b_protocol = b_folder / "final_protocol.json"
    if b_protocol.exists():
        protocol = read_json(b_protocol)
        validate_protocol(b_folder, protocol, labels)
        for cfg in protocol["configurations"]:
            record("B_bart_mnli", None, "validation", cfg["threshold_mode"], cfg["validation_metrics"], b_protocol)
        b_test = b_folder / "test_run_metadata.json"
        if b_test.exists():
            test = read_json(b_test)
            if test.get("validation_run_metadata_sha256") != protocol["run_metadata_sha256"]:
                raise ValueError("B test khác validation metadata")
            for cfg in test["configurations"]:
                record("B_bart_mnli", None, "test", cfg["threshold_mode"], cfg["metrics"], b_test)
        else:
            missing.append("B test")
    else:
        missing.append("B full validation + frozen protocol")

    for architecture in ARCHITECTURES:
        reference_run = None
        for seed in SEEDS:
            folder = run_folder(root, architecture, seed)
            metadata_path = folder / "run_metadata.json"
            if not metadata_path.exists() or not read_json(metadata_path).get("completed"):
                missing.append(f"C {architecture} seed {seed} full")
                continue
            metadata = load_transformer_run(folder, require_full=True)
            # Ba seed phải lặp cùng một thí nghiệm, không gộp learning rate/revision khác.
            if reference_run is None:
                reference_run = metadata
            elif (metadata["config"] != reference_run["config"] or
                  metadata.get("model_revision") != reference_run.get("model_revision")):
                raise ValueError(f"C {architecture}: các seed khác cấu hình hoặc revision checkpoint")
            frozen = folder / "final_protocol.json"
            if frozen.exists():
                protocol = read_json(frozen)
                validate_protocol(folder, protocol, labels)
                for cfg in protocol["configurations"]:
                    record("C_" + architecture, seed, "validation", cfg["threshold_mode"], cfg["validation_metrics"], frozen)
            else:
                record("C_" + architecture, seed, "validation", "fixed", metadata["metrics"], metadata_path)
                missing.append(f"C {architecture} seed {seed} frozen thresholds")
            result_path = folder / "test_results.json"
            if result_path.exists():
                value = read_json(result_path)
                if (not frozen.exists() or value["protocol_sha256"] != sha256(frozen)
                        or value["scores_sha256"] != sha256(folder / "test_scores.npz")):
                    raise ValueError("C test không khớp scores/protocol")
                for cfg in value["results"]:
                    record("C_" + architecture, seed, "test", cfg["threshold_mode"], cfg["metrics"], result_path)
            else:
                missing.append(f"C {architecture} seed {seed} test")
    if not (root / "data/processed/transformers/selected_model.json").exists():
        missing.append("D selected_model from 3 architectures × 3 seeds")
    return records, missing, sources


def summarize(records):
    """Std mẫu ddof=1, chỉ tính C đủ ba seed; A/B không giả độ lệch seed."""
    frame = pd.DataFrame(records)
    rows = []
    if frame.empty:
        return rows
    for (system, split, mode), group in frame.groupby(["system", "split", "threshold_mode"], sort=False):
        if system.startswith("C_"):
            if set(group["seed"].astype(int)) != set(SEEDS) or len(group) != len(SEEDS):
                continue
        elif len(group) != 1:
            raise ValueError("Trùng kết quả A/B")
        row = {"system": system, "split": split, "threshold_mode": mode, "n_runs": len(group)}
        for key in METRIC_NAMES:
            row[key + "_mean"] = float(group[key].mean())
            row[key + "_std"] = float(group[key].std(ddof=1)) if len(group) > 1 else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    records, missing, sources = collect()
    output = ROOT / "reports/project_results"
    output.mkdir(parents=True, exist_ok=True)
    averages = summarize(records)
    pd.DataFrame(records).to_csv(output / "all_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(averages).to_csv(output / "mean_std.csv", index=False, encoding="utf-8-sig")
    save_json(output / "summary.json", {"data_revision": REVISION, "seeds": SEEDS,
              "records": records, "averages": averages, "missing": missing,
              "complete": not missing, "sources": sources,
              "note": "Validation ngưỡng tuned là dữ liệu đã dùng chọn ngưỡng; không diễn giải thành test."})
    lines = ["# Kết quả thí nghiệm thực tế", "", "Std mẫu ddof=1; dấu — nghĩa là không áp dụng hoặc chưa đủ ba seed.", "",
             "| Hệ thống | Split | Ngưỡng | Số run | Macro-F1 mean±std | Micro-F1 mean±std | Hamming Loss |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in averages:
        def fmt(metric):
            value, std = row[metric + "_mean"], row[metric + "_std"]
            return f"{value:.4f}" + (f" ± {std:.4f}" if std is not None else "")
        lines.append(f"| {row['system']} | {row['split']} | {row['threshold_mode']} | {row['n_runs']} | {fmt('macro_f1')} | {fmt('micro_f1')} | {fmt('hamming_loss')} |")
    lines.extend(["", "**Bảng 5-3. Precision/Recall theo cùng split và cấu hình.**", "",
                  "| Hệ thống | Split | Ngưỡng | Macro-P | Macro-R | Micro-P | Micro-R |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for row in averages:
        cells = [row['system'], row['split'], row['threshold_mode']]
        for metric in ("macro_precision", "macro_recall", "micro_precision", "micro_recall"):
            mean, std = row[metric + "_mean"], row[metric + "_std"]
            cells.append(f"{mean:.4f}" + (f" ± {std:.4f}" if std is not None else ""))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Phần chưa có bằng chứng đầy đủ", "", *["- " + item for item in missing]])
    if not missing:
        lines.append("Đủ A/B/C, ba seed mỗi kiến trúc, test sau khóa protocol và lựa chọn mô hình C.")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ghi {len(records)} kết quả thật; còn thiếu {len(missing)} mục.")
    if args.require_complete and missing:
        raise SystemExit("Chưa đủ thí nghiệm; xem reports/project_results/RESULTS.md")


if __name__ == "__main__":
    main()
