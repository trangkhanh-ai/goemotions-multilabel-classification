"""So sánh ba nhóm lỗi kiểm chứng được giữa C1/C2/C3 trên cùng ID.

python -m scripts.analysis.analyze_project_errors --split validation
Không huấn luyện, không suy luận mới và không chọn seed/ngưỡng bằng test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baseline import load_aligned_scores
from src.datasets.goemotions import REVISION, load_goemotions, multi_hot, sha256
from src.evaluation.protocols import read_json, save_json, validate_protocol
from src.evaluation.metrics import evaluate_multilabel, validate_multilabel_inputs, validate_thresholds
from src.models.transformer import (ARCHITECTURES, configure_console, load_transformer_run,
                        run_folder, summarize_architectures)


from src.paths import ROOT
CATEGORIES = ("partial_multi_label", "rare_false_negative", "missed_extra_pair")


def representative_run(runs):
    """Chọn bằng val Macro-F1@0.5, hòa chọn seed nhỏ; không nhìn test."""
    if not runs or len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("Cần các run với seed khác nhau")
    return max(runs, key=lambda run: (run["metrics"]["macro_f1"], -run["seed"]))


def error_masks(truth, scores, rare_ids, threshold=0.5):
    """Trả mask theo câu; một câu có thể thuộc nhiều nhóm lỗi."""
    truth, scores = validate_multilabel_inputs(truth, scores)
    rare_ids = np.asarray(rare_ids, dtype=int)
    if rare_ids.ndim != 1 or not len(rare_ids) or len(set(rare_ids)) != len(rare_ids):
        raise ValueError("rare_ids phải là danh sách nhãn hiếm không rỗng và không trùng")
    if np.any((rare_ids < 0) | (rare_ids >= truth.shape[1])):
        raise ValueError("ID nhãn hiếm nằm ngoài mapping")
    limits = validate_thresholds(threshold, truth.shape[1])
    actual = truth.astype(bool)
    predicted = scores >= limits
    true_positive = actual & predicted
    missed = actual & ~predicted
    extra = ~actual & predicted
    masks = {
        "partial_multi_label": (actual.sum(axis=1) >= 2) & true_positive.any(axis=1) & missed.any(axis=1),
        "rare_false_negative": missed[:, rare_ids].any(axis=1),
        "missed_extra_pair": missed.any(axis=1) & extra.any(axis=1),
    }
    return masks, predicted, missed, extra


def load_representatives(root, seeds, weighted=False):
    """Cần đủ >=3 full seeds/kiến trúc; kiểm artifact bằng module C dùng chung."""
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Cần ít nhất 3 seed khác nhau cho mỗi kiến trúc")
    groups = {}
    for architecture in ARCHITECTURES:
        groups[architecture] = []
        for seed in seeds:
            folder = run_folder(root, architecture, seed, weighted=weighted)
            run = load_transformer_run(folder, require_full=True)
            if run["architecture"] != architecture or run["seed"] != seed:
                raise ValueError("Kiến trúc/seed không khớp folder")
            run["run_dir"] = folder.relative_to(root).as_posix()
            groups[architecture].append(run)
    # Kiểm cùng seeds/data và không trộn cấu hình/revision trong mỗi kiến trúc.
    summaries = summarize_architectures(groups)
    return {name: representative_run(runs) for name, runs in groups.items()}, summaries


def checked_threshold(root, run, labels, split, mode):
    """Chặn test trước khi mở dữ liệu nếu protocol/kết quả test chưa có hoặc bị sửa."""
    folder = Path(root) / run["run_dir"]
    needs_protocol = split == "test" or mode != "fixed"
    protocol = None
    if needs_protocol:
        path = folder / "final_protocol.json"
        protocol = read_json(path)
        validate_protocol(folder, protocol, labels)
        checkpoint = run.get("checkpoint", run.get("model_checkpoint"))
        revision = run.get("model_revision", run.get("checkpoint_revision"))
        if (protocol.get("checkpoint") != checkpoint or protocol.get("model_revision") != revision
                or protocol.get("architecture") != run["architecture"] or protocol.get("seed") != run["seed"]
                or protocol.get("selection_data") != "validation" or not protocol.get("frozen_at_utc")):
            raise ValueError("Protocol không khóa đúng checkpoint/seed hoặc không chọn trên validation")
    if split == "test":
        result = read_json(folder / "test_results.json")
        if (result.get("split") != "test" or result.get("method") != "C"
                or result.get("architecture") != run["architecture"] or result.get("seed") != run["seed"]
                or result.get("protocol_sha256") != sha256(folder / "final_protocol.json")
                or result.get("scores_sha256") != sha256(folder / "test_scores.npz")):
            raise ValueError("Kết quả test chưa hoàn tất hoặc hash/seed không khớp protocol")
        if mode not in [item.get("threshold_mode") for item in result.get("results", [])]:
            raise ValueError("Kết quả test không chứa cấu hình ngưỡng cần phân tích")
    threshold = 0.5
    if protocol is not None:
        selected = [item for item in protocol["configurations"] if item["threshold_mode"] == mode]
        if len(selected) != 1:
            raise ValueError("Protocol thiếu cấu hình ngưỡng đã yêu cầu")
        threshold = selected[0]["thresholds"]
        if mode == "fixed" and not np.all(np.asarray(threshold, dtype=float) == 0.5):
            raise ValueError("So sánh chính fixed phải dùng ngưỡng 0.5 theo kế hoạch")
    validate_thresholds(threshold, len(labels))
    return threshold


def analyze_arrays(frame, labels, rare_ids, train_support, runs, arrays, thresholds, max_examples=5):
    """Tách phần thống kê thuần dữ liệu để unit test không cần tải model."""
    if max_examples < 1:
        raise ValueError("max_examples phải dương")
    truth = multi_hot(frame["labels"].tolist(), len(labels))
    counts, pairs, rare_rows, computed = [], [], [], {}
    for architecture, run in runs.items():
        scores = arrays[architecture]
        masks, predicted, missed, extra = error_masks(truth, scores, rare_ids, thresholds[architecture])
        computed[architecture] = {"scores": scores, "masks": masks, "predicted": predicted,
                                  "missed": missed, "extra": extra}
        eligible = {"partial_multi_label": truth.sum(axis=1) >= 2,
                    "rare_false_negative": truth[:, rare_ids].any(axis=1),
                    "missed_extra_pair": np.ones(len(frame), dtype=bool)}
        for category, mask in masks.items():
            denominator = int(eligible[category].sum())
            counts.append({"architecture": architecture, "seed": run["seed"], "category": category,
                           "error_samples": int(mask.sum()), "total_samples": len(frame),
                           "eligible_samples": denominator,
                           "fraction_all_samples": float(mask.mean()),
                           "fraction_eligible": float(mask.sum() / denominator) if denominator else 0.0})
        # FN(label A) và FP(label B) trong cùng câu; một câu đóng góp nhiều cặp.
        pair_counts = missed.astype(np.int64).T @ extra.astype(np.int64)
        for a, b in np.argwhere(pair_counts > 0):
            occurrences = np.flatnonzero(missed[:, a] & extra[:, b])
            pairs.append({"architecture": architecture, "seed": run["seed"],
                          "missed_label": labels[a], "extra_label": labels[b],
                          "sample_count": int(pair_counts[a, b]),
                          "missed_label_total_fn": int(missed[:, a].sum()),
                          "fraction_of_missed_label_fn": float(pair_counts[a, b] / missed[:, a].sum()),
                          "example_id": str(frame.iloc[int(occurrences[0])]["id"])})
        metric = evaluate_multilabel(truth, scores, labels, thresholds[architecture])
        for j in rare_ids:
            per = metric["per_label"][j]
            rare_rows.append({"architecture": architecture, "seed": run["seed"], "label": labels[j],
                              "train_positive_count": int(train_support[j]),
                              "evaluation_positive_count": per["support"], "tp": per["tp"],
                              "fn": per["fn"], "fp": per["fp"], "precision": per["precision"],
                              "recall": per["recall"], "f1": per["f1"]})

    examples = []
    for category in CATEGORIES:
        # Một tập ID chung cho ba mô hình; chọn có thứ tự, không chọn câu thuận lợi.
        union = np.logical_or.reduce([item["masks"][category] for item in computed.values()])
        chosen = sorted(np.flatnonzero(union), key=lambda i: str(frame.iloc[int(i)]["id"]))[:max_examples]
        for i in chosen:
            for architecture, run in runs.items():
                item = computed[architecture]
                scores = item["scores"][i]
                predicted = item["predicted"][i]
                examples.append({"category": category, "id": str(frame.iloc[int(i)]["id"]),
                                 "architecture": architecture, "seed": run["seed"],
                                 "error_present": bool(item["masks"][category][i]),
                                 "text": str(frame.iloc[int(i)]["text"]),
                                 "true_labels": json.dumps([labels[j] for j in np.flatnonzero(truth[i])], ensure_ascii=False),
                                 "predicted_labels": json.dumps([labels[j] for j in np.flatnonzero(predicted)], ensure_ascii=False),
                                 "missed_labels": json.dumps([labels[j] for j in np.flatnonzero(item["missed"][i])], ensure_ascii=False),
                                 "extra_labels": json.dumps([labels[j] for j in np.flatnonzero(item["extra"][i])], ensure_ascii=False),
                                 "scores_by_label": json.dumps(dict(zip(labels, map(float, scores))), ensure_ascii=False),
                                 "manual_linguistic_notes": ""})
    return counts, pairs, rare_rows, examples


def analyze_project(root, *, split="validation", seeds=(42, 123, 2026), weighted=False,
                    threshold_mode="fixed", max_examples=5, output=None):
    root = Path(root).resolve()
    if split not in ("validation", "test") or threshold_mode not in ("fixed", "global", "tuned"):
        raise ValueError("split hoặc threshold_mode không hợp lệ")
    labels = read_json(root / "data/labels.json")
    runs, summaries = load_representatives(root, list(seeds), weighted)
    if any(run["label_names"] != labels for run in runs.values()):
        raise ValueError("Mapping nhãn đại diện không khớp dữ liệu chung")
    # Mọi protocol/test result phải được kiểm trước khi mở train + split cần phân tích.
    thresholds = {name: checked_threshold(root, run, labels, split, threshold_mode)
                  for name, run in runs.items()}
    frames, names, data_manifest = load_goemotions(root, write_metadata=False, splits=("train", split))
    if names != labels:
        raise ValueError("Nhãn dữ liệu không khớp")
    frame = frames[split]
    if frame["id"].duplicated().any():
        raise ValueError("ID split bị trùng")
    train_truth = multi_hot(frames["train"]["labels"].tolist(), len(labels))
    train_support = train_truth.sum(axis=0)
    rare_ids = sorted(range(len(labels)), key=lambda j: (int(train_support[j]), j))[:5]
    arrays = {name: load_aligned_scores(root / run["run_dir"] / f"{split}_scores.npz", frame["id"], labels)
              for name, run in runs.items()}
    counts, pairs, rare_rows, examples = analyze_arrays(frame, labels, rare_ids, train_support,
                                                       runs, arrays, thresholds, max_examples)
    variant = "weighted" if weighted else "standard"
    output = Path(output) if output else root / "reports" / f"errors_{split}_{variant}_{threshold_mode}"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts).to_csv(output / "counts.csv", index=False, encoding="utf-8-sig")
    pair_columns = ["architecture", "seed", "missed_label", "extra_label", "sample_count",
                    "missed_label_total_fn", "fraction_of_missed_label_fn", "example_id"]
    pd.DataFrame(pairs, columns=pair_columns).sort_values(
        ["architecture", "sample_count", "missed_label", "extra_label"], ascending=[True, False, True, True]
    ).to_csv(output / "pairs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rare_rows).to_csv(output / "rare_fn.csv", index=False, encoding="utf-8-sig")
    example_columns = ["category", "id", "architecture", "seed", "error_present", "text", "true_labels",
                       "predicted_labels", "missed_labels", "extra_labels", "scores_by_label", "manual_linguistic_notes"]
    pd.DataFrame(examples, columns=example_columns).to_csv(output / "examples.csv", index=False, encoding="utf-8-sig")
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "split": split,
                "data_revision": REVISION, "label_names": labels, "sample_count": len(frame),
                "data_sha256": {item["split"]: item["sha256"] for item in data_manifest["files"]},
                "variant": variant, "threshold_mode": threshold_mode,
                "representative_selection": "highest validation Macro-F1@0.5; smaller seed on tie",
                "rare_label_selection": "five smallest train positive counts; label ID on tie",
                "rare_labels": [{"label": labels[j], "train_support": int(train_support[j])} for j in rare_ids],
                "architecture_summary_validation": summaries,
                "runs": [{"architecture": name, "seed": run["seed"], "run_dir": run["run_dir"],
                          "validation_macro_f1_fixed": run["metrics"]["macro_f1"],
                          "thresholds": thresholds[name],
                          "run_metadata_sha256": sha256(root / run["run_dir"] / "run_metadata.json"),
                          "scores_sha256": sha256(root / run["run_dir"] / f"{split}_scores.npz")}
                         for name, run in runs.items()],
                "notes": "Error groups overlap. Representative seed diagnostics are not mean±std. Linguistic notes require human review."}
    save_json(output / "manifest.json", manifest)
    lines = [f"# So sánh lỗi C1/C2/C3 — {split}, {variant}, {threshold_mode}", "",
             "Đếm trên cùng ID. Seed đại diện chọn bằng validation @0,5; đây là phân tích "
             "checkpoint đại diện, không phải trung bình lỗi qua ba seed.", "",
             "| Kiến trúc | Seed | Nhóm lỗi | Số câu | Mẫu phù hợp định nghĩa | Tỷ lệ trong mẫu phù hợp |",
             "|---|---:|---|---:|---:|---:|"]
    for row in counts:
        lines.append(f"| {row['architecture']} | {row['seed']} | {row['category']} | "
                     f"{row['error_samples']} | {row['eligible_samples']} | {row['fraction_eligible']:.2%} |")
    lines += ["", "Ba nhóm có thể chồng lấp; không cộng tỷ lệ thành 100%. `pairs.csv` đếm "
              "FN(A)+FP(B) đồng thời, khác với nhãn thật đồng xuất hiện. Một câu có thể đóng góp nhiều cặp.",
              "", "`examples.csv` lấy cùng tập ID cho ba mô hình; `error_present=false` nghĩa "
              "là mô hình đó không gặp nhóm lỗi đang xét tại ID này. Đọc text, nhãn, điểm rồi "
              "điền `manual_linguistic_notes`; không tự quy kết mỉa mai/phủ định hoặc nguyên nhân.",
              "", "Định nghĩa và quy trình: `docs/EXPERIMENTS.md`. Manifest lưu nguồn/hashes."]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--threshold-mode", choices=("fixed", "global", "tuned"), default="fixed")
    parser.add_argument("--max-examples", type=int, default=5, help="Số ID chung tối đa cho mỗi nhóm lỗi")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_project(ROOT, split=args.split, seeds=args.seeds, weighted=args.weighted,
                             threshold_mode=args.threshold_mode, max_examples=args.max_examples, output=args.output)
    print(f"Đã phân tích {result['sample_count']:,} câu {args.split}, 3 kiến trúc, 3 nhóm lỗi.")


if __name__ == "__main__":
    main()
