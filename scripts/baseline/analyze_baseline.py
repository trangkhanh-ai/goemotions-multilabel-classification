"""Phân tích kết quả validation của baseline; không huấn luyện và không đọc test.

Chạy: python -m scripts.baseline.analyze_baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.baseline import (load_aligned_scores, load_run_metadata, tune_thresholds,
                          tune_global_threshold, label_error_pairs)
from src.datasets.goemotions import REVISION, load_goemotions, multi_hot
from src.evaluation.metrics import evaluate_multilabel


from src.paths import ROOT
BASE = ROOT / "data" / "processed" / "baseline"


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def error_examples(frame, truth, scores, labels, rare_ids):
    """Lấy ví dụ thật theo ID cho bốn kiểu lỗi. Người viết báo cáo cần đọc lại."""
    predicted = scores >= 0.5
    rows = []

    def add(category, sample_id, label_id):
        true_ids = np.flatnonzero(truth[sample_id])
        pred_ids = np.flatnonzero(predicted[sample_id])
        rows.append({
            "category": category,
            "id": str(frame.iloc[sample_id]["id"]),
            "text": str(frame.iloc[sample_id]["text"]),
            "focus_label": labels[label_id],
            "focus_score": round(float(scores[sample_id, label_id]), 4),
            "true_labels": ", ".join(labels[j] for j in true_ids),
            "predicted_labels": ", ".join(labels[j] for j in pred_ids),
        })

    # Nhãn hiếm bị bỏ sót: ưu tiên ví dụ sát ngưỡng 0,5 để phân tích quyết định.
    for label_id in rare_ids:
        candidates = np.flatnonzero(truth[:, label_id] & ~predicted[:, label_id])
        if len(candidates):
            closest = candidates[np.argmax(scores[candidates, label_id])]
            add("rare_false_negative", int(closest), label_id)

    # Nhãn dự đoán thừa: điểm cao vẫn có thể sai; cần xem cả khả năng thiếu nhãn gốc.
    fp = np.argwhere(predicted & ~truth.astype(bool))
    for index in np.argsort(scores[fp[:, 0], fp[:, 1]])[-5:][::-1]:
        sample_id, label_id = fp[index]
        add("false_positive", int(sample_id), int(label_id))

    # Mẫu nhiều cảm xúc: đo trường hợp đoán được một phần nhưng bỏ sót nhãn còn lại.
    partially_right = np.flatnonzero(
        (truth.sum(axis=1) > 1)
        & np.any(predicted & truth.astype(bool), axis=1)
        & np.any(truth.astype(bool) & ~predicted, axis=1)
    )
    for sample_id in partially_right[:5]:
        missed = np.flatnonzero(truth[sample_id] & ~predicted[sample_id])[0]
        add("partial_multi_label", int(sample_id), int(missed))

    # Không có nhãn nào vượt 0,5 dù mỗi mẫu trong split có ít nhất một nhãn thật.
    no_prediction = np.flatnonzero(~predicted.any(axis=1))
    near = no_prediction[np.argsort(scores[no_prediction].max(axis=1))[-5:][::-1]]
    for sample_id in near:
        label_id = int(np.argmax(scores[sample_id]))
        add("no_label_predicted", int(sample_id), label_id)
    return pd.DataFrame(rows)


def top_features(model, labels):
    """Từ/cặp từ có hệ số LR cao/thấp, chỉ là mối liên hệ thống kê."""
    vocabulary = model.named_steps["tfidf"].get_feature_names_out()
    classifiers = model.named_steps["classifier"].estimators_
    rows = []
    for label, classifier in zip(labels, classifiers):
        weights = classifier.coef_[0]
        for direction, feature_ids in (
            ("positive", np.argsort(weights)[-8:][::-1]),
            ("negative", np.argsort(weights)[:8]),
        ):
            for rank, feature_id in enumerate(feature_ids, start=1):
                rows.append({"label": label, "direction": direction, "rank": rank,
                             "feature": vocabulary[feature_id],
                             "coefficient": round(float(weights[feature_id]), 5)})
    return pd.DataFrame(rows)


def analyze_variant(name, folder, train, validation, labels, y_train, y_val, rare_ids):
    config = load_run_metadata(folder, name, labels)
    if config["n_train"] != len(train) or config["n_validation"] != len(validation):
        raise ValueError(f"Metadata {name} không khớp dữ liệu full đang mở")
    scores = load_aligned_scores(
        folder / "validation_scores.npz", validation["id"].tolist(), labels
    )
    model = joblib.load(folder / "model.joblib")  # Chỉ nạp file do chính nhóm vừa tạo.
    replay_scores = model.predict_proba(validation["text"].tolist())
    roundtrip_error = float(np.max(np.abs(scores - replay_scores)))
    if roundtrip_error > 1e-12:
        raise ValueError("Scores đã lưu không khớp dự đoán từ model đã nạp lại")
    fixed = evaluate_multilabel(y_val, scores, labels, threshold=0.5)
    global_threshold, curve = tune_global_threshold(y_val, scores)
    global_metrics = evaluate_multilabel(y_val, scores, labels, threshold=global_threshold)
    thresholds = tune_thresholds(y_val, scores)
    tuned = evaluate_multilabel(y_val, scores, labels, threshold=thresholds)

    save_json(folder / "thresholds_validation.json", {
        "variant": name,
        "data_revision": REVISION,
        "chosen_on": "validation",
        "grid": "0.05..0.95 step 0.05",
        "tie_break": "nearest to 0.5, then higher threshold",
        "warning": "F1 trên cùng validation dùng để chọn ngưỡng có thể lạc quan; test chưa đánh giá.",
        "label_names": labels,
        "artifact_sha256": config["artifact_sha256"],
        "global_threshold": global_threshold,
        "thresholds": thresholds.tolist(),
    })
    save_json(folder / "analysis_validation.json", {
        "variant": name,
        "artifact_sha256": config["artifact_sha256"],
        "fixed_0_5": fixed,
        "global_on_validation": global_metrics,
        "tuned_on_validation": tuned,
        "prediction_roundtrip_max_abs_error": roundtrip_error,
        "rare_labels_by_train_support": [labels[j] for j in rare_ids],
    })

    per_label = []
    for label_id, label in enumerate(labels):
        a, b = fixed["per_label"][label_id], tuned["per_label"][label_id]
        g = global_metrics["per_label"][label_id]
        per_label.append({
            "label_id": label_id, "label": label,
            "train_support": int(y_train[:, label_id].sum()),
            "validation_support": int(y_val[:, label_id].sum()),
            "threshold_fixed": 0.5, "f1_fixed": a["f1"],
            "precision_fixed": a["precision"], "recall_fixed": a["recall"],
            "tp_fixed": a["tp"], "fp_fixed": a["fp"], "fn_fixed": a["fn"], "tn_fixed": a["tn"],
            "threshold_global": global_threshold, "f1_global_val": g["f1"],
            "threshold_tuned": thresholds[label_id], "f1_tuned_val": b["f1"],
            "precision_tuned_val": b["precision"], "recall_tuned_val": b["recall"],
            "f1_change_on_val": b["f1"] - a["f1"],
        })
    pd.DataFrame(per_label).to_csv(folder / "per_label_validation.csv", index=False,
                                    encoding="utf-8-sig")
    examples = error_examples(validation, y_val, scores, labels, rare_ids)
    examples.to_csv(folder / "error_examples_validation.csv", index=False,
                    encoding="utf-8-sig")
    top_features(model, labels).to_csv(folder / "top_features.csv", index=False,
                                       encoding="utf-8-sig")
    pairs = label_error_pairs(y_val, scores, labels)
    for pair in pairs:
        pair["example_id"] = str(validation.iloc[pair.pop("example_row")]["id"])
    pd.DataFrame(pairs, columns=["missed_label", "extra_label", "count",
                                "missed_label_total_fn", "fraction_of_fn", "example_id"]).to_csv(
        folder / "label_error_pairs_validation.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(curve).to_csv(folder / "threshold_curve_validation.csv", index=False)
    return {"name": name, "folder": folder, "fixed": fixed, "tuned": tuned,
            "global": global_metrics, "global_threshold": global_threshold,
            "thresholds": thresholds, "per_label": per_label, "config": config,
            "examples": examples, "pairs": pairs}


def write_report(results, labels, rare_ids, y_train, y_val):
    """Tạo phần kết quả có số liệu thật để bạn dùng khi viết báo cáo tuần 4/7."""
    lines = [
        "# Kết quả phần A — baseline cổ điển GoEmotions",
        "",
        "Bản này sinh tự động từ các file validation đã chạy. Dữ liệu: GoEmotions "
        f"`simplified`, revision `{REVISION}`; train 43.410, validation 5.426, "
        "28 nhãn. **Chưa dùng nhãn test.**",
        "",
        "## Thiết lập",
        "",
        "- TF-IDF unigram và bigram, `min_df=2`, tối đa 100.000 đặc trưng; fit trên train.",
        "- One-vs-Rest Logistic Regression: 28 bộ phân loại nhị phân, `C=1`, "
        "`solver=liblinear`, `max_iter=1000`. Standard không có trọng số lớp; "
        "balanced dùng `class_weight='balanced'` tính từ train cho từng bộ phân loại.",
        "- Dùng cùng split và thứ tự nhãn, ngưỡng gốc 0,5. Macro-F1 tính trung bình "
        "F1 của đủ 28 nhãn; `zero_division=0`.",
        "",
        "## Bảng validation",
        "",
        "| Cấu hình | Ngưỡng | Macro-F1 | Micro-F1 | Micro-P | Micro-R | Macro-P | Macro-R | Hamming Loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for mode, metric in (("Cố định 0,5", result["fixed"]),
                             (f"Chung {result['global_threshold']:.2f}, chọn trên val", result["global"]),
                             ("Chọn theo từng nhãn trên val", result["tuned"])):
            lines.append(
                f"| {result['name']} | {mode} | {metric['macro_f1']:.4f} | "
                f"{metric['micro_f1']:.4f} | {metric['micro_precision']:.4f} | "
                f"{metric['micro_recall']:.4f} | {metric['macro_precision']:.4f} | "
                f"{metric['macro_recall']:.4f} | {metric['hamming_loss']:.4f} |"
            )
    lines += [
        "",
        "**Cách hiểu:** các hàng chọn ngưỡng được đo trên chính validation đã dùng "
        "để chọn ngưỡng; mức tăng ở đó có thể lạc quan. Chỉ dùng test một lần sau khi "
        "cả nhóm khóa cấu hình để xác nhận kết luận.",
        "",
        "## Môi trường và thời gian đo",
        "",
        "| Biến thể | Python | scikit-learn | Số đặc trưng TF-IDF | Fit (giây) | Dự đoán val (giây) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for result in results:
        config = result["config"]
        lines.append(f"| {result['name']} | {config['environment']['python']} | "
                     f"{config['environment']['scikit_learn']} | "
                     f"{config['n_tfidf_features']} | {config['fit_seconds']:.2f} | "
                     f"{config['predict_validation_seconds']:.2f} |")
    lines += [
        "",
        "Thời gian phụ thuộc máy và cache; xem JSON để có đủ phiên bản thư viện.",
        "",
        "## Năm nhãn hiếm nhất theo số mẫu dương trên train",
        "",
        "Danh sách được chọn bằng train trước khi nhìn kết quả validation.",
        "",
        "| Nhãn | Train + | Val + | " + " | ".join(
            f"{r['name']} F1@0,5 | {r['name']} F1 tuned-val" for r in results
        ) + " |",
        "|---|---:|---:|" + "---:|---:|" * len(results),
    ]
    for j in rare_ids:
        cells = [labels[j], str(int(y_train[:, j].sum())), str(int(y_val[:, j].sum()))]
        for result in results:
            row = result["per_label"][j]
            cells += [f"{row['f1_fixed']:.4f}", f"{row['f1_tuned_val']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Phân tích lỗi và khả năng giải thích",
        "",
        "Mỗi biến thể lưu `error_examples_validation.csv` với ID, văn bản, nhãn thật, "
        "nhãn dự đoán và điểm số. Bốn nhóm: bỏ sót nhãn hiếm, dự đoán nhãn thừa, "
        "đúng một phần ở mẫu đa nhãn, và không dự đoán nhãn nào. "
        "Cần đọc lại từng ví dụ trước khi trích vào báo cáo, vì nhãn gốc cũng có thể thiếu.",
        "",
        "Các nhóm FP/FN là dấu hiệu thống kê; khi trình bày cần giải thích thêm về "
        "ngữ cảnh, từ ngữ, nhiều cảm xúc hoặc ít mẫu của nhãn. Số ví dụ được chọn "
        "không phải tỷ lệ lỗi của toàn split.",
        "",
    ]
    # Một ví dụ thật cho mỗi nhóm, lấy từ validation của baseline gốc.
    preferred = {
        "rare_false_negative": "eczwil0",
        "false_positive": "ed832y6",
        "partial_multi_label": "eczdvun",
        "no_label_predicted": "eeoh5vh",
    }
    examples = results[0]["examples"]
    for category, wanted_id in preferred.items():
        subset = examples[examples["category"] == category]
        if subset.empty:
            continue
        row = subset[subset["id"] == wanted_id]
        item = (row.iloc[0] if not row.empty else subset.iloc[0])
        excerpt = str(item["text"]).replace("\n", " ").replace("`", "'")
        lines.append(f"- **{category}**, ID `{item['id']}`: `{excerpt}` "
                     f"— thật: {item['true_labels']}; đoán: "
                     f"{item['predicted_labels'] or '(không nhãn)'}. "
                     f"Điểm {item['focus_label']} = {item['focus_score']:.4f}.")
    lines += [
        "",
        "`top_features.csv` ghi từ/cặp từ có hệ số LR cao và thấp cho từng nhãn. "
        "Đây là liên hệ thống kê trong mô hình, không chứng minh nguyên nhân cảm xúc.",
        "",
        "## Cặp nhãn bị bỏ sót và dự đoán thừa trong cùng câu",
        "",
        "Đếm FN của nhãn thật A đồng thời FP của nhãn B. Đây là bảng lỗi đa nhãn, "
        "không phải ma trận nhầm lẫn một lớp và không phải bảng đồng xuất hiện nhãn thật. "
        "Một câu có thể đóng góp nhiều cặp; neutral được giữ nguyên theo nguồn.",
        "",
        "| Variant @0,5 | Nhãn bỏ sót | Nhãn thừa | Số câu | Tổng FN nhãn bỏ sót | Tỷ lệ trong FN |",
        "|---|---|---|---:|---:|---:|",
    ]
    for result in results:
        for pair in result["pairs"][:8]:
            lines.append(f"| {result['name']} | {pair['missed_label']} | "
                         f"{pair['extra_label']} | {pair['count']} | "
                         f"{pair['missed_label_total_fn']} | {pair['fraction_of_fn']:.1%} |")
    lines += [
        "",
        "Để đọc các cặp cảm xúc cụ thể, bảng phụ sau chỉ lấy cặp không có neutral. "
        "Đây là cách trình bày thêm; metric vẫn tính đủ 28 nhãn.",
        "",
        "| Variant @0,5 | Nhãn bỏ sót | Nhãn thừa | Số câu | ID ví dụ |",
        "|---|---|---|---:|---|",
    ]
    for result in results:
        emotion_pairs = [pair for pair in result["pairs"]
                         if "neutral" not in (pair["missed_label"], pair["extra_label"])]
        for pair in emotion_pairs[:5]:
            lines.append(f"| {result['name']} | {pair['missed_label']} | {pair['extra_label']} | "
                         f"{pair['count']} | {pair['example_id']} |")
    lines += [
        "",
        "## Số nhãn được dự đoán",
        "",
        "| Variant @0,5 | Câu không dự đoán nhãn | Trung bình nhãn/câu |",
        "|---|---:|---:|",
    ]
    for result in results:
        metric = result["fixed"]
        lines.append(f"| {result['name']} | {metric['empty_prediction_count']} | "
                     f"{metric['mean_predicted_labels']:.3f} |")
    lines += [
        "",
        "## Bằng chứng chạy lại và bàn giao",
        "",
    ]
    for result in results:
        folder = result["folder"].relative_to(ROOT).as_posix()
        lines.append(f"- `{result['name']}`: `{folder}/validation_metrics.json`, "
                     "`validation_scores.npz`, `per_label_validation.csv`, "
                     "`thresholds_validation.json`, `error_examples_validation.csv`, "
                     "`label_error_pairs_validation.csv`, `threshold_curve_validation.csv`, "
                     "`top_features.csv`, `model.joblib`.")
    lines += [
        "- `validation_scores.npz` gồm `ids`, `scores` N×28, `label_names`; "
        "ghép theo ID, không ghép theo thứ tự dòng. Các file lớn nằm trong "
        "`data/processed/` và được Git bỏ qua.",
        "- Môi trường, thời gian fit, số đặc trưng và cấu hình nằm trong "
        "`validation_metrics.json` của từng biến thể.",
        "- Hash model/scores đã được kiểm, nạp lại model dự đoán toàn validation "
        "và đối chiếu với scores lưu trước đó, sai số tối đa ≤1e-12.",
        "- Báo cáo này chỉ mô tả phần A. Nhóm vẫn cần B zero-shot, ba kiến trúc C "
        "mỗi kiến trúc ba seed, demo từ C tốt nhất và so sánh lỗi giữa C1/C2/C3.",
        "",
        "## Nguồn phương pháp",
        "",
        "- [Google Research: GoEmotions](https://github.com/google-research/google-research/blob/master/goemotions/README.md)",
        "- [scikit-learn: TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)",
        "- [scikit-learn: OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)",
        "- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)",
        "- [scikit-learn: Precision, Recall, F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)",
        "- [scikit-learn 1.7.2: chọn ngưỡng](https://scikit-learn.org/1.7/modules/classification_threshold.html)",
        "",
    ]
    output = ROOT / "reports" / "BASELINE_RESULTS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main():
    frames, labels, _ = load_goemotions(
        ROOT, write_metadata=False, splits=("train", "validation")
    )
    train, validation = frames["train"], frames["validation"]
    y_train = multi_hot(train["labels"].tolist(), len(labels))
    y_val = multi_hot(validation["labels"].tolist(), len(labels))
    rare_ids = np.argsort(y_train.sum(axis=0))[:5]
    results = []
    for name, folder in (("standard", BASE / "full"),
                         ("balanced", BASE / "balanced" / "full")):
        if (folder / "validation_scores.npz").exists():
            results.append(analyze_variant(name, folder, train, validation, labels,
                                           y_train, y_val, rare_ids))
    if not results:
        raise FileNotFoundError("Chạy python -m scripts.baseline.run_baseline trước")
    output = write_report(results, labels, rare_ids, y_train, y_val)
    print(f"Analyzed {len(results)} variants on validation; report: "
          f"{output.relative_to(ROOT).as_posix()}")
    for result in results:
        print(f"{result['name']}: Macro-F1 @0,5={result['fixed']['macro_f1']:.4f}; "
              f"tuned-val={result['tuned']['macro_f1']:.4f}")
    print("Test not evaluated. Tuned-validation gains require final confirmation after freeze.")


if __name__ == "__main__":
    main()
