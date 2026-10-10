"""Chèn bảng số từ artifacts thật vào báo cáo; không tính lại/chọn bằng test."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


ERROR_CATEGORIES = ("partial_multi_label", "rare_false_negative", "missed_extra_pair")


def select_error_case_ids(examples):
    """Chọn ID có đủ ba C, ưu tiên ID khác nhau; thiếu được giữ nguyên."""
    choices = []
    for category in ERROR_CATEGORIES:
        valid = []
        for sample_id in sorted({row["id"] for row in examples if row["category"] == category}):
            rows = [row for row in examples if row["category"] == category and row["id"] == sample_id]
            if len(rows) != 3 or {row["architecture"] for row in rows} != {"bert", "roberta", "distilbert"}:
                continue
            if len({(row["text"], row["true_labels"]) for row in rows}) != 1:
                raise ValueError(f"Ví dụ {sample_id}: text/nhãn thật khác nhau giữa ba C")
            valid.append(sample_id)
        choices.append(valid or [None])
    # Chỉ ba nhóm, mỗi nhóm tối đa vài ID: thử tổ hợp để tránh việc chọn
    # sớm một ID làm nhóm sau bị trùng dù thực ra có bộ ba khác nhau.
    best = max(itertools.product(*choices),
               key=lambda ids: len({sample_id for sample_id in ids if sample_id is not None}))
    return {category: sample_id for category, sample_id in zip(ERROR_CATEGORIES, best)
            if sample_id is not None}


def error_case_examples():
    """Fallback từ CSV thật khi chưa có hồ sơ đọc lỗi bằng văn bản."""
    path = ROOT / "reports/errors_test_standard_fixed/examples.csv"
    if not path.exists():
        return ["Chưa có CSV ví dụ test của ba C; chưa xác nhận đủ ba case đối chiếu."]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        examples = list(csv.DictReader(stream))
    selected = select_error_case_ids(examples)
    lines = ["Các ví dụ sau trích tự động theo ID từ CSV của checkpoint đại diện đã chọn bằng validation. "
             "Nhóm lỗi có thể chồng lấp. Khi trường manual_linguistic_notes còn trống, chưa có "
             "nhận xét ngôn ngữ được ghi trong CSV; không tự quy kết mỉa mai, phủ định hoặc nguyên nhân."]
    used = set()
    for index, category in enumerate(ERROR_CATEGORIES):
        if category not in selected:
            lines += ["", f"Nhóm {category}: chưa có một ID đủ dữ liệu đối chiếu ba C."]
            continue
        sample_id = selected[category]
        rows = [row for row in examples if row["category"] == category and row["id"] == sample_id]
        lines += ["", f"**Case {index + 1}: {category}, ID {sample_id}.**",
                  f"Văn bản: “{rows[0]['text']}”. Nhãn thật: {', '.join(json.loads(rows[0]['true_labels']))}.", ""]
        if sample_id in used:
            lines += ["ID trùng case trước vì trong nhóm này không còn ID khác đủ ba C; "
                      "đây là nhóm lỗi chồng lấp, không tính thành một câu mới.", ""]
        used.add(sample_id)
        lines += [f"**Bảng 5-2{chr(ord('g') + index)}. Ba C trên cùng ID {sample_id}.**", "",
                  "| C | Seed | Nhãn dự đoán | Bỏ sót | Nhãn thừa | Score nhãn bỏ sót/thừa | Gặp nhóm lỗi |",
                  "|---|---:|---|---|---|---|---|"]
        for row in sorted(rows, key=lambda item: ("bert", "roberta", "distilbert").index(item["architecture"])):
            predicted, missed, extra = (json.loads(row[key]) for key in ("predicted_labels", "missed_labels", "extra_labels"))
            scores = json.loads(row["scores_by_label"])
            related = ", ".join(f"{label}={scores[label]:.4f}" for label in dict.fromkeys(missed + extra)) or "—"
            name = {"bert": "C1 BERT", "roberta": "C2 RoBERTa", "distilbert": "C3 DistilBERT"}[row["architecture"]]
            cells = [name, row["seed"], ", ".join(predicted) or "không nhãn", ", ".join(missed) or "không",
                     ", ".join(extra) or "không", related, row["error_present"]]
            lines.append("| " + " | ".join(str(cell).replace("|", "/").replace("\n", " ") for cell in cells) + " |")
        for row in rows:
            if row.get("manual_linguistic_notes"):
                lines += ["", f"Nhận xét đã ghi ({row['architecture']}): {row['manual_linguistic_notes']}"]
    lines += ["", f"Số ID khác nhau được đối chiếu ở đây: {len(used)}. Đầy đủ scores 28 nhãn/những ví dụ khác "
              "nằm trong reports/errors_test_standard_fixed/examples.csv; nhóm cần đọc nội dung và ghi hồ sơ case study."]
    return lines


def final_abstract(summary):
    """Tóm tắt chỉ có số test khi bảng thực nghiệm đã qua đủ các điều kiện."""
    if not summary["complete"]:
        return None
    lookup = {(r["system"], r["split"], r["threshold_mode"]): r
              for r in summary["averages"]}
    selection = json.loads((ROOT / "data/processed/transformers/selected_model.json").read_text(encoding="utf-8"))
    system = "C_" + selection["architecture"]
    original = [lookup[("C_" + architecture, "validation", "fixed")]
                for architecture in ("bert", "roberta", "distilbert")]
    validation = lookup[(system, "validation", "fixed")]
    if (selection.get("method") != "C" or selection.get("smoke") is not False
            or any(row["n_runs"] != 3 for row in original)
            or not math.isclose(validation["macro_f1_mean"], max(row["macro_f1_mean"] for row in original),
                                rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Selection C không khớp winner mean validation của đủ ba seed")
    before = lookup[("A_standard", "test", "fixed")]
    after = lookup[("A_balanced", "test", "tuned")]
    selected = lookup[(system, "test", "fixed")]
    tuned = lookup[(system, "test", "tuned")]
    if selected["n_runs"] != 3 or tuned["n_runs"] != 3:
        raise ValueError("Tóm tắt C cần đủ ba seed; không thay bằng điểm checkpoint demo")
    return (f"Phần A đã được đo trên toàn bộ 5.427 mẫu test sau khi khóa cấu hình trên validation. "
            f"Bản standard với ngưỡng 0,5 đạt Macro-F1 {before['macro_f1_mean']:.4f}; "
            f"bản balanced với ngưỡng riêng đạt {after['macro_f1_mean']:.4f}. "
            f"Kiến trúc C được chọn bằng mean Macro-F1 validation @0,5 là {selection['architecture']} "
            f"({validation['macro_f1_mean']:.4f} ± {validation['macro_f1_std']:.4f}); "
            f"trên test, kiến trúc này đạt Macro-F1 {selected['macro_f1_mean']:.4f} ± {selected['macro_f1_std']:.4f} "
            f"và Micro-F1 {selected['micro_f1_mean']:.4f} ± {selected['micro_f1_std']:.4f}. "
            f"Ngưỡng riêng chọn trên validation đưa Macro-F1 test tới "
            f"{tuned['macro_f1_mean']:.4f} ± {tuned['macro_f1_std']:.4f} "
            f"(chênh lệch mean {tuned['macro_f1_mean'] - selected['macro_f1_mean']:+.4f}). "
            f"Mean và sample std tính giữa ba seed 42, 123, 2026; seed {selection['seed']} "
            "của demo là một checkpoint đại diện, không phải ensemble hay điểm trung bình. "
            "Báo cáo giữ cả các thay đổi F1 âm của nhãn hiếm, đối chiếu lỗi giữa ba C và thảo luận giá trị "
            "ứng dụng dự kiến. Kết quả này chưa xác nhận hiệu quả trên dữ liệu tiếng Việt hoặc ROI công nghiệp.")


def interpret_results(summary):
    """Nhận xét chỉ từ bảng thật; thứ hạng C dùng validation đã quy định."""
    rows = summary["averages"]
    original = [r for r in rows if r["system"].startswith("C_")
                and r["split"] == "validation" and r["threshold_mode"] == "fixed"]
    if len(original) != 3 or any(r["n_runs"] != 3 for r in original):
        return []
    ordered = sorted(original, key=lambda r: -r["macro_f1_mean"])
    selection_path = ROOT / "data/processed/transformers/selected_model.json"
    if selection_path.exists():
        selected_system = "C_" + json.loads(selection_path.read_text(encoding="utf-8"))["architecture"]
        selected = next((row for row in original if row["system"] == selected_system), None)
        if selected and math.isclose(selected["macro_f1_mean"], ordered[0]["macro_f1_mean"], rel_tol=0, abs_tol=1e-12):
            ordered = [selected] + [row for row in ordered if row["system"] != selected_system]
    best, worst = ordered[0], ordered[-1]
    stable = min(original, key=lambda r: r["macro_f1_std"])
    lines = ["", "### 5.2.6. Thứ hạng, độ ổn định và đánh đổi", "",
             f"Theo tiêu chí mean Macro-F1 validation @0,5 đã chốt, {best['system']} "
             f"đạt cao nhất ({best['macro_f1_mean']:.4f} ± {best['macro_f1_std']:.4f}); "
             f"{worst['system']} thấp nhất ({worst['macro_f1_mean']:.4f} ± {worst['macro_f1_std']:.4f}). "
             "Đây là thứ hạng trong ba cấu hình đã thử, không phải khẳng định một kiến trúc luôn tốt nhất.",
             f"{stable['system']} có sample std Macro-F1 nhỏ nhất trên validation "
             f"({stable['macro_f1_std']:.4f}). Std được tính giữa ba seed, khác biến động giữa các nhãn. "
             "Ba seed giúp mô tả độ ổn định trong lần đo nhưng chưa đủ để suy ra ý nghĩa thống kê hoặc bảo đảm tái hiện trên mọi máy."]
    for system in (r["system"] for r in original):
        test = {r["threshold_mode"]: r for r in rows
                if r["system"] == system and r["split"] == "test"}
        if "fixed" in test and "tuned" in test:
            before, after = test["fixed"], test["tuned"]
            change = after["macro_f1_mean"] - before["macro_f1_mean"]
            lines.append(f"{system} trên test: Macro-F1 @0,5 "
                         f"{before['macro_f1_mean']:.4f} ± {before['macro_f1_std']:.4f}; "
                         f"ngưỡng riêng {after['macro_f1_mean']:.4f} ± {after['macro_f1_std']:.4f} "
                         f"(chênh lệch mean {change:+.4f}). Micro-F1 tương ứng "
                         f"{before['micro_f1_mean']:.4f} và {after['micro_f1_mean']:.4f}. "
                         "Ngưỡng được chọn trên validation của từng seed, không chọn lại trên test.")
    lines.append("Nguyên nhân thứ hạng cần xét cùng dữ liệu, tokenizer, số tham số, learning rate, số epoch và ví dụ lỗi; "
                 "các kết quả này chưa tách riêng ảnh hưởng của từng yếu tố. Xem P/R và Hamming cùng F1: "
                 "hạ ngưỡng có thể tăng recall nhưng thêm false positives. Nhãn hiếm có support nhỏ nên F1 dễ thay đổi; "
                 "giữ cả nhãn giảm điểm trong bảng trước/sau. Thời gian BERT seed 42 và 2026 có gián đoạn máy ngủ, "
                 "RoBERTa seed 2026 cũng có gián đoạn máy ngủ; "
                 "vì vậy không dùng bảng elapsed để xếp hạng tốc độ các kiến trúc.")
    lines.extend(observed_tradeoffs(summary))
    return lines


def observed_tradeoffs(summary):
    """Ghi cả đánh đổi và giới hạn từ bảng cuối, không chọn lại bằng test."""
    if not summary.get("complete"):
        return []
    lookup = {(row["system"], row["split"], row["threshold_mode"]): row for row in summary["averages"]}
    bert_fixed = lookup[("C_bert", "test", "fixed")]
    bert_tuned = lookup[("C_bert", "test", "tuned")]
    balanced_fixed = lookup[("A_balanced", "test", "fixed")]
    standard = lookup[("A_standard", "test", "tuned")]
    global_a = lookup[("A_balanced", "test", "global")]
    tuned_a = lookup[("A_balanced", "test", "tuned")]
    direction_macro = "cao hơn" if global_a["macro_f1_mean"] > tuned_a["macro_f1_mean"] else "không cao hơn"
    direction_micro = "cao hơn" if standard["micro_f1_mean"] > tuned_a["micro_f1_mean"] else "không cao hơn"
    lines = [f"A balanced ngưỡng chung trên test đạt Macro-F1 {global_a['macro_f1_mean']:.4f}, "
             f"{direction_macro} ngưỡng riêng {tuned_a['macro_f1_mean']:.4f}. "
             f"A standard ngưỡng riêng đạt Micro-F1 {standard['micro_f1_mean']:.4f}, "
             f"{direction_micro} A balanced ngưỡng riêng {tuned_a['micro_f1_mean']:.4f}. "
             "Vì vậy weighting và ngưỡng riêng không làm mọi metric tăng. Các luật đã khóa trên validation; "
             "quan sát test này dùng để báo cáo đánh đổi, không dùng chọn lại cấu hình."]
    lines += [
        "Phân biệt mức chứng cứ: validation tuned F1 đo trên chính validation đã quét ngưỡng; "
        "test locked F1 đo trên test với checkpoint và ngưỡng đã khóa. Điểm tuned-validation có thể lạc quan, "
        "không dùng thay điểm test hoặc so trực tiếp với test của paper. Việc đánh giá test không cho phép "
        "điều chỉnh lại ngưỡng để chọn hàng đẹp hơn.",
        f"Ở C1 BERT trên test, mean Micro-Recall tăng từ {bert_fixed['micro_recall_mean']:.4f} "
        f"lên {bert_tuned['micro_recall_mean']:.4f}, mean Micro-Precision giảm từ "
        f"{bert_fixed['micro_precision_mean']:.4f} xuống {bert_tuned['micro_precision_mean']:.4f}, "
        f"mean Hamming Loss tăng từ {bert_fixed['hamming_loss_mean']:.4f} "
        f"lên {bert_tuned['hamming_loss_mean']:.4f} khi chuyển ngưỡng 0,5 sang ngưỡng riêng. "
        "Đây là đánh đổi quan sát trên toàn bộ 28 nhãn, không quy toàn bộ thay đổi cho riêng năm nhãn hiếm.",
        f"Chiều thay đổi không áp dụng cho mọi hệ thống: A balanced trên test sau tuning có Micro-Precision "
        f"{tuned_a['micro_precision_mean']:.4f}, cao hơn fixed {balanced_fixed['micro_precision_mean']:.4f}, "
        f"và Hamming Loss {tuned_a['hamming_loss_mean']:.4f}, thấp hơn fixed "
        f"{balanced_fixed['hamming_loss_mean']:.4f}. Vì vậy không viết rằng tuning luôn tăng Recall "
        "hoặc luôn làm Precision/Hamming xấu đi. Báo từng cấu hình và từng nhãn bằng TP/FP/FN, "
        "support và P/R/F1; không kết luận cả năm nhãn hiếm đều cải thiện."
    ]
    per_label_path = ROOT / "reports/project_results/per_label.csv"
    if per_label_path.exists():
        with per_label_path.open(encoding="utf-8-sig", newline="") as stream:
            per_label = list(csv.DictReader(stream))
        rare_labels = ("grief", "pride", "relief", "nervousness", "embarrassment")
        pairs = {(row["label"], row["threshold_mode"]): row for row in per_label
                 if row.get("method") == "A" and row.get("variant") == "balanced"
                 and row.get("split") == "test" and row.get("status") == "ok"
                 and row.get("label") in rare_labels}
        if all((label, mode) in pairs for label in rare_labels for mode in ("fixed", "tuned")):
            changes = []
            reduced = 0
            for label in rare_labels:
                before_f1, after_f1 = (float(pairs[(label, mode)]["f1"]) for mode in ("fixed", "tuned"))
                reduced += after_f1 < before_f1
                changes.append(f"{label} {before_f1:.4f}→{after_f1:.4f} ({after_f1-before_f1:+.4f})")
            lines.append(f"Đối chiếu riêng threshold ở A balanced trên test: {reduced}/5 nhãn hiếm giảm F1 "
                         "khi chuyển fixed→tuned; " + "; ".join(changes) + ". "
                         "Bảng standard fixed→balanced tuned là thay đổi kết hợp weighting/ngưỡng, "
                         "khác ablation này. Ngưỡng tốt trên validation có thể không giữ lợi thế trên test; "
                         "không quy mọi mức tăng của bảng kết hợp cho threshold.")
    b_fixed = lookup[("B_bart_mnli", "test", "fixed")]
    b_tuned = lookup[("B_bart_mnli", "test", "tuned")]
    lines.append(f"B zero-shot trên test đạt Macro-F1 {b_fixed['macro_f1_mean']:.4f} ở ngưỡng 0,5 "
                 f"và {b_tuned['macro_f1_mean']:.4f} với ngưỡng riêng. Ở ngưỡng 0,5, "
                 f"Micro-Precision {b_fixed['micro_precision_mean']:.4f} thấp trong khi "
                 f"Micro-Recall {b_fixed['micro_recall_mean']:.4f}, cho thấy nhiều nhãn dự đoán thừa. "
                 "Đây là kết quả của checkpoint, taxonomy và template đang dùng; chưa khảo sát prompt/model B khác. "
                 "Điểm yếu không tự chứng minh lỗi cài đặt hoặc mọi hệ zero-shot đều kém.")
    audit_path = ROOT / "reports/execution/zero_shot_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        test_audit = next((row for row in audit.get("runs", []) if row.get("split") == "test"), None)
        if test_audit:
            lines.append(f"Audit B ghi trung bình {test_audit['mean_predicted_labels_fixed']:.4f} nhãn dự đoán/câu "
                         f"ở ngưỡng 0,5, so với {test_audit['mean_truth_labels_from_saved_support']:.4f} nhãn thật/câu "
                         "trên test. Lượt audit đã kiểm code, cấu hình/head MNLI ba lớp, remap nhãn, "
                         "scores/checksum và metrics đã lưu; chưa phát hiện lỗi triển khai cụ thể trong phạm vi đó. "
                         "Lượt này không chạy inference mới, không đối chiếu raw logits và không kiểm lại toàn byte trọng số. "
                         "NLI-neutral khác candidate neutral của GoEmotions; score hai lớp entailment/contradiction "
                         "không mặc nhiên là xác suất cảm xúc đã được hiệu chuẩn. Hồ sơ: ZERO_SHOT_DIAGNOSTICS.md "
                         "và execution/zero_shot_audit.json; chưa chứng minh nguyên nhân của mọi FP.")
    c_tuned = [lookup[("C_" + arch, "test", "tuned")] for arch in ("bert", "roberta", "distilbert")]
    c_fixed_val = [lookup[("C_" + arch, "validation", "fixed")] for arch in ("bert", "roberta", "distilbert")]
    macro_winner = max(c_fixed_val, key=lambda row: row["macro_f1_mean"])
    micro_winner = max(c_tuned, key=lambda row: row["micro_f1_mean"])
    selected_tuned = lookup[(macro_winner["system"], "test", "tuned")]
    lines.append(f"Tiêu chí chọn C là mean Macro-F1 validation @0,5; {macro_winner['system']} thắng tiêu chí này. "
                 f"Trên test với ngưỡng riêng, {micro_winner['system']} đạt Micro-F1 cao nhất "
                 f"{micro_winner['micro_f1_mean']:.4f} ± {micro_winner['micro_f1_std']:.4f}, "
                 f"còn {macro_winner['system']} đạt {selected_tuned['micro_f1_mean']:.4f} ± "
                 f"{selected_tuned['micro_f1_std']:.4f}. Không gọi mô hình chọn cho demo là tốt nhất trên mọi metric; "
                 "không thay đổi rule lựa chọn sau khi đọc test.")
    lines.append("Chín run C đều là standard, chưa huấn luyện C với class weighting/pos_weight. "
                 "Nâng cao đã đo gồm class weighting ở A và ngưỡng riêng ở A/B/C. "
                 "Threshold tuning không cập nhật encoder; chưa có bằng chứng thực nghiệm về lợi ích weighting ở C. "
                 "Nhãn hiếm cần đọc cả mức tăng, giảm và không đổi, không chọn riêng những hàng có lợi.")
    return lines


def seed_table(summary):
    """Giữ từng seed bên cạnh bảng mean/std, kể cả khi mới có một run full."""
    records = summary["records"]
    originals = [r for r in records if r["system"].startswith("C_")
                 and r["split"] == "validation" and r["threshold_mode"] == "fixed"]
    if not originals:
        return []
    lookup = {(r["system"], r["seed"], r["split"], r["threshold_mode"]): r for r in records}
    lines = ["", "**Bảng 5-2f. Kết quả từng seed C đã hoàn tất; Macro-F1.**", "",
             "| Kiến trúc | Seed | Epoch chọn | Val @0,5 | Test @0,5 | Test ngưỡng riêng |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in originals:
        architecture = row["system"].removeprefix("C_")
        folder = ROOT / "data/processed/transformers" / architecture / f"seed_{row['seed']}" / "full/standard"
        metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
        def test_value(mode):
            value = lookup.get((row["system"], row["seed"], "test", mode))
            return f"{value['macro_f1']:.4f}" if value else "Chưa đo"
        lines.append(f"| {architecture} | {row['seed']} | {metadata['selected_epoch']} | "
                     f"{row['macro_f1']:.4f} | {test_value('fixed')} | {test_value('tuned')} |")
    lines.extend(["", "Epoch chọn theo validation @0,5 của đúng seed. File all_runs.csv giữ đủ bảy metrics "
                  "cho từng seed, split và luật ngưỡng; mean_std.csv giữ sample std. Những run chưa hoàn tất "
                  "không được tính vào bảng. Cấu hình/revision/hash nhỏ lưu trong reports/reproducibility; "
                  "trọng số lớn nằm trong data/processed để chạy demo hoặc chia sẻ riêng."])
    return lines


def demo_evidence_lines():
    """Giữ riêng kiểm suy luận và kiểm UI; ảnh phải khớp hash đã ghi."""
    lines = ["", "### 5.2.5. Kiểm suy luận và giao diện demo", ""]
    inference_path = ROOT / "reports/demo_verification.json"
    if inference_path.exists():
        evidence = json.loads(inference_path.read_text(encoding="utf-8"))
        lines += [f"Kiểm hàm suy luận lúc {evidence.get('checked_at_utc', '—')}: "
                  f"{evidence.get('model_inference_status', '—')}. Hồ sơ này đối chiếu scores/nhãn "
                  "với dữ liệu validation đã lưu và kiểm luồng nhập; không kiểm giao diện trình duyệt. "
                  "Đây là bằng chứng tại thời điểm ghi, không xác nhận server đang mở.",
                  "", "```json", json.dumps(evidence, ensure_ascii=False, indent=2), "```"]
    else:
        lines += ["Chưa có reports/demo_verification.json: chưa xác nhận kiểm suy luận demo hoàn tất."]
    ui_path = ROOT / "reports/demo_ui/evidence.json"
    if not ui_path.exists():
        return lines + ["", "Chưa có hồ sơ kiểm giao diện; không suy ra UI đã đạt từ kiểm suy luận."]
    ui = json.loads(ui_path.read_text(encoding="utf-8"))
    lines += ["", f"Kiểm giao diện lúc {ui.get('checked_at_utc', '—')}: "
              f"{ui.get('interface_status', '—')}; đã kiểm {ui.get('rendered_row_count', '—')}/28 hàng trong DOM. "
              f"Cờ kiểm đủ 28 hàng: {ui.get('all_28_rows_checked', '—')}. "
              f"Tương đương scores mô hình trong phép kiểm UI: {ui.get('model_score_equivalence', '—')}. "
              "Bằng chứng UI kiểm thao tác và hiển thị; phép đối chiếu scores thuộc hồ sơ suy luận riêng."]
    image_path = ui_path.parent / "demo_ui.png"
    if image_path.exists():
        if (ui.get("screenshot") != image_path.name
                or hashlib.sha256(image_path.read_bytes()).hexdigest() != ui.get("screenshot_sha256")):
            raise ValueError("Ảnh demo không khớp tên/hash trong hồ sơ kiểm UI")
        lines += ["", "![Giao diện demo được chụp sau thao tác thật](demo_ui/demo_ui.png)", "",
                  "**Hình 5.4. Giao diện demo trên app thật; ảnh khớp SHA-256 trong hồ sơ kiểm UI.**",
                  "", "Ảnh là bằng chứng hiển thị ở thời điểm chụp; không thay bảng test, "
                  "không chứng minh ROI và không xác nhận app đang mở ở thời điểm đọc báo cáo."]
    else:
        lines += ["", "Hồ sơ UI có nhưng thiếu ảnh minh chứng; chưa chèn ảnh vào báo cáo."]
    return lines


def main():
    summary = json.loads((ROOT / "reports/project_results/summary.json").read_text(encoding="utf-8"))
    content = (ROOT / "reports/project_results/RESULTS.md").read_text(encoding="utf-8")
    content = content.replace("# Kết quả thí nghiệm thực tế", "**Bảng 5-2a. Kết quả A/B/C thực tế.**", 1)
    content = content.replace("Bảng 5-3. Precision/Recall", "Bảng 5-2b. Precision/Recall")
    content = content.replace("## Phần chưa có bằng chứng đầy đủ", "### 5.2.1. Kiểm tra mức hoàn thành")
    additional = ["", "### 5.2.2. Cấu hình, seed và lựa chọn demo", ""]
    selection_path = ROOT / "data/processed/transformers/selected_model.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        additional.append("Danh tính mô hình được chọn lưu trong selected_model.json; lựa chọn dựa trên mean validation Macro-F1@0,5 của ba seed, không dựa trên test.")
        selected_summary = {key: selection[key] for key in
                            ("architecture", "seed", "run_dir", "selection_rule", "checkpoint_rule", "default_threshold")}
        additional.append("```json\n" + json.dumps(selected_summary, ensure_ascii=False, indent=2) + "\n```")
    else:
        additional.append("Chưa đủ hồ sơ để chọn demo C; không dùng kết quả smoke hoặc A/B để thay thế.")
    additional.extend(seed_table(summary))
    extra_paths = [("reports/errors_test_standard_fixed/summary.md", "### 5.2.3. Ba nhóm lỗi C1/C2/C3 trên test"),
                   ("reports/error_case_studies.md", "#### 5.2.3.1. Đọc và giải thích các ví dụ lỗi cụ thể"),
                   ("reports/project_results/ANALYSIS.md", "### 5.2.4. Nhãn hiếm và chi phí huấn luyện")]
    for filename, title in extra_paths:
        path = ROOT / filename
        if path.exists():
            extra = path.read_text(encoding="utf-8")
            extra = re.sub(r"^# .*\n", "", extra, count=1)
            extra = re.sub(r"^## ", "#### ", extra, flags=re.MULTILINE)
            def portable_image(match):
                image_path = Path(match[2])
                try:
                    target = image_path.relative_to(ROOT / "reports").as_posix()
                except ValueError:
                    target = image_path.as_posix()
                return f"![{match[1]}]({target})"
            extra = re.sub(r"!\[([^\]]*)\]\(<([^>]+)>\)", portable_image, extra)
            if "errors_test" in filename:
                extra = extra.replace("| Kiến trúc | Seed | Nhóm lỗi", "**Bảng 5-2c. So sánh ba nhóm lỗi trên test.**\n\n| Kiến trúc | Seed | Nhóm lỗi", 1)
            elif filename.endswith("ANALYSIS.md"):
                extra = extra.replace("| Nhãn | Mô hình |", "**Bảng 5-2d. F1 năm nhãn hiếm trước/sau cải tiến.**\n\n| Nhãn | Mô hình |", 1)
                extra = extra.replace("| Kiến trúc | Full seeds", "**Bảng 5-2e. Thời gian hoàn thành run C và số tham số.**\n\n| Kiến trúc | Full seeds", 1)
                extra = re.sub(r"(!\[[^\]]+\]\([^\n]+\))", r"\1\n\n**Hình 5.1. Đường học validation: mean và sample std theo epoch.**", extra, count=1)
                for split, figure_number in (("validation", "5.2"), ("test", "5.3")):
                    graph = ROOT / "reports/project_results/figures" / f"fixed_tuned_{split}.png"
                    if graph.exists():
                        target = graph.relative_to(ROOT / "reports").as_posix()
                        extra += (f"\n\n![So sánh ngưỡng trên {split}]({target})\n\n"
                                  f"**Hình {figure_number}. Macro-F1 ba C: ngưỡng 0,5 và ngưỡng riêng trên {split}.**\n\n"
                                  "Cột là mean và thanh sai số là sample std qua ba seed. "
                                  "Ngưỡng riêng được chọn trên validation của từng seed; "
                                  "điểm tuned-validation có thể lạc quan. Test sử dụng các ngưỡng đã khóa.")
            additional.extend(["", title, "", extra])
        elif filename == "reports/error_case_studies.md":
            additional.extend(["", title, "", *error_case_examples()])
    additional.extend(demo_evidence_lines())
    additional.extend(interpret_results(summary))
    replacement = "<!-- AUTO_RESULTS -->\n" + content + "\n" + "\n".join(additional) + "\n<!-- END_AUTO_RESULTS -->"
    source = ROOT / "reports/BAO_CAO_DO_AN_NOI_DUNG.md"
    report = source.read_text(encoding="utf-8")
    report, count = re.subn(r"<!-- AUTO_RESULTS -->.*?<!-- END_AUTO_RESULTS -->", lambda _: replacement,
                           report, flags=re.DOTALL)
    if count != 1:
        raise ValueError("Cần đúng một khối AUTO_RESULTS")
    if summary["complete"]:
        report, abstract_count = re.subn(r"Phần A đã được đo trên toàn bộ .*?(?=\n\n)",
                                        lambda _: final_abstract(summary), report, count=1, flags=re.DOTALL)
        if abstract_count != 1:
            raise ValueError("Không tìm thấy đoạn số liệu trong tóm tắt báo cáo")
        report = report.replace("Các kết quả B/C/D chỉ được bổ sung từ tệp chạy thật; không suy ra kết quả thực nghiệm từ việc có mã nguồn.",
                                "Toàn bộ A/B/C đã có kết quả full; mỗi kiến trúc C gồm ba seed 42,123,2026 với mean và sample std. Bảng test sử dụng ngưỡng/mô hình đã khóa trên validation.")
        report = report.replace("Kết quả A hiện dùng validation; tuning trên cùng validation có thể lạc quan.",
                                "Bảng test đã chạy sau khóa protocol; bảng tuned-validation dùng lại dữ liệu chọn ngưỡng nên có thể lạc quan.")
        report = report.replace("Ưu tiên hoàn tất đủ B/C, seed, lỗi đối chiếu và demo; khóa model/ngưỡng rồi đánh giá test theo protocol.",
                                "Duy trì hồ sơ A/B/C đủ seed, kiểm demo khi chuyển máy và đọc thủ công ví dụ lỗi; không chọn lại model/ngưỡng bằng test.")
        paragraph = "Các thí nghiệm A/B/C đã chạy full và khóa protocol trước test. Ba kiến trúc C được huấn luyện bằng cùng split và ba seed; bảng5-2 lưu từng cấu hình cùng mean±std. Phân tích lỗi đối chiếu checkpoint đại diện chọn trên validation. Kết quả demo cần được kiểm trực tiếp bằng hồ sơ đi kèm; bảng phân công và tỷ lệ đóng góp vẫn cần nhóm xác nhận."
        report = re.sub(r"Thiết kế toàn đồ án bao gồm B, ba C và D theo phân công bốn người\..*?(?=\n\n)", paragraph, report, count=1, flags=re.DOTALL)
        final_results = ("Nhóm đã tổ chức dữ liệu/mapping, EDA và các mô hình A/B/C với module đánh giá dùng chung. "
                         + final_abstract(summary))
        report, conclusion_count = re.subn(r"(## 6\.1\. Kết quả đạt được\n\n).*?(?=\n\n)",
                                          lambda match: match[1] + final_results, report, count=1, flags=re.DOTALL)
        if conclusion_count != 1:
            raise ValueError("Không tìm thấy đoạn kết quả trong mục 6.1")
    source.write_text(report, encoding="utf-8")
    print("Cập nhật báo cáo từ số thực nghiệm; complete =", summary["complete"])


if __name__ == "__main__":
    main()
