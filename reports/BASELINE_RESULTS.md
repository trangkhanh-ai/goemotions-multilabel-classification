# Kết quả phần A — baseline cổ điển GoEmotions

Bản này sinh tự động từ các file validation đã chạy. Dữ liệu: GoEmotions `simplified`, revision `add492243ff905527e67aeb8b80c082af02207c3`; train 43.410, validation 5.426, 28 nhãn. **Chưa dùng nhãn test.**

## Thiết lập

- TF-IDF unigram và bigram, `min_df=2`, tối đa 100.000 đặc trưng; fit trên train.
- One-vs-Rest Logistic Regression: 28 bộ phân loại nhị phân, `C=1`, `solver=liblinear`, `max_iter=1000`. Standard không có trọng số lớp; balanced dùng `class_weight='balanced'` tính từ train cho từng bộ phân loại.
- Dùng cùng split và thứ tự nhãn, ngưỡng gốc 0,5. Macro-F1 tính trung bình F1 của đủ 28 nhãn; `zero_division=0`.

## Bảng validation

| Cấu hình | Ngưỡng | Macro-F1 | Micro-F1 | Micro-P | Micro-R | Macro-P | Macro-R | Hamming Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| standard | Cố định 0,5 | 0.2025 | 0.3760 | 0.7254 | 0.2538 | 0.5571 | 0.1436 | 0.0354 |
| standard | Chung 0.10, chọn trên val | 0.4094 | 0.5100 | 0.4102 | 0.6741 | 0.4578 | 0.4455 | 0.0544 |
| standard | Chọn theo từng nhãn trên val | 0.4391 | 0.5427 | 0.4876 | 0.6119 | 0.5408 | 0.4513 | 0.0433 |
| balanced | Cố định 0,5 | 0.4562 | 0.5099 | 0.4158 | 0.6592 | 0.3858 | 0.5801 | 0.0532 |
| balanced | Chung 0.55, chọn trên val | 0.4660 | 0.5176 | 0.4524 | 0.6047 | 0.4142 | 0.5470 | 0.0473 |
| balanced | Chọn theo từng nhãn trên val | 0.4901 | 0.5467 | 0.4796 | 0.6356 | 0.4874 | 0.5219 | 0.0443 |

**Cách hiểu:** các hàng chọn ngưỡng được đo trên chính validation đã dùng để chọn ngưỡng; mức tăng ở đó có thể lạc quan. Chỉ dùng test một lần sau khi cả nhóm khóa cấu hình để xác nhận kết luận.

## Môi trường và thời gian đo

| Biến thể | Python | scikit-learn | Số đặc trưng TF-IDF | Fit (giây) | Dự đoán val (giây) |
|---|---|---|---:|---:|---:|
| standard | 3.13.9 | 1.7.2 | 58338 | 12.87 | 0.20 |
| balanced | 3.13.9 | 1.7.2 | 58338 | 22.63 | 0.24 |

Thời gian phụ thuộc máy và cache; xem JSON để có đủ phiên bản thư viện.

## Năm nhãn hiếm nhất theo số mẫu dương trên train

Danh sách được chọn bằng train trước khi nhìn kết quả validation.

| Nhãn | Train + | Val + | standard F1@0,5 | standard F1 tuned-val | balanced F1@0,5 | balanced F1 tuned-val |
|---|---:|---:|---:|---:|---:|---:|
| grief | 77 | 13 | 0.0000 | 0.0000 | 0.4118 | 0.4375 |
| pride | 111 | 15 | 0.0000 | 0.3333 | 0.5161 | 0.6087 |
| relief | 153 | 18 | 0.0000 | 0.1053 | 0.1277 | 0.1739 |
| nervousness | 164 | 21 | 0.0000 | 0.0909 | 0.3077 | 0.3125 |
| embarrassment | 303 | 35 | 0.1081 | 0.3636 | 0.5135 | 0.5507 |

## Phân tích lỗi và khả năng giải thích

Mỗi biến thể lưu `error_examples_validation.csv` với ID, văn bản, nhãn thật, nhãn dự đoán và điểm số. Bốn nhóm: bỏ sót nhãn hiếm, dự đoán nhãn thừa, đúng một phần ở mẫu đa nhãn, và không dự đoán nhãn nào. Cần đọc lại từng ví dụ trước khi trích vào báo cáo, vì nhãn gốc cũng có thể thiếu.

Các nhóm FP/FN là dấu hiệu thống kê; khi trình bày cần giải thích thêm về ngữ cảnh, từ ngữ, nhiều cảm xúc hoặc ít mẫu của nhãn. Số ví dụ được chọn không phải tỷ lệ lỗi của toàn split.

- **rare_false_negative**, ID `eczwil0`: `I am so proud of this community.` — thật: pride; đoán: (không nhãn). Điểm pride = 0.3776.
- **false_positive**, ID `ed832y6`: `"Homeopaths love it!"` — thật: neutral; đoán: love. Điểm love = 1.0000.
- **partial_multi_label**, ID `eczdvun`: `Thank you. I really appreciate your response` — thật: admiration, gratitude; đoán: gratitude. Điểm admiration = 0.4989.
- **no_label_predicted**, ID `eeoh5vh`: `And he even dared to reference fairly odd parents in there, wtf lol` — thật: amusement; đoán: (không nhãn). Điểm amusement = 0.4999.

`top_features.csv` ghi từ/cặp từ có hệ số LR cao và thấp cho từng nhãn. Đây là liên hệ thống kê trong mô hình, không chứng minh nguyên nhân cảm xúc.

## Cặp nhãn bị bỏ sót và dự đoán thừa trong cùng câu

Đếm FN của nhãn thật A đồng thời FP của nhãn B. Đây là bảng lỗi đa nhãn, không phải ma trận nhầm lẫn một lớp và không phải bảng đồng xuất hiện nhãn thật. Một câu có thể đóng góp nhiều cặp; neutral được giữ nguyên theo nguồn.

| Variant @0,5 | Nhãn bỏ sót | Nhãn thừa | Số câu | Tổng FN nhãn bỏ sót | Tỷ lệ trong FN |
|---|---|---|---:|---:|---:|
| standard | disapproval | neutral | 64 | 285 | 22.5% |
| standard | approval | neutral | 62 | 384 | 16.1% |
| standard | curiosity | neutral | 40 | 220 | 18.2% |
| standard | annoyance | neutral | 39 | 301 | 13.0% |
| standard | confusion | neutral | 30 | 149 | 20.1% |
| standard | anger | neutral | 28 | 169 | 16.6% |
| standard | caring | neutral | 26 | 148 | 17.6% |
| standard | realization | neutral | 22 | 126 | 17.5% |
| balanced | approval | neutral | 100 | 225 | 44.4% |
| balanced | neutral | annoyance | 85 | 455 | 18.7% |
| balanced | neutral | approval | 79 | 455 | 17.4% |
| balanced | annoyance | neutral | 78 | 171 | 45.6% |
| balanced | disapproval | neutral | 77 | 138 | 55.8% |
| balanced | neutral | disapproval | 70 | 455 | 15.4% |
| balanced | curiosity | neutral | 57 | 103 | 55.3% |
| balanced | neutral | curiosity | 55 | 455 | 12.1% |

Để đọc các cặp cảm xúc cụ thể, bảng phụ sau chỉ lấy cặp không có neutral. Đây là cách trình bày thêm; metric vẫn tính đủ 28 nhãn.

| Variant @0,5 | Nhãn bỏ sót | Nhãn thừa | Số câu | ID ví dụ |
|---|---|---|---:|---|
| standard | admiration | love | 10 | ee9xgzw |
| standard | joy | love | 8 | ed0ht04 |
| standard | joy | admiration | 7 | edopy68 |
| standard | approval | love | 5 | ed031mb |
| standard | gratitude | joy | 5 | eexk0cl |
| balanced | disapproval | annoyance | 23 | ed9fc3d |
| balanced | approval | disapproval | 22 | ee84bjg |
| balanced | approval | admiration | 21 | edcmnk3 |
| balanced | approval | annoyance | 21 | ee84bjg |
| balanced | annoyance | approval | 20 | eex5eeu |

## Số nhãn được dự đoán

| Variant @0,5 | Câu không dự đoán nhãn | Trung bình nhãn/câu |
|---|---:|---:|
| standard | 3271 | 0.411 |
| balanced | 123 | 1.864 |

## Bằng chứng chạy lại và bàn giao

- `standard`: `data/processed/baseline/full/validation_metrics.json`, `validation_scores.npz`, `per_label_validation.csv`, `thresholds_validation.json`, `error_examples_validation.csv`, `label_error_pairs_validation.csv`, `threshold_curve_validation.csv`, `top_features.csv`, `model.joblib`.
- `balanced`: `data/processed/baseline/balanced/full/validation_metrics.json`, `validation_scores.npz`, `per_label_validation.csv`, `thresholds_validation.json`, `error_examples_validation.csv`, `label_error_pairs_validation.csv`, `threshold_curve_validation.csv`, `top_features.csv`, `model.joblib`.
- `validation_scores.npz` gồm `ids`, `scores` N×28, `label_names`; ghép theo ID, không ghép theo thứ tự dòng. Các file lớn nằm trong `data/processed/` và được Git bỏ qua.
- Môi trường, thời gian fit, số đặc trưng và cấu hình nằm trong `validation_metrics.json` của từng biến thể.
- Hash model/scores đã được kiểm, nạp lại model dự đoán toàn validation và đối chiếu với scores lưu trước đó, sai số tối đa ≤1e-12.
- Báo cáo này chỉ mô tả phần A. Nhóm vẫn cần B zero-shot, ba kiến trúc C mỗi kiến trúc ba seed, demo từ C tốt nhất và so sánh lỗi giữa C1/C2/C3.

## Nguồn phương pháp

- [Google Research: GoEmotions](https://github.com/google-research/google-research/blob/master/goemotions/README.md)
- [scikit-learn: TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn: OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn: Precision, Recall, F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
- [scikit-learn 1.7.2: chọn ngưỡng](https://scikit-learn.org/1.7/modules/classification_threshold.html)
