# Baseline A — hiểu và chạy từng bước

Baseline là mô hình tham chiếu để nhóm so sánh với zero-shot và các Transformer.
Phần A dùng **TF-IDF + One-vs-Rest Logistic Regression**, phù hợp yêu cầu đề tài.

## 1. Bài toán đầu vào/đầu ra

- Đầu vào: một bình luận tiếng Anh.
- Nhãn: 28 vị trí theo mapping GoEmotions.
- Một câu có thể có nhiều nhãn: dùng vector multi-hot, không ép chỉ chọn một.
- Đầu ra model: 28 điểm độc lập; ngưỡng chuyển điểm thành tập nhãn.

Ví dụ minh họa với ba nhãn:

```text
nhãn       gratitude   joy   sadness
câu A          1        1       0
câu B          0        0       1
```

Ví dụ chỉ giải thích định dạng; dữ liệu thật luôn giữ đủ 28 nhãn.

## 2. TF-IDF biến câu thành số

`TfidfVectorizer` học từ/cặp từ từ **train**, tính độ quan trọng và tạo ma trận thưa.
Câu validation/test được `transform` bằng vocabulary/IDF đã học.

- `ngram_range=(1, 2)`: từ đơn và cặp từ.
- `min_df=2`: đặc trưng phải xuất hiện ở ít nhất hai văn bản train.
- `max_features=100_000`: giới hạn vocabulary.

Không fit lại TF-IDF trên validation hoặc test.
Xem hàm `build_model` trong [run_baseline.py](../scripts/baseline/run_baseline.py).

## 3. One-vs-Rest và Logistic Regression

Với mỗi nhãn j, model học câu nào có/không có nhãn đó.
28 Logistic Regression cùng dùng ma trận TF-IDF, nhưng có bộ trọng số riêng.

```text
TF-IDF → z_j = w_j · x + b_j → sigmoid(z_j) → score_j
```

Sigmoid đưa một logit về khoảng 0–1; 28 score không cần cộng thành 1.
Trong code, `predict_proba` của scikit-learn làm bước này.
Điểm model chưa mặc nhiên được hiệu chuẩn thành xác suất nghiệp vụ.

## 4. Code quan trọng

| File | Bạn đọc gì |
|---|---|
| [datasets/goemotions.py](../src/datasets/goemotions.py) | Tải snapshot, kiểm schema/hash, tạo multi-hot |
| [run_baseline.py](../scripts/baseline/run_baseline.py) | `build_model`, fit và scores validation |
| [models/baseline.py](../src/models/baseline.py) | Đọc model/metadata, căn ID, ngưỡng và cặp lỗi |
| [evaluation/metrics.py](../src/evaluation/metrics.py) | Macro/Micro P/R/F1, Hamming, support/TP/FP/FN |
| [analyze_baseline.py](../scripts/baseline/analyze_baseline.py) | So sánh standard/balanced và ba chế độ ngưỡng |
| [predict_baseline.py](../scripts/baseline/predict_baseline.py) | Dự đoán câu mới từ model đã tạo |
| [baseline_numpy.py](../src/learning/baseline_numpy.py) | TF-IDF/sigmoid/gradient viết nhỏ để học |

Chạy `python -m examples.baseline_from_scratch` trước khi đọc thí nghiệm thật.

## 5. Chạy phần của bạn

```bash
python -m scripts.baseline.run_baseline
python -m scripts.baseline.run_baseline --variant balanced
python -m scripts.baseline.analyze_baseline
python -m scripts.baseline.predict_baseline --variant balanced --threshold tuned --text "Thank you, but I am still worried."
```

Hai lần train tạo hai model độc lập. Analyze cần model full của cả hai.
`--smoke` chỉ tạo thư mục smoke; predictor trên dùng thư mục full.

Model/validation scores và metadata sinh tại:
`data/processed/baseline/full/` và `data/processed/baseline/balanced/full/`.
Bảng phân tích nằm trong `reports/` cục bộ và được Git bỏ qua.

## 6. Weighting và ngưỡng

`standard` dùng trọng số lớp mặc định; `balanced` tăng trọng số các lớp ít mẫu
trong từng bài toán nhị phân. Trọng số tính từ train.

Ba chế độ ngưỡng:

| Chế độ | Ý nghĩa |
|---|---|
| fixed | 0,5 cho mọi nhãn |
| global | Một ngưỡng chung được chọn trên validation |
| tuned | Một ngưỡng riêng cho mỗi nhãn, chọn trên validation |

Tham số `--threshold` của predictor nhận `fixed/global/tuned`.
Ngưỡng thuộc model đã tạo nó; không chuyển ngưỡng standard sang balanced.

Ngưỡng có thể đổi Precision/Recall/Hamming theo nhiều chiều.
Báo cả nhãn giảm điểm; việc tăng Macro-F1 không đảm bảo mọi metric đều tăng.
So standard fixed với balanced tuned thay cả weighting và ngưỡng, chưa cô lập
tác động riêng của một kỹ thuật.

## 7. Khóa protocol rồi test

Sau khi nhóm chốt cấu hình dựa trên validation:

```bash
python -m scripts.baseline.freeze_baseline
python -m scripts.baseline.evaluate_baseline_test
```

Tuned-validation dùng cùng tập đã quét ngưỡng nên có thể lạc quan.
Test dùng ngưỡng/model đã khóa; không dùng kết quả test để chọn lại.

## 8. Cần học gì và cần bàn giao gì?

Học theo thứ tự: Python/NumPy → pandas → multi-hot → TF-IDF →
Logistic Regression/sigmoid → train/validation/test → P/R/F1/Hamming →
weighting/ngưỡng → phân tích FP/FN.

Bàn giao code, thứ tự nhãn, cấu hình, seed, hash nguồn và artifact, scores theo ID,
ngưỡng đã chọn, chỉ số toàn bộ/từng nhãn và ví dụ lỗi.
Sản phẩm này giúp các nhánh B/C cùng đánh giá và so sánh đúng protocol.

Khi giải thích, trả lời: vì sao cần 28 bộ phân loại; vì sao TF-IDF chỉ fit train;
vì sao không dùng softmax; vì sao tuned-validation khác test; vì sao một nhãn
không được dự đoán chưa đủ kết luận cả model tốt/xấu.
