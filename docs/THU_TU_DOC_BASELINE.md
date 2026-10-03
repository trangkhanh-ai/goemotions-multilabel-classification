# Thứ tự đọc phần baseline của Bảo Duy Nguyễn

Cập nhật 03/10/2026. Vai trò trong nhóm 4 người: baseline A và điều phối B làm chung; GitHub `dzyuu1612`.
Đây là trang đánh dấu để bạn quay lại học sau. Mã và output đã có, nhưng bạn cần
tự đọc, chạy và giải thích được trước khi bảo vệ.

## Bốn tài liệu chính — đọc đúng thứ tự này

| Thứ tự | Tài liệu | Cách đọc | Sau khi đọc cần làm được |
|---|---|---|---|
| 1 | [Notebook baseline](../notebooks/baseline.ipynb) | Đọc từng cell và output; thử một câu tiếng Anh khác | Nối text → TF-IDF → 28 scores → ngưỡng → nhãn |
| 2 | [Hướng dẫn baseline](BASELINE.md) | Mục 1–5 trước, rồi 6–10; có lệnh cài và chạy | Chạy hai model, giải thích OvR, metric, weighting, tuning |
| 3 | [Bảng kết quả](../reports/BASELINE_RESULTS.md) | Đọc sáu hàng so sánh, năm nhãn hiếm và cặp lỗi | Phân biệt kết quả validation với test; viết phần A |
| 4 | [Hồ sơ đối chiếu yêu cầu cô](BASELINE_REVIEW.md) | Đọc yêu cầu, thiếu sót đã sửa và bằng chứng kiểm chứng | Biết phần nào A đã có và phần nào nhóm còn phải làm |

Nếu GitHub chưa hiển thị notebook ngay, tải file và mở bằng Jupyter/VS Code.
Notebook đã có output thật nên có thể đọc trước khi cài môi trường.

## Các kiến thức cần học

Đọc theo chủ đề dưới đây và tự kiểm bằng bài tập; không đặt thời lượng.

| Thứ tự chủ đề | Nội dung | Bài tập ngắn để tự kiểm |
|---|---|---|
| 1 | Python list, dict, hàm, import; NumPy shape và indexing | Với ba nhãn, đổi `[0, 2]` thành `[1, 0, 1]`; giải thích `Y.shape` |
| 2 | Train/validation/test và TF-IDF | Chỉ ra dòng `.fit` dùng train; giải thích tại sao val không mở rộng từ vựng |
| 3 | Logistic Regression và One-vs-Rest | Với score `[0.8, 0.3, 0.7]`, ngưỡng 0.5, chọn đúng hai nhãn |
| 4 | TP/FP/FN, Precision/Recall/F1 | Tính tay một ví dụ hai nhãn và đối chiếu `tests/test_metrics.py` |
| 5 | Class weighting và chọn ngưỡng | Giải thích Recall tăng nhưng Precision hoặc Hamming có thể xấu đi |
| 6 | Bảng kết quả và lỗi theo ID | Chọn một ví dụ, đọc nhãn thật/đoán, chỉ ra FN và FP, nêu giới hạn diễn giải |

Không cần học PyTorch/Transformer để chạy A. Cần hiểu khái quát B/C để giải thích
baseline là mốc so sánh trong đồ án. Bạn điều phối B làm chung, tiếp nhận phần
zero-shot Đức Trí đã nhận trước nếu có; Đức Trí sở hữu RoBERTa C2.
Để làm B, học thêm NLI/entailment, candidate labels, `multi_label=True`,
template và ánh xạ scores về thứ tự nhãn chuẩn. Tham khảo R07/R08/R18 trong
[kế hoạch nhóm](KE_HOACH_NHOM.md). A đã có kết quả validation; B chưa được
ghi nhận hoàn thành khi chưa có artifacts/số liệu bàn giao.

## Code đọc sau khi hiểu notebook

1. [Mô hình A](../scripts/run_baseline.py): `build_model`, fit train, predict validation.
2. [Chỉ số chung](../src/metrics.py): ngưỡng và TP/FP/FN/TN, micro/macro.
3. [Ngưỡng và ghép ID](../src/baseline.py): mỗi model có bộ ngưỡng riêng.
4. [Phân tích](../scripts/analyze_baseline.py): sáu cấu hình, nhãn hiếm, cặp lỗi.
5. [Dự đoán một câu](../scripts/predict_baseline.py): thử câu tiếng Anh.
6. [Xuất bảng lên GitHub](../scripts/export_baseline_results.py): JSON/CSV nhỏ để bàn giao.
7. [Khóa cấu hình](../scripts/freeze_baseline.py) và [đánh giá test](../scripts/evaluate_baseline_test.py): đọc sau, chỉ chạy khi nhóm chốt.

## Tài liệu nhóm và báo cáo cá nhân

- [Kế hoạch toàn nhóm](KE_HOACH_NHOM.md): A/B/C/D, phân công theo tên và sản phẩm.
- [Mục riêng của Duy trên Notion](https://app.notion.com/p/3ee7c277690281e693c7f1cf985d569d): tài liệu, bảng kết quả và báo cáo baseline.
- [Báo cáo tiến độ của bạn](../reports/BAO_CAO_BASELINE_BAO_DUY.md): phần đã làm, kết quả và việc còn lại.
- [Bảng validation dạng CSV](../reports/baseline_validation/comparison.csv): số đầy đủ để copy vào bảng báo cáo.

## Checklist tự học trước buổi báo cáo

- [ ] Tôi tự chạy được A và dự đoán một câu.
- [ ] Tôi giải thích được 27 cảm xúc + neutral và multi-hot N×28.
- [ ] Tôi biết TF-IDF chỉ fit trên train.
- [ ] Tôi phân biệt Micro-F1, Macro-F1, Precision, Recall, Hamming Loss.
- [ ] Tôi giải thích được tác dụng/đánh đổi của weighting và ngưỡng.
- [ ] Tôi đọc được một cặp FN/FP có ID và không gọi EDA là lỗi dự đoán.
- [ ] Tôi gọi các số hiện tại là validation; biết điểm tuned-val có thể lạc quan.
- [ ] Tôi phân biệt đóng góp A của mình với EDA và B/C/demo của cả nhóm.
