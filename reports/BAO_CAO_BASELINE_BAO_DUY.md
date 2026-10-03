# Báo cáo tiến độ cá nhân — baseline A

**Người phụ trách:** Bảo Duy Nguyễn — GitHub `dzyuu1612`.
**Ngày cập nhật:** 03/10/2026. **Phạm vi kết quả đã đo:** phần A của đề tài GoEmotions.
Vai trò cập nhật trong nhóm 4 người: A và điều phối B làm chung. B chưa có kết quả
được ghi nhận trong báo cáo này; tiếp nhận phần Đức Trí đã nhận trước nếu có,
trong khi Đức Trí sở hữu RoBERTa C2.

## Những việc đã thực hiện

- Tái sử dụng module dữ liệu/EDA của nhóm, kiểm split và thứ tự 28 nhãn; bổ sung
  giao diện chỉ mở train/validation và kiểm multi-hot, ID/hash.
- Xây module metric dùng chung: Micro/Macro Precision–Recall–F1, Hamming Loss,
  support và TP/FP/FN/TN từng nhãn; đối chiếu ví dụ tính tay.
- Triển khai và chạy đủ TF-IDF + 28 Logistic Regression ở hai bản standard/balanced
  trên 43.410 train, đánh giá 5.426 validation; 58.338 đặc trưng TF-IDF.
- So sánh sáu cấu hình từ hai model × ngưỡng 0.5/chung/riêng; chọn ngưỡng chỉ trên val.
- Phân tích năm nhãn hiếm từ train, cặp FN/FP và ví dụ có ID; ghi cả đánh đổi Precision,
  Recall/Hamming và nhãn không cải thiện. Kiểm model nạp lại khớp scores, sai số 0.
- Viết notebook có output thật, hướng dẫn chạy/học, hồ sơ đối chiếu yêu cầu cô;
  chuẩn bị script freeze/test và bảng nhỏ để đồng đội đọc trên GitHub.
- Kiểm mã: **16 bài test đạt**, notebook **11/11 cell mã đã chạy, không có output lỗi**.

## Kết quả validation đã đo

| Cấu hình | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---:|---:|---:|
| Standard @0.5 | 0.2025 | 0.3760 | 0.0354 |
| Standard ngưỡng chung 0.10 | 0.4094 | 0.5100 | 0.0544 |
| Standard ngưỡng riêng | 0.4391 | 0.5427 | 0.0433 |
| Balanced @0.5 | 0.4562 | 0.5099 | 0.0532 |
| Balanced ngưỡng chung 0.55 | 0.4660 | 0.5176 | 0.0473 |
| Balanced ngưỡng riêng | 0.4901 | 0.5467 | 0.0443 |

Weighting giúp Recall/Macro-F1 nhưng có thêm FP; Hamming không đồng thời tốt nhất.
Điểm tuning được đo trên chính validation dùng chọn ngưỡng, có thể lạc quan;
**chưa có kết quả test thật**. Không so trực tiếp với con số paper khi khác protocol.

## Bằng chứng bàn giao

[Notebook](../notebooks/baseline.ipynb) · [Hướng dẫn](../docs/BASELINE.md) ·
[Bảng kết quả/lỗi](BASELINE_RESULTS.md) · [Hồ sơ kiểm tra](../docs/BASELINE_REVIEW.md) ·
[CSV sáu cấu hình](baseline_validation/comparison.csv) ·
[Thứ tự đọc](../docs/THU_TU_DOC_BASELINE.md) · [Kế hoạch nhóm](../docs/KE_HOACH_NHOM.md).

Model và scores theo ID ở `data/processed/baseline/full/` và
`data/processed/baseline/balanced/full/`, cần chia sẻ riêng hoặc tái chạy.
EDA trước đó là đóng góp của đồng đội; không ghi toàn bộ EDA là công của tôi.
Mã đã push lên fork `dzyuu1612`, mở [PR #3 vào repo chung](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/pull/3),
chưa merge tại ngày cập nhật. Kế hoạch và báo cáo này cũng được lưu trong
[mục riêng của Duy trên Notion](https://app.notion.com/p/3ee7c277690281e693c7f1cf985d569d).

## Việc còn lại

Tự học/chạy lại để bảo vệ, điều phối B zero-shot và tiếp nhận bàn giao của Đức Trí nếu có, ghép bảng A/B/C khi nhận scores;
sau khi nhóm chốt protocol chạy test cuối và bổ sung F1 nhãn hiếm trước/sau.
Nhóm vẫn cần đủ ba C × ba seed, mean±std, demo best C và ≥3 nhóm lỗi đối chiếu C.
Chưa tự công bố phần trăm đóng góp khi nhóm chưa thống kê công việc thực tế.
