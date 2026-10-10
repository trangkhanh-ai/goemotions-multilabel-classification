# Báo cáo tiến độ cá nhân — baseline A

**Người phụ trách:** Bảo Duy Nguyễn — GitHub `dzyuu1612`.
**Ngày cập nhật:** 10/10/2026; số full ngày 08/10, kiểm mã sau ghép ngày 09/10.
**Phạm vi:** sản phẩm phần A của đề tài GoEmotions, vai trò A và điều phối B làm
chung trong nhóm 4 người. B đã có kết quả full; Đức Trí sở hữu RoBERTa C2.
Danh sách sản phẩm dưới đây giúp bàn giao phần A; nhóm cần xác nhận công việc
và% công sức thực tế từng người, Duy cần tự đọc/giải thích code khi bảo vệ.

**Mốc lịch sử 03/10:** bản cá nhân trước chỉ có validation, 16 tests và 11 cell mã.
Trạng thái đó được thay bằng kết quả test/proof mới bên dưới, không coi validation là test.

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
- Đã khóa protocol sáu cấu hình trước test, giữ `balanced_tuned` chọn bằng
  validation; đo đủ 5.427 test, lưu scores/metrics và F1 nhãn hiếm trước/sau.
- Notebook có output thật, hướng dẫn chạy/học và hồ sơ đối chiếu cô; thêm
  TF-IDF tính tay, sigmoid/từng dòng code và 20 câu hỏi bảo vệ.
- Lần kiểm mã 08/10: **69/69 tests toàn repo PASS**; notebook A **12/12**,
  B **4/4**, C **7/7** cell mã thực thi PASS. Đây là kiểm chung, không nhận tất cả
  là công cá nhân Duy. [Log/bằng chứng](execution/notebook_verification.json).

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
đã có test với protocol giữ nguyên. Không so trực tiếp với số paper khi khác protocol.

## Kết quả test đã đo — 5.427 mẫu

| Cấu hình | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---:|---:|---:|
| Standard @0,5 | 0.1963 | 0.3800 | 0.0348 |
| Standard chung 0,10 | 0.4096 | 0.5047 | 0.0553 |
| Standard riêng | 0.4134 | 0.5330 | 0.0444 |
| Balanced @0,5 | 0.4441 | 0.5024 | 0.0547 |
| Balanced chung 0,55 | 0.4530 | 0.5157 | 0.0480 |
| Balanced riêng — đã chọn bằng val | 0.4493 | 0.5277 | 0.0467 |

Nguồn: [all_runs.csv](project_results/all_runs.csv), [mean_std.csv](project_results/mean_std.csv).
Balanced global có Macro-F1 test cao hơn balanced tuned; standard tuned có
Micro-F1 cao hơn balanced tuned. Giữ lựa chọn đã khóa theo validation, không
chọn lại theo test. A chỉ một seed, không tự tạo mean±std.

### Nâng cao và năm nhãn hiếm test

| Nhãn | Test + | Standard fixed | Balanced tuned | Δ so standard fixed | Δ so balanced fixed |
|---|---:|---:|---:|---:|---:|
| grief | 6 | 0.0000 | 0.4615 | +0.4615 | +0.0330 |
| pride | 16 | 0.0000 | 0.4167 | +0.4167 | −0.0449 |
| relief | 11 | 0.0000 | 0.1176 | +0.1176 | −0.0157 |
| nervousness | 23 | 0.0000 | 0.1714 | +0.1714 | −0.1264 |
| embarrassment | 37 | 0.0000 | 0.2778 | +0.2778 | −0.0556 |

Tập hiếm chọn từ train; delta là chênh lệch F1 tuyệt đối. Kết hợp weighting và
tuning tăng so mốc standard fixed, nhưng tuning làm bốn nhãn giảm so chỉ weighting.
Grief của standard vẫn 0 ở fixed/global/tuned. Support thấp 6–37 nên không khái
quát quá mức. Bảng đầy đủ: [BASELINE_RESULTS.md](BASELINE_RESULTS.md),
[rare_before_after.csv](project_results/rare_before_after.csv).

## Bằng chứng bàn giao

[Notebook](../notebooks/baseline.ipynb) · [Hướng dẫn](../docs/BASELINE.md) ·
[Bảng kết quả/lỗi](BASELINE_RESULTS.md) · [Hồ sơ kiểm tra](../docs/BASELINE_REVIEW.md) ·
[CSV sáu cấu hình](baseline_validation/comparison.csv) ·
[Thứ tự đọc](../docs/THU_TU_DOC_BASELINE.md) · [Kế hoạch nhóm](../docs/KE_HOACH_NHOM.md).
[Hướng dẫn Duy giải thích với cô](../docs/HUONG_DAN_DUY_GIAI_THICH_BASELINE.md).

Model và scores theo ID ở `data/processed/baseline/full/` và
`data/processed/baseline/balanced/full/`, cần chia sẻ riêng hoặc tái chạy.
EDA trước đó là đóng góp của đồng đội; không ghi toàn bộ EDA là công của tôi.
Mã đã push lên fork `dzyuu1612`, mở [PR #3 vào repo chung](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/pull/3),
PR #3 đã merge 05/10/2026. Phần mở rộng toàn đồ án nằm ở
[PR #4](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/pull/4);
ba nội dung khoa học và hai báo cáo cập nhật ngày 10/10 nằm ở
[PR #5](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/pull/5),
đang nháp chờ nhóm duyệt.
Kế hoạch và báo cáo cá nhân cũng có
[mục riêng của Duy trên Notion](https://app.notion.com/p/3ee7c277690281e693c7f1cf985d569d).

### Nội dung bổ sung ngày 10/10

- Duy cần giải thích riêng validation đã chọn ngưỡng và test dùng ngưỡng khóa.
  Với A balanced trên test, tuning tăng Micro-Precision 0.4043 → 0.4561,
  giảm Hamming Loss 0.0547 → 0.0467 nhưng giảm Micro-Recall
  0.6631 → 0.6260; chiều đánh đổi khác BERT. Không kết luận tuning luôn có
  một chiều hoặc cả năm nhãn hiếm đều cải thiện.
- Đã bổ sung ba hướng ứng dụng cùng điều kiện/cách đo ROI vào cả hai báo cáo.
  Các ứng dụng là đề xuất chưa đo tại doanh nghiệp; số tiết kiệm nhà máy trong
  Case Study 4 của Lee không phải lợi ích tài chính NLP.
- Giải thích C1 tham khảo bài GoEmotions; C2/C3 là cấu hình nhóm. Ba seed
  đo biến động trong cùng cấu hình, chưa chứng minh đã tìm learning rate tối ưu.
- Báo cáo sáu chương 56 trang và IEEE 8 trang đã xuất lại; kiểm số/trích dẫn
  và xem 17 trang trọng yếu. Kết quả model/ngưỡng giữ nguyên; không có lượt
  huấn luyện hay kiểm mã mới ngày 10/10. Chi tiết tại
  [bản ghi nội dung](CAP_NHAT_NOI_DUNG_10_10_2026.md).

## Việc còn lại

Toàn nhóm đã đủ 9 run C, B full, **72 bản ghi/36 dòng tổng hợp**, 95 JSON metadata,
ba nhóm lỗi so C1/C2/C3. Demo BERT cased seed123 chọn theo validation đã có
[kiểm suy luận](demo_verification.json), [UI HTTP200/28 hàng](demo_ui/evidence.json)
và [ảnh thật](demo_ui/demo_ui.png). A cung cấp mốc so sánh, không làm input C;
phần B/C/D không tự được nhận là đóng góp cá nhân Duy.

Duy cần tự đọc/học/giải thích baseline, chọn ví dụ có ID để bảo vệ và bàn giao
đúng model/mapping/hash cho nhóm. Nhóm xác nhận phần việc/% đóng góp, điền
giảng viên/lớp/MSSV, đọc nguồn/nhận xét ngôn ngữ và kiểm/nộp bài.
Hai báo cáo tiến độ đã có tệp: lần 1 [Word](BAO_CAO_TIEN_DO_1.docx)/[PDF](BAO_CAO_TIEN_DO_1.pdf),
lần 2 [Word](BAO_CAO_TIEN_DO_2.docx)/[PDF](BAO_CAO_TIEN_DO_2.pdf).
Có tệp và test PASS không xác nhận nhóm đã nộp hay mọi người đã hiểu code.
