## Phạm vi

Tiếp tục đồ án GoEmotions theo yêu cầu giảng viên sau khi PR #3 đã được merge ngày 05/10/2026.

- A: TF-IDF + One-vs-Rest Logistic Regression; cấu hình chuẩn/cân bằng lớp, chọn ngưỡng trên validation.
- B: BART-large-MNLI zero-shot, 28 nhãn đúng thứ tự, không fine-tune trên GoEmotions.
- C: BERT, RoBERTa, DistilBERT; mỗi kiến trúc chạy ba seed 42/123/2026 và báo cáo mean ± sample std.
- D: chọn C bằng validation, khóa protocol trước test, dùng checkpoint thật trong demo.
- Module đánh giá chung, phân tích nhãn hiếm, ba nhóm lỗi, kiểm ID/checksum/config và nguồn tham khảo chính thức.
- Hướng dẫn baseline tiếng Việt, notebook theo thứ tự đọc, hồ sơ tái hiện và kế hoạch phân công bốn thành viên.
- Hai dạng báo cáo: sáu chương theo mẫu cô và bài báo IEEE hai cột; Word/PDF và hai báo cáo tiến độ.

## Trạng thái cuối đã kiểm chứng

Đã hoàn tất **9/9 run C full**, A/B validation và test sau khóa protocol; **72 bản ghi, 36 nhóm tổng hợp, missing=[]**. D dùng **BERT seed123**, chọn bằng validation. PR đang chờ nhóm review/merge, không dùng smoke thay benchmark.

**77/77 tests PASS** sau ghép code chung ngày 09/10/2026, 12.936s. Notebook A12/12, B4/4, C7/7 ô mã đã chạy; source hướng dẫn/code giữ nguyên. Demo đã đối chiếu checkpoint trên ba câu validation/28scores và kiểm UI thật HTTP200, nút bấm, 28/28 hàng, screenshot có checksum. Đây là bằng chứng đã lưu; lần sửa báo cáo ngày 10/10 không chạy lại model/tests hoặc cộng các lượt kiểm lịch sử thành số tests mới.

## Kết quả test chính

| Hệ thống | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---:|---:|---:|
| A standard @0.5 | 0.1963 | 0.3800 | 0.0348 |
| A balanced ngưỡng riêng từ val | 0.4493 | 0.5277 | 0.0467 |
| B MNLI @0.5 | 0.1035 | 0.1008 | 0.5315 |
| B MNLI ngưỡng riêng từ val | 0.1609 | 0.1752 | 0.2941 |
| C BERT @0.5, ba seed | 0.4720±0.0045 | 0.5820±0.0040 | 0.0318±0.0003 |
| C BERT ngưỡng riêng, ba seed | 0.5038±0.0097 | 0.5900±0.0029 | 0.0373±0.0003 |

Giữ kết quả bất lợi: balanced ngưỡng chung có test Macro hơn ngưỡng riêng; standard tuned Micro hơn balanced tuned; RoBERTa tuned Micro hơn BERT tuned; embarrassment của BERT giảm F1 sau tuning. B audit chưa phát hiện bug cụ thể nhưng fixed dự đoán15.3818 nhãn/câu so với truth1.1662. Không retune bằng test hoặc kết luận nguyên nhân nhân quả/calibration.

## Bản bàn giao và tích hợp kho chung

- Báo cáo sáu chương theo mẫu cô **56 trang**, IEEE hai cột **8 trang** đã xuất lại ngày 10/10; tiến độ1/2 giữ bản lịch sử **5/10 trang**. QA văn bản mọi trang/nguồn/case/std, xem 17 trang trọng yếu vừa đổi và giữ hồ sơ checksum cho các trang/PDF không đổi.
- Nguồn chính GoEmotions ACL2020; 26 nguồn trong báo cáo/20 trong bài hai cột. Case Study4 Lee2020 đã đối chiếu; không chuyển số tiết kiệm năng lượng thành ROI NLP.
- 95 JSON metadata và1CSV giữ đúng byte/hash qua Git; nguồn/history/revision/protocol được giữ, training commit không được suy từ export commit.
- Kiểm byte nguồn báo cáo/Word/PDF/ảnh, tổng hợp kết quả, log tests và notebook trong `reports/execution/delivery_artifact_bytes_10_10_2026.json`; giữ hồ sơ cũ. Tham chiếu phân tích lỗi dùng `reports/errors_test_standard_fixed/counts.csv`.
- Đã đồng bộ `origin/main` `3acdfc6`; giữ nghiên cứu C3 Nhật Huy riêng. 79 tệp config/notebook/bằng chứng Huy và demo đổi tên giữ nguyên byte.
- `app.py` là best-C Gradio; `app_distilbert_huy.py` là demo Streamlit riêng, verifier/hướng dẫn đã đổi tên. Hai C3 khác trainer/môi trường/protocol không cộng thành sáu seed hoặc dùng validation làm test; Streamlit riêng chưa suy luận lại trên máy hiện tại.
- Notion kế hoạch nhóm/mụcDuy đã cập nhật ba nội dung khoa học ngày 10/10, đọc lại xác nhận các đoạn sửa và trang con; bản ghi ở `reports/NOTION_CAP_NHAT_10_10_2026.md` và `reports/execution/notion_scientific_revision_10_10_2026.json`.

## Cập nhật nội dung khoa học ngày 10/10/2026

1. Giá trị ứng dụng: ba kịch bản hỗ trợ khách hàng, tín hiệu khủng hoảng thương hiệu và giảm công rà soát bình luận; định nghĩa MTTR, cách đo chi phí/ROI. Đây là đề xuất chưa triển khai/đo trong doanh nghiệp. Case Study 4 của Lee được giữ riêng, không chuyển số tiết kiệm nhà máy thành lợi nhuận NLP.
2. Đánh giá: tách validation tuned F1 và test locked F1. BERT tuning có Recall 0.5322→0.6437, Precision 0.6421→0.5446, Hamming 0.0318→0.0373; A balanced có chiều thay đổi khác. Giữ nhãn giảm F1, không quy thay đổi tổng thể cho riêng năm nhãn hiếm hoặc chọn lại ngưỡng bằng test.
3. Siêu tham số: BERT tham khảo mục 5.3 GoEmotions; RoBERTa/DistilBERT là cấu hình nhóm. Chín run/ba seed không phải tìm kiếm learning rate có hệ thống; không thêm giải thích gradient explosion/hội tụ nhanh khi thiếu thí nghiệm.

Hai nguồn Markdown, công cụ sinh nhận xét, hướng dẫn đọc và hai bản DOCX/PDF được đồng bộ. Mười hash của kết quả/ngưỡng/config/model-source giữ nguyên so với trước sửa; danh mục nguồn và hồ sơ kiểm 09/10 được giữ. Phạm vi lần này không điền thông tin hành chính hoặc thay phân công.

## Tệp để đọc

- `docs/HUONG_DAN_DUY_GIAI_THICH_BASELINE.md`
- `docs/THU_TU_DOC_DO_AN.md` và `docs/REPRODUCIBILITY.md`
- `reports/project_results/RESULTS.md`
- `reports/REFERENCE_CLAIM_AUDIT.md`
- `reports/CAP_NHAT_NOI_DUNG_10_10_2026.md`
- `reports/BAO_CAO_DO_AN_GOEMOTIONS_IEEE.docx` / `.pdf`
- `reports/BAI_BAO_GOEMOTIONS_IEEE.docx` / `.pdf`

Các trường giảng viên, lớp, MSSV và tỷ lệ đóng góp chưa được cung cấp được để trống theo yêu cầu. Trọng số model và dữ liệu nặng không đưa vào Git; metadata đã được kiểm tra giữ nguyên byte/hash qua Git transport.
