# Cập nhật nội dung khoa học GoEmotions — 10/10/2026

Áp dụng cho báo cáo sáu chương `BAO_CAO_DO_AN_GOEMOTIONS_IEEE.docx/pdf`, bài
IEEE hai cột `BAI_BAO_GOEMOTIONS_IEEE.docx/pdf`, tài liệu giải thích và kế hoạch
Notion. Phạm vi lần này là ba nội dung đã được nhóm yêu cầu; không bổ sung
thí nghiệm hoặc đổi kết quả đã công bố.

## 1. Giá trị công nghiệp và ROI

- Mục 5.6 của báo cáo và V.B của bài IEEE giữ đúng Case Study 4 của Lee (2020):
  ví dụ tiết kiệm năng lượng hơn 300.000 USD/năm thuộc case nhà máy, không thuộc
  mô hình NLP. Đây là nội dung sách cung cấp, chưa được nhóm kiểm toán độc lập.
- Bổ sung ba hướng ứng dụng: hỗ trợ định tuyến/ưu tiên khách hàng, theo dõi tín
  hiệu khủng hoảng thương hiệu và giảm công rà soát bình luận.
- Nêu rõ lớp tích hợp ngoài bộ phân loại: loại yêu cầu/luật nghiệp vụ, nhận diện
  thương hiệu và tổng hợp theo thời gian, cùng bước người đọc xác minh.
- Nêu chỉ số cần đo: chuyển đúng/sai, thời gian phản hồi/giải quyết, cảnh báo
  đúng/sai và bỏ sót, thời gian đọc/chất lượng, chi phí model/tích hợp/vận hành
  và công sửa dự đoán sai. MTTR được định nghĩa là mean time to resolution.
- Đây là đề xuất chưa thử nghiệm doanh nghiệp; không công bố tỷ lệ ROI,
  số tiền tiết kiệm NLP hoặc kết luận F1 tăng đã làm thời gian xử lý giảm.

## 2. Validation/test và nhãn hiếm

- Validation tuned F1 được đo trên tập đã chọn ngưỡng, có thể lạc quan.
  Test locked F1 dùng checkpoint/ngưỡng khóa từ validation; không chọn lại
  dựa trên kết quả test.
- Đánh đổi của C1 BERT trên test, mean ba seed:

| Metric | Ngưỡng 0,5 | Ngưỡng riêng |
|---|---:|---:|
| Micro-Recall | 0.5322 | 0.6437 |
| Micro-Precision | 0.6421 | 0.5446 |
| Hamming Loss | 0.0318 | 0.0373 |

- Đây là tổng hợp trên 28 nhãn; chưa cô lập đóng góp của năm nhãn hiếm vào
  toàn bộ mức thay đổi. Không diễn giải thành một quy luật cho mọi cấu hình.
- Đối chứng A balanced trên test sau tuning: Micro-Precision tăng
  0.4043 → 0.4561 và Hamming Loss giảm 0.0547 → 0.0467, trong khi
  Micro-Recall giảm 0.6631 → 0.6260.
- Giữ kết quả bất lợi: F1 embarrassment của BERT giảm 0.5089 → 0.4831;
  grief của RoBERTa/DistilBERT vẫn bằng 0; threshold riêng của A balanced
  làm bốn trong năm nhãn hiếm giảm F1 so với chính A balanced fixed.
- Bảng A standard fixed → A balanced tuned thay đổi cả weighting/ngưỡng;
  không quy toàn bộ mức tăng cho một kỹ thuật riêng.

Nguồn số: `reports/project_results/mean_std.csv`, `per_label.csv`,
`rare_before_after.csv` và `summary.json`; không tính lại ngưỡng trên test.

## 3. Siêu tham số và căn cứ tài liệu

- C1 BERT: learning rate 5×10⁻⁵, 4 epoch; tham khảo mục 5.3 GoEmotions.
- C2 RoBERTa/C3 DistilBERT: learning rate 2×10⁻⁵, 3 epoch; cấu hình nhóm.
- Chín run có ba seed cho mỗi cấu hình kiến trúc. Nhóm chưa có tìm kiếm
  siêu tham số có hệ thống; chọn checkpoint theo epoch trên validation
  không thay thế tìm kiếm learning rate.
- Chưa có bằng chứng để giải thích C2 bằng gradient explosion hoặc kết luận
  C3 hội tụ nhanh hơn. Giữ giới hạn: kiến trúc/tokenizer/lr/epoch cùng thay đổi.
- Trích năm công bố cho bài báo và ngày truy cập cho tài liệu API; không đổi
  năm truy cập thành năm phát hành phần mềm hoặc bịa năm edition IEEE Guide.

Nguồn nền tảng: [GoEmotions ACL 2020, mục 5.3](https://aclanthology.org/2020.acl-main.372.pdf).
Nguyên tắc ngưỡng: [scikit-learn 1.7](https://scikit-learn.org/1.7/modules/classification_threshold.html).
Hồ sơ nguồn: [REFERENCE_CLAIM_AUDIT.md](REFERENCE_CLAIM_AUDIT.md).

## 4. Đọc và kiểm bản cập nhật

Hai nguồn Markdown được sửa trước; các đoạn nhận xét số được tạo bằng
`tools/update_report_results.py` và dùng chung cho hai báo cáo để tránh lệch số.
Hồ sơ cuối `reports/execution/final_report_verification.json` kiểm Word/PDF thực
tế, nguồn trích dẫn và các trang đã render. Lượt 77/77 kiểm mã được giữ là
bằng chứng ngày 09/10, không gọi là một lượt tests mới ngày 10/10.

Thông tin hành chính và tỷ lệ đóng góp nằm ngoài phạm vi ba nội dung lần này.

Kết quả xuất cuối: báo cáo sáu chương **56 trang**, bài IEEE **8 trang**.
Đã kiểm văn bản toàn bộ PDF và xem trực tiếp 17 trang trọng yếu vừa đổi;
trang bìa và hai báo cáo tiến độ được đối chiếu checksum với bản đã kiểm ngày 09/10.
Notion kế hoạch nhóm/mục Duy đã cập nhật và đọc lại; hồ sơ tại
`reports/NOTION_CAP_NHAT_10_10_2026.md` và
`reports/execution/notion_scientific_revision_10_10_2026.json`.
