# Rà soát phần A của đồ án — 01/10/2026

Nguồn đối chiếu: ảnh đề tài 1, nội dung yêu cầu chung của giảng viên do bạn cung cấp,
bản phân công ngày 30/09 và kế hoạch nhóm đã cập nhật. Hướng dẫn kỹ thuật dùng
scikit-learn 1.7.2 đúng môi trường đã chạy. Đây là hồ sơ kiểm tra **phần A**, không
phải xác nhận đã hoàn thành toàn bộ A/B/C/D của nhóm.

## 1. Đúng yêu cầu cô ở đâu?

| Yêu cầu | Bằng chứng trong phần A | Trạng thái |
|---|---|---|
| TF-IDF + Logistic Regression cổ điển, 1 SV phụ trách | `scripts/run_baseline.py`, Pipeline + OneVsRestClassifier | Đã chạy full |
| Bài toán đa nhãn, 27 cảm xúc + neutral | `multi_hot`, N×28, ngưỡng độc lập; không ép argmax | Đạt trên train/validation |
| Split chính thức | 43.410 train/5.426 val; revision và SHA-256 cố định | Đã kiểm |
| Macro/Micro-F1, Precision/Recall, Hamming Loss | `src/metrics.py`, đủ micro/macro P/R, F1 từng nhãn và support | Đã kiểm bằng ví dụ tính tay |
| Phân tích cặp cảm xúc dễ nhầm | Bảng FN(A)+FP(B) cùng câu, có ID ví dụ | Đã bổ sung; không dùng heatmap EDA thay cho lỗi |
| Nâng cao và F1 nhãn hiếm | Hai model × ba luật ngưỡng; bảng năm nhãn hiếm từ train | Có số validation, chưa kết luận test |
| Báo cáo tiến độ tuần 4 có số A thực | `reports/BASELINE_RESULTS.md` | Có thể lấy số validation kèm cấu hình |
| Đánh giá cuối có số test | `freeze_baseline.py` + `evaluate_baseline_test.py` | Mã đã chuẩn bị; nhóm chốt protocol rồi chạy |

Ba seed trong PDF áp dụng cho từng kiến trúc C; không tự thêm quy định ba seed
bắt buộc cho A. PDF đề tài không chỉ định nâng cao phải nằm riêng ở C; áp dụng
weighting/ngưỡng cho C tốt nhất là đề xuất của kế hoạch nhóm. B zero-shot có thể
làm chung hoặc gộp vào vai trò A; mã B chưa nằm trong phần triển khai này.

## 2. Những thiếu sót đã sửa

1. **Ngưỡng có thể cũ sau khi retrain:** metadata trước đây chỉ kiểm nhãn/revision.
   Nay cả model, scores và file ngưỡng được liên kết bằng SHA-256. Nạp nhầm hoặc
   dùng ngưỡng của model khác bị từ chối với hướng dẫn chạy lại phân tích.
2. **Thiếu ngưỡng chung:** đã thêm lưới 0,05–0,95, chọn theo Macro-F1 trên val;
   giữ nguyên mốc 0,5 và ngưỡng riêng để so sánh đầy đủ.
3. **Thiếu cặp lỗi thực tế:** đã đếm FN của nhãn A cùng FP của nhãn B, lưu số câu
   và ID. Bảng này khác co-occurrence của nhãn thật. Không gọi đó là confusion
   matrix một lớp và không giả định một câu chỉ góp một cặp.
4. **Test trước đây chỉ cho một cấu hình:** như vậy thiếu baseline gốc hoặc thiếu
   bảng trước/sau nâng cao. Quy trình mới khóa cả sáu cấu hình trước test, giữ
   lựa chọn dựa trên val và tính đủ các hàng đã khóa trong cùng lần đánh giá.
5. **Kiểm tra đầu vào/hội tụ còn thiếu:** chặn ma trận rỗng, NaN, score ngoài [0,1],
   nhãn sai, ngưỡng sai số chiều; dừng khi optimizer báo chưa hội tụ.
6. **Mô tả tiền xử lý chưa đủ:** đã ghi rõ lowercase, token pattern mặc định,
   bỏ punctuation/emoji và hạn chế với apostrophe/phủ định. Không tuyên bố
   baseline đã hiểu ngữ cảnh như Transformer.
7. **Học qua script còn khó:** đã có notebook 22 cell, 11 cell mã với output thật,
   đi từ multi-hot, TF-IDF, OvR, score, metric đến ngưỡng và cặp lỗi.
8. **Hướng dẫn lệnh lỗi trên Windows:** đã sửa phần trợ giúp dòng lệnh để không
   phụ thuộc mã hóa tiếng Việt của terminal; kiểm cả năm lệnh `--help` thành công.

## 3. Bằng chứng kiểm chứng

- Hai model đều fit đủ 43.410 dòng train; không mở split test trong run/analyze/notebook.
- 28 Logistic Regression đều hội tụ; số vòng lặp tối đa: standard 12, balanced 9,
  thấp hơn `max_iter=1000`. TF-IDF có 58.338 đặc trưng ở cả hai model.
- Nạp lại model, dự đoán toàn bộ 5.426 validation, ghép theo ID và đối chiếu score
  lưu trước đó. Sai số tối đa hiện tại bằng 0; tiêu chí chấp nhận ≤1e-12.
- 16 test mã nguồn đạt: công thức metric tính tay, ngưỡng từng nhãn/ngưỡng chung,
  ID bị xáo thứ tự, artifact/ngưỡng cũ, từ vựng train không đổi khi transform val,
  đếm cặp FN/FP, luồng cuối giữ baseline gốc và tái sử dụng kết quả.
- Test luồng cuối dùng dữ liệu giả ở thư mục tạm; không mở nhãn GoEmotions test thật.
- Notebook đã thực thi 11/11 cell mã, không có output lỗi.
- Thử tạo protocol trong bộ nhớ: có đủ sáu cấu hình; lựa chọn dựa trên val là
  `balanced_tuned`. Chưa tạo `final_protocol.json`, chưa có kết quả test thật.

## 4. Các số validation cần trình bày

| Model | Luật ngưỡng | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---|---:|---:|---:|
| standard | 0,5 | 0,2025 | 0,3760 | 0,0354 |
| standard | chung 0,10 chọn trên val | 0,4094 | 0,5100 | 0,0544 |
| standard | riêng từng nhãn chọn trên val | 0,4391 | 0,5427 | 0,0433 |
| balanced | 0,5 | 0,4562 | 0,5099 | 0,0532 |
| balanced | chung 0,55 chọn trên val | 0,4660 | 0,5176 | 0,0473 |
| balanced | riêng từng nhãn chọn trên val | 0,4901 | 0,5467 | 0,0443 |

Weighting thay đổi model; tuning thay đổi quyết định từ score. Giữ hai bước này
riêng khi giải thích. Điểm tuned-val được đo trên tập đã dùng chọn ngưỡng, có
thể lạc quan và không được gọi là kết quả test. Năm nhãn hiếm chỉ có 13–35 mẫu
dương validation; báo cáo support, cả nhãn tăng lẫn nhãn không tăng.

Standard không dự đoán nhãn nào ở 3.271/5.426 câu; balanced chỉ còn 123 câu nhưng
đoán trung bình 1,864 nhãn/câu so với 0,411 của standard. Điều này giúp giải thích
Recall tăng và FP/Hamming Loss cũng tăng. Không chỉ chọn một chỉ số đẹp để báo cáo.

## 5. Cách trình bày lỗi bằng bằng chứng

- Từ chỉ cảm xúc có thể diễn đạt đối tượng hoặc sở thích: ID `ee9xgzw`,
  “I love Wood Witch, still saving to buy that.” Nhãn thật admiration; standard
  bỏ sót admiration và đoán thừa love. Đây là lỗi theo nhãn nguồn; không chứng
  minh love hoàn toàn vô lý về ngữ nghĩa.
- Cảm xúc tiêu cực gần nhau: balanced có 23 câu bỏ sót disapproval và đoán thừa
  annoyance, ví dụ ID `ed9fc3d`. Cần đọc văn bản và nhãn khác trước khi diễn giải.
- Câu nhiều cảm xúc: ID `eczdvun` thật admiration + gratitude; standard chỉ đoán
  gratitude, score admiration 0,4989 vừa dưới mốc 0,5.
- Nhãn hiếm: ID `eczwil0` thật pride; score pride của standard 0,3776 nên không
  được chọn ở 0,5. Weighting/ngưỡng là giả thuyết sửa, phải đối chiếu số đo thực.

Các ví dụ là minh họa, không thay cho tần suất toàn split. Phân tích yêu cầu chung
vẫn cần so sánh cùng các nhóm lỗi giữa C1/C2/C3 khi nhóm có dự đoán C.

## 6. Việc bạn làm tiếp

1. Mở `notebooks/baseline.ipynb`, đọc từng cell và thử đổi câu ở cell dự đoán.
2. Đọc `docs/BASELINE.md` mục 5–7, tự giải thích TF-IDF, OvR, metric, weighting và ngưỡng.
3. Dùng `reports/BASELINE_RESULTS.md` để viết phần A của báo cáo; gọi đúng là validation.
4. Bàn giao code, mapping nhãn và scores theo ID cho nhóm; model/scores trong
   `data/processed/` bị Git bỏ qua và cần nơi chia sẻ riêng.
5. Khi nhóm chốt cuối, chạy freeze rồi evaluate; ghép bảng test A với B và C.

Nguồn kỹ thuật:
[chọn ngưỡng](https://scikit-learn.org/1.7/modules/classification_threshold.html),
[LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html),
[metric đa nhãn](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html).
