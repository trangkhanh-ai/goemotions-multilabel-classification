# Kế hoạch và phân công nhóm 4 người — GoEmotions

Cập nhật **03/10/2026** theo xác nhận mới nhất của Duy. Nhóm có đúng **4 thành viên: Bảo Duy Nguyễn, Quốc Khánh, Đức Trí, Nhật Huy**. Kế hoạch chia theo đầu việc, người phụ trách, sản phẩm và điều kiện nghiệm thu.
Kho chung: [goemotions-multilabel-classification](https://github.com/trangkhanh-ai/goemotions-multilabel-classification).

## 1. Căn cứ và cách phân công lại

- **Yêu cầu cô:** A do một sinh viên phụ trách; ba sinh viên còn lại mỗi người làm trọn một kiến trúc C; B được làm chung hoặc gộp với người làm A; D dùng mô hình tốt nhất trong ba C.
- **Ghi nhận trao đổi trước:** Đức Trí chính là “Thợ Săn Thập Cẩm”, đã nhận zero-shot; Quốc Khánh đã nhận phần đầu báo cáo và fine-tune một mô hình.
- **Phân công mới cho nhóm 4 người:** Duy giữ A và điều phối B làm chung; Quốc Khánh làm BERT C1; Đức Trí làm RoBERTa C2; Nhật Huy làm DistilBERT C3 và tích hợp demo.
- Đức Trí bàn giao code/prompt/kết quả B nếu đã thực hiện; Duy tiếp nhận để hoàn thiện B với hỗ trợ của nhóm. Việc nhận B trước đây không thay thế trách nhiệm C2 trong kế hoạch mới.
- BERT/RoBERTa/DistilBERT thuộc nhóm kiến trúc cô gợi ý cho phân loại văn bản tiếng Anh. Checkpoint, seed cụ thể và thông số thử nghiệm bên dưới là lựa chọn triển khai của nhóm.

## 2. Bảng phân công chính

| Người | Phần chính | Việc phối hợp | Sản phẩm cần bàn giao |
|---|---|---|---|
| **Duy — Bảo Duy Nguyễn** | **A: TF-IDF + One-vs-Rest Logistic Regression; điều phối B zero-shot làm chung** | Data/metrics, bảng so sánh A/B/C, tổng hợp nhãn hiếm/nâng cao và cặp lỗi | Code/scores/config/metrics A và B; phần phương pháp, kết quả, hạn chế A/B; chi tiết cá nhân nằm trong mục riêng |
| **Quốc Khánh** | **C1: BERT-base; phần đầu và tổng hợp báo cáo** | Script fine-tune chung; bố cục, trích nguồn và nhận phần viết của các bạn | BERT ≥3 seed; log/checkpoint/scores; bảng từng seed và mean ± std; mục tiêu, paper, dữ liệu/EDA |
| **Đức Trí — Thợ Săn Thập Cẩm** | **C2: RoBERTa-base** | Bàn giao/hỗ trợ B đã nhận trước; đối chiếu lỗi ba C và viết phần C2 | RoBERTa ≥3 seed; log/checkpoint/scores; bảng từng seed và mean ± std; phương pháp và lỗi C2 |
| **Nhật Huy** | **C3: DistilBERT-base; demo D** | Tích hợp checkpoint và ngưỡng của C thắng theo validation; hướng dẫn chạy | DistilBERT ≥3 seed và mean ± std; demo Gradio/Streamlit chạy được cùng minh chứng |
| **Cả nhóm** | So sánh A/B/C, nâng cao và ≥3 loại lỗi giữa C1/C2/C3 | Đọc nguồn, kiểm số liệu, thống kê đóng góp và chuẩn bị phản biện | Báo cáo, bảng thực nghiệm, code, demo, nguồn và đóng góp thực tế |

Ba người sở hữu ba C lần lượt là **Quốc Khánh / Đức Trí / Nhật Huy**. Mỗi người tự thực hiện trọn kiến trúc của mình: dữ liệu → tokenizer → fine-tune → đủ seed → scores → metrics → phân tích lỗi → phần báo cáo.
Nhật Huy tích hợp **C thắng theo validation**, kể cả checkpoint do Quốc Khánh hoặc Đức Trí huấn luyện.

## 3. Đối chiếu chi tiết yêu cầu cô

Căn cứ là ảnh đề tài 1 và văn bản yêu cầu chung cô do Duy cung cấp. Nguồn kỹ thuật ở mục 13 giải thích cách thực hiện; quy định chấm/nộp bài lấy từ tài liệu cô.

| Yêu cầu | Cách thực hiện của nhóm | Bằng chứng cần có |
|---|---|---|
| Đọc paper và tìm hiểu dữ liệu | Tóm tắt GoEmotions bằng lời nhóm; nguồn, taxonomy, mẫu và split | Tóm tắt có trích dẫn, EDA/phân bố nhãn, ví dụ có ID |
| A cổ điển, 1 sinh viên | Duy: TF-IDF + OvR Logistic Regression | Code, cấu hình, model, scores và bảng A |
| B pretrained trực tiếp, không fine-tune | BART-large-MNLI qua HF pipeline; B làm chung, Duy điều phối | Revision, template, candidate labels, scores/metrics; không cập nhật trọng số trên GoEmotions |
| C đủ 3 kiến trúc, 3 người | Khánh BERT; Trí RoBERTa; Huy DistilBERT | Ba phương pháp và ba người chịu trách nhiệm đầy đủ |
| Mỗi C ≥3 random seed; mean ± std | Đề xuất seed 42, 123, 2026, cùng cấu hình đã chốt | ≥9 run C; kết quả từng seed và mean ± sample std |
| D giao diện dùng C tốt nhất | Huy tích hợp Gradio/Streamlit | App, checkpoint/ngưỡng đúng, hướng dẫn và link/ảnh/video |
| Multi-label 27 cảm xúc + neutral | Multi-hot N×28; 28 scores; ngưỡng độc lập | Mapping và kiểm thử; giữ đủ 28 nhãn trong bảng chính |
| Đánh giá và cặp cảm xúc dễ nhầm | Micro/Macro-F1, Precision/Recall, Hamming Loss, per-label | Bảng số thật, support, cặp FN/FP và câu có ID |
| ≥3 loại lỗi, đối chiếu ba C | Cảm xúc gần nghĩa; thiếu cảm xúc thứ hai; hàm ý/phủ định; có thể thêm nhãn hiếm | Cùng ID đối chiếu C1/C2/C3 và nhận xét nguyên nhân |
| Nâng cao và F1 nhãn hiếm | Weighting và/hoặc ngưỡng riêng; contrastive là lựa chọn mở rộng | Bảng trước/sau, F1/support nhãn hiếm, đánh đổi và hạn chế |
| Báo cáo tiến độ và cuối cùng | Đủ 2 báo cáo tiến độ và 1 báo cáo cuối | Nội dung theo mục 12; hạn/kênh nộp theo thông báo cô |
| Độ ổn định, đóng góp, phản biện | Giải thích thứ hạng, std, hạn chế và công việc thực tế | Bảng đóng góp có tỷ lệ theo thống kê nhóm; GitHub/demo và phần thảo luận |

Phần nâng cao của đề tài cần hoàn thành để đáp ứng tiêu chí điểm cao; không bảo đảm điểm số chỉ bằng việc có code. Tài liệu cô không giới hạn nâng cao riêng ở C. Mở rộng tuning sang C là lựa chọn hợp lý của nhóm để có so sánh trên các mô hình chính.

## 4. Dữ liệu và quy ước dùng chung

| Hạng mục | Quy ước |
|---|---|
| Dataset | `google-research-datasets/go_emotions`, config `simplified` |
| Revision dữ liệu | `add492243ff905527e67aeb8b80c082af02207c3` |
| Split chính thức | Train 43.410 / validation 5.426 / test 5.427; tổng 54.263 |
| Nhãn | 27 cảm xúc + neutral; giữ đúng thứ tự `data/labels.json` |
| Nhãn thật | Multi-hot N×28; labels float khi dùng BCE cho C |
| Văn bản | Tiếng Anh, giữ nguồn; biểu diễn/tokenizer riêng theo phương pháp |
| Dữ liệu dùng chung | `src/data.py`; kiểm ID, split, revision và SHA-256 |
| Scores bàn giao | NPZ gồm `ids`, `scores` N×28, `label_names`; ghép theo ID |
| Metric | `src/metrics.py`; đủ 28 nhãn, `zero_division=0`; P/R/F1 micro và macro, Hamming, per-label |
| Train | Học A/C, fit TF-IDF và tính weighting |
| Validation | Chọn cấu hình/checkpoint/prompt nếu có/ngưỡng; ghi rõ quá trình chọn |
| Test | Chạy sau khi khóa các lựa chọn; không chọn lại bằng test |

Nguồn thô có 58.009 bình luận; simplified dùng ở đây có 54.263. Số mẫu khác nhau do dạng dữ liệu, không phải nhóm tự chia lại. [Google Research](https://github.com/google-research/google-research/blob/master/goemotions/README.md).
Giữ neutral theo annotation; không tự gán neutral khi không nhãn nào vượt ngưỡng. Mỗi câu có thể có nhiều nhãn; không dùng argmax để ép một cảm xúc.
EDA cần phân bố nhãn, số nhãn/câu, độ dài, đồng xuất hiện và kiểm trùng văn bản/ID giữa split. Nếu giữ split chính thức, ghi nhận các trường hợp trùng và giới hạn đánh giá; không âm thầm xóa hoặc chia lại.
Checkpoint và scores lớn chia sẻ riêng hoặc tái chạy; code, cấu hình và bảng nhỏ lưu trên GitHub. Ghi thư viện, thiết bị, thời gian chạy và revision để tái hiện.

## 5. Zero-shot B — làm chung, Duy điều phối

1. Nhận phần B từ Đức Trí nếu có; kiểm phiên bản, mapping, template và số liệu trước khi dùng.
2. Dùng `facebook/bart-large-mnli` qua `pipeline("zero-shot-classification")`; không fine-tune trên GoEmotions. Model đã được huấn luyện MNLI trước khi nhóm sử dụng.
3. Candidate labels gồm đủ 28 nhãn; bật `multi_label=True`. Template khởi đầu đề xuất: `This text expresses {}.`.
4. Pilot một tập nhỏ validation để kiểm đúng luồng; pilot không thay cho kết quả full.
5. Ánh xạ output về `data/labels.json`: HF trả labels theo score giảm dần, không theo thứ tự nhãn chuẩn.
6. Chạy full validation theo batch; lưu dần scores cùng ID và cấu hình để có thể tiếp tục.
7. Báo hàng gốc @0.5. Nếu dùng nhãn validation chọn template hoặc ngưỡng, giữ hàng riêng và ghi rõ có hiệu chỉnh bằng nhãn đích dù không cập nhật trọng số.
8. Khóa checkpoint revision/template/ngưỡng trước test; dùng module metric chung.
9. Đọc lỗi B, viết cách hoạt động NLI, kết quả và giới hạn. Duy nhận bàn giao cuối; mỗi người ghi đóng góp thật.

Một nhãn cảm xúc được đặt vào giả thuyết để kiểm mức entailment của câu. `neutral` trong suy luận NLI khác nhãn cảm xúc `neutral` của GoEmotions. [Model card BART-MNLI](https://huggingface.co/facebook/bart-large-mnli), [HF pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.ZeroShotClassificationPipeline).
Full validation cần 5.426 × 28 = 151.928 cặp câu/giả thuyết; đo tài nguyên thay vì giả định chạy tức thì.
**Nghiệm thu B:** full scores đúng ID/mapping; đủ metrics; template/revision truy được; mô tả đúng không fine-tune; số gốc và hiệu chỉnh tách rõ; phần viết và ví dụ lỗi.

## 6. Quốc Khánh — C1 BERT và phần đầu báo cáo

### C1 BERT

1. Checkpoint đề xuất `google-bert/bert-base-uncased`; tokenizer tương ứng; khóa revision.
2. Tạo script fine-tune chung nhận checkpoint/seed/config; chia sẻ cho Đức Trí và Nhật Huy. Mỗi bạn vẫn chạy, kiểm và sở hữu kết quả kiến trúc mình.
3. Pilot multi-hot float, head 28 logits, loss và lưu checkpoint/scores; không dùng nhãn đơn.
4. Chốt cấu hình sau pilot; chạy ≥3 seed đề xuất **42, 123, 2026**.
5. Mỗi seed chọn checkpoint bằng Macro-F1 validation @0.5; lưu log/cấu hình/revision và scores.
6. Tính đủ metrics từng seed; bảng mean ± sample std `ddof=1`; giữ số chi tiết.
7. Phân tích lỗi BERT và đối chiếu cùng ID với RoBERTa/DistilBERT; tự viết phương pháp, kết quả, hạn chế C1.
8. Bàn giao checkpoint và hàm suy luận cho Huy nếu BERT được chọn làm demo.

### Phần đầu và tổng hợp báo cáo

- Thông tin đề tài, nhóm 4 người, mục tiêu và bố cục.
- Tóm tắt paper GoEmotions bằng lời nhóm: nguồn, taxonomy, annotation, baseline trong paper và đóng góp.
- Dữ liệu/split/EDA; trích đúng người và artifact tạo ra bảng.
- Thiết kế so sánh A/B/C/D; tiền xử lý và protocol dùng chung.
- Nhận A/B và bảng tổng hợp từ Duy, C2 từ Trí, C3/demo từ Huy. Mỗi người tự viết và kiểm phần mình.
- Thống nhất trích nguồn; tổng hợp 2 báo cáo tiến độ, báo cáo cuối và slide.

**Nghiệm thu Khánh:** BERT ≥3 seed có log/checkpoint/scores, từng seed và mean±std; phần đầu/báo cáo ghép đủ nguồn, không dùng số dự kiến như số thực nghiệm.

## 7. Đức Trí — C2 RoBERTa và hỗ trợ bàn giao B

1. Checkpoint đề xuất `FacebookAI/roberta-base`; tokenizer đúng RoBERTa; khóa revision.
2. Dùng script chung; kiểm tokenization, truncation, multi-hot và output 28 logits riêng cho C2.
3. Pilot rồi chốt cấu hình; tự chạy ≥3 seed **42, 123, 2026**, lưu đầy đủ từng run.
4. Xuất validation scores/checkpoint/log; tính metrics và mean ± std cho C2.
5. Đọc lỗi RoBERTa, đối chiếu BERT/DistilBERT theo ID; tự viết phương pháp/kết quả/hạn chế C2.
6. Chọn ngưỡng riêng trên validation của đúng model/seed khi thực hiện nâng cao; cung cấp bảng trước/sau và F1 nhãn hiếm cho Duy.
7. Bàn giao phần B đã nhận trước nếu đã có; hỗ trợ giải thích NLI và mapping. C2 là phần fine-tune chính cần hoàn thành.
8. Nếu RoBERTa thắng, chuyển checkpoint/tokenizer/ngưỡng/hàm suy luận cho Huy.

**Nghiệm thu Trí:** RoBERTa đủ ≥3 seed, mean±std và phần C2; lỗi có ID; phần B bàn giao ghi đúng việc thực tế đã làm.

## 8. Nhật Huy — C3 DistilBERT và demo D

1. Checkpoint đề xuất `distilbert/distilbert-base-uncased`; tokenizer và revision tương ứng.
2. Pilot bằng script chung; chốt cấu hình rồi tự chạy ≥3 seed **42, 123, 2026**.
3. Lưu checkpoint/log/config/scores; tính metrics từng seed, mean±std; viết phương pháp và lỗi C3.
4. Nhận checkpoint của kiến trúc C thắng theo validation cùng mapping/ngưỡng đã khóa; không mặc định DistilBERT thắng.
5. Dựng Gradio/Streamlit: nhập văn bản tiếng Anh, hiển thị nhiều nhãn cùng score và thông tin model/ngưỡng.
6. Xử lý input rỗng, văn bản dài/truncation và trường hợp không nhãn nào đạt ngưỡng.
7. Kiểm cùng input cho cùng score/nhãn giữa app và script suy luận của checkpoint đó.
8. Bàn giao mã app, dependencies, hướng dẫn khởi động, model revision và link hoặc ảnh/video minh chứng demo chạy.

**Nghiệm thu Huy:** DistilBERT ≥3 seed và mean±std; demo từ best C chạy/khởi động lại được, mapping/ngưỡng đúng và có hướng dẫn.

## 9. Quy tắc fine-tune, seed và chọn C

- Tokenizer riêng → encoder pretrained → head **28 logits**.
- Multi-hot float; dùng `BCEWithLogitsLoss`, truyền logits trực tiếp vào loss. Dùng sigmoid khi xuất scores. [PyTorch BCE](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).
- HF text-classification tutorial dùng để tham khảo luồng huấn luyện; cần điều chỉnh nhãn/loss/metric sang multi-label. Không sao chép nguyên ví dụ sentiment hai lớp.
- Cấu hình pilot đề xuất: max_length 128, batch 16, learning rate 2e-5, epoch 3. Đây là điểm bắt đầu; chốt theo GPU/validation và ghi mọi thay đổi.
- Nếu tích lũy gradient, ghi effective batch; dùng cùng cấu hình giữa các seed của một thí nghiệm.
- Seed tác động khởi tạo head, dropout và thứ tự batch. Ghi seed Python/NumPy/PyTorch cùng thiết bị/thư viện; cùng seed không bảo đảm mọi máy cho số giống hệt. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
- Bảng chính: A/B/C @0.5; C có từng seed và **mean ± sample std**. Dùng `np.std(values, ddof=1)`; ghi n. Không chỉ báo seed đẹp nhất. [NumPy std](https://numpy.org/doc/stable/reference/generated/numpy.std.html).
- Chọn kiến trúc C bằng **mean Macro-F1 validation @0.5** trên ba seed; giữ quy tắc này trước khi đọc test. Khi bằng nhau, xét std thấp hơn, rồi chi phí suy luận; ghi cách xử lý.
- Checkpoint demo đề xuất là checkpoint có Macro-F1 val tốt nhất trong kiến trúc thắng. Điểm checkpoint demo khác điểm trung bình kiến trúc.
- Ngưỡng cải tiến chọn trên validation của đúng checkpoint/seed. Điểm tuned-val có thể lạc quan; không dùng test chọn kiến trúc/seed/ngưỡng.
- Sau khi khóa cấu hình, đánh giá các run/config đã khai báo trên test; báo C mean±std test và A/B theo protocol. Không thay cấu hình theo thứ hạng test.
- Ba seed giúp mô tả độ dao động; std thấp không tự chứng minh mô hình hiểu ngôn ngữ tốt hơn. Thảo luận cùng F1, lỗi và chi phí.

## 10. Nâng cao và phân tích lỗi

### Nâng cao có số liệu trước/sau

1. Duy điều phối bảng chung; chủ sở hữu mỗi mô hình tạo scores/ngưỡng của mình.
2. Chọn danh sách nhãn hiếm dựa trên train trước khi xét hiệu quả: grief 77, pride 111, relief 153, nervousness 164, embarrassment 303.
3. Giữ hàng gốc @0.5, thêm ngưỡng riêng từng nhãn chọn bằng validation. Lưới khởi đầu 0.05–0.95, bước 0.05; chốt luật hòa trước khi so sánh.
4. A có thể dùng `class_weight="balanced"`; C có thể dùng `pos_weight` từ train nếu chọn thí nghiệm weighting. Weighting cần train lại; tuning chỉ thay luật quyết định.
5. Không dùng ngưỡng A cho B/C hoặc model weighted mới; lưu ngưỡng kèm model/revision/seed.
6. Báo Macro/Micro-F1, P/R, Hamming cùng F1/P/R/support của từng nhãn hiếm trước/sau; giữ cả nhãn tăng, không tăng hoặc giảm.
7. Với C cải tiến, nếu kết luận bằng kết quả kiến trúc, chạy và báo đủ ≥3 seed; nếu chỉ pilot một seed, ghi là pilot.
8. Contrastive là hướng mở rộng nếu nhóm chọn, không bắt buộc làm thêm khi đã hoàn thành phương án nâng cao cô cho phép. Giải thích hiệu quả bằng số thật, không hứa sẵn sẽ tăng.
9. Kết luận nâng cao cuối xác nhận trên test sau khóa protocol; không coi tuned-val là đánh giá độc lập.

### Ít nhất ba nhóm lỗi so sánh C1/C2/C3

| Nhóm lỗi | Cách xác định | Bằng chứng |
|---|---|---|
| Cảm xúc gần nghĩa | FN nhãn thật đồng thời FP nhãn khác trong cùng câu | Cặp nhãn, số đếm, ID/text, true/pred ba C |
| Thiếu cảm xúc thứ hai | Câu có ≥2 nhãn thật, dự đoán chỉ tìm một phần | Nhãn thật/đoán, scores/ngưỡng và đối chiếu ba C |
| Hàm ý, phủ định, mỉa mai hoặc thiếu ngữ cảnh | Lọc FN/FP, đọc thủ công để xác nhận cách diễn giải | Ví dụ có ID và lý do ba C sai giống/khác nhau |
| Nhãn hiếm hoặc neutral | Support thấp hoặc lỗi neutral cụ thể | F1/P/R/support, ID và biến động qua seed |

Khánh/Trí/Huy tự phân tích C mình và nộp ví dụ theo mẫu chung; Duy ghép bảng, cả nhóm đọc lại ít nhất ba nhóm có minh chứng thật. A/B được bổ sung để đối chiếu.
Giữ cùng ID và cách chọn run để so sánh; nêu rõ ví dụ đến từ checkpoint demo hay một seed đại diện.
Đồng xuất hiện nhãn thật trong EDA khác cặp lỗi FN/FP. Một câu có thể đóng góp nhiều cặp; không cộng số cặp để suy ra tổng câu sai.

## 11. Trình tự triển khai theo đầu việc

1. Đọc paper, kiểm EDA; thống nhất split, label mapping, metric, scores, seed và cách chọn C.
2. Duy hoàn thiện A và điều phối B; Trí bàn giao phần B có sẵn. Khánh/Trí/Huy pilot C riêng, dùng script chung.
3. Chốt cấu hình; mỗi người C chạy đủ seed, xuất validation scores/log/checkpoint và phần báo cáo riêng.
4. Duy ghép bảng A/B/C, đủ ≥9 run C, mean±std; ba chủ C cùng phân tích ≥3 nhóm lỗi.
5. Thực hiện nâng cao và bảng nhãn hiếm trước/sau; chọn C theo quy tắc validation đã chốt.
6. Huy tích hợp demo từ checkpoint C đã chọn; so score app với script.
7. Khóa model/revision/config/seed/template/ngưỡng và danh sách thí nghiệm; đánh giá test, không chọn lại.
8. Khánh ghép báo cáo; cả nhóm kiểm nguồn, số liệu, công việc thực tế và chuẩn bị demo/phản biện.

Đầu việc có thể phối hợp đồng thời khi đủ đầu vào. Hạn nộp và kênh báo cáo theo thông báo cô; kế hoạch này không đặt tuần/ngày hoặc thời lượng thực hiện.

## 12. Báo cáo, bảng kết quả và bộ sản phẩm cuối

| Sản phẩm | Nội dung theo yêu cầu cô | Người tổng hợp và đầu vào |
|---|---|---|
| Báo cáo tiến độ lần 1 | Paper/mục tiêu; EDA; A/B có số cụ thể; tiến độ từng C và seed, môi trường/vướng mắc; 4 vai trò và mức hoàn thành | Khánh tổng hợp; Duy A/B, Trí C2, Huy C3 |
| Báo cáo tiến độ lần 2 | Đủ 3 C × ≥3 seed, mean±std so với A/B; ≥3 nhóm lỗi; minh chứng demo; kế hoạch/kết quả nâng cao bước đầu | Khánh ghép; Duy bảng nâng cao/lỗi, chủ C cấp số, Huy demo |
| Báo cáo cuối docx/pdf | Toàn bộ phương pháp/cấu hình/kết quả; nâng cao và nhãn hiếm; best/worst C và std; hạn chế/hướng phát triển; đóng góp có tỷ lệ; mã và demo | Cả nhóm kiểm; Khánh tổng hợp |
| Artifacts | Checkpoint/tokenizer, log/config/revision, scores theo ID, metrics JSON/CSV, ngưỡng đúng model/seed | Chủ sở hữu từng mô hình |
| Slide và phản biện | Bài toán, thiết kế, bảng số, lỗi, nâng cao, demo, kết luận và giới hạn | Mỗi người trình bày phần mình |

### Các bảng phải có

- A/B/C gốc, ghi rõ split và luật ngưỡng; C có mean±std.
- Từng seed của cả ba C: ≥9 run, config/revision truy được.
- Nâng cao trước/sau, cùng model/seed/split và quy tắc chọn.
- F1/P/R/support nhãn hiếm và thay đổi theo từng nhãn.
- Ít nhất ba loại lỗi đối chiếu C1/C2/C3 theo ID.
- Đóng góp 4 thành viên: việc thực tế, artifact/commit và tỷ lệ nhóm thống nhất.

Chưa có số thì ghi **chưa đo/chưa bàn giao**, không điền số giả định. Không tự chia đều tỷ lệ công sức. Code baseline đã có không đồng nghĩa B, ba C, demo và báo cáo cuối đã hoàn thành.

### Checklist nghiệm thu cả nhóm

- [ ] A và B có code chạy được, scores và số liệu cụ thể.
- [ ] Ba C, mỗi C do một người phụ trách và ≥3 seed; đủ từng seed + mean±std.
- [ ] Cùng split/mapping/metric; khóa lựa chọn bằng validation trước test.
- [ ] Demo từ C tốt nhất chạy được, có hướng dẫn và minh chứng.
- [ ] ≥3 loại lỗi đối chiếu ba C có ID và giải thích.
- [ ] Nâng cao có bảng trước/sau và F1 nhãn hiếm.
- [ ] Đủ hai báo cáo tiến độ và báo cáo cuối theo kênh cô.
- [ ] Nguồn đúng phần dùng, công việc thực tế và hạn chế được ghi rõ.

## 13. Tài liệu tham khảo NLP đã đối chiếu

Kiểm tra nguồn ngày **03/10/2026**. Dùng paper gốc, model card của bên phát hành, tài liệu thư viện và giáo trình từ tác giả. Nguồn phương pháp không thay thế số thực nghiệm nhóm.

### Paper, dữ liệu và kiến trúc

| Mã | Nguồn | Liên quan tới đồ án / người đọc |
|---|---|---|
| R01 | [Demszky et al. (2020), GoEmotions: A Dataset of Fine-Grained Emotions — ACL](https://aclanthology.org/2020.acl-main.372/) | Paper nền tảng; tất cả đọc, Khánh tóm tắt nguồn/annotation/taxonomy và giới hạn |
| R02 | [Google Research — GoEmotions README/data](https://github.com/google-research/google-research/blob/master/goemotions/README.md) | Đối chiếu 58.009 nguồn thô, official split, nhãn và baseline BERT trong paper |
| R03 | [HF dataset card — google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Config simplified, schema và nguồn dữ liệu nhóm tải |
| R04 | [Devlin et al. (2019), BERT — NAACL](https://aclanthology.org/N19-1423/) | C1, Khánh: bidirectional pretraining và fine-tune |
| R05 | [Liu et al. (2019), RoBERTa](https://arxiv.org/abs/1907.11692) | C2, Trí: thay đổi cách pretrain so với BERT; kiểm bằng thực nghiệm nhóm |
| R06 | [Sanh et al. (2019), DistilBERT](https://arxiv.org/abs/1910.01108) | C3, Huy: distillation, mô hình nhỏ và đánh đổi hiệu quả/chi phí |
| R07 | [Yin, Hay, Roth (2019), Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) | B, Duy/nhóm: cách dùng entailment cho zero-shot |
| R08 | [facebook/bart-large-mnli — model card](https://huggingface.co/facebook/bart-large-mnli) | B: checkpoint đã học MNLI, candidate labels và dùng nhiều nhãn |
| R09 | [google-bert/bert-base-uncased — model card](https://huggingface.co/google-bert/bert-base-uncased) | Checkpoint/tokenizer C1; log revision tải thực tế |
| R10 | [FacebookAI/roberta-base — model card](https://huggingface.co/FacebookAI/roberta-base) | Checkpoint/tokenizer C2; log revision |
| R11 | [distilbert/distilbert-base-uncased — model card](https://huggingface.co/distilbert/distilbert-base-uncased) | Checkpoint/tokenizer C3; log revision |

### Triển khai, đánh giá và demo

| Mã | Nguồn chính thức | Dùng ở đâu |
|---|---|---|
| R12 | [scikit-learn 1.7 — TfidfVectorizer](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) | A: unigram/bigram, min_df, vocab fit trên train |
| R13 | [scikit-learn 1.7 — OneVsRestClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) | A: 28 bộ phân loại nhị phân độc lập |
| R14 | [scikit-learn 1.7 — LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html) | A: solver, regularization, class_weight và hội tụ |
| R15 | [scikit-learn 1.7 — Precision/Recall/F1](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.precision_recall_fscore_support.html) | A/B/C: micro/macro, per-label, support và zero_division |
| R16 | [scikit-learn 1.7 — Hamming Loss](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.hamming_loss.html) | A/B/C: sai quyết định nhãn; đọc cùng F1/P/R |
| R17 | [scikit-learn 1.7 — Decision threshold tuning](https://scikit-learn.org/1.7/modules/classification_threshold.html) | Chọn ngưỡng bằng validation; nhóm tự triển khai vòng lặp từng nhãn |
| R18 | [HF — ZeroShotClassificationPipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.ZeroShotClassificationPipeline) | B: multi_label=True, template, output labels được sắp theo score |
| R19 | [HF — Text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) | C: tham khảo training loop; đổi sang 28 multi-hot/BCE/sigmoid/metrics chung |
| R20 | [PyTorch — BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html) | C: binary loss cho mỗi nhãn, logits trực tiếp, pos_weight nếu chọn |
| R21 | [PyTorch — Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) | C: seed, deterministic settings và giới hạn tái hiện |
| R22 | [NumPy — std](https://numpy.org/doc/stable/reference/generated/numpy.std.html) | Bảng mean±std, dùng ddof=1 và giữ từng seed |
| R23 | [Gradio — Quickstart](https://www.gradio.app/guides/quickstart) | D: giao diện nhập văn bản nối hàm suy luận; Huy đọc |

### Giáo trình và hướng đọc

| Mã | Nguồn | Phần nên đọc |
|---|---|---|
| R24 | [Jurafsky & Martin (2026), Speech and Language Processing, bản thảo 3rd edition ngày 19/08/2026](https://web.stanford.edu/~jurafsky/slp3/) | Chương 2 Words and Tokens, 4 Logistic Regression/Text Classification; chương 7 Transformers/Pretraining, 9 Masked Language Models |
| R25 | [Hugging Face — LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Transformer, tokenizer, datasets và fine-tune; học phần liên quan, không cần đọc toàn bộ |

**Thứ tự đọc theo vai trò:**
- Tất cả: R01–R03 và quy ước đánh giá của nhóm.
- Duy: R12–R18, R07–R08; R24 chương 2/4 để hiểu A, NLI/pipeline để làm B.
- Khánh: R04/R09/R19–R22; R24 chương 7/9.
- Trí: R05/R10/R19–R22; R07–R08 khi bàn giao B.
- Huy: R06/R11/R19–R23 và phần suy luận checkpoint thắng.

**Quy tắc trích dẫn và kiểm chứng:**
- Viết mục tiêu/đóng góp paper bằng lời nhóm; trích paper khi nói về phương pháp, model card khi nói về checkpoint, docs khi nói về API.
- Ghi ngày truy cập và phiên bản thư viện dùng thực tế. Docs PyTorch 2.14 là bản đọc được khi kiểm nguồn; môi trường chạy C phải ghi phiên bản đã cài, không suy ra từ link.
- Số F1 trong paper không phải số nhóm đo. Khác checkpoint/split/ngưỡng/protocol thì không kết luận vượt paper chỉ bằng việc so hai con số.
- Không khẳng định RoBERTa hay DistilBERT sẽ thắng trước khi có kết quả.
- Teacher requirement, lựa chọn triển khai và trạng thái đã chạy phải được ghi riêng.

**Tài liệu nội bộ:** [Notebook EDA](../notebooks/eda.ipynb), [EDA bổ sung](../notebooks/eda_extra.ipynb), [báo cáo dữ liệu](../reports/THONG_KE_DU_LIEU.md), [hồ sơ đối chiếu phần A](BASELINE_REVIEW.md).
[Trang kế hoạch Notion](https://app.notion.com/p/3ed7c27769028185af2dfbaac4c4586b).

## Mục riêng của Duy

[Phần của Duy — baseline, zero-shot, tài liệu và báo cáo](https://app.notion.com/p/3ee7c277690281e693c7f1cf985d569d).
Notebook, hướng dẫn, bảng kết quả, hồ sơ kiểm chứng và báo cáo cá nhân nằm trong mục này.
