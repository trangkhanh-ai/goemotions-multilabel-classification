# Kiểm chứng nguồn và nhận định trong báo cáo GoEmotions

**Ngày kiểm tra:** 08/10/2026. **Phạm vi:** nguồn nền tảng, phương pháp, số liệu nguồn và liên hệ Case Study 4. Đây là hồ sơ kiểm chứng tài liệu; trạng thái huấn luyện và kết quả cuối phải lấy từ artifacts của các lần chạy thực tế.

**Cập nhật bản bàn giao 09/10/2026:** đã xuất báo cáo sáu chương **55 trang** và bài IEEE hai cột **8 trang**, cùng hai báo cáo tiến độ **5/10 trang**. Kết quả đầy đủ có **9/9 run C**, 72 bản ghi và 36 nhóm tổng hợp; kiểm thử sau tích hợp **77/77 PASS**. Đối chiếu số cuối tại [bảng kết quả](project_results/RESULTS.md) và [hồ sơ kiểm bản Word/PDF](execution/final_report_verification.json). Các số trang 45/6 bên dưới ghi lại lượt đọc nguồn ban đầu, không phải bản cuối. Lần cập nhật này chỉ đối chiếu trạng thái bàn giao, không tuyên bố đã đọc lại nguồn ngoài.

## 1. Tài liệu đã đối chiếu

**Bổ sung diễn giải ngày 10/10/2026:** hai báo cáo cuối đã thêm ba kịch bản ứng
dụng dưới dạng đề xuất chưa đo ROI; validation tuned/test locked được tách rõ;
đánh đổi của BERT và đối chứng A balanced được lấy từ cùng bảng test đã khóa.
Phần siêu tham số chỉ ghi C1 tham khảo mục 5.3 GoEmotions, C2/C3 là cấu hình
nhóm; chín run/ba seed không được mô tả thành tìm kiếm learning rate tối ưu.
Chưa có bằng chứng để quy cấu hình cho gradient explosion hoặc hội tụ nhanh
hơn. Số trang và checksum của bản xuất hiện tại lấy từ
`execution/final_report_verification.json`, thay các mốc trang lịch sử ở trên.
Không thay năm truy cập API thành năm phát hành phần mềm; không tự điền năm
edition IEEE Reference Guide chưa xác minh được. Căn cứ và phạm vi nguồn đã
đọc vẫn như bảng bên dưới; lần bổ sung diễn giải không tuyên bố đọc lại mọi PDF.

- `reports/references_ieee.json`: danh mục 26 nguồn của báo cáo sáu chương.
- `reports/BAO_CAO_DO_AN_NOI_DUNG.md`: các phần dữ liệu, nền tảng, phương pháp, đánh giá, lựa chọn mô hình và giá trị ứng dụng.
- `reports/BAI_BAO_GOEMOTIONS_IEEE_NOI_DUNG.md`: bản bài viết hai cột; có danh mục riêng 20 nguồn.
- `reports/BAI_BAO_GOEMOTIONS_IEEE_KIEM_CHUNG.json`: hồ sơ định dạng và thời điểm xuất bài viết.
- Hai bản PDF tương ứng ở lượt đọc nguồn ban đầu ngày 08/10: bản sáu chương 45 trang và bài viết 6 trang tại thời điểm đó; đã kiểm tra nội dung Case Study 4 trong bản xuất. Bản cuối 55/8 trang được kiểm riêng trong hồ sơ bàn giao nêu trên.
- PDF người dùng cung cấp `C:\Users\dzyuu\Downloads\lee2020.pdf`: đọc trực tiếp bảy trang PDF 98–104, tương ứng trang in 82–88. Không dùng bản tóm tắt thứ cấp để xác nhận nội dung case.

Đối với GoEmotions, đã đọc bài PDF gốc cùng README chính thức. Đối với BERT, RoBERTa, DistilBERT, Yin và BART, kiểm tra metadata/abstract của nguồn gốc và model card chính thức cho các nhận định khái quát; không tuyên bố đã đọc lại toàn bộ các PDF đó trong lượt kiểm tra này. Tài liệu phần mềm được đối chiếu tại nội dung API liên quan. Việc kiểm tra không chạy model, không sử dụng GPU và không thay đổi quá trình huấn luyện đang hoạt động.

## 2. Kết luận

Các nhận định nền tảng được kiểm tra phù hợp với nguồn: bài toán GoEmotions có 28 nhãn, giữ official split, phân loại đa nhãn bằng scores độc lập, dùng BERT theo hướng bài gốc và bổ sung các hệ thống so sánh theo đồ án. Phần công nghiệp dùng Case Study 4 đúng trường hợp và ghi rõ giới hạn khi liên hệ sang NLP.

Quy tắc hòa khi chọn kiến trúc đã được sửa cho khớp code; hai citation PyTorch đã đổi sang đúng tag v2.13.0; mốc edition IEEE Guide chưa xác minh lại được đã được bỏ khỏi metadata. Các sửa đổi này nằm trong nguồn báo cáo và references. Tại lượt kiểm nguồn ban đầu, DOCX/PDF là bản nháp chờ tổng hợp cuối; trạng thái này đã được thay bằng bản xuất 55/8 trang và kết quả full nêu ở cập nhật 09/10. Bằng chứng hoàn tất thí nghiệm lấy từ artifacts thực tế, không suy từ việc có file Word/PDF.

## 3. Bảng nguồn → nhận định → căn cứ → giới hạn

| Nguồn gốc | Nhận định trong báo cáo | Căn cứ đã đọc | Điều kiện và giới hạn |
|---|---|---|---|
| [Demszky và cộng sự, ACL 2020](https://aclanthology.org/2020.acl-main.372/) và [PDF gốc](https://aclanthology.org/2020.acl-main.372.pdf) | Reddit tiếng Anh; 27 cảm xúc + neutral; đa nhãn. | §3; metadata: năm 2020, tr. 4040–4054, DOI `10.18653/v1/2020.acl-main.372`. | Không đổi năm công bố thành 2022–2026. Chất lượng trên tiếng Việt/doanh nghiệp cần đánh giá riêng. |
| [README Google Research](https://raw.githubusercontent.com/google-research/google-research/master/goemotions/README.md) | Khoảng 58.000 bình luận ban đầu; official train/validation/test là 43.410/5.426/5.427. | README ghi 58.009 bản raw và các số mẫu từng split; phần dữ liệu dùng sự đồng thuận của ít nhất hai người gán nhãn. | Tổng ba split là 54.263, không phải 58.009. Không chia lại ngẫu nhiên rồi gọi đó là official split. |
| [Dataset card chính thức](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Sử dụng cấu hình simplified và danh sách labels cho multi-hot. | Card phân biệt simplified với full; simplified có text, labels và id. | Dataset card hiện tại không chứng minh revision cụ thể đã chạy; revision và hash phải kiểm trong manifest của repo. |
| [GoEmotions PDF](https://aclanthology.org/2020.acl-main.372.pdf), §5; [README](https://raw.githubusercontent.com/google-research/google-research/master/goemotions/README.md) | BERT-base-cased bám hướng bài gốc. | §5.2: BERT/dense/sigmoid/cross-entropy; §5.3: batch 16, learning rate 5e−5, ít nhất 4 epoch. README xác nhận cased. | TF-IDF + LR, zero-shot và ba C là thiết kế bổ sung của đồ án. |
| [GoEmotions PDF](https://aclanthology.org/2020.acl-main.372.pdf), Table4 | F1 trung bình khoảng 0,46 là kết quả được bài gốc công bố. | Table4 và phần thảo luận kết quả. | Độ phân tán khoảng 0,19 ở bảng gốc là giữa các nhãn. Không biến thành std giữa seed; không so điểm test của paper với tuned-validation của nhóm để tuyên bố vượt paper. |
| [BERT, NAACL 2019](https://aclanthology.org/N19-1423/) và [bert-base-cased card](https://huggingface.co/google-bert/bert-base-cased) | BERT học biểu diễn theo ngữ cảnh hai chiều và có thể fine-tune cho phân loại. | Abstract/metadata bài gốc; model card của checkpoint được lựa chọn. | Năm2019, tr. 4171–4186, DOI `10.18653/v1/N19-1423`. Model card không chứng minh head GoEmotions đã huấn luyện; cần checkpoint thực tế. |
| [RoBERTa, arXiv:1907.11692](https://arxiv.org/abs/1907.11692) và [roberta-base card](https://huggingface.co/FacebookAI/roberta-base) | RoBERTa nghiên cứu điều chỉnh quy trình pretraining của BERT; dùng làm C2. | Abstract nhấn mạnh thiết kế pretraining, dữ liệu và siêu tham số; metadata đăng đầu năm 2019. | Kết quả trên GLUE/RACE/SQuAD trong nguồn không phải kết quả GoEmotions của nhóm. Không kết luận C2 tốt nhất trước khi đo. |
| [DistilBERT, arXiv:1910.01108](https://arxiv.org/abs/1910.01108) và [checkpoint card](https://huggingface.co/distilbert/distilbert-base-uncased) | Knowledge distillation tạo mô hình nhỏ hơn; C3 khảo sát chất lượng/tài nguyên. | Abstract và metadata: đăng đầu 02/10/2019; bản cập nhật v4 là 01/03/2020. | Trích năm 2019 cho preprint đầu là hợp lý. Lợi thế tốc độ trong paper không tự là tốc độ đo trên máy nhóm. Không dùng số tham số làm bằng chứng đã benchmark suy luận. |
| [Yin, Hay, Roth, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1404/) | Có thể diễn đạt zero-shot text classification như textual entailment. | Abstract/metadata xác nhận phương pháp entailment; tr. 3914–3923, DOI `10.18653/v1/D19-1404`. | Bài này hỗ trợ hướng phương pháp, không phải kết quả BART-MNLI trên GoEmotions của repo. |
| [BART, ACL 2020](https://aclanthology.org/2020.acl-main.703/) và [bart-large-mnli card](https://huggingface.co/facebook/bart-large-mnli) | B dùng checkpoint BART đã học MNLI, đưa text thành premise và nhãn thành hypothesis; `multi_label=True`. | BART abstract xác nhận pretraining denoising; model card xác nhận checkpoint MNLI và cách đánh giá nhãn độc lập. | “Không fine-tune” nghĩa là nhóm không cập nhật trọng số trên GoEmotions. Checkpoint đã được fine-tune MNLI trước đó. Biến thể tuning threshold có dùng nhãn validation phải báo riêng. Neutral trong NLI không phải nhãn neutral của GoEmotions. |
| [TF-IDF API, scikit-learn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) | TF-IDF biểu diễn từ/cụm từ; `sublinear_tf`, smoothing và chuẩn hóa L2. | Các tham số ngram_range, sublinear_tf, smooth_idf, norm được API mô tả. | Vocabulary/IDF phải fit trên train. Biểu diễn không tự hiểu cảm xúc; các trọng số LR học quan hệ đặc trưng–nhãn. |
| [LogisticRegression API](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html) và [OneVsRestClassifier API](https://scikit-learn.org/1.7/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) | A có 28 bộ phân loại nhị phân độc lập; balanced weighting tùy tần suất từng lớp. | OvR nhận ma trận chỉ thị nhãn 2D; LR balanced dùng số mẫu chia số lớp và tần suất lớp. | LR `class_weight` cho cả hai lớp khác BCE `pos_weight` chỉ trọng số phần dương. Solver liblinear không nên được mô tả thành đúng vòng gradient descent minh họa bằng tay. |
| [BCEWithLogitsLoss tại tag PyTorch v2.13.0](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/nn/modules/loss.py) | C dùng logits + BCEWithLogitsLoss; sigmoid khi suy luận; có thể dùng pos_weight. | Đọc docstring lớp BCEWithLogitsLoss trong nguồn tag chính thức: đầu vào logits, target cùng shape; phần multi-label/pos_weight. | Có nguồn đúng runtime 2.13.0; tránh dùng tên “PyTorch 2.14 documentation” để mô tả môi trường 2.13. BCEWithLogitsLoss đã tích hợp sigmoid trong phép loss, không sigmoid trước loss lần nữa. |
| [Threshold tuning, scikit-learn 1.7](https://scikit-learn.org/1.7/modules/classification_threshold.html) | Threshold cần dữ liệu validation riêng và metric lựa chọn phù hợp. | §3.3.1.2 cảnh báo dùng lại dữ liệu train để tìm ngưỡng; prefit cần validation mới. | Lưới 19 điểm và quy tắc hòa là thiết kế repo, không được nguồn này quy định. Điểm trên chính validation chọn threshold có thể lạc quan; benchmark cuối cần test sau khi khóa quyết định. |
| [Precision/Recall/F-score API](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.precision_recall_fscore_support.html) và [Hamming Loss API](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.hamming_loss.html) | Macro trung bình theo nhãn; Micro gộp TP/FP/FN; Hamming là tỷ lệ sai quyết định nhãn. | API mô tả rõ các loại average và phân biệt Hamming với sai toàn bộ tập nhãn. | Macro-F1 không là tỷ lệ câu đúng hoàn toàn. Hamming thấp chưa chứng minh bắt được nhãn hiếm. Thứ tự cột nhãn và zero_division cần thống nhất giữa mọi hệ thống. |
| [NumPy std](https://numpy.org/doc/stable/reference/generated/numpy.std.html) | Mean±sample std giữa seed dùng `ddof=1`. | Công thức mẫu số N−ddof và phần sample standard deviation. | Một run không có sample std; không điền0 để giả ổn định. Trang stable hiện là manual 2.5, chỉ dùng ở đây cho định nghĩa ddof; phiên bản thư viện đã chạy cần lấy metadata, không suy ra từ URL stable. |
| [Reproducibility tại tag PyTorch v2.13.0](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/docs/source/notes/randomness.md) | Seed giảm nguồn ngẫu nhiên nhưng không cam kết tái hiện từng bit trên mọi máy/phiên bản. | Đã đọc toàn bộ tệp 241 dòng: seed Python/NumPy/PyTorch, thuật toán không xác định, SDPA và DataLoader. | SDPA với `warn_only=True` có thể cảnh báo nhưng vẫn chạy backend không xác định. Ghi3seed cùng cấu hình và giới hạn CUDA là phù hợp; không gọi cảnh báo này là bằng chứng tái hiện tuyệt đối. |
| [Springer: Industrial AI, Jay Lee](https://link.springer.com/book/10.1007/978-981-15-2144-7) + PDF người dùng | Case Study 4, §4.2.3.1, tr. in 82–88: nhà máy LCD Shenzhen, FMCS/SCADA, dự báo/tối ưu/health. | Publisher xác nhận sách 2020/DOI; nội dung case đọc trực tiếp trong PDF 98–104. | Website publisher xác nhận metadata; phần case đầy đủ xác nhận từ PDF. Liên hệ quy trình phản hồi NLP là suy luận ứng dụng của nhóm, không phải thực nghiệm NLP trong sách. |
| [IEEE Author Center](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/) và [IEEE Reference Guide](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf) | Bản bài viết trình bày hai cột và dùng số tài liệu tham khảo. | Trang Author Center cung cấp template; hồ sơ xuất ghi các hướng dẫn IEEE-hosted đã dùng. | Bản viết môn học theo bố cục IEEE chưa phải bài được IEEE xuất bản/chấp nhận. Lượt truy cập độc lập này nhận chuyển hướng Reference Guide sang Google Docs; nguồn báo cáo đã bỏ mốc edition chưa kiểm lại được. |

## 4. Case Study 4: dấu vết để tự kiểm tra

Nguồn đầy đủ: Jay Lee, *Industrial AI: Applications with Sustainable Performance*, Springer Singapore, 2020, DOI [10.1007/978-981-15-2144-7](https://link.springer.com/book/10.1007/978-981-15-2144-7), §4.2.3.1.

| Trang in | Trang PDF người dùng | Nội dung kiểm chứng |
|---|---|---|
| 82 | 98 | Mở đầu Case Study 4; nhà máy LCD Shenzhen và thiết bị dịch vụ nhà máy. |
| 83–84 | 99–100 | FMCS/SCADA, giám sát/cảnh báo, vận hành theo kinh nghiệm; hạn chế của điều khiển phản ứng theo ngưỡng. |
| 85–86 | 101–102 | Mô hình cơ chế, dự báo nhu cầu, tối ưu quyết định, theo dõi sức khỏe thiết bị. |
| 87 | 103 | Luồng nền tảng từ dữ liệu/mô hình đến ứng dụng vận hành. |
| 88 | 104 | Tính toán lợi ích từ dữ liệu lịch sử 2018: 10 máy nén khí 1500 HP tiết kiệm hơn 300.000 USD/năm;12 “ice compressors” 136 cold tons hơn 70.000 USD/năm. Số ghi là 136, không phải 1360. |

Báo cáo dùng “chiller” để gọi nhóm thiết bị nước lạnh; khi cần đối chiếu nguyên văn có thể kèm tên “ice compressors” ở trang 88. Các số tiền là tính toán sách cung cấp; nhóm chưa kiểm toán độc lập và chưa đo ROI GoEmotions. Giữ nguyên điều kiện này khi rút gọn nội dung vào slide hoặc Notion.

Quan hệ áp dụng sang đồ án: văn bản → scores 28 nhãn → ngưỡng → gợi ý phân loại/ưu tiên → người xử lý → chỉ số nghiệp vụ. Đo thời gian xử lý, phản hồi bỏ sót, gợi ý sai và chi phí mới có thể đánh giá giá trị doanh nghiệp. Tăng F1 trên Reddit không tự chứng minh tiết kiệm năng lượng hoặc lợi nhuận.

## 5. Các điểm sửa và điều kiện khi xuất báo cáo cuối

| Mã | Mức | Phát hiện | Trạng thái/hành động |
|---|---|---|---|
| F01 | P2 | Bản sáu chương từng nói hòa mean/std thì chọn theo “chi phí suy luận”, trong khi code chọn ít tham số rồi tên kiến trúc. | **Đã sửa trong nguồn:** mục 4.2.3 ghi mean Macro-F1 val@0,5 → std nhỏ → ít tham số → tên. Cần mang sửa đổi này vào DOCX/PDF khi xuất lại. |
| F02 | P2 | Danh mục từng có nguồn PyTorch 2.14, trong khi run C ghi PyTorch 2.13.0+cu130. | **Đã sửa trong nguồn:** đã tìm/đọc và cập nhật references 15/23 về đúng tag v2.13.0. Cần xuất lại DOCX/PDF từ nguồn mới; không cần đổi môi trường/train vì sửa citation. |
| F03 | Điều kiện bản cuối | Hai báo cáo đọc tại thời điểm kiểm tra có snapshot tiến độ khác nhau; bài IEEE ghi 14:09 UTC+7/C full 2/9, bản sáu chương được xuất ở trạng thái trước đó. | Có thể chấp nhận trong bản nháp đã ghi trạng thái. Trước bản nộp, tổng hợp artifacts rồi xuất cả hai báo cáo từ cùng summary/protocol; không viết đủ 3 seed khi bảng chưa có đủ 3 run full. |
| F04 | Giới hạn kiểm chứng | Link IEEE Reference Guide ở lượt kiểm tra này chuyển hướng; chưa xác nhận lại edition 29/11/2023 bằng chính tài nguyên đó. | **Đã sửa metadata:** bỏ mốc edition cụ thể, đặt year=null trong JSON; giữ official URL/ngày truy cập 08/10/2026. Không tự nhận đã đọc lại toàn văn link này trong lượt audit. |

Nguồn 24 (Hugging Face text classification) và 25 (Gradio Quickstart) hỗ trợ triển khai chung. Audit này không chạy lại tutorial/UI để xác nhận phiên bản API; kiểm thử code, khởi động demo và artifacts có hồ sơ riêng. Template IEEE download được hồ sơ xuất ghi nhận trả HTTP 202/rỗng; không tự nhận đã đọc nội dung DOCX tải không thành công.

## 6. Tài liệu tham khảo và kết quả cần đọc đúng

- Số [1]–[26] của báo cáo sáu chương và [1]–[20] của bài viết IEEE là hai hệ đánh số cục bộ. Bài viết đã rút các nguồn được dùng và đánh lại theo lần xuất hiện; không yêu cầu Lee phải giữ số [26] trong cả hai.
- Năm 2019/2020 là năm công bố của các bài nền tảng. “Accessed: Oct. 8, 2026” của tài liệu/API là ngày truy cập, không làm các tài liệu đó thành bài báo nghiên cứu 2026.
- Class weighting, ngưỡng riêng và phân tích nhãn hiếm là các nội dung đồ án cần báo bằng phép so sánh đã chạy. Một citation về phương pháp không thay được bằng chứng cải thiện F1 của nhóm.
- Thay learning rate/epoch giữa BERT và RoBERTa/DistilBERT phải tiếp tục được công khai: đây là so sánh các hệ thống với cấu hình đã khai báo, chưa tách riêng ảnh hưởng kiến trúc trong ablation hoàn toàn đồng nhất.
- Ví dụ lỗi/ngữ nghĩa là phần phân tích; FP/FN tự động chỉ tạo ứng viên để đọc. Không chuyển ví dụ A thành bằng chứng lỗi cả ba C khi chưa có dự đoán cùng ID.
- Sau mỗi lần tổng hợp kết quả mới, kiểm lại split, threshold, số seed và thời điểm snapshot trong abstract, bảng, kết luận, DOCX/PDF và nội dung Notion. Các nguồn nền tảng đúng không làm một bảng kết quả chưa đủ run trở thành benchmark cuối.

**Phạm vi xác nhận cuối:** đã kiểm chứng các nhận định và nguồn được liệt kê ở bảng; chưa xác nhận mọi thí nghiệm hoàn tất, chưa kiểm toán lợi ích nhà máy và chưa xác nhận ROI của ứng dụng NLP.
