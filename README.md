# GoEmotions: Phân loại cảm xúc đa nhãn trên văn bản mạng xã hội

## Kế hoạch và phân công nhóm

Xem [kế hoạch đầy đủ theo công việc và sản phẩm](docs/KE_HOACH_NHOM.md) hoặc
[trang Notion của nhóm](https://app.notion.com/p/3ed7c27769028185af2dfbaac4c4586b).

| Người | Phần phụ trách |
|---|---|
| Duy | Baseline A; điều phối zero-shot B làm chung; data/metrics và bảng nâng cao/lỗi |
| Đức Trí — Thợ Săn Thập Cẩm | RoBERTa C2; bàn giao/hỗ trợ B đã nhận trước |
| Quốc Khánh | BERT C1, phần đầu báo cáo và script fine-tune chung |
| Nhật Huy | DistilBERT C3, demo của mô hình C tốt nhất |

**Cập nhật nhóm 4 người:** Duy phụ trách A và điều phối B làm chung;
Khánh/Trí/Huy mỗi người làm trọn BERT/RoBERTa/DistilBERT, mỗi kiến trúc ≥3 seed.
Đức Trí đã nhận zero-shot trong trao đổi trước, bàn giao phần đã làm nếu có.
Quốc Khánh giữ phần đầu báo cáo. Xem mục 13 của kế hoạch để đọc 25 nguồn NLP.

<details>
<summary>Phần của Duy — baseline, tài liệu và báo cáo</summary>

[Mục riêng của Duy trên Notion](https://app.notion.com/p/3ee7c277690281e693c7f1cf985d569d)
và [thứ tự đọc lưu trong repo](docs/THU_TU_DOC_BASELINE.md).

1. [Notebook baseline](notebooks/baseline.ipynb).
2. [Hướng dẫn chạy và giải thích code](docs/BASELINE.md).
3. [Bảng kết quả và phân tích lỗi](reports/BASELINE_RESULTS.md).
4. [Hồ sơ đối chiếu yêu cầu cô](docs/BASELINE_REVIEW.md).

[Báo cáo baseline của Duy](reports/BAO_CAO_BASELINE_BAO_DUY.md) ·
[CSV sáu cấu hình validation](reports/baseline_validation/comparison.csv) ·
[PR #3 vào repo chung](https://github.com/trangkhanh-ai/goemotions-multilabel-classification/pull/3).
Chưa chạy test thật; bảng hiện tại là validation.

</details>

## Thống kê dữ liệu đã khám phá — đã hoàn thành

Phần EDA đã chạy trên **54.263 mẫu GoEmotions simplified**, giữ nguyên train/validation/test
43.410/5.426/5.427 mẫu. Các sản phẩm đáp ứng ba nội dung yêu cầu:

| Nội dung yêu cầu | Kết quả và tệp xem trực tiếp |
|---|---|
| Số mẫu | [Số mẫu và tỷ lệ từng split](reports/tables/split_sizes.csv) |
| Phân bố nhãn/lớp | [Số mẫu dương và tỷ lệ của đủ 28 nhãn](reports/tables/label_distribution.csv), gồm từng split và tổng |
| Ví dụ mẫu dữ liệu | [30 ví dụ nguyên văn, có ID, nhãn và nhận xét](reports/tables/examples_30.csv), phủ đủ 28 nhãn |

- [Notebook giải thích từng bước bằng tiếng Việt, có output](notebooks/eda.ipynb).
- [Báo cáo thống kê chi tiết và biểu đồ](reports/THONG_KE_DU_LIEU.md).
- [Hướng dẫn cài đặt và chạy lại EDA](docs/EDA.md).
- [Bản HTML đã chạy](reports/eda.html): tải về và mở bằng trình duyệt.

Đã đối chiếu số mẫu, support/tỷ lệ từng nhãn và 30 ví dụ với dữ liệu nguồn.
Với bài toán đa nhãn, một bình luận được tính vào nhiều nhãn nên tổng support có thể
lớn hơn số mẫu. Trên nhánh triển khai baseline, đã có mã A và số đo validation đầu tiên;
zero-shot và ba mô hình fine-tune vẫn là kế hoạch của đồ án.

## Bài toán và bài báo nền tảng

GoEmotions được giới thiệu trong bài báo của Demszky và cộng sự tại ACL 2020. Bài báo xây dựng dữ liệu bình luận Reddit với 27 cảm xúc và neutral, đồng thời nghiên cứu mô hình dựa trên BERT.

Hai đặc điểm của bài toán cần phân biệt:

- **Fine-grained:** phân biệt nhiều loại cảm xúc chi tiết, thay vì chỉ tích cực/tiêu cực.
- **Multi-label:** một bình luận có thể mang nhiều nhãn cùng lúc.

Đây là hai khái niệm khác nhau. Đề tài này có cả hai đặc điểm. Các thí nghiệm TF-IDF + Logistic Regression và zero-shot BART-MNLI là thiết kế của đồ án; không mô tả chúng như toàn bộ phương pháp của bài báo gốc.

## Dữ liệu sử dụng

Nguồn GoEmotions ban đầu có 58.009 bình luận. Đồ án sử dụng phiên bản `simplified` đã lọc theo mức đồng thuận của người gán nhãn, với 54.263 mẫu:

| Split | Số mẫu | Vai trò |
|---|---:|---|
| Train | 43.410 | Fit TF-IDF; huấn luyện mô hình có giám sát; tính trọng số nhãn |
| Validation | 5.426 | Chọn cấu hình, checkpoint và ngưỡng theo thiết kế đã công bố |
| Test | 5.427 | Đánh giá cuối sau khi đã khóa các lựa chọn |

Giữ split chính thức và thứ tự 28 nhãn từ metadata. Biểu diễn nhãn bằng vector multi-hot: mỗi vị trí là 0 hoặc 1; một hàng có thể có nhiều số 1. Không thay nhãn gốc và không tự ép neutral loại trừ các nhãn khác.

Dataset revision trong mã EDA hiện tại: `add492243ff905527e67aeb8b80c082af02207c3`.

## Trạng thái triển khai

Repo hiện có mã và sản phẩm EDA. Nhánh baseline đã chạy TF-IDF + One-vs-Rest
Logistic Regression ở hai cấu hình: chuẩn và cân bằng lớp. Xem [hướng dẫn từng bước](docs/BASELINE.md)
và [bảng kết quả validation thật](reports/BASELINE_RESULTS.md). Cấu hình chuẩn ở ngưỡng
0,5 có Macro-F1 **0,2025**, Micro-F1 **0,3760**; cấu hình cân bằng lớp có Macro-F1
**0,4562**, Micro-F1 **0,5099**. Test chưa được đánh giá; B và C chưa có kết quả trong repo.
Có [notebook baseline với output đã chạy](notebooks/baseline.ipynb) để học từng bước.
Xem [hồ sơ rà soát phần A ngày 01/10](docs/BASELINE_REVIEW.md) để biết các lỗi đã sửa,
bằng chứng kiểm chứng và phần việc còn cần cả nhóm hoàn thành.
Phần A còn có so sánh ngưỡng chung/ngưỡng riêng, bảng cặp FN/FP theo ID và protocol
khóa sáu cấu hình trước khi đánh giá test.

- [Notebook EDA](notebooks/eda.ipynb)
- [Báo cáo EDA](reports/THONG_KE_DU_LIEU.md)
- [Số mẫu](reports/tables/split_sizes.csv)
- [Phân bố nhãn](reports/tables/label_distribution.csv)
- [Ví dụ dữ liệu](reports/tables/examples_30.csv)
- [Cách chạy EDA](docs/EDA.md)

## Ba hướng tiếp cận đã thống nhất

| Hướng | Hệ thống dự kiến | Học thêm từ nhãn GoEmotions? |
|---|---|---|
| A — Baseline | TF-IDF + One-vs-Rest Logistic Regression | Có, trên train |
| B — Zero-shot | BART-large-MNLI | Không cập nhật trọng số |
| C — Fine-tuning | BERT-base, RoBERTa-base, DistilBERT-base | Có, mỗi mô hình trên train |

TF-IDF là cách biểu diễn văn bản cho A, không phải một mô hình phân loại riêng. Zero-shot nghĩa là không fine-tune mô hình trên GoEmotions; BART-MNLI đã được huấn luyện trước trên MNLI.


Ba hướng tiếp cận cùng giải quyết một bài toán. Mỗi hướng tự tạo dự đoán từ văn bản; đầu ra của A không phải đầu vào bắt buộc của B hoặc các mô hình fine-tune. Nhóm dự kiến so sánh tổng cộng **năm hệ thống**: một baseline, một zero-shot và ba mô hình fine-tune.

```text
GoEmotions: split gốc + 28 nhãn + quy tắc đánh giá chung
    ├── A: TF-IDF + One-vs-Rest Logistic Regression
    ├── B: zero-shot facebook/bart-large-mnli
    └── C: fine-tune BERT-base / RoBERTa-base / DistilBERT-base
                          ↓
            So sánh trên cùng tập test và phân tích lỗi
```

Danh sách checkpoint dự kiến: `google-bert/bert-base-uncased`, `FacebookAI/roberta-base`, `distilbert/distilbert-base-uncased`. Ba mô hình này thuộc **cùng hướng fine-tuning**, nhưng được huấn luyện và đánh giá riêng.

### A — TF-IDF + One-vs-Rest Logistic Regression

TF-IDF chuyển văn bản thành vector đặc trưng từ/cặp từ. One-vs-Rest huấn luyện một Logistic Regression nhị phân cho mỗi nhãn: tổng cộng 28 bộ phân loại.

```text
Huấn luyện: text train → fit TF-IDF → X_train
                                      + Y_train multi-hot
                                      → fit 28 Logistic Regression

Dự đoán: text → transform bằng TF-IDF đã fit
             → 28 điểm dự đoán → ngưỡng từng nhãn → tập nhãn
```

Chỉ fit từ vựng/IDF và mô hình trên train. Khởi đầu có thể dùng ngưỡng 0,5; nếu điều chỉnh, chọn bằng validation và ghi rõ. TF-IDF là bộ biểu diễn đặc trưng, không phải mô hình tự phân loại cảm xúc.

### B — Zero-shot BART-large-MNLI

Checkpoint `facebook/bart-large-mnli` đã được huấn luyện trên tác vụ suy luận ngôn ngữ tự nhiên MNLI. Đồ án sử dụng checkpoint đó mà không huấn luyện thêm trọng số trên GoEmotions.

```text
Text + 28 nhãn ứng viên + hypothesis template
    → tạo cặp text/giả thuyết cho từng nhãn
    → BART-MNLI → 28 score → ngưỡng → tập nhãn
```

Nhãn ứng viên là đầu vào do nhóm cung cấp, không phải kết quả mô hình sinh ra. Bật `multi_label=True` để đánh giá độc lập từng nhãn. Score NLI không phải xác suất cảm xúc đã được hiệu chuẩn. Nhãn neutral của tác vụ MNLI và nhãn neutral của GoEmotions có ý nghĩa khác nhau.

Ví dụ suy luận một câu — không phải mã đánh giá benchmark:

```python
import json
from pathlib import Path
from transformers import pipeline

# Chạy từ thư mục gốc repo; file này đã có trong repo.
labels = json.loads(Path("data/labels.json").read_text(encoding="utf-8"))
assert len(labels) == 28

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)
result = classifier(
    "I am really grateful for everything you have done.",
    candidate_labels=labels,
    hypothesis_template="This text expresses {}.",
    multi_label=True,
)
# 0,5 là ngưỡng khởi đầu cho ví dụ, chưa phải ngưỡng đã tối ưu.
predicted = [label for label, score in zip(result["labels"], result["scores"])
             if score >= 0.5]
print(predicted)
```

Khi đánh giá, ánh xạ score về thứ tự nhãn chuẩn vì pipeline trả nhãn theo score giảm dần. Báo cáo B ban đầu có thể dùng ngưỡng cố định. Nếu dùng nhãn validation để chỉnh template/ngưỡng, phải mô tả là zero-shot về trọng số, có hiệu chỉnh bằng dữ liệu đích, và tách khỏi thiết lập không sử dụng nhãn đích.

### C — Fine-tune BERT-base, RoBERTa-base và DistilBERT-base

Ba checkpoint dự kiến được nêu ở trên. Với **mỗi** mô hình, thêm đầu phân loại có 28 đầu ra và huấn luyện riêng trên GoEmotions train; trong thiết lập fine-tune toàn bộ, cập nhật cả encoder và đầu phân loại. Sơ đồ dưới đây dùng BERT để minh họa; RoBERTa và DistilBERT có luồng tương tự nhưng dùng tokenizer/encoder tương ứng.

```text
Text → tokenizer BERT → encoder BERT → đầu phân loại → 28 logits
Huấn luyện: logits + nhãn multi-hot → BCEWithLogitsLoss → cập nhật trọng số
Suy luận: logits → sigmoid từng nhãn → ngưỡng từng nhãn → tập nhãn
```

Không dùng softmax chung 28 nhãn để ép chọn duy nhất một cảm xúc. `BCEWithLogitsLoss` nhận logits trực tiếp; không áp sigmoid trước loss này. Sigmoid được dùng khi chuyển logits thành điểm để dự đoán.

## Quy trình đánh giá chung

1. Thống nhất split, thứ tự nhãn và ID mẫu; giữ cùng văn bản nguồn, mỗi phương pháp có bộ biểu diễn/tokenizer phù hợp.
2. Huấn luyện A/C trên train; B giữ nguyên trọng số.
3. Với nhánh sử dụng validation, chọn cấu hình/checkpoint/ngưỡng trên validation; công bố rõ việc dùng nhãn validation cho B nếu có.
4. Khóa cấu hình rồi đo trên cùng test. Không dùng test để chọn mô hình hoặc ngưỡng.
5. Báo cáo Micro-F1, Macro-F1, Precision/Recall có ghi cách lấy trung bình, Hamming Loss và P/R/F1/support từng nhãn. Macro-F1 tính trên đủ 28 nhãn, thống nhất cách xử lý mẫu số bằng 0.
6. Ghi checkpoint/revision, seed, phần cứng, thư viện, thời gian fit và suy luận. Với nhiều seed fine-tune, báo cáo kết quả từng seed và trung bình/độ lệch chuẩn.

Không giả định trước zero-shot tốt hơn baseline hoặc fine-tuning tốt nhất. Thứ hạng phải dựa trên số liệu thực nghiệm.

## Phần nâng cao và nhãn hiếm

Nhánh A đã chạy class weighting và khảo sát ngưỡng riêng từng nhãn trên validation.
Kế hoạch nhóm đề xuất thử tiếp cho C tốt nhất. PDF đề tài không quy định nâng cao
phải áp dụng riêng cho C; yêu cầu là có thí nghiệm và số cải thiện F1 nhãn hiếm:

- A: đã thử trọng số lớp trong từng Logistic Regression nhị phân; xem số thật trong báo cáo A.
- C: thử `pos_weight` trong BCEWithLogitsLoss; tính từ train, chọn cách xử lý trọng số quá lớn bằng validation nếu cần.
- Chọn ngưỡng từng nhãn từ validation của chính mô hình đó. Không chuyển nguyên ngưỡng A sang B/C.
- Xác định tập nhãn hiếm từ train và công bố tiêu chí trước khi so sánh.
- So sánh thiết lập gốc, chỉ weighting, chỉ tuning và kết hợp; báo cáo F1/support của tất cả nhãn hiếm đã xác định, kể cả nhãn giảm điểm.

Trên validation, A cân bằng lớp cải thiện F1 của năm nhãn hiếm xác định từ train; kết quả
test chưa có. Điểm sau khi chọn ngưỡng trên cùng validation có thể lạc quan. Contrastive
representation là hướng khác được đề bài nêu; nhóm không cần thực hiện đồng thời cả ba hướng.

## Phân tích lỗi

Phân biệt **đồng xuất hiện** (nhãn thật cùng có trong một mẫu) với **nhầm lẫn** (nhãn dự đoán thừa/bỏ sót so với ground truth). Các cặp như anger/annoyance chỉ nên ghi là ví dụ cần khảo sát trước khi có minh chứng.

Dùng dự đoán theo ID, thống kê lỗi và ví dụ nguyên văn để chứng minh; tránh gán một cặp là thường xuyên nhầm chỉ dựa trên trực giác. Trong bài toán đa nhãn, confusion matrix dạng single-label không đủ mô tả toàn bộ lỗi.

## Kết quả thực nghiệm

| Hệ thống | Thiết lập dự kiến | Trạng thái |
|---|---|---|
| A | TF-IDF + One-vs-Rest Logistic Regression | Đã có hai biến thể và bảng validation trong `reports/BASELINE_RESULTS.md`; chưa đo test |
| B | BART-large-MNLI, multi_label=True | Dự kiến, chưa có số đo trong repo |
| C1 | BERT-base-uncased, 28 đầu ra | Dự kiến, chưa có số đo trong repo |
| C2 | RoBERTa-base, 28 đầu ra | Dự kiến, chưa có số đo trong repo |
| C3 | DistilBERT-base-uncased, 28 đầu ra | Dự kiến, chưa có số đo trong repo |

Khi B/C có kết quả, bổ sung số cùng split/metric và link log chạy thật; không điền số giả định.

## Môi trường và cài đặt

Mã EDA được repo ghi nhận đã kiểm tra trên Python 3.12.6. Xem `docs/EDA.md` để chạy lại.
Baseline A đã kiểm tra trên Python 3.13.9 với `requirements-baseline.txt`; xem `docs/BASELINE.md`.
`requirements.txt` hiện phục vụ EDA, chưa đủ để chạy A/B/C.

Nhóm cần bổ sung môi trường mô hình riêng, ghi phiên bản đã kiểm tra và lựa chọn PyTorch phù hợp thiết bị. Không dùng lệnh placeholder `python=3.x` như một lệnh cài đặt hoàn chỉnh. Ví dụ ở phần B cần môi trường đã cài transformers và PyTorch.

## Tài liệu tham khảo

- [Bài báo GoEmotions, ACL 2020](https://aclanthology.org/2020.acl-main.372/)
- [GoEmotions của Google Research](https://github.com/google-research/google-research/tree/master/goemotions)
- [OneVsRestClassifier và nhãn multi-hot](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
- [TF-IDF trong scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [BART-large-MNLI model card](https://huggingface.co/facebook/bart-large-mnli)
- [Zero-shot pipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline)
- [BERT-base-uncased model card](https://huggingface.co/google-bert/bert-base-uncased)
- [RoBERTa-base model card](https://huggingface.co/FacebookAI/roberta-base)
- [DistilBERT-base-uncased model card](https://huggingface.co/distilbert/distilbert-base-uncased)
- [BCEWithLogitsLoss trong mã nguồn PyTorch](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py)

Trích dẫn bài báo nền tảng: Demszky, D., Movshovitz-Attias, D., Ko, J.,
Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained
Emotions*. Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics, 4040–4054. https://doi.org/10.18653/v1/2020.acl-main.372

Giấy phép cho mã của repo chưa được nhóm công bố; xem giấy phép tại nguồn cho dữ liệu
GoEmotions và các checkpoint sử dụng.
