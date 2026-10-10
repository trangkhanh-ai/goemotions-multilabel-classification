# Duy giải thích baseline A với cô

Đối chiếu mã và bảng validation ngày **08/10/2026**. Phần này bổ sung cách tính
tay, giải thích code và trả lời bảo vệ. Cách cài/chạy, bảng sáu cấu hình và hồ sơ
bàn giao đã có ở [BASELINE.md](BASELINE.md), [thứ tự đọc](THU_TU_DOC_BASELINE.md)
và [bảng kết quả](../reports/BASELINE_RESULTS.md); không cần tạo mô hình mới để học.

## 1. Bài toán: một câu có thể có nhiều nhãn

**Đầu vào:** một bình luận tiếng Anh. **Đầu ra trước ngưỡng:** 28 điểm, cùng thứ
tự trong `data/labels.json`. **Đầu ra sau ngưỡng:** một tập nhãn được chọn trong
27 cảm xúc và `neutral`.

Ví dụ minh họa với **ba cột giả định** `[joy, gratitude, neutral]`: câu
“Thank you! I am happy.” có nhãn `[1, 1, 0]`. Đây là vector **multi-hot** vì có
hai số 1; dữ liệu thật dùng 28 cột và nhãn do người gán, không tự gán bằng từ khóa.
`Y_train` có kích thước **43.410 × 28**; scores validation là **5.426 × 28**.

`neutral` là nhãn thật của bộ dữ liệu. Nếu cả 28 điểm đều dưới ngưỡng, kết quả
là tập nhãn rỗng; mã không tự gán neutral, không ép lấy nhãn điểm cao nhất và
không tự thay đổi các tổ hợp nhãn gốc.

**Câu giải thích ngắn:** “Em giải 28 câu hỏi có/không cho cùng một bình luận,
nên câu có thể được nhận nhiều cảm xúc.”

## 2. TF-IDF: tính một ví dụ đúng cấu hình sklearn

TF-IDF đổi câu thành vector số. Nó chưa tự quyết định câu có cảm xúc gì.
Mã dùng từ đơn và cặp từ liền nhau: `ngram_range=(1, 2)`, `min_df=2`,
`max_features=100000`. Các mặc định đang dùng gồm `lowercase=True`,
`smooth_idf=True`, `sublinear_tf=False`, `norm='l2'`.

### Một tập train nhỏ để tính tay

Các câu sau chỉ minh họa toán, **không phải mẫu/kết quả GoEmotions**:

| Câu | Văn bản |
|---|---|
| D1 | `happy happy thank` |
| D2 | `happy thank` |
| D3 | `sad thank` |
| D4 | `sad thank` |

`df` đếm **số văn bản** có đặc trưng. D1 có hai lần `happy` nhưng chỉ góp một
vào `df(happy)`. `happy happy` chỉ có trong D1, nên bị `min_df=2` loại.

Với mặc định trên, trước bước chuẩn hóa:

```text
tf(t,d)  = số lần đặc trưng t xuất hiện trong câu d
idf(t)   = ln((1 + số câu train) / (1 + df(t))) + 1
raw(t,d) = tf(t,d) × idf(t)
vector cuối = raw / sqrt(tổng bình phương các phần tử raw)
```

Đây là **TF theo số đếm**, không phải số đếm chia độ dài câu. Sau đó vector
được chuẩn hóa L2; dùng một công thức TF-IDF khác sẽ cho số minh họa khác mã hiện tại.
Nguồn công thức/mặc định: [TfidfTransformer, sklearn 1.7.2](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).

| Đặc trưng còn lại | df | idf | tf trong D1 | raw trong D1 | Sau L2 |
|---|---:|---:|---:|---:|---:|
| `happy` | 2 | 1,5108 | 2 | 3,0217 | 0,8576 |
| `happy thank` | 2 | 1,5108 | 1 | 1,5108 | 0,4288 |
| `sad` | 2 | 1,5108 | 0 | 0 | 0 |
| `sad thank` | 2 | 1,5108 | 0 | 0 | 0 |
| `thank` | 4 | 1,0000 | 1 | 1,0000 | 0,2838 |

Norm của vector raw D1 là khoảng **3,5232**. `thank` xuất hiện trong mọi câu
nên idf thấp hơn `happy`; `happy` xuất hiện hai lần trong D1 nên tf cao hơn.
Trong train thật, lần fit đã lưu **58.338 đặc trưng**. 100.000 là giới hạn trên,
không phải số đặc trưng bắt buộc hoặc số token tối đa của một bình luận.

### Khi gặp câu mới

Vectorizer dùng từ vựng và IDF **đã học từ train**, không tính lại trên câu mới
hay validation/test. Từ ngoài từ vựng bị bỏ qua. Nếu vector toàn 0, Logistic
Regression vẫn có thể cho điểm nhờ bias; không kết luận câu đó tự động neutral.

Bigram có thể giữ một phần thứ tự như `not happy`. Mặc định token pattern chỉ
giữ token gồm ít nhất hai ký tự dạng từ; dấu câu/emoji không thành đặc trưng.
Ví dụ `don't` có thể thành `don` và bỏ token `t`. Vì vậy mô hình vẫn có hạn chế
với phủ định, mỉa mai và nghĩa cần ngữ cảnh dài.

## 3. One-vs-Rest và Logistic Regression hoạt động thế nào?

Với mỗi nhãn `j`, mô hình học một vector trọng số `w_j` và bias `b_j`:

```text
z_j = w_j · x + b_j
p_j = sigmoid(z_j) = 1 / (1 + exp(-z_j))
chọn nhãn j khi p_j >= ngưỡng_j
```

`x` là TF-IDF của câu. `z` là điểm chưa sigmoid; `p` là điểm nằm trong [0,1].
Ví dụ minh họa `z_joy=2`, `z_gratitude=1`, `z_neutral=-1` cho các điểm tương
ứng khoảng **0,8808; 0,7311; 0,2689**. Ngưỡng 0,5 chọn joy và gratitude.
Đây là số để hiểu sigmoid, **không phải dự đoán của checkpoint đã chạy**.

**28 mô hình nhị phân** được học riêng từ 28 cột Y. Tổng 28 điểm không cần bằng
1; softmax chung hoặc argmax sẽ làm mất khả năng chọn nhiều nhãn. OvR không
mô hình hóa trực tiếp phụ thuộc giữa các nhãn; các bộ phân loại cùng dùng đặc
trưng văn bản. Điểm `predict_proba` chưa được hiệu chuẩn riêng trong đồ án,
nhất là sau weighting, nên không nói “chắc chắn 88%” chỉ vì score là 0,88.
Nguồn OvR đa nhãn: [OneVsRestClassifier, sklearn 1.7.2](https://scikit-learn.org/1.7/modules/generated/sklearn.multiclass.OneVsRestClassifier.html).

Trong A, sklearn tối ưu logistic loss có regularization. A không dùng tokenizer
BERT, encoder Transformer hay loss PyTorch; `BCEWithLogitsLoss` thuộc phần C.

## 4. Giải thích các dòng code thật

### Hàm dựng mô hình

Trích [scripts/run_baseline.py](../scripts/run_baseline.py), không cần chạy lại:

```python
def build_model(variant="standard"):
    """Hai bước dễ đọc: biểu diễn TF-IDF -> 28 Logistic Regression nhị phân."""
    if variant not in ("standard", "balanced"):
        raise ValueError("variant phải là standard hoặc balanced")
    class_weight = "balanced" if variant == "balanced" else None
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(C=1.0, solver="liblinear", max_iter=1000,
                               class_weight=class_weight, random_state=42), n_jobs=1
        )),
    ])
```

| Dòng/thành phần | Duy giải thích |
|---|---|
| `def build_model(...)` | Hàm trả một Pipeline chưa fit; mặc định chọn mốc standard. |
| Chuỗi ba dấu nháy | Mô tả hàm, không huấn luyện hoặc thay kết quả. |
| `if variant not in ...` | Kiểm tên biến thể hợp lệ. |
| `raise ValueError(...)` | Báo lỗi rõ khi gõ nhầm; không âm thầm đổi mô hình. |
| `class_weight = ...` | Balanced có trọng số lớp; standard dùng None. |
| `return Pipeline([...])` | Ghép hai bước theo thứ tự: TF-IDF rồi classifier. |
| `("tfidf", TfidfVectorizer(...))` | Tên bước và bộ chuyển văn bản thành vector; tham số đã tính ở mục 2. |
| `("classifier", OneVsRestClassifier(...))` | Nhận Y đa nhãn để dựng một bộ phân loại cho mỗi cột. |
| `LogisticRegression(C=1.0, ...)` | Mô hình tuyến tính có regularization; C càng lớn thì regularization càng yếu. |
| `solver="liblinear"` | Bộ tối ưu phù hợp các bài toán nhị phân của OvR. |
| `max_iter=1000` | Giới hạn vòng tối ưu, không phải 1.000 epoch Transformer; thực tế có thể hội tụ sớm. |
| `class_weight=class_weight` | Truyền lựa chọn standard/balanced vào từng bài toán nhị phân. |
| `random_state=42` | Ghi seed của bộ tối ưu để hỗ trợ tái lập; không bảo đảm giống bit ở mọi môi trường. |
| `n_jobs=1` | OvR xử lý tuần tự, giúp hạn chế nhiều tác vụ chiếm RAM cùng lúc. |
| Các dấu `)`, `]` đóng cuối | Đóng constructor/classifier/list/Pipeline; không thêm một bước học nào khác. |

### Dữ liệu, fit và dự đoán

Các dòng sau cũng nằm trong cùng script; `train`, `validation`, `labels` đã
được đọc/kiểm trước đó. Đây là trích đoạn để đọc, không phải script mới:

```python
y_train = multi_hot(train["labels"].tolist(), len(labels))
y_validation = multi_hot(validation["labels"].tolist(), len(labels))
model = build_model(args.variant)
model.fit(train["text"].tolist(), y_train)
scores = model.predict_proba(validation["text"].tolist())
metrics = evaluate_multilabel(y_validation, scores, labels, threshold=0.5)
```

| Dòng | Giải thích |
|---|---|
| `y_train = multi_hot(...)` | Đổi danh sách ID nhãn thành ma trận train N×28; hàm kiểm nhãn rỗng/lặp/ngoài miền. |
| `y_validation = multi_hot(...)` | Chuẩn bị ground truth validation để đo; không truyền nó vào `fit`. |
| `model = build_model(...)` | Dựng đúng standard/balanced, chưa học dữ liệu. |
| `model.fit(...)` | Pipeline fit từ vựng/IDF trên train, transform train rồi học 28 LR bằng Y_train. |
| `scores = model.predict_proba(...)` | Transform val bằng TF-IDF đã fit, sau đó trả 28 điểm từng câu; không fit lại. |
| `metrics = evaluate_multilabel(...)` | Áp ngưỡng 0,5 rồi tính chỉ số. Hàm này đo, không tự tìm ngưỡng. |

Code còn kiểm mapping, ID trùng/giao nhau, văn bản rỗng, đủ mẫu dương/âm mỗi
nhãn, kích thước scores và hội tụ. Nếu có `ConvergenceWarning`, script dừng;
không dùng số từ mô hình chưa hội tụ. Model/scores lưu kèm revision/hash và
thời gian; hash giúp phát hiện file thay đổi, không tự chứng minh chất lượng cao.

## 5. Train, validation và test có ba vai trò

| Split | Số mẫu | Được làm gì? |
|---|---:|---|
| Train | 43.410 | Fit từ vựng/IDF, học trọng số LR, tính class weighting và chọn tập nhãn hiếm. |
| Validation | 5.426 | Đo baseline gốc, chọn ngưỡng chung/riêng và chốt cấu hình. |
| Test | 5.427 | Đo cuối với các cấu hình/ngưỡng đã khóa. |

1. Chạy A standard và balanced chỉ mở train/validation.
2. `tune_global_threshold` thử **19 mức 0,05–0,95, bước 0,05**; chọn Macro-F1
   validation cao nhất cho một ngưỡng chung.
3. `tune_thresholds` dùng cùng lưới nhưng chọn F1 từng nhãn: kết quả là 28 ngưỡng.
   Hòa chọn mức gần 0,5; nếu vẫn hòa, mức lớn hơn. Không thay trọng số LR khi tune.
4. `freeze_baseline` giữ **sáu cấu hình = hai model × ba luật ngưỡng**, với hash
   và lựa chọn từ validation. `evaluate_baseline_test` kiểm protocol rồi đo test.

Sáu hàng kết quả không có nghĩa đã train sáu mô hình. A có **hai Pipeline đã fit**;
mỗi Pipeline được đo theo fixed/global/tuned. Điểm tuned trên chính validation
dùng tìm ngưỡng thường lạc quan; kết luận cải tiến cuối phải xem test sau khóa.
Không chuyển ngưỡng của standard sang balanced, hoặc từ A sang B/C.

**Giải thích đánh đổi bằng số test, cập nhật 10/10/2026.** Với A balanced,
tuning làm Micro-Precision tăng 0,4043 → 0,4561, Hamming Loss giảm
0,0547 → 0,0467 nhưng Micro-Recall giảm 0,6631 → 0,6260. Vì vậy không học
thuộc câu “tuning luôn tăng Recall và tăng nhãn thừa”. Chiều thay đổi phụ thuộc
ngưỡng từng nhãn và cấu hình; giải thích bằng bảng đã đo. So standard fixed với
balanced tuned là thay cả weighting và ngưỡng, không cô lập một kỹ thuật.
Đọc [ba nội dung cập nhật trong báo cáo](../reports/CAP_NHAT_NOI_DUNG_10_10_2026.md)
để phân biệt giá trị ứng dụng dự kiến, validation/test và ba seed với tìm kiếm
siêu tham số.

EDA của nhóm đã xem các split. Câu đúng khi bảo vệ là **“không dùng test để
fit/chọn mô hình/ngưỡng”**, không nói chưa từng nhìn bất kỳ dữ liệu test nào.
Trạng thái test hiện tại xem [bảng tổng hợp thực nghiệm](../reports/project_results/RESULTS.md).

## 6. Macro/Micro, nhãn hiếm và weighting

Với mỗi nhãn: TP = đoán có và thật có; FP = đoán có nhưng thật không; FN =
thật có nhưng đoán không. Precision = TP/(TP+FP), Recall = TP/(TP+FN),
F1 = 2TP/(2TP+FP+FN). Mã dùng `zero_division=0` khi mẫu số bằng 0.

Ví dụ giả định 12 câu, hai nhãn, để tính tay:

| Nhãn | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Nhãn phổ biến | 8 | 2 | 2 | 0,8000 |
| Nhãn ít mẫu hơn | 1 | 1 | 3 | 0,3333 |

- **Macro-F1:** (0,8 + 0,3333)/2 ≈ **0,5667**. Thực tế lấy trung bình đều
  **28 F1**, kể cả nhãn F1 bằng 0; không tính F1 lại từ macro Precision/Recall.
- **Micro-F1:** gom TP=9, FP=3, FN=5 → 18/(18+3+5) ≈ **0,6923**.
  Nhãn nhiều mẫu thường ảnh hưởng mạnh hơn.
- **Hamming Loss:** ví dụ có 8 quyết định nhãn sai trên 12×2 ô → **0,3333**.
  Thực tế mẫu số N×28. Vì nhiều ô là 0, Hamming thấp vẫn có thể bỏ sót nhãn hiếm.

### Cân bằng lớp làm thay đổi phần học

Với từng bộ LR nhị phân, `class_weight='balanced'` dùng:

```text
trọng số mẫu dương = N / (2 × số mẫu dương)
trọng số mẫu âm    = N / (2 × số mẫu âm)
```

Ví dụ giả định N=10, dương=2, âm=8: trọng số dương=2,5; âm=0,625. Lỗi ở lớp
ít mẫu bị phạt mạnh hơn. Đây là đổi objective khi **fit**, khác với đổi ngưỡng
sau khi đã có scores. Có thể tăng Recall và cả FP; không bảo đảm mọi nhãn đều tốt hơn.
Nguồn công thức balanced: [LogisticRegression, sklearn 1.7.2](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html).

Mã xác định **năm support thấp nhất của train**, không chọn theo nhãn nào tăng F1:
grief (77), pride (111), relief (153), nervousness (164), embarrassment (303).
Khi báo cáo phải giữ support, F1 trước/sau và cả nhãn giảm hoặc không cải thiện.

Hai hàng **validation @0,5** đã lưu trong
[comparison.csv](../reports/baseline_validation/comparison.csv):

| Variant | Macro-F1 | Micro-F1 | Micro-Precision | Micro-Recall | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| standard | 0,2025 | 0,3760 | 0,7254 | 0,2538 | 0,0354 |
| balanced | 0,4562 | 0,5099 | 0,4158 | 0,6592 | 0,0532 |

Diễn giải: balanced tìm được nhiều nhãn thật hơn, đồng thời đoán thừa nhiều
hơn nên Precision giảm và Hamming tăng ở cùng ngưỡng 0,5. Đây là số validation
của A; không gọi Macro-F1 0,4562 là “45,62% câu đúng” hay số test.

## 7. Đầu ra A liên hệ phần còn lại của đồ án

```text
Cùng text/split/ID/28 nhãn
  A: TF-IDF → 28 LR → scores_A
  B: text + candidate labels/template → BART-MNLI → scores_B
  C: tokenizer đúng checkpoint → encoder fine-tune → sigmoid → scores_C
        ↓
  Ngưỡng val riêng từng run → đo cùng test → bảng và phân tích lỗi
  C thắng theo mean validation của 3 seed → D dùng checkpoint C đó
```

A cung cấp **mốc so sánh** về chất lượng, chi phí và lỗi của phương pháp cổ
điển. A không cấp TF-IDF hoặc nhãn dự đoán cho BERT/RoBERTa/DistilBERT học.
B/C chạy trực tiếp trên văn bản; D hiện dùng C thắng, không tự fallback sang A.
`validation_scores.npz`/scores test lưu `ids`, `scores`, `label_names`; ghép
theo **ID và mapping**, không mặc định các file cùng thứ tự dòng.

Bài [GoEmotions ACL 2020](https://aclanthology.org/2020.acl-main.372/) là nguồn
dữ liệu, taxonomy và hướng BERT. A TF-IDF+LR là baseline cổ điển cô yêu cầu
nhóm xây, không mô tả đó là toàn bộ phương pháp bài gốc. Kết quả khác paper
cần xem split/nhãn/cấu hình/ngưỡng/cách đo trước khi diễn giải.

Yêu cầu nâng cao có các lựa chọn weighting **hoặc** ngưỡng riêng **hoặc**
contrastive; nhóm vẫn phải có số F1 nhãn hiếm trước/sau. Ba kiến trúc C cần
mỗi kiến trúc ≥3 seed và mean±sample std; ba luật ngưỡng A không thay ba seed C.

## 8. Câu hỏi bảo vệ và cách trả lời

| Cô hỏi | Trả lời dựa đúng đồ án |
|---|---|
| Baseline em làm là gì? | TF-IDF unigram/bigram và OvR 28 Logistic Regression. Có mốc standard và biến thể class weighting balanced. |
| Em tự viết Logistic Regression không? | Em dùng triển khai sklearn mã nguồn mở; phần nhóm làm là luồng dữ liệu đa nhãn, cấu hình, protocol, tuning, đánh giá và phân tích. |
| Vì sao không chỉ đoán tích cực/tiêu cực? | GoEmotions phân biệt 27 cảm xúc + neutral và cho phép nhiều nhãn; gộp ba sentiment sẽ đổi đề tài/ground truth. |
| Vì sao min_df=2? | Loại đặc trưng chỉ có trong một văn bản train để hạn chế đặc trưng quá ít xuất hiện. Đây là lựa chọn đã khai báo, chưa có ablation chứng minh luôn tối ưu. |
| Vì sao dùng bigram? | Có thêm đặc trưng cặp từ như not happy; chi phí/đặc trưng tăng và vẫn không giải hết ngữ cảnh dài. |
| BERT có dùng TF-IDF của em không? | Không. C dùng tokenizer/encoder của checkpoint và nhãn train. A là đối chứng trên cùng dữ liệu/chỉ số. |
| Một câu có hai nhãn thì học thế nào? | Hàng multi-hot có hai số 1; hai bộ LR tương ứng nhận mẫu dương, 26 bộ còn lại nhận mẫu âm. |
| Vì sao không softmax? | Softmax chung làm các nhãn cạnh tranh trong một phân phối; OvR sigmoid độc lập cho phép nhiều nhãn đạt ngưỡng. |
| Score là nhãn hay độ chắc chắn? | Score phải qua ngưỡng mới thành nhãn; chưa có thí nghiệm hiệu chuẩn để coi score là độ chắc chắn đáng tin cậy. |
| Sao không luôn gán neutral khi không có nhãn? | Neutral cũng phải được mô hình chọn theo điểm/ngưỡng; tự gán sẽ thêm một luật làm đổi đánh giá. |
| Macro-F1 sao thấp hơn Micro-F1? | Có các nhãn hiếm F1 thấp; Macro chia trọng số đều 28 nhãn. Micro gom quyết định nên nhãn phổ biến ảnh hưởng mạnh. |
| Balanced có luôn tốt hơn không? | Không. Validation @0,5 tăng Macro/Recall nhưng giảm Precision và tăng Hamming; xem thêm từng nhãn và test đã khóa. |
| Tune có phải train lại không? | Không. Giữ nguyên scores/model, chỉ chọn ngưỡng trên validation. Weighting mới làm thay phần fit. |
| Vì sao không chọn ngưỡng bằng test? | Khi dùng test để chọn, điểm cuối không còn phép đánh giá độc lập của các lựa chọn; nhóm chọn trên val rồi khóa trước test. |
| Hai baseline, ba ngưỡng có sáu model không? | Hai Pipeline đã fit, sáu cấu hình đánh giá. Ngưỡng không tạo thêm encoder hoặc bộ LR. |
| C=1 và max_iter=1000 nghĩa gì? | C là nghịch đảo độ mạnh regularization; max_iter là trần vòng tối ưu liblinear, không phải epoch C. Đây là config cố định đã lưu. |
| Em dùng bao nhiêu seed? | A dùng random_state42; yêu cầu ≥3 seed áp dụng từng kiến trúc C với42/123/2026. A không được tự ghi mean±std từ một lần fit. |
| Vì sao không nói tái lập chính xác paper? | TF-IDF+LR là baseline đồ án; C có lựa chọn triển khai được khai báo. Chỉ gọi tái lập chính xác khi kiểm đủ thiết lập của paper. |
| Nhãn thật có thể thiếu thì lỗi còn ý nghĩa không? | Benchmark đo theo ground truth có sẵn. Em đưa câu/ID/scores và giới hạn chú giải; không tự sửa nhãn chỉ vì không đồng ý. |
| Đồ án giúp gì trong thực tế? | Có thể nhóm/ưu tiên đọc phản hồi cảm xúc. F1 trên Reddit tiếng Anh chưa chứng minh hiệu quả doanh nghiệp hoặc tiếng Việt; cần đánh giá đúng miền. |

Trước buổi bảo vệ, tự tính lại ví dụ ở mục 2/6, chỉ đúng dòng `fit` và
`predict_proba`, mở một lỗi có ID rồi giải thích FN/FP. Đọc số mới nhất từ
artifacts/bảng tổng hợp thay vì nhớ một thứ hạng chưa hoàn tất.

## 9. Nguồn và đường dẫn đối chiếu

- [TfidfVectorizer, sklearn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
  và [TfidfTransformer](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html):
  số đếm, smooth IDF, L2 và mặc định. Công thức/mặc định đã đối chiếu source
  sklearn cài tại môi trường mô hình, không thay bằng công thức TF khác.
- [LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html),
  [OneVsRestClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.multiclass.OneVsRestClassifier.html):
  sigmoid nhị phân, regularization, weighting và OvR đa nhãn.
- [P/R/F1](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.precision_recall_fscore_support.html),
  [Hamming Loss](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.hamming_loss.html),
  [chọn ngưỡng](https://scikit-learn.org/1.7/modules/classification_threshold.html).
- Code thực: [run_baseline.py](../scripts/run_baseline.py), [metrics.py](../src/metrics.py),
  [baseline.py](../src/baseline.py), [analyze_baseline.py](../scripts/analyze_baseline.py).
- Yêu cầu toàn nhóm: [hồ sơ đối chiếu cô](DOI_CHIEU_YEU_CAU_CO.md) và
  [thứ tự đọc đồ án](THU_TU_DOC_DO_AN.md). Tài liệu này hướng dẫn hiểu A;
  không thay việc chạy đủ C/B/D, báo cáo và minh chứng mà cô yêu cầu.

## 10. Đọc ba lỗi thật: cùng câu, hai baseline, hai cách đặt ngưỡng

Phần này dùng **ba ID validation có thật**, lấy text và nhãn chuẩn từ
`data/processed/baseline/full/error_examples_validation.csv`. Điểm bên dưới
được đọc lại từ `validation_scores.npz` của từng mô hình; CSV đối chiếu giữ
độ chính xác của điểm lưu, không lấy cột điểm đã làm tròn trong CSV nguồn.
Hai file điểm đã được kiểm SHA-256 với metadata, đủ 5.426 ID không trùng,
ma trận 5.426 × 28, đúng thứ tự nhãn và các điểm hữu hạn trong [0, 1].
Ngưỡng riêng được kiểm thuộc **đúng model và file điểm của model đó**.

Đọc [bảng 12 hàng](../reports/baseline_validation/paired_examples.csv) và
[manifest đối chiếu](../reports/baseline_validation/paired_examples_manifest.json)
để thấy điểm, ngưỡng, nhãn được chọn, FN/FP, hash nguồn và thời điểm tạo thật.
Ba câu được chọn để học từ lỗi standard; **không phải mẫu ngẫu nhiên**, không
phải kết quả test, không đủ để kết luận model nào tốt hơn trên toàn tập.
Tuned dùng chính validation để chọn ngưỡng nên đây là minh họa sau hiệu chỉnh.

### 10.1. Cách tự đọc từng hàng

1. Giữ nguyên **ID, text và tập nhãn chuẩn** khi chuyển standard ↔ balanced.
   Đối chiếu file điểm bằng ID, không bằng vị trí dòng.
2. Chọn một model. Model đó tạo 28 điểm trước khi đặt ngưỡng.
3. Với fixed, xét mỗi điểm ≥ 0,50. Với tuned, xét điểm của nhãn j ≥ ngưỡng
   j đã lưu cho **chính model đang đọc**; không chuyển ngưỡng giữa hai model.
4. Gom tất cả nhãn đạt ngưỡng. Đây là đa nhãn, không lấy duy nhất điểm cao nhất.
   Không có nhãn đạt ngưỡng thì tập dự đoán rỗng; code không tự gán neutral.
5. **FN = nhãn chuẩn − nhãn dự đoán**; **FP = nhãn dự đoán − nhãn chuẩn**.
   Dự đoán trúng một nhãn vẫn có thể còn nhãn thừa hoặc nhãn thiếu.

`focus_label` trong CSV là nhãn đang soi kỹ, không giới hạn mô hình chỉ đoán
nhãn đó. Câu thứ ba soi `love` vì model chọn thừa love; nhãn chuẩn vẫn là neutral.
**Bảng và lời giải dưới đây làm tròn điểm đến 4 chữ số thập phân, ngưỡng
đến 2 chữ số cho dễ đọc.** CSV giữ độ chính xác số đã lưu để đối chiếu;
FN/FP và quyết định chọn nhãn được tính từ điểm/ngưỡng đầy đủ, không từ
số đã làm tròn trong hướng dẫn. Điểm hiển thị `1.0000` có thể vẫn nhỏ hơn 1.

### 10.2. Câu pride: sửa nhãn thiếu nhưng xuất hiện nhãn thừa

**ID `eczwil0`:** “I am so proud of this community.”
**Nhãn chuẩn:** `{pride}`.

| Model | Luật ngưỡng | Điểm pride | Ngưỡng pride | Toàn bộ nhãn được chọn | FN | FP |
|---|---|---:|---:|---|---|---|
| standard | fixed | 0.3776 | 0.50 | ∅ | pride | ∅ |
| standard | tuned | 0.3776 | 0.15 | admiration, pride | ∅ | admiration |
| balanced | fixed | 0.9994 | 0.50 | admiration, pride | ∅ | admiration |
| balanced | tuned | 0.9994 | 0.80 | admiration, pride | ∅ | admiration |

**Tự diễn giải:** ở standard fixed, điểm khoảng 0,3776 < 0,50 nên pride không được
chọn: có FN pride. Khi dùng standard tuned, điểm pride **không đổi**; ngưỡng
giảm xuống 0,15 nên pride được chọn. Tuy nhiên còn chọn admiration, trong khi
nhãn chuẩn chỉ có pride, nên vẫn có FP admiration. Không nói “đã sửa hết lỗi”.

Balanced có điểm pride khác vì đó là **một model đã fit với objective có
class weighting**, không phải bản standard được đổi tên. Tại câu này cả fixed
và tuned đều nhận pride, nhưng đều thêm admiration. Số liệu này mô tả quyết
định đã xảy ra; chưa chỉ ra từ/đặc trưng nào gây ra điểm cao hoặc FP.

**Câu trả lời ngắn khi bảo vệ:** “Tune chỉ đổi luật quyết định, không train
lại. Cùng điểm khoảng 0,3776, hạ ngưỡng pride từ 0,50 xuống 0,15 sửa FN pride,
nhưng tập dự đoán vẫn thừa admiration so với nhãn chuẩn.”

### 10.3. Câu grief: weighting nhận được grief nhưng neutral vẫn thừa

**ID `edwloev`:** “He died 4 days later of dehydration”
**Nhãn chuẩn:** `{grief}`.

| Model | Luật ngưỡng | Điểm grief | Ngưỡng grief | Toàn bộ nhãn được chọn | FN | FP |
|---|---|---:|---:|---|---|---|
| standard | fixed | 0.0189 | 0.50 | neutral | grief | neutral |
| standard | tuned | 0.0189 | 0.50 | neutral | grief | neutral |
| balanced | fixed | 0.9858 | 0.50 | grief, neutral | ∅ | neutral |
| balanced | tuned | 0.9858 | 0.55 | grief, neutral | ∅ | neutral |

**Tự diễn giải:** ở standard, điểm grief khoảng 0,0189 chưa đạt 0,50. Ngưỡng
tuned của **riêng grief vẫn là 0,50**, nên tuning không sửa FN này. Không phải
mọi ngưỡng tuned đều thấp hơn 0,50. Neutral standard có điểm khoảng
**0,5799**, vượt cả fixed 0,50 và ngưỡng neutral tuned 0,30;
do đó vẫn được chọn và vẫn là FP theo nhãn chuẩn của câu.

Balanced cho grief khoảng **0,9858**, vượt 0,50 và 0,55 nên không còn
FN grief. Nhưng neutral balanced có điểm khoảng **0,7136**, vượt cả
0,50 và ngưỡng neutral tuned 0,45; vì vậy vẫn còn FP neutral. Các bộ LR xét
độc lập: code không cưỡng ép neutral loại trừ mọi cảm xúc khác.

**Câu trả lời ngắn khi bảo vệ:** “Ở câu này balanced giúp nhận được nhãn
grief mà standard bỏ sót, nhưng chưa đúng toàn bộ tập nhãn vì vẫn đoán thừa
neutral. Kết quả từng câu không chứng minh weighting luôn tốt hơn.”

### 10.4. Câu có từ love: điểm rất cao vẫn sai theo ground truth

**ID `ed832y6`:** văn bản nguyên gốc có dấu ngoặc kép: `"Homeopaths love it!"`.
**Nhãn chuẩn:** `{neutral}`. Nhãn đang soi là **love**.

| Model | Luật ngưỡng | Điểm love | Ngưỡng love | Toàn bộ nhãn được chọn | FN | FP |
|---|---|---:|---:|---|---|---|
| standard | fixed | 1.0000 | 0.50 | love | neutral | love |
| standard | tuned | 1.0000 | 0.15 | love | neutral | love |
| balanced | fixed | 1.0000 | 0.50 | love | neutral | love |
| balanced | tuned | 1.0000 | 0.70 | love | neutral | love |

**Tự diễn giải:** cả bốn cấu hình đều chọn love và không chọn neutral.
Điểm neutral standard khoảng **0,0015**, dưới cả 0,50 và 0,30;
điểm neutral balanced khoảng **0,0021**, dưới cả 0,50 và 0,45.
Vì nhãn chuẩn chỉ có neutral, mỗi hàng có một FN neutral và một FP love.

Điểm love gần 1 **không đồng nghĩa dự đoán đúng hoặc đã có độ chắc chắn
được hiệu chuẩn**. Ta biết score, nhãn chuẩn và lỗi đo được; chưa có kiểm
tra ngữ cảnh/chú giải hay phân tích đóng góp đặc trưng để kết luận nguyên
nhân. Không tự gọi câu này là mỉa mai, không tự sửa neutral thành love và
không khẳng định model “chỉ nhìn chữ love” từ riêng kết quả này.

**Câu trả lời ngắn khi bảo vệ:** “Mô hình cho điểm love gần 1 nhưng benchmark
ghi neutral. Theo nhãn chuẩn, love là FP và neutral là FN ở cả bốn cấu hình.
Điểm cao chưa bảo đảm đúng; em cần đọc thêm ngữ cảnh/chú giải trước khi
kết luận nguyên nhân ngôn ngữ.”

### 10.5. Ba điều Duy phải phân biệt khi trình bày

- **Đổi ngưỡng:** cùng model, cùng scores; chỉ đổi tập nhãn được chọn.
  Ví dụ pride standard sửa FN nhưng thêm FP; grief standard vẫn chưa sửa.
- **Đổi model bằng weighting:** objective lúc fit khác nên scores có thể khác.
  Ví dụ grief được nhận ở balanced, nhưng vẫn đoán thừa neutral. Không suy ra
  nguyên nhân từ ba câu hay coi điểm hai model đã được hiệu chuẩn giống nhau.
- **Đo lỗi theo nhãn chuẩn:** phải ghi đủ FN/FP, kể cả khi câu có vẻ dễ hoặc
  điểm rất cao. Không dùng ba câu để tính thứ hạng toàn tập hay số liệu test.

Phần này giúp Duy giải thích **A**. Nghĩa vụ đồ án về ít nhất ba nhóm lỗi và
so sánh C1/C2/C3 vẫn cần bộ phân tích chung trên các run C thật, cùng ID và
protocol; ba ví dụ A này không thay yêu cầu đó.
