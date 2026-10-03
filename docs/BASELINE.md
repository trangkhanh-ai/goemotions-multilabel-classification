# Phần A của bạn: baseline GoEmotions từ đầu đến lúc bàn giao

Tài liệu này dành cho **thành viên phụ trách baseline**. Mục tiêu là bạn tự chạy được,
hiểu từng bước, giải thích số liệu và giao đúng đầu ra cho nhóm. Mã dùng các thư viện
quen thuộc `scikit-learn`, `numpy`, `pandas`; không thêm mô hình phức tạp không cần thiết.

## 1. Đối chiếu yêu cầu cô

| Yêu cầu đã đối chiếu | Phần A thực hiện |
|---|---|
| Baseline cổ điển TF-IDF/luật + mô hình scikit-learn; 1 sinh viên phụ trách | TF-IDF + One-vs-Rest Logistic Regression |
| Đề tài 1: GoEmotions đa nhãn | 28 đầu ra độc lập trên split chính thức |
| Macro/Micro-F1, Precision/Recall, Hamming Loss | Metric chung, bảng validation và F1 từng nhãn |
| Báo cáo cuối tuần 4 phải có số A cụ thể | `reports/BASELINE_RESULTS.md` có số validation thật |
| Nâng cao: class weighting hoặc ngưỡng riêng hoặc contrastive, có F1 nhãn hiếm | Có thí nghiệm `balanced` và ngưỡng riêng trên validation |

Zero-shot B có thể làm chung với người làm A, nhưng **không phải baseline A**. Ba mô
hình C, ba seed mỗi mô hình, phân tích lỗi giữa C1/C2/C3 và demo từ C tốt nhất là
việc bắt buộc của **cả nhóm**.
PDF đề tài không quy định nâng cao phải áp dụng riêng cho C: thí nghiệm weighting/
ngưỡng trên A có thể là bằng chứng nâng cao của đề tài nếu được đánh giá đúng.
Áp dụng thêm cho C tốt nhất là **đề xuất của kế hoạch nhóm**, không phải một yêu
cầu bổ sung đã ghi trong PDF. Cô vẫn yêu cầu đủ ba kiến trúc C × ba seed và demo.

## 2. Bài toán và dữ liệu

Một câu có thể vừa `gratitude` vừa `admiration`. Vì vậy nhãn thật là vector 28 ô
0/1, với nhiều ô bằng 1. Đây là **đa nhãn**, khác chọn một nhãn duy nhất. `neutral`
là một trong 28 nhãn của nguồn; không tự xóa khi nó xuất hiện cùng nhãn khác.

Nguồn thô trong bài báo có **58.009** bình luận Reddit; bản `simplified` dùng ở
đây có **54.263** mẫu sau xử lý của nguồn. Nó gồm train **43.410**, validation
**5.426**, test **5.427**
dòng. Train dùng để fit TF-IDF và Logistic Regression; validation dùng để chọn
cấu hình/ngưỡng; test chỉ đo một lần sau khi chốt quyết định. Các tệp nguồn được
khóa revision và SHA-256 trong `src/data.py`. Mọi hệ thống A/B/C phải giữ cùng thứ
tự 28 nhãn trong `data/labels.json` và ghép điểm dự đoán theo `id`.

## 3. File cần đọc theo thứ tự

| File | Việc của file |
|---|---|
| `src/data.py` | Tải/kiểm hash, lấy text/labels/id, đổi nhãn thành multi-hot |
| `src/metrics.py` | Micro/Macro Precision–Recall–F1, Hamming Loss, từng nhãn |
| `src/baseline.py` | Ghép scores theo ID, chọn ngưỡng riêng từ validation |
| `scripts/run_baseline.py` | Fit TF-IDF + LR; lưu model, scores và metric validation |
| `scripts/analyze_baseline.py` | So sánh ngưỡng, nhãn hiếm, lỗi có ví dụ, từ quan trọng |
| `scripts/predict_baseline.py` | Nhập một câu tiếng Anh để xem điểm và nhãn |
| `scripts/freeze_baseline.py` | Khóa cả bảng cấu hình A trước khi mở test |
| `scripts/evaluate_baseline_test.py` | Chỉ chạy test **sau khi khóa cấu hình** |
| `notebooks/baseline.ipynb` | Học từng bước bằng dữ liệu và output thật |
| `reports/BASELINE_RESULTS.md` | Bảng kết quả thật, sinh lại bằng script phân tích |

## 4. Cài đặt và lệnh chạy

Mở PowerShell ở **gốc repo** `goemotions-multilabel-classification`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.run_baseline --smoke
.\.venv\Scripts\python.exe -m scripts.run_baseline
.\.venv\Scripts\python.exe -m scripts.run_baseline --variant balanced
.\.venv\Scripts\python.exe -m scripts.analyze_baseline
.\.venv\Scripts\python.exe -m scripts.export_baseline_results
```

Nếu không có `py -3.13`, dùng Python 3.13 đã cài để tạo `.venv`. Trên máy đang làm
đồ án có thể thay `.\.venv\Scripts\python.exe` bằng
`C:\Users\dzyuu\anaconda3\python.exe`. `--smoke` chỉ dùng 5.000 train/1.000
validation để bắt lỗi; **không lấy số smoke đưa vào báo cáo**. Lần chạy đầu có thể
tải dữ liệu; hash được kiểm trước khi dùng. Chạy lại sẽ lấy cache local.

Notebook có output thật đã chạy sẵn. Nếu muốn chạy từng cell, cài
`requirements-notebook.txt` và mở `notebooks/baseline.ipynb` bằng VS Code/Jupyter.
Để sinh và chạy lại notebook từ script:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m scripts.build_baseline_notebook --execute
```

Thử một câu sau khi fit:

```powershell
.\.venv\Scripts\python.exe -m scripts.predict_baseline --text "I am so proud of you!"
.\.venv\Scripts\python.exe -m scripts.predict_baseline --variant balanced --threshold tuned --text "I am so proud of you!"
```

## 5. Mã hoạt động như thế nào?

### Bước 1: dữ liệu và nhãn

`load_goemotions(..., splits=("train", "validation"))` chỉ mở hai split cần thiết.
Mỗi dòng có `text`, `labels` (danh sách ID nhãn) và `id` (khóa ghép kết quả).
`multi_hot` biến `[3, 7]` thành 28 ô, chỉ ô 3 và 7 bằng 1; nhãn rỗng, lặp hoặc
ngoài miền bị từ chối. Test không được đọc trong quá trình huấn luyện/phân tích.

### Bước 2: TF-IDF

`TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000)` lấy từ đơn
và cặp từ, bỏ đặc trưng chỉ xuất hiện trong một văn bản. Lần fit full hiện có
**58.338 đặc trưng**. TF-IDF làm nổi bật từ quan trọng trong một bình luận nhưng
không quá phổ biến trên toàn train. Không cần xóa stopword hay stem tùy tiện:
phủ định như `not` có thể đổi nghĩa. `Pipeline` bảo đảm TF-IDF chỉ `fit` trên
train rồi dùng cùng bộ từ vựng để `transform` validation.
Vectorizer mặc định bỏ dấu câu, emoji và token một ký tự; apostrophe có thể khiến
`don't` thành `don`. Vì vậy việc giữ từ `not` không đồng nghĩa đã xử lý tốt mọi
phủ định hoặc mỉa mai. Đây là một hạn chế cụ thể để bạn ghi trong báo cáo.

### Bước 3: 28 Logistic Regression

`OneVsRestClassifier(LogisticRegression(...))` học một bài toán có/không cho từng
nhãn: tổng cộng 28 bộ phân loại nhị phân. Mỗi bộ trả score 0–1; tổng 28 score
không cần bằng 1. Hai biến thể dùng cùng `C=1`, `solver=liblinear`,
`max_iter=1000`, dữ liệu và cách đo:

- `standard`: không dùng trọng số lớp, là mốc gốc.
- `balanced`: dùng `class_weight='balanced'` **cho từng nhãn**, tăng trọng số lỗi
  của lớp ít mẫu dựa trên tần suất trong train. Đây là thí nghiệm A nâng cao.

Với một nhãn có `n+` mẫu dương trong `N` dòng train, scikit-learn đặt trọng số
phía dương xấp xỉ `N/(2×n+)` và phía âm `N/(2×(N−n+))`. Vì `grief` chỉ có 77 mẫu
dương, lỗi bỏ sót `grief` được phạt mạnh hơn. Mức phạt này có thể tăng cả TP lẫn FP.
`C` là nghịch đảo độ mạnh regularization; ở đây giữ `C=1` để hai biến thể so sánh
công bằng. Mã dùng `random_state=42` để giữ điều kiện tái lập. Yêu cầu ít nhất ba
seed trong PDF áp dụng cho từng kiến trúc C; PDF không yêu cầu ba seed cho A.
Mã dừng nếu Logistic Regression báo chưa hội tụ và lưu số vòng lặp từng nhãn.

### Bước 4: score, ngưỡng và nhãn

`predict_proba` tạo ma trận **5.426 × 28** trên validation. Score chưa phải
nhãn. Ngưỡng gốc là `>=0,5`; nếu không score nào đạt, kết quả là **không nhãn**.
Không ép chọn nhãn cao nhất vì như vậy đổi luật đánh giá.

`tune_thresholds` thử 0,05–0,95 theo bước 0,05 **cho từng nhãn riêng**, lấy F1
nhãn đó cao nhất trên validation; nếu hòa, lấy ngưỡng gần 0,5. Ngưỡng phải đi
cùng đúng variant. Không sao chép ngưỡng A sang B/C.
`tune_global_threshold` dùng cùng lưới để chọn **một ngưỡng chung** theo Macro-F1.
Nếu hai mức cách 0,5 bằng nhau và F1 hòa, chọn mức lớn hơn để kết quả xác định.
Lưu ý lưới này không thử mức dưới 0,05; standard có score grief rất thấp nên
ngưỡng riêng chưa giúp grief trong thí nghiệm hiện tại. Không che nhãn không tăng.
File ngưỡng lưu hash model/scores; nếu bạn huấn luyện lại, phải chạy lại phân tích
trước khi suy luận với ngưỡng `global` hoặc `tuned`.

### Bước 5: chỉ số

- Precision: trong nhãn dự đoán có, tỷ lệ đúng.
- Recall: trong nhãn thật, tỷ lệ tìm được.
- F1: trung bình điều hòa của Precision và Recall.
- Micro-F1: cộng TP/FP/FN của 28 nhãn rồi tính F1; nhãn phổ biến ảnh hưởng mạnh.
- Macro-F1: F1 từng nhãn rồi lấy trung bình đều 28 nhãn; nhãn hiếm có trọng số ngang nhau.
- Hamming Loss: số quyết định nhãn sai chia cho `số mẫu × 28`; **thấp hơn** là tốt.

`zero_division=0` khi một nhãn không được đoán dương. Phần lớn ô trong ma trận
N×28 là 0; Hamming Loss thấp **không chứng minh** mô hình bắt tốt nhãn hiếm.

## 6. Số liệu validation đã đo

| A | Ngưỡng | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---|---:|---:|---:|
| standard | 0,5 | 0,2025 | 0,3760 | 0,0354 |
| standard | chung 0,10, chọn trên val | 0,4094 | 0,5100 | 0,0544 |
| standard | riêng từng nhãn, chọn trên val | 0,4391 | 0,5427 | 0,0433 |
| balanced | 0,5 | 0,4562 | 0,5099 | 0,0532 |
| balanced | chung 0,55, chọn trên val | 0,4660 | 0,5176 | 0,0473 |
| balanced | riêng từng nhãn, chọn trên val | 0,4901 | 0,5467 | 0,0443 |

Đây là **validation**, chưa phải test. Weighting tăng Macro-F1 và Recall nhưng
Hamming Loss xấu đi ở ngưỡng 0,5: mô hình đoán nhiều nhãn hơn và tạo thêm FP.
Hàng tuned được đo trên chính validation dùng để tìm ngưỡng, nên có thể lạc quan.
Không tuyên bố “đã cải thiện trên test” hoặc “đạt điểm paper”. Bảng P/R/F1 đủ
28 nhãn ở `data/processed/baseline/.../per_label_validation.csv`.

Năm nhãn ít mẫu nhất theo **train**, xác định trước khi nhìn validation: `grief`
(77), `pride` (111), `relief` (153), `nervousness` (164), `embarrassment` (303).
Với standard ở 0,5, bốn nhãn đầu có F1 bằng 0. Balanced cải thiện cả năm nhãn
trên validation ở 0,5, nhưng mỗi nhãn chỉ có 13–35 mẫu dương val. Hãy nêu support
và giới hạn này khi giải thích.

## 7. Phân tích lỗi có ví dụ thật

`error_examples_validation.csv` có văn bản, ID, nhãn thật, nhãn dự đoán và score:

1. **Bỏ sót nhãn hiếm:** ID `eczwil0`, “I am so proud of this community.” Nhãn thật
   `pride`, standard đoán không nhãn, điểm pride 0,3776 dưới 0,5.
2. **Dự đoán nhãn thừa:** ID `ed832y6`, “Homeopaths love it!” Nhãn thật `neutral`,
   standard đoán `love`, score gần 1. Từ `love` có thể làm mô hình bám mặt chữ;
   đồng thời nhãn gốc có thể chưa đầy đủ.
3. **Đúng một phần câu đa nhãn:** ID `eczdvun`, “Thank you. I really appreciate
   your response”. Nhãn thật `admiration, gratitude`, mô hình chỉ trả `gratitude`.
4. **Không nhãn nào vượt ngưỡng:** ID `eeoh5vh`, câu có “wtf lol”, nhãn thật
   `amusement`, điểm cao nhất khoảng 0,4999, vừa dưới ngưỡng.

Đây là lỗi của **A**. Yêu cầu chung còn cần so sánh ít nhất ba loại lỗi giữa
**C1/C2/C3**; nhóm làm thêm khi có score các mô hình đó. `top_features.csv` ghi
từ/cặp từ có hệ số LR cao/thấp cho mỗi nhãn; không coi đó là quan hệ nhân quả.

`label_error_pairs_validation.csv` đếm câu có **FN nhãn A + FP nhãn B** và lưu
ID ví dụ. Khác với heatmap nhãn thật đồng xuất hiện của EDA. Ví dụ một câu thật
`anger`, dự đoán thừa `annoyance` và bỏ sót anger sẽ góp cho cặp này. Một câu có
thể góp nhiều cặp, nên không cộng bảng cặp để lấy tổng số câu lỗi. Báo cáo A giữ
bảng neutral và một bảng phụ các cặp không có neutral; metric vẫn đủ 28 nhãn.

## 8. File đầu ra để bàn giao

- Standard: `data/processed/baseline/full/`
- Balanced: `data/processed/baseline/balanced/full/`

| File | Cách dùng |
|---|---|
| `validation_metrics.json` | Cấu hình, revision, hash model/scores, phiên bản, hội tụ, chỉ số và 28 nhãn |
| `validation_scores.npz` | `ids`, `scores` N×28, `label_names`; nhóm ghép theo ID |
| `per_label_validation.csv` | Support, P/R/F1 ở 0,5 và ngưỡng riêng |
| `thresholds_validation.json` | 28 ngưỡng, luật chọn; chỉ dùng đúng variant |
| `error_examples_validation.csv` | Ví dụ thật để đọc và chọn đưa vào báo cáo |
| `label_error_pairs_validation.csv` | Cặp FN/FP cùng câu, số đếm và ID minh chứng |
| `threshold_curve_validation.csv` | So sánh 19 ngưỡng chung trên validation |
| `top_features.csv` | Hệ số LR cho từ/cặp từ của từng nhãn |
| `model.joblib` | Pipeline đã fit; **chỉ mở file do nhóm tạo** |

`data/processed/` được `.gitignore` bỏ qua vì model/dữ liệu lớn. Chia sẻ thư mục
này với đồng đội qua nơi lưu trữ chung kèm revision, variant và thứ tự nhãn.
Mã và `reports/BASELINE_RESULTS.md` nằm trong repo.
Các bảng JSON/CSV nhỏ được xuất bằng `scripts.export_baseline_results` vào
`reports/baseline_validation/` và đưa lên Git để đồng đội xem số liệu ngay.
Script kiểm hash model/scores/ngưỡng trước khi xuất; không mở test, không chép model.

## 9. Khi nào mới chạy test?

**Chưa chạy test.** Khi cả nhóm chốt variant và luật ngưỡng bằng validation, ghi
quyết định vào báo cáo/commit rồi khóa **cả bảng so sánh trước/sau**:

```powershell
.\.venv\Scripts\python.exe -m scripts.freeze_baseline
.\.venv\Scripts\python.exe -m scripts.evaluate_baseline_test
```

Lệnh freeze không mở test: nó ghi sáu cấu hình (hai model × ba luật ngưỡng), hash
model/scores/ngưỡng và lựa chọn tốt nhất **theo validation** vào `final_protocol.json`.
Lệnh evaluate kiểm mọi hash rồi dự đoán test một lần cho mỗi model, tính đủ sáu
hàng đã khóa để giữ baseline gốc và so sánh nâng cao. File `rare_labels_test.csv`
ghi F1/support và chênh lệch của cả năm nhãn hiếm, kể cả nhãn giảm.

Chạy lại cùng protocol sẽ đọc kết quả đã có. Nếu model/ngưỡng đổi sau freeze,
mã báo lỗi trước khi mở test. Không dùng test để chọn lại cấu hình. Các test mã
nguồn về luồng cuối sử dụng dữ liệu giả ở thư mục tạm, không mở GoEmotions test.

## 10. Bạn cần tự giải thích được khi bảo vệ

1. Vì sao một mẫu có thể có nhiều cảm xúc, và `Y` có 28 cột?
2. TF-IDF và Logistic Regression làm hai việc khác nhau thế nào?
3. Vì sao TF-IDF chỉ `fit` trên train, không trên validation/test?
4. One-vs-Rest giải bài toán đa nhãn bằng 28 bài toán nhị phân thế nào?
5. Vì sao weighting tăng Recall nhưng Hamming Loss có thể xấu đi?
6. Macro-F1 khác Micro-F1; vì sao bốn nhãn hiếm F1 bằng 0 ở A gốc?
7. Vì sao ngưỡng được chọn trên validation và điểm tuned-val có thể lạc quan?
8. Vì sao một dự đoán `love` sai theo nhãn gốc chưa đủ kết luận ngữ nghĩa hoàn toàn sai?
9. Ghép score A với B/C bằng gì, vì sao không ghép theo số dòng?
10. Vì sao chỉ mở test sau khi cả nhóm khóa quyết định?

## 11. Nguồn để học và trích dẫn

- [Bài báo GoEmotions, ACL 2020](https://aclanthology.org/2020.acl-main.372/): nguồn gốc và bài toán.
- [Google Research README](https://github.com/google-research/google-research/blob/master/goemotions/README.md): split và dữ liệu.
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html): đặc trưng.
- [OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html): chiến lược đa nhãn.
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): trọng số lớp, `C`, solver.
- [precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html): P/R/F1, micro/macro.
- [Hamming Loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html): tỷ lệ quyết định nhãn sai.
- [Chọn ngưỡng, đúng phiên bản scikit-learn 1.7.2](https://scikit-learn.org/1.7/modules/classification_threshold.html): tách học model, chọn ngưỡng và đánh giá cuối.

Học theo thứ tự: multi-hot → TF-IDF → Logistic Regression → One-vs-Rest →
Precision/Recall/F1 → ngưỡng → class weighting → phân tích lỗi. Sau mỗi lệnh ở
mục 4, mở JSON/CSV vừa tạo để nối khái niệm với số liệu thật.
