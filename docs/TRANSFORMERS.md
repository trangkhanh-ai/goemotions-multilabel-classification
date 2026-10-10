# Phần C và demo D — hướng dẫn đọc, chạy và giải thích

Phụ trách C1 **Quốc Khánh**, C2 **Đức Trí**, C3 và demo **Nhật Huy**.
Đây là ba mô hình đã học ngôn ngữ trước, được fine-tune trên cùng GoEmotions.
Script hỗ trợ toàn bộ luồng; trạng thái thực nghiệm phải đọc trong artifacts và bảng kết quả,
không suy ra “đã có kết quả” chỉ vì file code tồn tại.

## 1. Bài toán trong một câu

Một bình luận tiếng Anh có thể cùng thể hiện `joy` và `gratitude`. Mỗi mô hình
phải trả **28 điểm độc lập**, tương ứng 27 cảm xúc và `neutral`, đúng thứ tự
`data/labels.json`. Không dùng softmax hoặc argmax ép câu chỉ có một nhãn.

1. **Tokenizer** đổi văn bản thành token ID và attention mask đúng checkpoint.
2. **Encoder** BERT/RoBERTa/DistilBERT tạo biểu diễn ngữ cảnh.
3. **Classification head** trả 28 logits, là các số thực chưa sigmoid.
4. **Huấn luyện:** `BCEWithLogitsLoss(logits, multi_hot_float)` học từng nhãn.
5. **Dự đoán:** `sigmoid(logits)` trả điểm trong `[0,1]`; so điểm với ngưỡng.

`neutral` là một nhãn đã được con người gán; khi không điểm nào vượt ngưỡng,
không tự thay bằng neutral. Các điểm sigmoid chưa được hiệu chuẩn nên không
viết “mô hình chắc chắn 90%” chỉ vì điểm là 0,90.

Nguồn kỹ thuật: [BERT trong Transformers](https://huggingface.co/docs/transformers/en/model_doc/bert),
[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

## 2. Cấu hình đã khai báo

| Hạng mục | C1: BERT | C2: RoBERTa | C3: DistilBERT |
|---|---|---|---|
| Checkpoint | `google-bert/bert-base-cased` | `FacebookAI/roberta-base` | `distilbert/distilbert-base-uncased` |
| Người phụ trách | Quốc Khánh | Đức Trí | Nhật Huy |
| Learning rate | 0,00005 | 0,00002 | 0,00002 |
| Epoch tối đa | 4 | 3 | 3 |
| Max tokens | 128 | 128 | 128 |
| Batch thực tế | 16 | 16 | 16 |
| Gradient accumulation | 1 | 1 | 1 |
| Batch hiệu dụng | 16 | 16 | 16 |
| Seeds | 42, 123, 2026 | 42, 123, 2026 | 42, 123, 2026 |

C1 chọn checkpoint cased, batch hiệu dụng 16, learning rate 5e-5 và 4 epoch để
tham khảo thực nghiệm BERT trong bài GoEmotions. AdamW/weight decay 0,01,
warmup tuyến tính 10%, max tokens 128, mixed precision và chọn checkpoint trên
validation là các lựa chọn triển khai được khai báo của nhóm. C2/C3 dùng cấu
hình của nhóm. **Không ghi “tái lập chính xác bài gốc” nếu cấu hình khác.**
Nguồn bài gốc: [GoEmotions, ACL 2020](https://aclanthology.org/2020.acl-main.372/).

**Căn cứ giải thích cập nhật 10/10/2026:** ba seed kiểm tra biến động kết quả trong
cùng cấu hình, không thay thế tìm kiếm siêu tham số. Hồ sơ chín run chính chưa có
grid/random search learning rate và epoch; chọn checkpoint qua các epoch trên
validation là một quyết định khác. Không giải thích C2 bằng gradient explosion
hoặc C3 bằng hội tụ nhanh hơn khi chưa có thí nghiệm chứng minh. Các C dùng
learning rate/epoch khác nhau nên chưa cô lập ảnh hưởng kiến trúc. Test đã công bố
không được dùng chọn lại cấu hình hoặc ngưỡng.

Tokenizer cache mỗi câu tối đa 128 token. Khi tạo batch, `trim_padding_collate`
bỏ phần padding bên phải sau câu dài nhất trong batch. Không cắt token có
attention mask=1, không thay nhãn N×28. Config ghi `padding=dynamic_batch_trim`;
run cũ padding cố định không được coi là cùng cấu hình khi resume.

Checkpoint và tokenizer được khóa cùng commit SHA trước khi tải. Gieo seed Python,
NumPy, CPU, CUDA; tắt cuDNN benchmark; bật deterministic `warn_only`.
Kết quả có thể khác theo GPU, thư viện hoặc phép toán không deterministic.
`run_metadata.json` ghi thiết bị, thư viện, SHA, seed, cấu hình và log thật.
`history` ghi thời gian từng epoch gồm huấn luyện/validation, loss và F1 thực đo;
chỉ log run hoàn tất mới là bằng chứng thực nghiệm. `warn_only` cho phép PyTorch
cảnh báo phép toán không deterministic thay vì dừng; không bảo đảm kết quả giống bit.

## 3. Môi trường

Dùng môi trường Python đã có numpy/pandas/pyarrow/scikit-learn của dự án.
Cài **PyTorch phù hợp CPU/GPU** theo [trang cài đặt chính thức](https://pytorch.org/get-started/locally/),
sau đó cài `requirements-transformer.txt`. RTX 50xx cần bản PyTorch/CUDA hỗ trợ
kiến trúc GPU đang dùng; kiểm `torch.cuda.is_available()` trước full run.
File requirements bổ sung không tự thay bản Torch CUDA đã chọn bằng wheel CPU.

```powershell
python -m pip install -r requirements-transformer.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA dùng AMP float16 và GradScaler; CPU dùng float32. Không bật `trust_remote_code`.
Mỗi run chạy tuần tự để ba mô hình không cùng chiếm VRAM. Nếu thiếu VRAM, giảm
`--batch-size 8 --gradient-accumulation 2` (hoặc 4/4) để giữ batch hiệu dụng 16 và ghi cấu hình
thực tế của toàn bộ ba seed kiến trúc đó. Không trộn batch/config khác nhau trong bảng seed.
Nguồn: [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html).

## 4. Kiểm tra một run nhỏ trước

```powershell
python -m scripts.train_transformer --architecture bert --seed 42 --smoke --device auto
```

Smoke dùng **64 train, 32 validation, 1 epoch**. Nó kiểm tokenization, loss, optimizer,
sigmoid, lưu checkpoint, metrics và hash. Nó có thể thiếu các nhãn hiếm, không đủ dữ liệu
để kết luận chất lượng. Folder `smoke/` tách khỏi `full/` và không được chọn cho demo cuối.

## 5. Chạy thực nghiệm theo phân công

Mỗi thành viên chạy đủ ba seed kiến trúc của mình, kiểm từng run và đọc lỗi.

```powershell
# Quốc Khánh
python -m scripts.run_transformer_seeds --architectures bert --seeds 42 123 2026 --device auto --resume
# Đức Trí
python -m scripts.run_transformer_seeds --architectures roberta --seeds 42 123 2026 --device auto --resume
# Nhật Huy
python -m scripts.run_transformer_seeds --architectures distilbert --seeds 42 123 2026 --device auto --resume
```

Chạy toàn bộ tuần tự trên một máy:

```powershell
python -m scripts.run_transformer_seeds --device auto --resume
```

- `--resume` bỏ qua **run đã hoàn tất** đúng cấu hình và hash; không resume optimizer
  giữa epoch. Run bị ngắt cần chạy lại riêng seed với `--overwrite`.
- `--overwrite` chỉ cho phép thay artifacts trong folder kiến trúc/seed/variant được chỉ định.
- `--revision <SHA>` khóa một revision cụ thể; mặc định `main` được resolve SHA và lưu lại.
- Chỉ đọc **train + validation** khi huấn luyện. Không xem nhãn test để chọn epoch hoặc ngưỡng.
- Sau mỗi epoch, chọn checkpoint có **Macro-F1 validation @0,5** cao nhất. Hòa giữ epoch sớm hơn.
- Warning về classification head mới được khởi tạo là bình thường: head này cần học GoEmotions.

## 6. Artifact bàn giao

Ví dụ run C1 seed 42:

```text
data/processed/transformers/bert/seed_42/full/standard/
  checkpoint/                 weights + config + tokenizer của epoch được chọn
  label_mapping.json          28 nhãn, label2id, id2label
  validation_scores.npz       ids, scores N×28, label_names
  validation_metrics.json     metrics @0.5 của checkpoint được chọn
  run_metadata.json           config, SHA, seed, dữ liệu, môi trường, history, hashes
```

Điểm được lưu đúng ID thứ tự validation, nhưng khi ghép mô hình vẫn ghép theo ID.
`completed=true` chỉ được ghi sau khi artifacts đủ; hash bắt lỗi thay model,
trộn score hoặc mapping. Full phải đủ **43.410 train / 5.426 validation**.
Checkpoint lớn nằm trong `.gitignore`; không tự đưa weights lên GitHub.

## 7. Mean ± std và chọn mô hình cho demo

```powershell
python -m scripts.select_best_transformer --seeds 42 123 2026
```

Script từ chối thiếu kiến trúc/seed, run smoke, config khác nhau hoặc hash sai.
Tính mean và **sample standard deviation, `ddof=1`**, gồm Macro/Micro-F1,
Precision/Recall micro và macro, Hamming Loss. Ba kiến trúc dùng cùng tập seed.

1. Chọn **kiến trúc** theo mean Macro-F1 validation @0,5 cao nhất.
2. Nếu hòa, ưu tiên sample std nhỏ hơn; nếu tiếp tục hòa, ít tham số hơn.
3. Trong kiến trúc thắng, chọn checkpoint seed có Macro-F1 validation cao nhất;
   hòa chọn seed nhỏ hơn. Demo này dùng một checkpoint, không phải ensemble.

Kết quả lưu `seed_summary.json`, `seed_summary.md`, `selected_model.json`.
Không chọn lại kiến trúc hoặc seed theo test. Phần root đánh giá chung khóa ngưỡng
và cấu hình trước khi mở test; bảng test cần đủ từng seed và mean±std cuối.

## 8. Nâng cao trên C

### Class weighting

Thêm `--weighted` để BCE dùng `pos_weight[j] = N_negative[j]/N_positive[j]`
**chỉ từ train**. Lưu ở `full/weighted/`, giữ `standard` làm đối chứng.
So sánh cùng seed/config; báo F1 nhãn hiếm trước/sau và đánh đổi precision/recall.
Weighting có thể làm tăng false positives; không mặc định nó luôn tốt hơn.

```powershell
python -m scripts.train_transformer --architecture bert --seed 42 --weighted --device auto
```

### Ngưỡng riêng từng nhãn

Dùng scores validation của **đúng checkpoint/seed** để chọn ngưỡng. Có thể dùng
module chung `src/baseline.py` (`tune_thresholds`, `tune_global_threshold`) và
protocol đánh giá chung của dự án. Không dùng test để tune. Để app dùng ngưỡng
khác 0,5, JSON phải gồm `thresholds` (28 số), `label_names` và
`run_metadata_sha256` của đúng model được chọn. Không chuyển threshold giữa seeds.

Nếu chỉ chạy weighting 1 seed, ghi là ablation 1 seed, không ghi mean±std.
Ba kiến trúc chính vẫn phải đủ ít nhất ba seed theo yêu cầu cô.

## 9. Demo D

```powershell
python app.py --device auto
# Nếu có file ngưỡng đúng model đã chọn:
python app.py --device auto --thresholds path/to/demo_thresholds.json
```

Mở `http://127.0.0.1:7860`. App load checkpoint C **full** trong selection,
kiểm hash và mapping, hiển thị nhiều nhãn + điểm/ngưỡng đủ 28 nhãn. Có xử lý câu rỗng,
câu quá dài, cắt token và không có nhãn vượt ngưỡng. App không fallback sang baseline A.

Ví dụ người dùng nhập “Thank you so much! I am really happy with your help.”
Các nhãn hiển thị là **kết quả tính thật của model**, không gán sẵn joy/gratitude
cho ví dụ này. Khi viết báo cáo, chụp kết quả thực tế và lưu seed/checkpoint/ngưỡng.
App bind localhost và không tự công khai qua Gradio share.

### Kiểm giao diện thật và lưu ảnh

Sau khi đủ C full, chọn checkpoint và chạy `app.py`, mở terminal thứ hai tại gốc
repo. Trên máy hiện tại dùng Node/Playwright đã có sẵn; không cần cài thêm:

```powershell
& 'C:\Users\dzyuu\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' tools/capture_demo.js --url http://127.0.0.1:7860 --output reports/demo_ui
```

Script mở app thật, nhập mẫu tiếng Anh, bấm **Nhận diện cảm xúc**, chờ thông báo
dự đoán và bảng **Điểm của 28 nhãn** cùng bốn tiêu đề cột. Nó kiểm điểm/ngưỡng các
hàng đang có trong DOM, rồi lưu `reports/demo_ui/demo_ui.png` và `evidence.json`
khi đạt. Số hàng đã kiểm được ghi rõ; bảng có cuộn/virtualize không được tự xem
là đã đối chiếu đủ 28 điểm. Script không khởi động app, tải model hoặc tạo dự đoán giả.

Máy khác có thể truyền `--playwright-path` thư mục Playwright đã cài và
`--browser-path` executable Chromium có sẵn; `--help` in các lựa chọn.
`interface_status=PASS` chỉ chứng minh tương tác/giao diện. Kiểm tính nhất quán
scores với checkpoint vẫn dùng `python -m scripts.verify_demo --device cuda`
riêng; cả hai phép kiểm không thay thế chỉ số test. Nếu công cụ báo lỗi, không
có evidence mới: đọc thời điểm/hash ảnh trong evidence để tránh nhầm bản cũ.

## 10. Kiểm thử và thứ tự đọc code

```powershell
python -m unittest tests.test_transformer -v
```

Tests kiểm BCE đa nhãn float, weighting theo count, full/smoke, mean/sample std,
config/seed, thay weights và ngưỡng khác model. Fixture chỉ là dữ liệu kiểm thử,
không có điểm thực nghiệm. Không tải model trong unit tests.

Thứ tự đọc: `src/neural.py` → `scripts/train_transformer.py` →
`scripts/run_transformer_seeds.py` → `scripts/select_best_transformer.py` → `app.py`.
Notebook `notebooks/transformers.ipynb` giải thích và đọc artifacts, không tự chạy
chín lần huấn luyện khi mở notebook.
