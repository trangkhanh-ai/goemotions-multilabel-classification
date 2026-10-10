# Chạy toàn bộ A/B/C/D

## 1. Pipeline và điều kiện

Đã cài extra `eda,transformers,dev`, chọn wheel CPU/CUDA đúng máy.
Bản clone mới chưa có trọng số hoặc kết quả full.
Chạy từ gốc repo; một tác vụ GPU tại một thời điểm.

```bash
python -m scripts.baseline.run_baseline
python -m scripts.baseline.run_baseline --variant balanced
python -m scripts.baseline.analyze_baseline
python -m scripts.pipeline.complete_project --device cuda
```

Máy CPU đổi `--device cpu`. Chi phí chạy full phụ thuộc máy.
Pipeline: ba C × ba seed → B validation → chọn C → khóa protocol A/B/C →
test → tổng hợp → metadata/ba nhóm lỗi.
Lệnh pipeline cần đã tạo hai baseline full và phân tích ngưỡng A.

## 2. Chạy từng nhánh

### A

[Baseline](BASELINE.md) có các bước train/tune/freeze/test/predict đầy đủ.

### B

```bash
python -m scripts.zero_shot.run_zero_shot --device cpu --dtype float32
```

CUDA dùng `--device cuda --dtype float16 --batch-size 16`.
B giữ trọng số BART-large-MNLI, dùng hypothesis template và 28 candidate labels.
Scores phải về thứ tự nhãn chuẩn, không theo thứ tự pipeline sắp xếp giảm dần.
Nếu dùng validation chọn ngưỡng thì B là zero-shot về trọng số, có hiệu chỉnh nhãn đích.
NLI score chưa mặc nhiên là xác suất cảm xúc được hiệu chuẩn.

### C

```bash
python -m scripts.transformers.train_transformer --architecture bert --seed 42 --device cuda
python -m scripts.transformers.run_transformer_seeds --device cuda --resume
```

| Kiến trúc | Checkpoint | LR | Epoch tối đa |
|---|---|---:|---:|
| BERT | google-bert/bert-base-cased | 5e-5 | 4 |
| RoBERTa | FacebookAI/roberta-base | 2e-5 | 3 |
| DistilBERT | distilbert/distilbert-base-uncased | 2e-5 | 3 |

Cùng batch 16, max length 128, gradient accumulation 1; seed 42/123/2026.
Logits đi thẳng vào BCEWithLogitsLoss; sigmoid dùng ở suy luận.
Checkpoint mỗi seed chọn bằng validation Macro-F1 @0,5, hòa chọn epoch sớm hơn.
BERT tham khảo GoEmotions; C2/C3 là cấu hình nhóm. Ba seed chưa là tìm kiếm LR;
khác LR/epoch/tokenizer nên chưa cô lập ảnh hưởng kiến trúc.

`--resume` bỏ qua run hoàn tất đã kiểm hash, không phục hồi optimizer của run dở.
`--overwrite` cho phép chạy lại folder được chỉ định; kiểm đúng run trước khi dùng.
Smoke lưu riêng, không đưa vào bảng full.

## 3. Validation, khóa protocol và test

Chọn C bằng mean validation Macro-F1 @0,5 của ba seed cho mỗi kiến trúc.
Ngưỡng fixed/global/tuned lấy từ validation của chính run đó.
Protocol lưu mapping, revision/checksum và quyết định đã khóa.

Tuned-validation đo trên tập đã quét ngưỡng, có thể lạc quan.
Test giữ model/ngưỡng đã khóa; không retune để lấy hàng đẹp hơn.
Xem [REPRODUCIBILITY](REPRODUCIBILITY.md).

Đã đủ train/validation thì có thể:
`python -m scripts.pipeline.complete_project --device cuda --skip-training`.
Chỉ dùng khi các run full và metadata/hash hợp lệ.

## 4. Đầu ra cục bộ

| Đường dẫn | Nội dung |
|---|---|
| `data/processed/baseline/` | Model A, score và ngưỡng |
| `data/processed/zero_shot/` | Scores/checkpoint batch B |
| `data/processed/transformers/` | Checkpoint C, metadata, selected_model |
| `reports/project_results/` | CSV từng run, mean/std, per-label |
| `reports/errors_test_standard_fixed/` | Ba nhóm lỗi với ID chung |
| `reports/reproducibility/` | Metadata nhỏ từ artifact nguồn |

Những đường dẫn này sinh sau khi chạy và không nằm trong bản Git source.
Không gộp nghiên cứu C3 riêng của Huy thành sáu seed của C3 chính.
Báo support và nhãn giảm F1, không hứa cả năm nhãn hiếm đều tăng.

## 5. Demo D

Sau khi đủ C và đã chọn model:

```bash
python -m scripts.transformers.verify_demo --device auto
python app.py --device auto
```

Demo dùng model/selection ở máy, không tải một model giả để thay C đã chọn.
Ngưỡng mặc định 0,5; ngưỡng riêng cần `--thresholds` trỏ đúng protocol run.
Dùng câu tiếng Anh; không có nhãn vượt ngưỡng thì không tự gán neutral.
