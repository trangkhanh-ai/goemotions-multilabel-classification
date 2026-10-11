# Đức Trí — C2 RoBERTa: học, chạy và báo cáo

Phần của **Đức Trí (GitHub: Ductri2006)** là RoBERTa C2. Notebook riêng:
[notebooks/roberta_duc_tri.ipynb](../notebooks/roberta_duc_tri.ipynb).

## Bạn cần hoàn thành gì?

1. Hiểu dữ liệu và mã hóa multi-hot 28 nhãn.
2. Kiểm tokenizer, tensor đầu vào và head 28 logits của RoBERTa.
3. Fine-tune bằng trainer chung; kiểm ba seed **42, 123, 2026**.
4. Báo từng seed và mean ± sample std (`ddof=1`), đủ Macro/Micro-F1, Precision/Recall và Hamming Loss.
5. Chọn ngưỡng trên validation của đúng run, khóa protocol trước test.
6. Phân tích lỗi, nhãn hiếm, đối chiếu cùng ID với C1/C3; viết phần phương pháp/kết quả/hạn chế C2.

Zero-shot B là phần hỗ trợ/bàn giao Duy. BERT C1 thuộc Khánh; DistilBERT C3 và demo thuộc Huy.
Bạn dùng code chung của nhóm để metrics/artifacts ghép được, không cần viết lại A/B/C1/C3/D.

## Hai loại output trong notebook

**Output có sẵn:** đọc snapshot bảng số liệu/metadata nhóm, tính lại mean/std, vẽ bốn hình,
bảng nhãn hiếm và ba case lỗi. Các nguồn và commit xuất được ghi trong notebook.
Đây là hồ sơ nhóm đã chạy, không phải một lượt train mới do notebook này thực hiện.

**Output khi bạn bật chạy:** tokenizer, forward model thật, smoke, full ba seed,
chấm local scores, khóa protocol, test, lỗi validation và suy luận câu mới.
Mọi cờ mặc định `False`; Run All không tự tải/train model. Không có output dự đoán giả
ở các ô model chưa chạy. Các tensor số học minh họa được ghi riêng.

Notebook/giải thích này là phần bổ sung của Đức Trí. Trainer, metric và các kết quả
chung giữ nguyên nguồn/tác giả lịch sử; không đổi attribution của run người khác.

## Cách mở và thứ tự học

1. Mở toàn bộ repo bằng VS Code, mở `notebooks/roberta_duc_tri.ipynb`.
2. Chọn kernel của môi trường mô hình. Để chỉ xem báo cáo cần NumPy/Pandas/Matplotlib;
   bài minh họa backward dùng thêm PyTorch. `requirements-notebook.txt` phục vụ notebook.
3. Đọc Markdown rồi chạy ô code ngay dưới bằng **Shift+Enter**.
4. Đọc mục 1–9: dữ liệu, nhãn, loss, learning curves, kết quả, nhãn hiếm và lỗi.
5. Sau khi có code/môi trường/checkpoint, thực hành mục 10 theo thứ tự dưới đây.

## Chạy lại C2

Các thư viện/môi trường: xem [CHAY_THUC_NGHIEM.md](CHAY_THUC_NGHIEM.md),
[TRANSFORMERS.md](TRANSFORMERS.md) và các requirements của repo. Không cài đè môi trường đang train.
Các weights/scores lớn không được Git theo dõi; nhận từ máy chạy hoặc tạo lại bằng full run.
Notebook tự tìm repo từ thư mục hiện tại/cha; kiểm `PROJECT_ROOT` trước khi bật thao tác ghi.

| Cờ | Mục đích | Điều kiện |
|---|---|---|
| `RUN_TOKENIZER` | Xem token IDs/attention mask thật | Transformers + truy cập/cache tokenizer |
| `RUN_FORWARD` | Kiểm logits/labels/loss của model thật | Đã chạy tokenizer; đủ RAM/VRAM |
| `RUN_SMOKE` | Pilot 64 train/32 validation, 1 epoch | Repo/môi trường đúng; không dùng điểm này như benchmark |
| `RUN_FULL` | Chạy đủ ba seed C2 | Đủ dữ liệu/model, thời gian và tài nguyên |
| `READ_LOCAL_RUNS` | Kiểm hash, ghép ID, tự chấm scores | Ba artifact full hoàn tất trên máy bạn |
| `FREEZE_VALIDATION` | Khóa ba cấu hình ngưỡng từ validation | Scores/model đúng; không ghi đè protocol |
| `RUN_LOCKED_TEST` | Đánh giá test sau khóa | Nhóm đã khóa lựa chọn; có final_protocol.json |
| `ANALYZE_LOCAL_ERRORS` | Bảng FN/FP, ID và heatmap validation | Scores/checkpoint validation đầy đủ |
| `RUN_INFERENCE` | Suy luận câu mới từ C2 fine-tuned | Checkpoint/tokenizer của full run seed 2026 |

Full C2 dùng checkpoint `FacebookAI/roberta-base`, revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`, LR `2e-5`, 3 epoch, tối đa 128 token,
batch 16, accumulation 1; chọn epoch theo validation Macro-F1 @0.5.
Nếu GPU 4 GB thiếu VRAM, có thể batch 4/accumulation 4 để giữ effective batch 16;
ghi đây là cấu hình chạy riêng và dùng thống nhất cả ba seed. Không trộn với run khác config.

Trainer ghi vào `data/processed/transformers/roberta/seed_<seed>/full/standard/`.
`--resume` chỉ bỏ qua run hoàn tất đủ hash/config; không phục hồi optimizer giữa epoch.
Không dùng `--overwrite` trước khi kiểm folder và giữ kết quả cũ cần thiết.

## Điểm cần nói khi báo cáo

- RoBERTa test Macro-F1 fixed **0.4219 ± 0.0088**, tuned **0.4872 ± 0.0156**.
- Micro-F1 tăng, nhưng Precision giảm và Hamming Loss tăng; tuning có đánh đổi.
- Nhãn `grief` vẫn có F1 0; không nói mọi nhãn hiếm đều cải thiện.
- BERT được chọn demo bằng mean Macro-F1 validation @0.5; không đổi tiêu chí bằng test.
- Ba seed không thay tìm kiếm siêu tham số; LR/epoch khác nhau giữa C là giới hạn so sánh.
- Kết quả nhóm có sẵn không tự chứng minh cá nhân đã trực tiếp train; kể đúng đóng góp đã làm.

## Tạo lại notebook và kiểm output

```powershell
python -m scripts.build_roberta_notebook --execute
```

Builder đọc các CSV và metadata nhỏ có sẵn trong repo, sinh lại notebook với output phần
đọc/giải thích. Nó không bật training/suy luận hoặc mở test mới. Nếu chỉ muốn tạo ô chưa chạy,
bỏ `--execute`. Snapshot nhúng giúp bản notebook đã xuất đọc độc lập; khi tự train, đọc local
artifacts ở phần 10, không dùng snapshot thay kết quả của bạn.

## GitHub attribution

Notebook mới được commit qua tài khoản **Ductri2006**. GitHub liên kết tác giả theo email
commit; Contributors của repo thường cập nhật sau khi nhánh được nhập vào nhánh mặc định.
Giữ các commit/tác giả cũ, không đổi tên người chạy các thí nghiệm đã có.
