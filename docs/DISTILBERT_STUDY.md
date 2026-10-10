# Nghiên cứu DistilBERT riêng của Nhật Huy

Source riêng được giữ để ghi nhận bàn giao và tránh trộn protocol:
`src/models/distilbert_study.py`, `src/evaluation/distilbert_thresholds.py`,
`scripts/distilbert/`, `configs/distilbert_*.json` và notebook 05.

Cài wheel PyTorch phù hợp, rồi:
`python -m pip install -e ".[study,eda,notebooks]"`.

Các lệnh tham khảo:

```bash
python -m scripts.distilbert.train_distilbert --help
python -m scripts.distilbert.predict_distilbert --help
```

Chọn config và thư mục output theo `--help`/JSON; các config pilot/full là
thiết kế riêng. `scripts/transformers/` vẫn là trainer dùng chung cho ba C.

`scripts/distilbert/run_c3_full.py` chạy các run full trực tiếp, không có
`--help`; đọc source và các config trước khi chạy lệnh đó.

`app.py` là Gradio của C thắng theo bộ chính.
`app_distilbert_huy.py` là Streamlit riêng, cần bundle C3 được tạo/nhận riêng.
Đọc `--help` của predictor/config trước khi nạp; bản source không có bundle.

Kết quả khác trainer, thiết bị hoặc protocol được đọc riêng:
không cộng thành sáu seed, không đổi điểm validation thành test.
