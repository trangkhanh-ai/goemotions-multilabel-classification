# Bản đồ mã nguồn

## Đọc từ dữ liệu đến dự đoán

| Thứ tự | Mục | Trách nhiệm |
|---|---|---|
| 1 | [src/datasets](../src/datasets/) | Tải GoEmotions, revision/hash, schema, multi-hot |
| 2 | [src/models](../src/models/) | Logic từng họ mô hình |
| 3 | [src/evaluation](../src/evaluation/) | Kiểm score, metric, ngưỡng và protocol |
| 4 | [scripts](../scripts/) | Lệnh điều phối; không chứa dữ liệu/model |
| 5 | [examples](../examples/) | Ví dụ nhỏ tự tạo |
| 6 | [tests](../tests/) | Các tình huống đúng/sai dùng dữ liệu giả |

`src/paths.py` xác định gốc bản clone cho tất cả script.
`src/` chứa hàm dùng lại; `scripts/` nhận tham số, gọi hàm và lưu output.

## Lệnh theo mục

| Phần | Lệnh chính |
|---|---|
| EDA | `python -m scripts.data.run_eda` |
| Train A | `python -m scripts.baseline.run_baseline` |
| Dự đoán A | `python -m scripts.baseline.predict_baseline --text "Thank you!"` |
| Tune A | `python -m scripts.baseline.analyze_baseline` |
| Khóa/test A | `python -m scripts.baseline.freeze_baseline` → `evaluate_baseline_test` cùng mục |
| Zero-shot B | `python -m scripts.zero_shot.run_zero_shot` |
| Train một C | `python -m scripts.transformers.train_transformer --architecture bert --seed 42` |
| Ba kiến trúc × ba seed | `python -m scripts.transformers.run_transformer_seeds --device cuda` |
| Chọn C bằng validation | `python -m scripts.transformers.select_best_transformer` |
| Ghép pipeline | `python -m scripts.pipeline.complete_project --device cuda` |
| Tổng hợp | `python -m scripts.analysis.summarize_project --require-complete` |
| Phân tích lỗi | `python -m scripts.analysis.analyze_project_errors --split test` |

Builder notebook giữ tại đúng mục cho phần tương ứng.
Các lệnh học phần đã đổi đường dẫn trong nhánh này; dùng bảng mới và `--help`.
[Quy trình chạy đầy đủ](EXPERIMENTS.md).

## Nghiên cứu C3 riêng

Nhóm giữ source của Nhật Huy tại `scripts/distilbert/`,
`src/models/distilbert_study.py` và `configs/distilbert_*.json`.
Đây là một trainer/protocol riêng; xem [hướng dẫn](DISTILBERT_STUDY.md).

## Tiện ích

`tools/check_repository.py` kiểm tệp đăng Git, syntax, notebook sạch và link local.
Các tiện ích còn lại giúp chụp UI, kiểm metadata hoặc giữ Windows thức theo PID;
chúng không tự thay thế huấn luyện và đánh giá.
