# Notebook theo thứ tự

| Notebook | Bạn học gì | Điều kiện chạy |
|---|---|---|
| [01_eda](../notebooks/01_eda.ipynb) | Split, 28 nhãn, mất cân bằng, multi-hot, ví dụ dữ liệu | Extra `eda,notebooks`; tải dataset lần đầu |
| [02_baseline](../notebooks/02_baseline.ipynb) | Model A, weighting, ngưỡng và metric | Chạy hai model A/analyze để đọc đủ kết quả |
| [03_zero_shot](../notebooks/03_zero_shot.ipynb) | NLI, candidate labels, scores và trạng thái B | Đọc/in lệnh; chạy B ở terminal để có artifact |
| [04_transformers](../notebooks/04_transformers.ipynb) | BCE, seed/checkpoint/protocol và trạng thái ba C | Đọc/in lệnh; train C ở terminal |
| [05_distilbert_huy](../notebooks/05_distilbert_huy.ipynb) | Nghiên cứu C3 riêng của Nhật Huy | Artifact riêng của trainer C3 |
| [06_eda_extra](../notebooks/06_eda_extra.ipynb) | EDA bổ sung và các câu hỏi dữ liệu | Sau 01_eda; extra EDA |

Mở Jupyter từ gốc repo và chọn đúng kernel như [GETTING_STARTED](GETTING_STARTED.md).
Bản Git chỉ giữ source/giải thích; output tạo trên máy người chạy.

B/C là notebook đọc artifact và in lệnh, không tự train full khi bấm Run All.
Các trạng thái thiếu artifact có thể xuất hiện trên bản clone mới; đọc hướng dẫn
để tạo dữ liệu/model trước. Notebook EDA cần Internet ở lần tải dataset đầu.
Model nặng và output nằm trong các thư mục Git bỏ qua.

Các runner EDA/C3 và builder baseline với `--execute` lưu bản có output trong
`reports/notebooks/`, giữ notebook source sạch. Nếu chạy trực tiếp bằng Jupyter,
xóa output và execution count trước khi commit notebook.
