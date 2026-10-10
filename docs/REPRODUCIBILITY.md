# Giữ thực nghiệm tái hiện được

1. **Dữ liệu:** snapshot `add492243ff905527e67aeb8b80c082af02207c3`;
   schema, số dòng và SHA-256 kiểm trong `src/datasets/goemotions.py`.
2. **Mapping:** giữ 28 nhãn trong `data/labels.json`, multi-hot theo đúng thứ tự.
3. **Train:** chỉ fit train; ghi hyperparameter, seed, tokenizer/model revision.
4. **Validation:** chọn checkpoint/cấu hình/ngưỡng; ghi luật hòa điểm.
5. **Freeze:** model, mapping và hash phải khớp trước khi mở test.
6. **Test:** scores theo ID và cùng split; không chọn lại cấu hình bằng test.
7. **Tổng hợp:** C đủ ba seed mới có mean ± sample std (`ddof=1`);
   A/B một run không biến thành thí nghiệm đa seed.

Các script ghi metadata và kiểm hash trong output cục bộ.
Để xuất metadata sau thực nghiệm:
`python -m scripts.analysis.export_run_metadata`.

Để tổng hợp số:
`python -m scripts.analysis.summarize_project --require-complete`.
Thiếu hoặc sai hash là lỗi cần xử lý, không tự đổi thành số 0.

Nguồn baseline cố định `random_state=42`. Với Torch, seed không bảo đảm mọi
kernel CUDA giống từng bit; báo điều kiện chạy và biến động giữa seed.
Ba seed không thay tìm kiếm siêu tham số hoặc kiểm định ý nghĩa thống kê.

Nhánh source không chứa kết quả benchmark hoặc checkpoint đã chạy trước.
Người clone tạo output bằng scripts hoặc nhận artifact riêng từ nhóm.
`requirements-models-lock.txt` lưu phiên bản môi trường mô hình trước;
thiết bị/wheel phù hợp vẫn cần chọn trên máy chạy.
