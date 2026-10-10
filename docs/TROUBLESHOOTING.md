# Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `No module named src` hoặc scripts | Đứng tại thư mục có pyproject.toml; dùng `python -m ...` và cài `pip install -e .` |
| Notebook import sai môi trường | Kiểm `sys.executable`; chọn kernel của venv |
| PowerShell chặn Activate.ps1 | Dùng trực tiếp `.venv/Scripts/python.exe`; không cần đổi policy toàn máy |
| Không có model baseline | Chạy `scripts.baseline.run_baseline`; smoke không tạo model full |
| Thiếu ngưỡng tuned | Train đúng variant và chạy analyze_baseline |
| Thiếu artifact B/C trong notebook | Chạy các lệnh terminal ở EXPERIMENTS; notebook B/C chủ yếu đọc trạng thái |
| App báo chưa có best C | Chạy C đủ seed và select_best_transformer; không có weights trong Git |
| CUDA unavailable | Cài wheel từ trang PyTorch phù hợp máy; có thể chọn CPU |
| Lỗi hash/schema | Kiểm nguồn/đúng revision và file đích được báo; đừng bỏ kiểm hash |
| Run dở | Giữ log, xác định đúng folder; resume chỉ bỏ qua run complete |
| Không nhãn vượt ngưỡng | Đây là kết quả có thể xảy ra; không tự ép neutral |

Khi báo lỗi, gửi phiên bản Python/thư viện, hệ điều hành, lệnh và traceback.
[Hướng dẫn cài](GETTING_STARTED.md) · [Pipeline](EXPERIMENTS.md).
