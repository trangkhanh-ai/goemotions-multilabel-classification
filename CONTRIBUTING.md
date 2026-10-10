# Đóng góp vào mã nguồn

1. Tạo nhánh cho phần cần sửa; dùng mục tương ứng trong `src/` và `scripts/`.
2. Giữ hàm nhỏ, tên rõ và giải thích những quyết định NLP bằng comment tiếng Việt.
3. Giữ cùng thứ tự 28 nhãn; fit trên train, chọn cấu hình/ngưỡng trên validation.
4. Khi sửa đường dẫn hoặc lệnh, cập nhật hướng dẫn và notebook liên quan.
5. Kiểm trước khi gửi PR:

```bash
python tools/check_repository.py
python -m unittest discover -s tests -v
```

Unit test dùng dữ liệu tự tạo và mock; không tự tải checkpoint hoặc train full.
Khi thay logic mô hình, ghi phép kiểm cần thêm và kết quả đã đo.

Notebook trong Git giữ code/giải thích, không giữ output và execution count.
Báo cáo học phần, PDF/Word, trọng số và dữ liệu tải về nằm ngoài nhánh source.
Giữ tên tác giả/đóng góp của đồng đội khi sửa phần họ đã bàn giao.
