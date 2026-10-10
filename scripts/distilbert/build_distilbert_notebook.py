"""Tương thích lệnh cũ: tạo notebook C3 full; số liệu lấy từ artifact thật."""
import runpy

if __name__ == "__main__":
    runpy.run_module("scripts.distilbert.build_c3_full_notebook", run_name="__main__")
