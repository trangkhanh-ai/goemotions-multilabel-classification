"""Một đường dẫn gốc dùng chung, độc lập với thư mục đang đứng.

Dự án chạy từ bản clone hoặc cài editable bằng pip install -e .
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
