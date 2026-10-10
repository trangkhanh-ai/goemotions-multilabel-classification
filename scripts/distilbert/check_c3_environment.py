"""Kiểm khả năng dùng CUDA bằng phép tính thật, ghi phiên bản môi trường C3."""
import platform
import subprocess
import importlib.metadata
from src.models.distilbert_study import ROOT, write_json
import torch


def main():
    packages = ["torch", "transformers", "numpy", "pandas", "pyarrow", "scikit-learn", "streamlit", "tokenizers"]
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in packages},
              "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda}
    if report["cuda_available"]:
        gpu = torch.cuda.get_device_properties(0)
        a = torch.eye(16, device="cuda")
        torch.testing.assert_close(a @ a, a)
        report.update(gpu=gpu.name, vram_mib=gpu.total_memory / 2**20, cuda_matmul="passed")
    else:
        report["cuda_matmul"] = "not run"
    folder = ROOT / "reports/c3_distilbert"
    write_json(folder / "environment.json", report)
    lock = subprocess.check_output([__import__('sys').executable, '-m', 'pip', 'freeze'], text=True)
    (folder / "environment-lock.txt").write_text(lock, encoding="utf-8")
    print(report)
    if not report["cuda_available"]:
        raise SystemExit("Chưa có CUDA khả dụng trong interpreter hiện tại")


if __name__ == "__main__":
    main()
