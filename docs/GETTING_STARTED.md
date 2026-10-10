# Cài đặt và chạy lần đầu

## 1. Chọn Python

Dùng Python 3.12 hoặc 3.13. Đứng tại thư mục có `pyproject.toml`.

```bash
python --version
python -m venv .venv
```

Windows PowerShell: `./.venv/Scripts/Activate.ps1`.
CMD: `.venv\Scripts\activate.bat`.
Linux/macOS: `source .venv/bin/activate`.

Nếu PowerShell chặn kích hoạt, vẫn chạy trực tiếp:
`./.venv/Scripts/python.exe -m pip install -e .`.
Kiểm `python -c "import sys; print(sys.executable)"` để biết đang dùng đúng môi trường.

## 2. Baseline A: không cần GPU

```bash
python -m pip install -e .
python -m examples.baseline_from_scratch
python -m scripts.baseline.run_baseline
python -m scripts.baseline.predict_baseline --text "I really appreciate your help."
```

`pip install -e .` cài editable: bạn sửa source trong bản clone và chạy lại.
Thư mục repo chứa cấu hình/dữ liệu cache; giữ bản clone khi sử dụng.
Thay thế khi chỉ muốn cài dependencies: `python -m pip install -r requirements-baseline.txt`.

Script tự tải snapshot GoEmotions ở lần đầu và kiểm SHA; cần Internet cho lần tải đó.
Chạy cả biến thể để so sánh:

```bash
python -m scripts.baseline.run_baseline --variant balanced
python -m scripts.baseline.analyze_baseline
python -m scripts.baseline.predict_baseline --variant balanced --threshold tuned --text "I am happy and grateful."
```

`--threshold tuned` cần kết quả chọn ngưỡng của đúng biến thể từ bước analyze.
[Cách baseline hoạt động](BASELINE.md).

## 3. EDA và notebook

```bash
python -m pip install -e ".[eda,notebooks]"
python -m jupyterlab
```

Chọn kernel dùng đúng `.venv`; mở `notebooks/01_eda.ipynb`.
Có thể đăng ký kernel:

```bash
python -m ipykernel install --user --name goemotions --display-name "GoEmotions"
```

Mở Jupyter từ gốc repo, chọn kernel **GoEmotions**.
Notebook trong Git không giữ output; chạy các ô để tạo output ở máy mình.
[Thứ tự notebook và bước nào có tải dữ liệu](NOTEBOOKS.md).

## 4. Zero-shot, Transformer và Gradio

Máy chỉ có CPU:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[transformers]"
```

Máy NVIDIA: chọn lệnh wheel theo hệ điều hành/GPU tại
[PyTorch Get Started](https://pytorch.org/get-started/locally/), cài trước extra.
Không dùng nguyên lệnh CUDA của máy khác nếu GPU/driver khác.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m scripts.transformers.train_transformer --help
python -m scripts.zero_shot.run_zero_shot --help
```

Các lệnh smoke có tải checkpoint và chạy một phần dữ liệu:

```bash
python -m scripts.transformers.train_transformer --architecture distilbert --seed 42 --smoke --device cpu
python -m scripts.zero_shot.run_zero_shot --smoke --device cpu --dtype float32
```

Smoke giúp kiểm luồng, không dùng làm kết quả cuối hoặc chọn demo.
Chạy full trên CPU có thể lâu; xem [toàn bộ thực nghiệm](EXPERIMENTS.md).
[Phiên bản khóa đã dùng](../requirements-models-lock.txt) ghi môi trường nghiên cứu trước;
wheel CPU/CUDA vẫn cần phù hợp máy hiện tại.

## 5. Kiểm trước khi thực nghiệm

```bash
python -m pip install -e ".[dev]"
python tools/check_repository.py
python -m unittest discover -s tests -v
```

Bộ tests không cần checkpoint hoặc dữ liệu GoEmotions.
Các kiểm Torch cần đã cài extra Transformer.
