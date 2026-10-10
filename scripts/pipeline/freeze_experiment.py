"""python -m scripts.pipeline.freeze_experiment --run-dir data/processed/zero_shot/full"""
import argparse
from pathlib import Path
from src.evaluation.protocols import freeze_run, save_json

from src.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description="Khóa ngưỡng B/C chỉ từ validation")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--threshold-mode", choices=("fixed", "global", "tuned"), default="tuned")
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    output = folder / "final_protocol.json"
    if output.exists():
        raise FileExistsError("Protocol đã khóa; giữ nguyên để đánh giá test")
    save_json(output, freeze_run(ROOT, folder, args.threshold_mode))
    print(f"Đã khóa ba cấu hình từ validation: {output}")


if __name__ == "__main__":
    main()
