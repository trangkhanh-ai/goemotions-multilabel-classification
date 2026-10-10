"""Chạy toàn bộ thí nghiệm theo thứ tự, một tác vụ GPU tại một thời điểm.

python -m scripts.pipeline.complete_project --device cuda
Không cần chạy lại A đã có full train/val. C --resume chỉ bỏ qua run đã kiểm hash.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from src.datasets.goemotions import sha256
from src.evaluation.protocols import read_json, save_json, freeze_run, validate_protocol
from src.models.transformer import ARCHITECTURES, run_folder, configure_console

from src.paths import ROOT
SEEDS = (42, 123, 2026)


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-training", action="store_true", help="Chỉ khi đã đủ C full và B validation")
    parser.add_argument("--restart-incomplete", action="store_true", help="Chạy lại run C dở; giữ các run hoàn tất đã kiểm hash")
    args = parser.parse_args()
    # Yêu cầu chỉ sống cùng thread/process này; không thay power plan của máy.
    if os.name == "nt":
        import ctypes
        state = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        print("Giữ máy thức trong thời gian thực nghiệm." if state else
              "Không đặt được yêu cầu giữ máy thức; kiểm chế độ ngủ của máy.", flush=True)
    output = ROOT / "reports/execution"
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    with (output / "full_pipeline.log").open("a", encoding="utf-8") as log:
        def run(module, *arguments):
            command = [sys.executable, "-m", module, *map(str, arguments)]
            header = f"\n{datetime.now(timezone.utc).isoformat()} | {' '.join(command[1:])}\n"
            print(header, end="", flush=True)
            log.write(header)
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(f"Bước thất bại: {module}. Giữ artifacts để kiểm/tiếp tục.")

        if not args.skip_training:
            flags = ["--device", args.device, "--resume"]
            if args.restart_incomplete:
                flags.append("--overwrite")
            run("scripts.transformers.run_transformer_seeds", *flags)
            # CPU dùng float32; GPU dùng float16 để phù hợp 8 GB, được ghi trong metadata.
            import torch
            dtype = "float16" if args.device != "cpu" and torch.cuda.is_available() else "float32"
            run("scripts.zero_shot.run_zero_shot", "--device", args.device, "--batch-size", 16, "--dtype", dtype)

        # Chọn best C và khóa TẤT CẢ cấu hình trước bước test đầu tiên.
        run("scripts.transformers.select_best_transformer")
        labels = read_json(ROOT / "data/labels.json")
        folders = [ROOT / "data/processed/zero_shot/full"]
        folders += [run_folder(ROOT, architecture, seed)
                    for architecture in ARCHITECTURES for seed in SEEDS]
        for folder in folders:
            frozen = folder / "final_protocol.json"
            if frozen.exists():
                validate_protocol(folder, read_json(frozen), labels)
            else:
                save_json(frozen, freeze_run(ROOT, folder))
        a_protocol = ROOT / "data/processed/baseline/final_protocol.json"
        if not a_protocol.exists():
            run("scripts.baseline.freeze_baseline")
        # Lưu dấu vết đóng băng toàn bộ bảng thí nghiệm, không dùng test đổi người thắng.
        protocols = [a_protocol, *[folder / "final_protocol.json" for folder in folders]]
        save_json(output / "frozen_protocols.json", {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_data": "validation", "seeds": SEEDS,
            "protocols": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in protocols]})
        run("scripts.baseline.evaluate_baseline_test")
        b_folder = folders[0]
        dtype = read_json(b_folder / "run_metadata.json").get("dtype", "float32")
        run("scripts.zero_shot.run_zero_shot", "--split", "test", "--protocol", b_folder / "final_protocol.json",
            "--device", args.device, "--batch-size", 16, "--dtype", dtype)
        for folder in folders[1:]:
            run("scripts.transformers.evaluate_transformer_test", "--run-dir", folder, "--device", args.device)
        run("scripts.analysis.summarize_project", "--require-complete")
        run("scripts.analysis.export_project_analysis")
        run("scripts.analysis.export_run_metadata")
        run("scripts.analysis.analyze_project_errors", "--split", "validation")
        run("scripts.analysis.analyze_project_errors", "--split", "test")
    print("Đã hoàn tất thí nghiệm. Xem reports/project_results/RESULTS.md và full_pipeline.log.")


if __name__ == "__main__":
    main()
