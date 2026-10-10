"""Chạy tuần tự ba kiến trúc × ba seed; --resume không chạy lại run hợp lệ."""
import argparse
import subprocess
import sys

from src.models.transformer import ARCHITECTURES, configure_console


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.architectures)) != len(args.architectures):
        parser.error("Kiến trúc và seed không được lặp")
    for architecture in args.architectures:
        for seed in args.seeds:
            command = [sys.executable, "-m", "scripts.transformers.train_transformer", "--architecture", architecture,
                       "--seed", str(seed), "--device", args.device, "--batch-size", str(args.batch_size),
                       "--gradient-accumulation", str(args.gradient_accumulation),
                       "--max-length", str(args.max_length)]
            command += ["--" + flag for flag in ("weighted", "smoke", "resume", "overwrite")
                        if getattr(args, flag)]
            print(f"Bắt đầu {architecture}, seed {seed}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
