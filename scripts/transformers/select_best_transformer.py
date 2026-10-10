"""Tạo bảng mean±std và chọn best C theo validation, không nhìn test."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.goemotions import sha256
from src.models.transformer import (ARCHITECTURES, METRIC_NAMES, load_transformer_run, run_folder,
                        summarize_architectures, write_json, configure_console)

from src.paths import ROOT


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--weighted", action="store_true", help="Bảng C weighted riêng, không trộn standard")
    args = parser.parse_args()
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Cần ít nhất 3 seed khác nhau")
    runs = {}
    for architecture in ARCHITECTURES:
        runs[architecture] = []
        for seed in args.seeds:
            folder = run_folder(ROOT, architecture, seed, weighted=args.weighted)
            metadata = load_transformer_run(folder, require_full=True)
            if metadata["seed"] != seed or metadata["architecture"] != architecture:
                raise ValueError("Artifact nằm sai folder seed/kiến trúc")
            metadata["run_dir"] = folder.relative_to(ROOT).as_posix()
            runs[architecture].append(metadata)
    summaries = summarize_architectures(runs)
    winner = summaries[0]["architecture"]
    # Checkpoint minh họa là seed tốt nhất của kiến trúc thắng; không giả làm ensemble.
    best = max(runs[winner], key=lambda r: (r["metrics"]["macro_f1"], -r["seed"]))
    folder = ROOT / best["run_dir"]
    selection = {
        "artifact_version": 1, "method": "C", "smoke": False,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "highest validation mean Macro-F1@0.5; tie: lower sample std, fewer parameters, name",
        "checkpoint_rule": "best validation Macro-F1 seed in winning architecture; smaller seed on tie",
        "architecture": winner, "seed": best["seed"], "run_dir": best["run_dir"],
        "run_metadata_sha256": sha256(folder / "run_metadata.json"),
        "label_names": best["label_names"], "default_threshold": 0.5,
        "variant": "weighted" if args.weighted else "standard", "architecture_summary": summaries,
        "run_sources": [{"architecture": arch, "seed": run["seed"], "run_dir": run["run_dir"],
                         "run_metadata_sha256": sha256(ROOT / run["run_dir"] / "run_metadata.json")}
                        for arch, group in runs.items() for run in group],
    }
    destination = ROOT / "data/processed/transformers"
    filename = "selected_model_weighted.json" if args.weighted else "selected_model.json"
    write_json(destination / filename, selection)
    write_json(destination / ("seed_summary_weighted.json" if args.weighted else "seed_summary.json"),
               {"metric_names": list(METRIC_NAMES), "ddof": 1, "summaries": summaries,
                "runs": [{"architecture": arch, "seed": r["seed"], "metrics": r["metrics"],
                          "selected_epoch": r["selected_epoch"], "run_dir": r["run_dir"]}
                         for arch, group in runs.items() for r in group]})
    lines = ["| Kiến trúc | Seed | Macro-F1 mean ± std | Micro-F1 mean ± std | Hamming mean ± std |",
             "|---|---|---|---|---|"]
    for row in summaries:
        cells = [row["architecture"], ", ".join(map(str, row["seeds"]))]
        cells += [f"{row['metrics'][key]['mean']:.4f} ± {row['metrics'][key]['std']:.4f}"
                  for key in ("macro_f1", "micro_f1", "hamming_loss")]
        lines.append("| " + " | ".join(cells) + " |")
    (destination / ("seed_summary_weighted.md" if args.weighted else "seed_summary.md")).write_text(
        "\n".join(lines) + f"\n\nBest C theo validation: **{winner}**, demo seed **{best['seed']}**.\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Demo chọn {winner}, seed {best['seed']}; lưu {destination / filename}")


if __name__ == "__main__":
    main()
