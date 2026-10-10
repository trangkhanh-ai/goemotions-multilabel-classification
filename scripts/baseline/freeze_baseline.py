"""Khóa cả bảng thí nghiệm A trước khi mở test: python -m scripts.baseline.freeze_baseline.

Chỉ chạy khi nhóm đã thống nhất protocol cuối. Lệnh này KHÔNG mở dữ liệu test.
"""

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from src.models.baseline import load_run_metadata, load_thresholds
from src.datasets.goemotions import REVISION, sha256

from src.paths import ROOT


def build_protocol(root):
    labels = json.loads((root / "data/labels.json").read_text(encoding="utf-8"))
    runs, configurations = {}, []
    for variant in ("standard", "balanced"):
        folder = root / "data/processed/baseline"
        if variant == "balanced":
            folder /= "balanced"
        folder /= "full"
        metadata = load_run_metadata(folder, variant, labels)
        analysis = json.loads((folder / "analysis_validation.json").read_text(encoding="utf-8"))
        if analysis.get("artifact_sha256") != metadata["artifact_sha256"]:
            raise ValueError("Phân tích validation đã cũ. Chạy lại scripts.baseline.analyze_baseline.")
        runs[variant] = {
            "folder": folder.relative_to(root).as_posix(),
            "artifact_sha256": metadata["artifact_sha256"],
            "threshold_file_sha256": sha256(folder / "thresholds_validation.json"),
            "train_positive_counts": metadata["train_positive_counts"],
        }
        for mode, metric_key in (("fixed", "fixed_0_5"), ("global", "global_on_validation"),
                                 ("tuned", "tuned_on_validation")):
            thresholds = np.broadcast_to(load_thresholds(folder, metadata, mode), (len(labels),))
            configurations.append({"name": f"{variant}_{mode}", "variant": variant,
                                   "threshold_mode": mode, "thresholds": thresholds.tolist(),
                                   "validation_macro_f1": analysis[metric_key]["macro_f1"]})
    selected = max(configurations, key=lambda row: row["validation_macro_f1"])["name"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                            text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    rare_ids = np.argsort(runs["standard"]["train_positive_counts"])[:5]
    return {"schema_version": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_revision": REVISION, "label_names": labels,
            "git_head": commit, "working_tree_dirty": dirty,
            "selection_rule": "highest Macro-F1 on validation, not test",
            "selected_configuration": selected, "runs": runs,
            "rare_label_ids_from_train": rare_ids.tolist(), "configurations": configurations}


def main():
    argparse.ArgumentParser(description="Freeze six baseline configurations before opening test").parse_args()
    output = ROOT / "data/processed/baseline/final_protocol.json"
    if output.exists():
        raise FileExistsError("Protocol đã khóa. Giữ file này để mọi kết quả test cùng một protocol.")
    protocol = build_protocol(ROOT)
    output.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Frozen {len(protocol['configurations'])} configurations without opening test.")
    print(f"Selected on validation: {protocol['selected_configuration']}")
    print("Protocol: data/processed/baseline/final_protocol.json")


if __name__ == "__main__":
    main()
