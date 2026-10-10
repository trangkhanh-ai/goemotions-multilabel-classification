"""Kiểm và khóa một thí nghiệm B/C trước khi đánh giá test.

Giữ các bước riêng: đọc scores -> ghép ID -> chọn ngưỡng trên val -> khóa.
Không dùng điểm test để chọn cấu hình.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.models.baseline import load_aligned_scores, tune_global_threshold, tune_thresholds
from src.datasets.goemotions import REVISION, load_goemotions, multi_hot, sha256
from src.evaluation.metrics import evaluate_multilabel, validate_thresholds


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_full_run(run_dir, labels):
    """Chặn smoke, nhãn khác, file bị đổi hoặc chạy chưa hoàn tất."""
    run_dir = Path(run_dir)
    metadata = read_json(run_dir / "run_metadata.json")
    if metadata.get("smoke") is not False or metadata.get("data_revision") != REVISION:
        raise ValueError("Cần thí nghiệm full đúng snapshot; smoke không được vào bảng chính")
    if metadata.get("label_names") != list(labels):
        raise ValueError("Mapping nhãn không khớp dữ liệu chung")
    if metadata.get("method") not in ("C", "zero_shot"):
        raise ValueError("File này chỉ dùng cho thí nghiệm B hoặc C")
    if metadata["method"] == "C":
        from src.models.transformer import load_transformer_run
        load_transformer_run(run_dir, require_full=True)
    elif metadata.get("status") != "complete":
        raise ValueError("Thí nghiệm zero-shot chưa hoàn tất")
    for name, expected in metadata.get("artifact_sha256", {}).items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("Đường dẫn artifact vượt ra ngoài thí nghiệm")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact đã thay đổi: {name}")
    if not (run_dir / "validation_scores.npz").is_file():
        raise ValueError("Chưa có scores validation đầy đủ")
    return metadata


def freeze_run(root, run_dir, threshold_mode="tuned"):
    """Chỉ mở validation; lưu cả ba cấu hình trước test để so sánh công bằng."""
    if threshold_mode not in ("fixed", "global", "tuned"):
        raise ValueError("threshold_mode phải là fixed/global/tuned")
    root, run_dir = Path(root), Path(run_dir)
    labels = read_json(root / "data/labels.json")
    metadata = check_full_run(run_dir, labels)
    frames, actual_labels, _ = load_goemotions(root, write_metadata=False, splits=("validation",))
    if actual_labels != labels:
        raise ValueError("Nhãn dữ liệu không khớp")
    frame = frames["validation"]
    scores = load_aligned_scores(run_dir / "validation_scores.npz", frame["id"], labels)
    truth = multi_hot(frame["labels"].tolist(), len(labels))
    global_threshold, curve = tune_global_threshold(truth, scores)
    thresholds = {"fixed": 0.5, "global": global_threshold,
                  "tuned": tune_thresholds(truth, scores).tolist()}
    configurations = [{"threshold_mode": mode, "thresholds": limits,
                       "validation_metrics": evaluate_multilabel(truth, scores, labels, limits)}
                      for mode, limits in thresholds.items()]
    checkpoint = metadata.get("checkpoint", metadata.get("model_checkpoint"))
    revision = metadata.get("model_revision", metadata.get("checkpoint_revision"))
    protocol = {"protocol_version": 1, "method": metadata["method"], "smoke": False,
                "checkpoint": checkpoint, "model_revision": revision,
                "hypothesis_template": metadata.get("hypothesis_template"),
                "label_names": labels, "data_revision": REVISION,
                "run_metadata_sha256": sha256(run_dir / "run_metadata.json"),
                "validation_scores_sha256": sha256(run_dir / "validation_scores.npz"),
                "threshold_mode": threshold_mode, "thresholds": thresholds[threshold_mode],
                "configurations": configurations, "global_threshold_curve": curve,
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection_data": "validation", "architecture": metadata.get("architecture"),
                "seed": metadata.get("seed"),
                "note": "Tuned-validation dùng cùng dữ liệu để chọn và đo ngưỡng; test là đánh giá cuối."}
    return protocol


def validate_protocol(run_dir, protocol, labels):
    """Kiểm trước khi mở test; thay model/ngưỡng sau khóa sẽ bị từ chối."""
    run_dir = Path(run_dir)
    metadata = check_full_run(run_dir, labels)
    if (protocol.get("protocol_version") != 1 or protocol.get("smoke") is not False
            or protocol.get("label_names") != list(labels)
            or protocol.get("data_revision") != REVISION
            or protocol.get("method") != metadata["method"]):
        raise ValueError("Protocol không khớp thí nghiệm")
    for field, filename in (("run_metadata_sha256", "run_metadata.json"),
                            ("validation_scores_sha256", "validation_scores.npz")):
        if protocol.get(field) != sha256(run_dir / filename):
            raise ValueError("Artifacts đã đổi sau khi khóa protocol")
    modes = set()
    for configuration in protocol.get("configurations", []):
        mode = configuration["threshold_mode"]
        if mode not in ("fixed", "global", "tuned") or mode in modes:
            raise ValueError("Cấu hình ngưỡng không hợp lệ")
        modes.add(mode)
        validate_thresholds(configuration["thresholds"], len(labels))
        if mode == "fixed" and not np.all(np.asarray(configuration["thresholds"]) == 0.5):
            raise ValueError("Cấu hình fixed phải giữ ngưỡng 0,5")
    if modes != {"fixed", "global", "tuned"}:
        raise ValueError("Protocol phải khai báo đủ ba cấu hình ngưỡng")
    validate_thresholds(protocol["thresholds"], len(labels))
    selected = next((cfg for cfg in protocol["configurations"]
                     if cfg["threshold_mode"] == protocol.get("threshold_mode")), None)
    if selected is None or not np.array_equal(
            np.broadcast_to(protocol["thresholds"], (len(labels),)),
            np.broadcast_to(selected["thresholds"], (len(labels),))):
        raise ValueError("Ngưỡng chính không khớp cấu hình đã khóa")
    if (protocol.get("checkpoint") != metadata.get("checkpoint", metadata.get("model_checkpoint"))
            or protocol.get("model_revision") != metadata.get("model_revision", metadata.get("checkpoint_revision"))):
        raise ValueError("Checkpoint/revision không khớp protocol")
    if metadata["method"] == "C" and any(protocol.get(key) != metadata.get(key) for key in ("architecture", "seed")):
        raise ValueError("Kiến trúc/seed không khớp protocol")
    return metadata
