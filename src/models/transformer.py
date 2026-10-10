"""Các bước nhỏ của phần C: cấu hình, loss, artifact và chọn mô hình.

Chỉ import PyTorch khi thực sự huấn luyện hoặc suy luận. Các hàm kiểm
artifact/chọn kiến trúc có thể kiểm thử mà không tải bất kỳ checkpoint nào.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

from src.datasets.goemotions import EXPECTED_ROWS, REVISION, sha256
from src.evaluation.metrics import validate_thresholds


ARCHITECTURES = {
    "bert": {"checkpoint": "google-bert/bert-base-cased", "epochs": 4,
             "learning_rate": 5e-5, "owner": "Quốc Khánh"},
    "roberta": {"checkpoint": "FacebookAI/roberta-base", "epochs": 3,
                "learning_rate": 2e-5, "owner": "Đức Trí"},
    "distilbert": {"checkpoint": "distilbert/distilbert-base-uncased", "epochs": 3,
                   "learning_rate": 2e-5, "owner": "Nhật Huy"},
}
METRIC_NAMES = ("macro_f1", "micro_f1", "macro_precision", "macro_recall",
                "micro_precision", "micro_recall", "hamming_loss")


def configure_console():
    """Windows pipe thường mặc định cp1252; tránh lỗi khi in hướng dẫn tiếng Việt."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def training_config(architecture, *, smoke=False, weighted=False, epochs=None,
                    learning_rate=None, batch_size=16, gradient_accumulation=1,
                    max_length=128, device="auto", revision="main"):
    """C1 dùng thông số tham khảo bài gốc; C2/C3 là cấu hình của nhóm."""
    if architecture not in ARCHITECTURES:
        raise ValueError("Kiến trúc phải là bert, roberta hoặc distilbert")
    defaults = ARCHITECTURES[architecture]
    config = {
        "checkpoint": defaults["checkpoint"], "requested_revision": revision,
        "epochs": 1 if smoke else (epochs if epochs is not None else defaults["epochs"]),
        "learning_rate": learning_rate if learning_rate is not None else defaults["learning_rate"],
        "batch_size": batch_size, "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "max_length": max_length, "padding": "dynamic_batch_trim",
        "weighted": bool(weighted), "device": device,
        "weight_decay": 0.01, "warmup_ratio": 0.1, "max_grad_norm": 1.0,
        "checkpoint_selection": "validation macro_f1 at threshold 0.5; earlier epoch on tie",
    }
    for field in ("epochs", "batch_size", "gradient_accumulation", "max_length"):
        if not isinstance(config[field], int) or config[field] < 1:
            raise ValueError(f"{field} phải là số nguyên dương")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate phải hữu hạn và dương")
    if max_length > 512 or device not in ("auto", "cpu", "cuda"):
        raise ValueError("max_length <= 512 và device là auto/cpu/cuda")
    return config


def run_folder(root, architecture, seed, *, smoke=False, weighted=False):
    return (Path(root) / "data/processed/transformers" / architecture /
            f"seed_{seed}" / ("smoke" if smoke else "full") /
            ("weighted" if weighted else "standard"))


def write_json(path, value):
    """Ghi tạm rồi thay file đích để không nhận nhầm JSON đang ghi dở."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    """Gieo seed cả Python, NumPy, CPU và CUDA; không hứa giống bit mọi máy."""
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested):
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA chưa dùng được. Kiểm PyTorch/GPU hoặc chọn --device cpu")
    return torch.device("cuda" if requested == "cuda" or
                        (requested == "auto" and torch.cuda.is_available()) else "cpu")


def positive_weights(y_train):
    """BCE pos_weight[j] = số âm / số dương, chỉ tính bằng nhãn train."""
    truth = np.asarray(y_train)
    if truth.ndim != 2 or len(truth) == 0 or not np.isin(truth, (0, 1)).all():
        raise ValueError("y_train phải là multi-hot N × L không rỗng")
    support = truth.sum(axis=0)
    # Smoke nhỏ có thể thiếu một cảm xúc; clamp tránh chia 0, không dùng làm kết quả.
    return ((len(truth) - support) / np.maximum(support, 1)).astype(np.float32)


def multilabel_loss(logits, targets, pos_weight=None):
    """Logits N×28 + nhãn float N×28 -> BCE; chưa sigmoid ở bước loss."""
    import torch
    if logits.ndim != 2 or logits.shape != targets.shape or not targets.is_floating_point():
        raise ValueError("logits và nhãn float phải cùng shape N × L")
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )


def artifact_hashes(folder):
    """Hash cả weights/tokenizer/config và điểm; metadata ghi sau cùng."""
    folder = Path(folder)
    paths = [folder / name for name in ("label_mapping.json", "validation_scores.npz",
                                        "validation_metrics.json")]
    paths += sorted((folder / "checkpoint").rglob("*"))
    return {p.relative_to(folder).as_posix(): sha256(p) for p in paths if p.is_file()}


def load_transformer_run(folder, *, require_full=False, expected_config=None):
    """Từ chối run dở, smoke dùng cho bảng cuối, nhãn sai hoặc files bị trộn."""
    folder = Path(folder).resolve()
    metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != 1 or metadata.get("method") != "C":
        raise ValueError("Artifact này không phải run Transformer C đúng phiên bản")
    if metadata.get("architecture") not in ARCHITECTURES:
        raise ValueError("Kiến trúc trong metadata không thuộc ba C đã khai báo")
    if not metadata.get("completed") or metadata.get("data_revision") != REVISION:
        raise ValueError("Run chưa hoàn tất hoặc revision dữ liệu sai")
    if require_full and (metadata.get("smoke") or metadata["sizes"] !=
                         {"train": EXPECTED_ROWS["train"], "validation": EXPECTED_ROWS["validation"]}):
        raise ValueError("Chỉ run full được dùng để chọn demo và báo cáo cuối")
    if expected_config is not None and metadata.get("config") != expected_config:
        raise ValueError("Cấu hình hiện tại khác run đã có; dùng folder riêng hoặc --overwrite")
    mapping = json.loads((folder / "label_mapping.json").read_text(encoding="utf-8"))
    labels = metadata.get("label_names", [])
    if len(labels) != 28 or len(set(labels)) != 28 or mapping.get("label_names") != labels:
        raise ValueError("Run phải giữ đúng thứ tự 28 nhãn GoEmotions")
    if mapping.get("label2id") != {name: i for i, name in enumerate(labels)}:
        raise ValueError("label2id không khớp thứ tự label_names")
    config_path = folder / "checkpoint/config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if (model_config.get("model_type") != metadata["architecture"] or
            model_config.get("problem_type") != "multi_label_classification" or
            model_config.get("id2label") != {str(i): name for i, name in enumerate(labels)}):
        raise ValueError("Head hoặc id2label của checkpoint không đúng bài toán đa nhãn")
    hashes = metadata.get("artifact_sha256", {})
    required = {"label_mapping.json", "validation_scores.npz", "validation_metrics.json",
                "checkpoint/config.json"}
    if not required.issubset(hashes) or not any(
            p.endswith((".safetensors", ".bin")) and p.startswith("checkpoint/") for p in hashes):
        raise ValueError("Thiếu hash checkpoint/điểm/mapping")
    for name, expected in hashes.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact bị thay đổi hoặc thiếu: {name}")
    with np.load(folder / "validation_scores.npz", allow_pickle=False) as saved:
        ids, scores = saved["ids"].astype(str), saved["scores"]
        if (saved["label_names"].astype(str).tolist() != labels or
                ids.ndim != 1 or len(set(ids)) != len(ids) or
                scores.shape != (metadata["sizes"]["validation"], 28) or len(ids) != len(scores) or
                not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
            raise ValueError("Scores/ID không khớp run")
    metrics = json.loads((folder / "validation_metrics.json").read_text(encoding="utf-8"))
    if metrics.get("threshold") != 0.5 or metrics.get("n_labels") != 28:
        raise ValueError("Bảng chọn C phải dùng đủ 28 nhãn với ngưỡng 0.5")
    if metrics.get("n_samples") != metadata["sizes"]["validation"]:
        raise ValueError("Metric có số mẫu khác scores validation")
    for metric in METRIC_NAMES:
        if not math.isfinite(metrics.get(metric, math.nan)) or not 0 <= metrics[metric] <= 1:
            raise ValueError(f"Metric không hợp lệ: {metric}")
    metadata["metrics"] = metrics
    return metadata


def summarize_architectures(runs_by_architecture, min_seeds=3):
    """Mean và sample std ddof=1; cùng seed/config/data giữa ba kiến trúc."""
    if min_seeds < 3:
        raise ValueError("Yêu cầu cô: ít nhất 3 seed cho mỗi kiến trúc")
    summaries, expected_seeds, expected_labels = [], None, None
    for architecture, runs in runs_by_architecture.items():
        seeds = sorted(run["seed"] for run in runs)
        if len(seeds) < min_seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"{architecture}: thiếu seed hoặc seed lặp")
        if expected_seeds is None:
            expected_seeds = seeds
            expected_labels = runs[0]["label_names"]
        if seeds != expected_seeds:
            raise ValueError("Các kiến trúc phải dùng cùng tập seed để so sánh")
        reference_config = runs[0]["config"]
        reference_revision = runs[0].get("model_revision")
        for run in runs:
            if (run.get("method") != "C" or run.get("smoke") or not run.get("completed") or
                    run["architecture"] != architecture or run["label_names"] != expected_labels or
                    run["config"] != reference_config or run["data_revision"] != REVISION or
                    run.get("model_revision") != reference_revision or
                    run["parameter_count"] != runs[0]["parameter_count"]):
                raise ValueError("Không trộn smoke, config, nhãn hoặc revision trong bảng seed")
        summaries.append({
            "architecture": architecture, "seeds": seeds, "n_seeds": len(seeds),
            "metrics": {key: {"mean": float(np.mean([r["metrics"][key] for r in runs])),
                               "std": float(np.std([r["metrics"][key] for r in runs], ddof=1))}
                        for key in METRIC_NAMES},
            "parameter_count": int(runs[0]["parameter_count"]),
        })
    # Thắng bởi mean Macro-F1, hòa dùng std nhỏ, rồi ít tham số, cuối cùng tên.
    summaries.sort(key=lambda row: (-row["metrics"]["macro_f1"]["mean"],
                                   row["metrics"]["macro_f1"]["std"],
                                   row["parameter_count"], row["architecture"]))
    return summaries


def load_demo_selection(root, selection_path, *, thresholds_path=None):
    """Demo chỉ được lấy C full đã chọn, gắn cùng hash run và mapping."""
    root = Path(root).resolve()
    selected = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selected.get("method") != "C" or selected.get("smoke") or selected.get("artifact_version") != 1:
        raise ValueError("Demo yêu cầu selected_model.json của phần C full")
    folder = (root / selected["run_dir"]).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("run_dir phải nằm trong repo")
    if sha256(folder / "run_metadata.json") != selected.get("run_metadata_sha256"):
        raise ValueError("Run đã đổi sau khi chọn mô hình demo")
    metadata = load_transformer_run(folder, require_full=True)
    if metadata["architecture"] != selected["architecture"] or metadata["seed"] != selected["seed"]:
        raise ValueError("Kiến trúc/seed trong selection không khớp checkpoint")
    if "label_names" in selected and selected["label_names"] != metadata["label_names"]:
        raise ValueError("Mapping trong selection khác checkpoint")
    limits = 0.5
    if thresholds_path:
        saved = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
        if (saved.get("run_metadata_sha256") != selected["run_metadata_sha256"] or
                saved.get("label_names") != metadata["label_names"]):
            raise ValueError("Ngưỡng phải thuộc chính run C được dùng cho demo")
        limits = validate_thresholds(saved["thresholds"], 28)
    return folder, metadata, limits
